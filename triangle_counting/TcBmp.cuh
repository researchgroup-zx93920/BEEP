#pragma once

#include <cstdint>
#include <driver_types.h>
#include <cuda_runtime_api.h>
#include "../include/Logger.cuh"
#include "../include/Timer.h"
#include "../include/utils_cuda.cuh"
#include "../include/cub_wrappers.cuh"

// #include <immintrin.h>
#include <cuda_runtime.h>
#include <cuda_runtime.h>
#include "../include/CGArray.cuh"
#include "../include/GraphDataStructure.cuh"



#include "omp.h"



__global__ void construct_bsr_row_ptr_per_thread(uint32_t* d_offsets, uint32_t* d_dsts,
    uint32_t num_vertices, uint32_t* bmp_offs) {
    uint32_t u = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (u >= num_vertices) return;

    constexpr int word_in_bits = sizeof(uint32_t) * 8;
    auto prev_blk_id = -1;
    auto num_blks = 0;
    for (auto off = d_offsets[u]; off < d_offsets[u + 1]; off++) {
        auto v = d_dsts[off];
        int cur_blk_id = v / word_in_bits;
        if (cur_blk_id != prev_blk_id) {
            prev_blk_id = cur_blk_id;
            num_blks++;
        }
    }
    if ((d_offsets[u + 1] - d_offsets[u]) >= 16 && (d_offsets[u + 1] - d_offsets[u]) / num_blks > 2) {
        bmp_offs[u] = num_blks;
    }
    else {
        bmp_offs[u] = 0;
    }
}

__global__ void construct_bsr_content_per_thread(uint32_t* d_offsets, uint32_t* d_dsts, uint32_t num_vertices,
    uint32_t* bmp_offs, uint* bmp_word_indices,
    uint* bmp_words) 

{
    uint32_t u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_vertices) return;

    auto size = bmp_offs[u + 1] - bmp_offs[u];
    if (size == 0) return;

    auto curr_size = 0;
    auto write_off = bmp_offs[u];
    auto bmp_word_idx_ptr = bmp_word_indices + write_off;
    auto bmp_words_ptr = bmp_words + write_off;
    auto prev_blk_id = -1;
    constexpr int word_in_bits = sizeof(uint32_t) * 8;
    for (auto off = d_offsets[u]; off < d_offsets[u + 1]; off++) {
        auto v = d_dsts[off];
        int cur_blk_id = v / word_in_bits;
        if (cur_blk_id != prev_blk_id) {
            prev_blk_id = cur_blk_id;
            bmp_words_ptr[curr_size] = 0;
            bmp_word_idx_ptr[curr_size++] = cur_blk_id;
        }
        bmp_words_ptr[curr_size - 1] |= static_cast<uint>(1u) << (v % word_in_bits);
    }
}



__global__ void bmp_kernel(uint32_t* d_offsets, /*card: |V|+1*/
    int32_t* d_dsts, /*card: 2*|E|*/
    uint32_t* d_bitmaps, /*the global bitmaps*/
    uint32_t* d_bitmap_states, /*recording the usage of the bitmaps on the SM*/
    uint32_t* vertex_count, /*for sequential block execution*/
    uint32_t conc_blocks_per_SM, /*#concurrent blocks per SM*/
    uint* eid, /*card: 2*|E|*/
    int32_t* d_intersection_count_GPU) /*card: |E|*/
{
    const uint32_t tid = threadIdx.x + blockDim.x * threadIdx.y; /*threads in a warp are with continuous threadIdx.x */
    const uint32_t tnum = blockDim.x * blockDim.y;
    const uint32_t num_nodes = gridDim.x; /*#nodes=#blocks*/
    const uint32_t elem_bits = sizeof(uint32_t) * 8; /*#bits in a bitmap element*/
    const uint32_t val_size_bitmap = (num_nodes + elem_bits - 1) / elem_bits;
    const uint32_t val_size_bitmap_indexes = (val_size_bitmap + BITMAP_SCALE - 1) >> BITMAP_SCALE_LOG;

    __shared__ uint32_t node_id, sm_id, bitmap_ptr;
    __shared__ uint32_t start_src, end_src, start_src_in_bitmap, end_src_in_bitmap;

    extern __shared__ uint32_t bitmap_indexes[];

    if (tid == 0) {
        node_id = atomicAdd(vertex_count, 1); /*get current vertex id*/
        start_src = d_offsets[node_id];
        end_src = d_offsets[node_id + 1];
        start_src_in_bitmap = d_dsts[start_src] / elem_bits;
        end_src_in_bitmap = (start_src == end_src) ? d_dsts[start_src] / elem_bits : d_dsts[end_src - 1] / elem_bits;
    }
    else if (tid == tnum - 1) {
        uint32_t temp = 0;
        asm("mov.u32 %0, %smid;" : "=r"(sm_id));
        /*get current SM*/
        while (atomicCAS(&d_bitmap_states[sm_id * conc_blocks_per_SM + temp], 0, 1) != 0)
            temp++;
        bitmap_ptr = temp;
    }
    /*initialize the 2-level bitmap*/
    for (uint32_t idx = tid; idx < val_size_bitmap_indexes; idx += tnum)
        bitmap_indexes[idx] = 0;
    __syncthreads();

    uint32_t* bitmap = &d_bitmaps[val_size_bitmap * (conc_blocks_per_SM * sm_id + bitmap_ptr)];

    /*construct the source node neighbor bitmap*/
    for (uint32_t idx = start_src + tid; idx < end_src; idx += tnum) {
        uint32_t src_nei = d_dsts[idx];
        const uint32_t src_nei_val = src_nei / elem_bits;
        atomicOr(&bitmap[src_nei_val], (0b1 << (src_nei & (elem_bits - 1)))); /*setting the bitmap*/
        atomicOr(&bitmap_indexes[src_nei_val >> BITMAP_SCALE_LOG],
            (0b1 << ((src_nei >> BITMAP_SCALE_LOG) & (elem_bits - 1)))); /*setting the bitmap index*/
    }
    __syncthreads();

    /*loop the neighbors*/
    /* x dimension: warp-size
     * y dimension: number of warps
     * */
    auto du = d_offsets[node_id + 1] - d_offsets[node_id];
    for (uint32_t idx = start_src + threadIdx.y; idx < end_src; idx += blockDim.y) {
        /*each warp processes a node*/
        uint32_t private_count = 0;
        uint32_t src_nei = d_dsts[idx];
        auto dv = d_offsets[src_nei + 1] - d_offsets[src_nei];
        if (dv > du || ((du == dv) && node_id > src_nei))continue;
        uint32_t start_dst = d_offsets[src_nei];
        uint32_t end_dst = d_offsets[src_nei + 1];
        for (uint32_t dst_idx = start_dst + threadIdx.x; dst_idx < end_dst; dst_idx += blockDim.x) {
            uint32_t dst_nei = d_dsts[dst_idx];
            const uint32_t dst_nei_val = dst_nei / elem_bits;
            if ((bitmap_indexes[dst_nei_val >> BITMAP_SCALE_LOG] >> ((dst_nei >> BITMAP_SCALE_LOG) & (elem_bits - 1)))
                & 0b1 == 1)
                if ((bitmap[dst_nei_val] >> (dst_nei & (elem_bits - 1))) & 0b1 == 1)
                    private_count++;
        }
        __syncwarp();
        /*warp-wise reduction*/
        WARP_REDUCE(private_count);
        if (threadIdx.x == 0)
            d_intersection_count_GPU[eid[idx]] = private_count;
    }
    __syncthreads();

    /*clean the bitmap*/
    if (end_src_in_bitmap - start_src_in_bitmap + 1 <= end_src - start_src) {
        for (uint32_t idx = start_src_in_bitmap + tid; idx <= end_src_in_bitmap; idx += tnum) {
            bitmap[idx] = 0;
        }
    }
    else {
        for (uint32_t idx = start_src + tid; idx < end_src; idx += tnum) {
            uint32_t src_nei = d_dsts[idx];
            bitmap[src_nei / elem_bits] = 0;
        }
    }
    __syncthreads();

    /*release the bitmap lock*/
    if (tid == 0)
        atomicCAS(&d_bitmap_states[sm_id * conc_blocks_per_SM + bitmap_ptr], 1, 0);
}

__global__ void bmp_bsr_count_kernel(uint32_t* d_offsets, /*card: |V|+1*/
    uint32_t* d_dsts, /*card: 2*|E|*/
    uint32_t* d_bitmaps, /*the global bitmaps*/
    uint32_t* d_bitmap_states, /*recording the usage of the bitmaps on the SM*/
    uint32_t* vertex_count, /*for sequential block execution*/
    uint32_t conc_blocks_per_SM, /*#concurrent blocks per SM*/
    uint* eid, /*card: 2*|E|*/
    uint32_t* bmp_offs,
    uint* bmp_word_indices,
    uint* bmp_words,
    uint64 *count
) {
    const uint32_t tid = threadIdx.x + blockDim.x * threadIdx.y; /*threads in a warp are with continuous threadIdx.x */ //per block
    const uint32_t num_threads = blockDim.x * blockDim.y; // per block
    const uint32_t num_nodes = gridDim.x; /*#nodes=#blocks*/
    
    constexpr uint32_t elem_bits = sizeof(uint32_t) * 8; /*#bits in a bitmap element*/
    const uint32_t val_size_bitmap = (num_nodes + elem_bits - 1) / elem_bits;
    const uint32_t val_size_bitmap_indexes = (val_size_bitmap + BITMAP_SCALE - 1) >> BITMAP_SCALE_LOG;

    __shared__ uint32_t u, sm_id, bitmap_ptr;
    __shared__ uint32_t off_u, off_u_plus_one, start_src_in_bitmap, end_src_in_bitmap;

    extern __shared__ uint32_t bitmap_indexes[];

    if (tid == 0) {
        u = atomicAdd(vertex_count, 1); /*get current vertex id*/
        off_u = d_offsets[u];
        off_u_plus_one = d_offsets[u + 1];
        start_src_in_bitmap = d_dsts[off_u] / elem_bits;
        end_src_in_bitmap = (off_u == off_u_plus_one) ? d_dsts[off_u] / elem_bits :
            d_dsts[off_u_plus_one - 1] / elem_bits;
    }
    else if (tid == num_threads - 1) {
        uint32_t temp = 0;
        asm("mov.u32 %0, %smid;" : "=r"(sm_id));
        /*get current SM*/
        while (atomicCAS(&d_bitmap_states[sm_id * conc_blocks_per_SM + temp], 0, 1) != 0)
            temp++;
        bitmap_ptr = temp;
    }
    /*initialize the 2-level bitmap*/
    for (uint32_t idx = tid; idx < val_size_bitmap_indexes; idx += num_threads)
        bitmap_indexes[idx] = 0;
    __syncthreads();

    uint32_t* bitmap = &d_bitmaps[val_size_bitmap * (conc_blocks_per_SM * sm_id + bitmap_ptr)];

    /*construct the source node neighbor bitmap*/
    for (uint32_t idx = off_u + tid; idx < off_u_plus_one; idx += num_threads) {
        uint32_t v = d_dsts[idx];
        const uint32_t src_nei_val = v / elem_bits;
        atomicOr(&bitmap[src_nei_val], (0b1 << (v & (elem_bits - 1)))); /*setting the bitmap*/
        atomicOr(&bitmap_indexes[src_nei_val >> BITMAP_SCALE_LOG],
            (0b1 << ((v >> BITMAP_SCALE_LOG) & (elem_bits - 1)))); /*setting the bitmap index*/
    }
    __syncthreads();

    auto du = d_offsets[u + 1] - d_offsets[u];
    for (uint32_t idx = off_u + threadIdx.y; idx < off_u_plus_one; idx += blockDim.y) {
        uint32_t v = d_dsts[idx];

        /*each warp processes an edge (u, v), v: v */
        auto dv = d_offsets[v + 1] - d_offsets[v];
        //if (dv > du || ((du == dv) && u > v))continue; //for full graph

        uint64 private_count = 0;
        auto size_nv = bmp_offs[v + 1] - bmp_offs[v];
        if (size_nv > 0) {
            for (uint32_t wi = bmp_offs[v] + threadIdx.x; wi < bmp_offs[v + 1]; wi += blockDim.x) {
                private_count += __popc(bmp_words[wi] & bitmap[bmp_word_indices[wi]]);
            }
        }
        else {
            for (uint32_t dst_idx = d_offsets[v] + threadIdx.x; dst_idx < d_offsets[v + 1]; dst_idx += blockDim.x) {
                uint32_t w = d_dsts[dst_idx];
                const uint32_t dst_nei_val = w / elem_bits;
                if ((bitmap_indexes[dst_nei_val >> BITMAP_SCALE_LOG]
                    >> ((w >> BITMAP_SCALE_LOG) & (elem_bits - 1))) & 0b1 == 1)
                    if ((bitmap[dst_nei_val] >> (w & (elem_bits - 1))) & 0b1 == 1)
                        private_count++;
            }
        }

        __syncwarp();
        /*warp-wise reduction*/
        WARP_REDUCE(private_count);
        if (threadIdx.x == 0)
        {
            atomicAdd(count, private_count);
        }
    }
    __syncthreads();

    /*clean the bitmap*/
    if (end_src_in_bitmap - start_src_in_bitmap + 1 <= off_u_plus_one - off_u) {
        for (uint32_t idx = start_src_in_bitmap + tid; idx <= end_src_in_bitmap; idx += num_threads) {
            bitmap[idx] = 0;
        }
    }
    else {
        for (uint32_t idx = off_u + tid; idx < off_u_plus_one; idx += num_threads) {
            uint32_t src_nei = d_dsts[idx];
            bitmap[src_nei / elem_bits] = 0;
        }
    }
    __syncthreads();

    /*release the bitmap lock*/
    if (tid == 0)
        atomicCAS(&d_bitmap_states[sm_id * conc_blocks_per_SM + bitmap_ptr], 1, 0);

}




__global__ void bmp_bsr_kernel(uint32_t* d_offsets, /*card: |V|+1*/
    uint32_t* d_dsts, /*card: 2*|E|*/
    uint32_t* d_bitmaps, /*the global bitmaps*/
    uint32_t* d_bitmap_states, /*recording the usage of the bitmaps on the SM*/
    uint32_t* vertex_count, /*for sequential block execution*/
    uint32_t conc_blocks_per_SM, /*#concurrent blocks per SM*/
    uint* eid, /*card: 2*|E|*/
    int32_t* d_intersection_count_GPU,
    uint32_t* bmp_offs,
    uint* bmp_word_indices,
    uint* bmp_words
) {
    const uint32_t tid = threadIdx.x + blockDim.x * threadIdx.y; /*threads in a warp are with continuous threadIdx.x */ //per block
    const uint32_t num_threads = blockDim.x * blockDim.y; // per block
    const uint32_t num_nodes = gridDim.x; /*#nodes=#blocks*/

    constexpr uint32_t elem_bits = sizeof(uint32_t) * 8; /*#bits in a bitmap element*/
    const uint32_t val_size_bitmap = (num_nodes + elem_bits - 1) / elem_bits;
    const uint32_t val_size_bitmap_indexes = (val_size_bitmap + BITMAP_SCALE - 1) >> BITMAP_SCALE_LOG;

    __shared__ uint32_t u, sm_id, bitmap_ptr;
    __shared__ uint32_t off_u, off_u_plus_one, start_src_in_bitmap, end_src_in_bitmap;

    extern __shared__ uint32_t bitmap_indexes[];

    if (tid == 0) {
        u = atomicAdd(vertex_count, 1); /*get current vertex id*/
        off_u = d_offsets[u];
        off_u_plus_one = d_offsets[u + 1];
        start_src_in_bitmap = d_dsts[off_u] / elem_bits;
        end_src_in_bitmap = (off_u == off_u_plus_one) ? d_dsts[off_u] / elem_bits :
            d_dsts[off_u_plus_one - 1] / elem_bits;
    }
    else if (tid == num_threads - 1) {
        uint32_t temp = 0;
        asm("mov.u32 %0, %smid;" : "=r"(sm_id));
        /*get current SM*/
        while (atomicCAS(&d_bitmap_states[sm_id * conc_blocks_per_SM + temp], 0, 1) != 0)
            temp++;
        bitmap_ptr = temp;
    }
    /*initialize the 2-level bitmap*/
    for (uint32_t idx = tid; idx < val_size_bitmap_indexes; idx += num_threads)
        bitmap_indexes[idx] = 0;
    __syncthreads();

    uint32_t* bitmap = &d_bitmaps[val_size_bitmap * (conc_blocks_per_SM * sm_id + bitmap_ptr)];

    /*construct the source node neighbor bitmap*/
    for (uint32_t idx = off_u + tid; idx < off_u_plus_one; idx += num_threads) {
        uint32_t v = d_dsts[idx];
        const uint32_t src_nei_val = v / elem_bits;
        atomicOr(&bitmap[src_nei_val], (0b1 << (v & (elem_bits - 1)))); /*setting the bitmap*/
        atomicOr(&bitmap_indexes[src_nei_val >> BITMAP_SCALE_LOG],
            (0b1 << ((v >> BITMAP_SCALE_LOG) & (elem_bits - 1)))); /*setting the bitmap index*/
    }
    __syncthreads();

    auto du = d_offsets[u + 1] - d_offsets[u];
    for (uint32_t idx = off_u + threadIdx.y; idx < off_u_plus_one; idx += blockDim.y) {
        uint32_t v = d_dsts[idx];

        /*each warp processes an edge (u, v), v: v */
        auto dv = d_offsets[v + 1] - d_offsets[v];
        if (dv > du || ((du == dv) && u > v))continue; //for full graph

        uint32_t private_count = 0;
        auto size_nv = bmp_offs[v + 1] - bmp_offs[v];
        if (size_nv > 0) {
            for (uint32_t wi = bmp_offs[v] + threadIdx.x; wi < bmp_offs[v + 1]; wi += blockDim.x) {
                private_count += __popc(bmp_words[wi] & bitmap[bmp_word_indices[wi]]);
            }
        }
        else {
            for (uint32_t dst_idx = d_offsets[v] + threadIdx.x; dst_idx < d_offsets[v + 1]; dst_idx += blockDim.x) {
                uint32_t w = d_dsts[dst_idx];
                const uint32_t dst_nei_val = w / elem_bits;
                if ((bitmap_indexes[dst_nei_val >> BITMAP_SCALE_LOG]
                    >> ((w >> BITMAP_SCALE_LOG) & (elem_bits - 1))) & 0b1 == 1)
                    if ((bitmap[dst_nei_val] >> (w & (elem_bits - 1))) & 0b1 == 1)
                        private_count++;
            }
        }

        __syncwarp();
        /*warp-wise reduction*/
        WARP_REDUCE(private_count);
        if (threadIdx.x == 0)
        {
            d_intersection_count_GPU[eid[idx]] = private_count;
        }
    }
    __syncthreads();

    /*clean the bitmap*/
    if (end_src_in_bitmap - start_src_in_bitmap + 1 <= off_u_plus_one - off_u) {
        for (uint32_t idx = start_src_in_bitmap + tid; idx <= end_src_in_bitmap; idx += num_threads) {
            bitmap[idx] = 0;
        }
    }
    else {
        for (uint32_t idx = off_u + tid; idx < off_u_plus_one; idx += num_threads) {
            uint32_t src_nei = d_dsts[idx];
            bitmap[src_nei / elem_bits] = 0;
        }
    }
    __syncthreads();

    /*release the bitmap lock*/
    if (tid == 0)
        atomicCAS(&d_bitmap_states[sm_id * conc_blocks_per_SM + bitmap_ptr], 1, 0);

}


template<typename T>
uint32_t LinearSearch(T* array, uint32_t offset_beg, uint32_t offset_end, T val) {
    // linear search fallback
    for (auto offset = offset_beg; offset < offset_end; offset++) {
        if (array[offset] >= val) {
            return offset;
        }
    }
    return offset_end;
}

template<typename T>
uint32_t BranchFreeBinarySearch(T* a, uint32_t offset_beg, uint32_t offset_end, T x) {
    int32_t n = offset_end - offset_beg;
    using I = uint32_t;
    const T* base = a + offset_beg;
    while (n > 1) {
        I half = n / 2;
        // _mm_prefetch((char*)(base + half / 2), _MM_HINT_T0);
        // _mm_prefetch((char*)(base + half + half / 2), _MM_HINT_T0);
        base = (base[half] < x) ? base + half : base;
        n -= half;
    }
    return (*base < x) + base - a;
}

// require sizeof(T) to be 4
template<typename T>
uint32_t BinarySearchForGallopingSearch(const T* array, uint32_t offset_beg, uint32_t offset_end, T val) {
    while (offset_end - offset_beg >= 32) {
        auto mid = static_cast<uint32_t>((static_cast<unsigned long>(offset_beg) + offset_end) / 2);
        // _mm_prefetch((char*)&array[(static_cast<unsigned long>(mid + 1) + offset_end) / 2], _MM_HINT_T0);
        // _mm_prefetch((char*)&array[(static_cast<unsigned long>(offset_beg) + mid) / 2], _MM_HINT_T0);
        if (array[mid] == val) {
            return mid;
        }
        else if (array[mid] < val) {
            offset_beg = mid + 1;
        }
        else {
            offset_end = mid;
        }
    }

    // linear search fallback
    for (auto offset = offset_beg; offset < offset_end; offset++) {
        if (array[offset] >= val) {
            return offset;
        }
    }
    return offset_end;
}

// Assuming (offset_beg != offset_end)
template<typename T>
uint32_t GallopingSearch(T* array, uint32_t offset_beg, uint32_t offset_end, T val) {
    if (array[offset_end - 1] < val) {
        return offset_end;
    }
    // galloping
    if (array[offset_beg] >= val) {
        return offset_beg;
    }
    if (array[offset_beg + 1] >= val) {
        return offset_beg + 1;
    }
    if (array[offset_beg + 2] >= val) {
        return offset_beg + 2;
    }

    auto jump_idx = 4u;
    while (true) {
        auto peek_idx = offset_beg + jump_idx;
        if (peek_idx >= offset_end) {
            return BranchFreeBinarySearch(array, (jump_idx >> 1u) + offset_beg + 1, offset_end, val);
        }
        if (array[peek_idx] < val) {
            jump_idx <<= 1u;
        }
        else {
            return array[peek_idx] == val ? peek_idx :
                BranchFreeBinarySearch(array, (jump_idx >> 1u) + offset_beg + 1, peek_idx + 1, val);
        }
    }
}


template<typename T>
inline int FindSrc(int n, int m, graph::GPUArray<T> rowPtr, graph::GPUArray<T> colInd, int u, uint32_t edge_idx) {
    if (edge_idx >= rowPtr.cdata()[u + 1]) {
        // update last_u, preferring galloping instead of binary search because not large range here
        u = GallopingSearch(rowPtr.cdata(), static_cast<uint32_t>(u) + 1, n + 1, edge_idx);
        // 1) first > , 2) has neighbor
        if (rowPtr.cdata()[u] > edge_idx) {
            while (rowPtr.cdata()[u] - rowPtr.cdata()[u - 1] == 0) { u--; }
            u--;
        }
        else {
            // g->num_edges[u] == i
            while (rowPtr.cdata()[u + 1] - rowPtr.cdata()[u] == 0) {
                u++;
            }
        }
    }
    return u;
}


//
//struct Edge {
//    uint u;
//    uint v;
//
//    __host__ __device__
//        Edge() {
//        this->u = 0;
//        this->v = 0;
//    }
//
//    __host__ __device__
//        Edge(uint u, uint v) {
//        this->u = u;
//        this->v = v;
//    }
//};


namespace graph {

    template<typename T>
    class BmpGpu
    {

       


    public:
        int deviceId;
        GPUArray<T> d_bitmaps;
        GPUArray<T> d_bitmap_states;
        GPUArray<T> d_vertex_count;
        GPUArray<T> bmp_offs;

        uint conc_blocks_per_SM;

        T* bmp_word_indices;
        T* bmp_words;

        GPUArray<int> edge_sup_gpu;
        GPUArray<Edge> idToEdge;
        GPUArray<T> eid;

        void getEidAndEdgeList(graph::COOCSRGraph<T> g) 
        {
            Timer t;

            T m = g.numEdges;
            T n = g.numNodes;

            //Allocate space for eid -- size g->m
            idToEdge.initialize("id2edge", AllocationTypeEnum::unified, m / 2, deviceId);
            eid.initialize("EID", AllocationTypeEnum::unified, m, deviceId);

            //Edge upper_tri_start of each edge
            auto* num_edges_copy = (T*)malloc((n + 1) * sizeof(T));
            assert(num_edges_copy != nullptr);

            auto* upper_tri_start = (T*)malloc(n * sizeof(T));

            num_edges_copy[0] = 0;
            #pragma omp parallel 
            {
                #pragma omp for
                // Histogram (Count).
                for (int u = 0; u < n; u++) {
                    upper_tri_start[u] = (g.rowPtr->cdata()[u + 1] - g.rowPtr->cdata()[u] > 256)
                        ? GallopingSearch<T>(g.colInd->cdata(), g.rowPtr->cdata()[u], g.rowPtr->cdata()[u + 1], u)
                        : LinearSearch<T>(g.colInd->cdata(), g.rowPtr->cdata()[u], g.rowPtr->cdata()[u + 1], u);
                    num_edges_copy[u + 1] = g.rowPtr->cdata()[u + 1] - upper_tri_start[u];
                }
                // Scan.
                #pragma omp single
                {
                    Timer local_timer;
                    for (auto i = 0; i < n; i++) {
                        num_edges_copy[i + 1] += num_edges_copy[i];
                    }
                    Log(LogPriorityEnum::info, "SCAN Time: %.9lf s", local_timer.elapsed());
                }

                // Transform.
                auto u = 0;
                #pragma omp for schedule(dynamic, 6000)
                for (int j = 0; j < m; j++) 
                {
                    u = FindSrc(n,m,*g.rowPtr,*g.colInd, u, j);
                    if (j < upper_tri_start[u]) 
                    {
                        auto v = g.colInd->cdata()[j];
                        auto offset = BranchFreeBinarySearch<T>(g.colInd->cdata(), g.rowPtr->cdata()[v], g.rowPtr->cdata()[v + 1], u);
                        auto eids = num_edges_copy[v] + (offset - upper_tri_start[v]);
                        eid.cdata()[j] = eids;
                        idToEdge.cdata()[eids] = std::make_pair(v, u);
                    }
                    else 
                    {
                        eid.cdata()[j] = num_edges_copy[u] + (j - upper_tri_start[u]);
                    }
                }
            }


            eid.switch_to_gpu(0);
            idToEdge.switch_to_gpu(0);
            free(upper_tri_start);

            free(num_edges_copy);

            #pragma omp single
            Log(LogPriorityEnum::info, "EID and Edge list: %.9lfs", t.elapsed());
        }
        BmpGpu(int d)
        {
            deviceId = d;

        }

        void InitBMP(
           graph::COOCSRGraph_d<T> g
        )
        {
            Timer t;
            T m = g.numEdges;
            T n = g.numNodes;
            CUDAContext context;
            const uint32_t elem_bits = sizeof(uint32_t) * 8; /*#bits in a bitmap element*/
            const uint32_t num_words_bmp = (n + elem_bits - 1) / elem_bits;

            auto num_SMs = context.num_SMs;
            auto conc_blocks_per_SM = context.GetConCBlocks(512);

            /*initialize the bitmaps*/
            d_bitmaps.initialize("bmp bitmap", AllocationTypeEnum::gpu, conc_blocks_per_SM * num_SMs * num_words_bmp, deviceId);
            d_bitmaps.setAll(0,true);

            ///*initialize the bitmap states*/
            d_bitmap_states.initialize("bmp bitmap stats", AllocationTypeEnum::gpu, num_SMs * conc_blocks_per_SM, deviceId);
            d_bitmap_states.setAll(0, true);

                ///vertex count for sequential block execution
            d_vertex_count.initialize("d_vertex count", AllocationTypeEnum::gpu, 1, deviceId);
            d_vertex_count.setAll(0, true);

            Log(LogPriorityEnum::info, "BMP Inialization: %.9lfs", t.elapsed());
 
        }
  


        void bmpConstruct(graph::COOCSRGraph_d<T> g, AllocationTypeEnum at)
        {
            
            Timer t;
            T m = g.numEdges;
            T n = g.numNodes;
            bmp_offs.initialize("bmp offset", AllocationTypeEnum::unified, n + 1, deviceId);
            edge_sup_gpu.initialize("Edge Support", AllocationTypeEnum::unified, m/2, deviceId);


           execKernel(construct_bsr_row_ptr_per_thread, 
                (n + 127) / 128, 
                128,
               deviceId,
                true, 
                g.rowPtr, g.colInd, n, bmp_offs.gdata());


            auto word_num = CUBScanExclusive<T, T>(bmp_offs.gdata(), bmp_offs.gdata(), n, deviceId, 0, unified);
            if (word_num > 0)
            {
                bmp_offs.setSingle(n, word_num,true); //unified !!
                Log(LogPriorityEnum::info, "Word Num: %d", word_num);

                cudaMalloc(&bmp_word_indices, sizeof(T) * word_num);
                cudaMalloc(&bmp_words, sizeof(T) * word_num);
                execKernel(construct_bsr_content_per_thread, (n + 127) / 128, 128, deviceId,
                    true, g.rowPtr, g.colInd, n, bmp_offs.gdata(), bmp_word_indices, bmp_words);


                //Log(LogPriorityEnum::info, "Finish BSR construction");
            }
            else
                Log(LogPriorityEnum::warn, "No BSR blocks");



            bmp_offs.advicePrefetch(true);

            Log(LogPriorityEnum::info, "BMP Construction: %.9lfs", t.elapsed());
        }


        double Count_Set(graph::COOCSRGraph_d<T> g)
        {
            CUDA_RUNTIME(cudaSetDevice(deviceId));
            
            Timer t;
            T m = g.numEdges;
            T n = g.numNodes;

            const T elem_bits = sizeof(T) * 8; /*#bits in a bitmap element*/
            const T num_words_bmp = (n + elem_bits - 1) / elem_bits;
            const T num_word_bmp_idx = (num_words_bmp + BITMAP_SCALE - 1) / BITMAP_SCALE;


            //printf("%d, %u, %u, %u\n", n, elem_bits, num_words_bmp, num_word_bmp_idx);

            uint32_t block_size = 512; // maximally reduce the number of bitmaps
            dim3 t_dimension(32, block_size / 32); /*2-D*/
            CUDAContext context;
            conc_blocks_per_SM = context.GetConCBlocks(block_size);

            uint* count;
            CUDA_RUNTIME(cudaMallocManaged(&count, sizeof(*count)));

            
            execKernelDynamicAllocation(bmp_bsr_kernel, n, t_dimension,
                num_word_bmp_idx * sizeof(uint32_t), deviceId, false,
                g.rowPtr, g.colInd, d_bitmaps.gdata(), d_bitmap_states.gdata(),
                d_vertex_count.gdata(), conc_blocks_per_SM, eid.gdata(), edge_sup_gpu.gdata(),
                bmp_offs.gdata(), bmp_word_indices, bmp_words);
            CUDA_RUNTIME(cudaDeviceSynchronize());    // ensure the kernel execution finish

           // // 4th: Free Memory.
            d_bitmaps.freeGPU();
            d_bitmap_states.freeGPU();
            d_vertex_count.freeGPU();

          
            Log(LogPriorityEnum::info, "TC Count = %u, End-To-End Time: %.9lfs", *count, t.elapsed());

            return t.elapsed();

        }


        double Count(graph::COOCSRGraph_d<T> g)
        {
            CUDA_RUNTIME(cudaSetDevice(deviceId));

            T m = g.numEdges;
            T n = g.numNodes;
            uint64* count;
            CUDA_RUNTIME(cudaMallocManaged(&count, sizeof(*count)));
            uint32_t block_size = 512; // maximally reduce the number of bitmaps
            dim3 t_dimension(32, block_size / 32); /*2-D*/
            CUDAContext context;
            conc_blocks_per_SM = context.GetConCBlocks(block_size);


            Log<false>(LogPriorityEnum::info, "Kernel Time= ");
            Timer t;
            const T elem_bits = sizeof(T) * 8; /*#bits in a bitmap element*/
            const T num_words_bmp = (n + elem_bits - 1) / elem_bits;
            const T num_word_bmp_idx = (num_words_bmp + BITMAP_SCALE - 1) / BITMAP_SCALE;
            execKernelDynamicAllocation(bmp_bsr_count_kernel, n, t_dimension,
                num_word_bmp_idx * sizeof(uint32_t), deviceId,true,
                g.rowPtr, g.colInd, d_bitmaps.gdata(), d_bitmap_states.gdata(),
                d_vertex_count.gdata(), conc_blocks_per_SM, eid.gdata(),
                bmp_offs.gdata(), bmp_word_indices, bmp_words, count);
            CUDA_RUNTIME(cudaDeviceSynchronize());    // ensure the kernel execution finished

            // // 4th: Free Memory.
            d_bitmaps.freeGPU();
            d_bitmap_states.freeGPU();
            d_vertex_count.freeGPU();

           printf("s Count = %u\n", *count);

            return t.elapsed();

        }

      };
}