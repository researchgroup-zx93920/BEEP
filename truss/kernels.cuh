#pragma once
#include <cuda_runtime.h>
#include "../include/utils_cuda.cuh"
#include "../include/defs.cuh"

#define INT_INVALID  (INT32_MAX)

#define LEVEL_SKIP_SIZE (16)
#define KCL_NODE_LEVEL_SKIP_SIZE (1024)
#define KCL_EDGE_LEVEL_SKIP_SIZE (1024)

#define INBUCKET_BOOL
#ifndef INBUCKET_BOOL
using InBucketWinType = int;
#define InBucketTrue (1)
#define InBucketFalse (0)
#else
using InBucketWinType = bool;
#define InBucketTrue (true)
#define InBucketFalse (false)
#endif

template<typename DataType, typename CntType>
__global__
void init_asc(DataType* data, CntType count) {
    auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gtid < count) data[gtid] = (DataType)gtid;
}


static __inline__ __device__ bool atomicCASBool(bool* address, bool compare, bool val) {
    unsigned long long addr = (unsigned long long) address;
    unsigned pos = addr & 3;  // byte position within the int
    int* int_addr = (int*)(addr - pos);  // int-aligned address
    int old = *int_addr, assumed, ival;

    do {
        assumed = old;
        if (val)
            ival = old | (1 << (8 * pos));
        else
            ival = old & (~((0xFFU) << (8 * pos)));
        old = atomicCAS(int_addr, assumed, ival);
    } while (assumed != old);

    return (bool)(old & ((0xFFU) << (8 * pos)));
}


/*
 * Expensive operation: atomic update of a single address, totally synchronized.
 */
__inline__ __device__
void process_support(
    uint32_t edge_idx, int level, int* EdgeSupport,
    int* next, int* next_cnt, bool* inNext,
    InBucketWinType* in_bucket_window_, int* bucket_buf_, int* window_bucket_buf_size_,
    int bucket_level_end_) 
{
    auto cur = atomicSub(&EdgeSupport[edge_idx], 1);
    if (cur == (level + 1)) {
        auto insert_idx = atomicAdd(next_cnt, 1);
        next[insert_idx] = edge_idx;
        inNext[edge_idx] = true;
    }
    if (cur <= level) {
        atomicAdd(&EdgeSupport[edge_idx], 1);
    }

#ifndef LEGACY_SCAN
    // Update the Bucket.
    auto latest = cur - 1;
    if (latest > level && latest < bucket_level_end_) {
        auto old_token = atomicCASBool(in_bucket_window_ + edge_idx, InBucketFalse, InBucketTrue);
        if (!old_token) {
            auto insert_idx = atomicAdd(window_bucket_buf_size_, 1);
            bucket_buf_[insert_idx] = edge_idx;
        }
    }
#endif
}

/*
 * Expensive operation: relatively random access of inCurr and processed and afterwards EdgeSupport.
 */
__inline__ __device__
void PeelTriangle(
    int level, bool* inCurr,
    int* next, int* next_cnt, bool* inNext, //next_cnt is init as 0
    int* EdgeSupport, bool* processed,
    InBucketWinType* in_bucket_window_, int* bucket_buf_, int* window_bucket_buf_size_,
    int bucket_level_end_,
    uint e1_idx, uint e2_idx, uint e3_idx) 
{
    bool is_peel_e2 = !inCurr[e2_idx];
    bool is_peel_e3 = !inCurr[e3_idx];

    if (is_peel_e2 || is_peel_e3) {
        if ((!processed[e2_idx]) && (!processed[e3_idx])) 
        {
            if (is_peel_e2 && is_peel_e3) {
                process_support(e2_idx, level, EdgeSupport, next, next_cnt, inNext,
                    in_bucket_window_, bucket_buf_, window_bucket_buf_size_, bucket_level_end_);
                process_support(e3_idx, level, EdgeSupport, next, next_cnt, inNext,
                    in_bucket_window_, bucket_buf_, window_bucket_buf_size_, bucket_level_end_);

            }
            else if (is_peel_e2) {
                if (e1_idx < e3_idx) {
                    process_support(e2_idx, level, EdgeSupport, next, next_cnt, inNext,
                        in_bucket_window_, bucket_buf_, window_bucket_buf_size_,
                        bucket_level_end_);
                }
            }
            else {
                if (e1_idx < e2_idx) {
                    process_support(e3_idx, level, EdgeSupport, next, next_cnt, inNext,
                        in_bucket_window_, bucket_buf_, window_bucket_buf_size_,
                        bucket_level_end_);
                }
            }
        }
    }
}

template<typename NeedleType, typename HaystackType>
__device__
int binary_search(
    NeedleType needle, HaystackType* haystacks,
    int hay_begin, int hay_end) {
    while (hay_begin <= hay_end) {
        int middle = hay_begin + (hay_end - hay_begin) / 2;
        if (needle > haystacks[middle])
            hay_begin = middle + 1;
        else if (needle < haystacks[middle])
            hay_end = middle - 1;
        else
            return middle;
    }
    return INT_INVALID;  //not found
}

__global__
void sub_level_process(
    int level, int* curr, uint32_t curr_cnt, bool* inCurr,
    int* next, int* next_cnt, bool* inNext, //next_cnt is init as 0
    uint* offsets, uint* adj, uint* eid,
    Edge* edge_list, int* EdgeSupport, bool* processed,
    InBucketWinType* in_bucket_window_, int* bucket_buf_, int* window_bucket_buf_size_,
    int bucket_level_end_) {

    auto tid = threadIdx.x;
    auto tnum = blockDim.x;
    auto bid = blockIdx.x;
    auto bnum = gridDim.x;

    __shared__ int size;
    extern __shared__ int shared[];
    int* e1_arr = shared;
    int* e2_arr = shared + tnum * 2;
    int* e3_arr = shared + tnum * 2 * 2;
    if (tid == 0) {
        size = 0;
    }
    __syncthreads();
    /*block-wise process*/
    for (auto i = bid; i < curr_cnt; i += bnum) {
        auto e1_idx = curr[i];
        Edge e1 = edge_list[e1_idx];
        uint u = e1.first;
        uint v = e1.second;

        int u_start = offsets[u], u_end = offsets[u + 1];
        int v_start = offsets[v], v_end = offsets[v + 1];

        if (u_end - u_start > v_end - v_start) {
            swap_ele(u, v);
            swap_ele(u_start, v_start);
            swap_ele(u_end, v_end);
        }

        /*u neighbor set is smaller than v neighbor*/
        for (auto t = u_start + tid; t < u_start + (u_end - u_start + tnum - 1) / tnum * tnum; t += tnum) 
        {
            __syncthreads();
            if (size >= tnum) {
                for (auto i = tid; i < size; i += tnum) {
                    auto e1_idx = e1_arr[i];
                    auto e2_idx = eid[e2_arr[i]];
                    auto e3_idx = eid[e3_arr[i]];

                    PeelTriangle(level, inCurr,
                        next, next_cnt, inNext, //next_cnt is init as 0
                        EdgeSupport, processed,
                        in_bucket_window_, bucket_buf_, window_bucket_buf_size_, bucket_level_end_,
                        e1_idx, e2_idx, e3_idx);
                }
                __syncthreads();
                if (tid == 0) {
                    size = 0;
                }
                __syncthreads();
            }

            int match = t >= u_end ? INT_INVALID : binary_search(adj[t], adj, v_start, v_end - 1);
            if (match != INT_INVALID) {
                auto pos = atomicAdd(&size, 1);
                e1_arr[pos] = e1_idx;
                e2_arr[pos] = t;
                e3_arr[pos] = match;
            }
        }
    }
    __syncthreads();
    for (auto i = tid; i < size; i += tnum) {
        auto e1_idx = e1_arr[i];
        auto e2_idx = eid[e2_arr[i]];
        auto e3_idx = eid[e3_arr[i]];

        PeelTriangle(level, inCurr,
            next, next_cnt, inNext, //next_cnt is init as 0
            EdgeSupport, processed,
            in_bucket_window_, bucket_buf_, window_bucket_buf_size_, bucket_level_end_,
            e1_idx, e2_idx, e3_idx);
    }
}




__global__
void sub_level_process2(
    int level, int* curr, uint32_t curr_cnt, bool* inCurr,
    int* next, int* next_cnt, bool* inNext, //next_cnt is init as 0
    uint* offsets, uint* adj, uint* eid,
    Edge* edge_list, int* EdgeSupport, bool* processed,
    InBucketWinType* in_bucket_window_, uint* bucket_buf_, uint32_t* window_bucket_buf_size_,
    int bucket_level_end_, int max) {

    auto tid = threadIdx.x;
    auto tnum = blockDim.x;
    auto bid = blockIdx.x;
    auto bnum = gridDim.x;

    __shared__ int size;
    extern __shared__ int shared[];
    int* e1_arr = shared;
    int* e2_arr = shared + tnum * 2;
    int* e3_arr = shared + tnum * 2 * 2;
    if (tid == 0) {
        size = 0;
    }
    __syncthreads();
    /*block-wise process*/
    for (auto i = bid; i < curr_cnt; i += bnum) 
    {
        auto e1_idx = curr[i];
        Edge e1 = edge_list[e1_idx];
        uint u = e1.first;
        uint v = e1.second;

        int u_start = offsets[u], u_end = offsets[u + 1];
        int v_start = offsets[v], v_end = offsets[v + 1];

        if (u_end - u_start > v_end - v_start) {
            swap_ele(u, v);
            swap_ele(u_start, v_start);
            swap_ele(u_end, v_end);
        }

        ///*u neighbor set is smaller than v neighbor*/
        for (auto t = u_start + tid; t < u_start + (u_end - u_start + tnum - 1) / tnum * tnum; t += tnum)
        {
            __syncthreads();
            if (size >= tnum) {
                //for (auto i = tid; i < size; i += tnum) {
                //    auto e1_idx = e1_arr[i];
                //    auto e2_idx = eid[e2_arr[i]];
                //    auto e3_idx = eid[e3_arr[i]];

                //    PeelTriangle(level, inCurr,
                //        next, next_cnt, inNext, //next_cnt is init as 0
                //        EdgeSupport, processed,
                //        in_bucket_window_, bucket_buf_, window_bucket_buf_size_, bucket_level_end_,
                //        e1_idx, e2_idx, e3_idx);
                //}
                //__syncthreads();
                //if (tid == 0) {
                //    size = 0;
                //}
                //__syncthreads();
            }


            if (t > max )
                printf("Index out of range (search val) %d, %d\n", t , max);
           
            int match = t >= u_end ? INT_INVALID : binary_search(adj[t], adj, v_start, v_end - 1);
           /* if (match != INT_INVALID) {
                auto pos = atomicAdd(&size, 1);
                e1_arr[pos] = e1_idx;
                e2_arr[pos] = t;
                e3_arr[pos] = match;
            }*/
        }
    }
    __syncthreads();
    //for (auto i = tid; i < size; i += tnum) {
    //    auto e1_idx = e1_arr[i];
    //    auto e2_idx = eid[e2_arr[i]];
    //    auto e3_idx = eid[e3_arr[i]];

    //    PeelTriangle(level, inCurr,
    //        next, next_cnt, inNext, //next_cnt is init as 0
    //        EdgeSupport, processed,
    //        in_bucket_window_, bucket_buf_, window_bucket_buf_size_, bucket_level_end_,
    //        e1_idx, e2_idx, e3_idx);
    //}
}

template<typename T>
__global__
void update_processed(T* curr, T curr_cnt, bool* inCurr, bool* processed) {
    auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gtid < curr_cnt) {
        auto edge_off = curr[gtid];
        processed[edge_off] = true;
        inCurr[edge_off] = false;
    }
}




__global__
void output_edge_support(
    int* output, int* curr, uint32_t curr_cnt,
    uint* edge_off_origin, uint start_pos) {
    auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gtid < curr_cnt) {
        output[gtid + start_pos] = edge_off_origin[curr[gtid]];
    }
}

template<typename T>
__global__
void warp_detect_deleted_edges(
    T* old_offsets, T old_offset_cnt,
    T* eid, bool* old_processed,
    T* histogram, bool* focus) 
{

    __shared__ uint32_t cnts[WARPS_PER_BLOCK];

    auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
    auto gtnum = blockDim.x * gridDim.x;
    auto gwid = gtid >> WARP_BITS;
    auto gwnum = gtnum >> WARP_BITS;
    auto lane = threadIdx.x & WARP_MASK;
    auto lwid = threadIdx.x >> WARP_BITS;

    for (auto u = gwid; u < old_offset_cnt; u += gwnum) {
        if (0 == lane) cnts[lwid] = 0;
        __syncwarp();

        auto start = old_offsets[u];
        auto end = old_offsets[u + 1];
        for (auto v_idx = start + lane; v_idx < end; v_idx += WARP_SIZE) {
            auto target_edge_idx = eid[v_idx];
            focus[v_idx] = !old_processed[target_edge_idx];
            if (focus[v_idx])
                atomicAdd(&cnts[lwid], 1);
        }
        __syncwarp();

        if (0 == lane) histogram[u] = cnts[lwid];
    }
}
//
//__global__
//void reverse_bits(bool* boolean_input, bool* boolean_output, uint32_t cnt) {
//    auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
//    if (gtid < cnt) boolean_output[gtid] = !boolean_input[gtid];
//}
//
//__global__
//void update_eid(uint* eid, uint* scanned_processed, uint32_t cnt) {
//    auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
//    if (gtid < cnt) {
//        eid[gtid] -= scanned_processed[eid[gtid]];
//    }
//}
//

template<typename T, typename PeelT>
__global__
void filter_window(PeelT* edge_sup, T count, InBucketWinType* in_bucket, T low, T high) {
    auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gtid < count) {
        auto v = edge_sup[gtid];
        in_bucket[gtid] = (v >= low && v < high);
    }
}

template<typename T, typename PeelT>
__global__
void filter_pointer_window(PeelT* edge_sup, T count, InBucketWinType* in_bucket, T low, T high) {
    auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gtid < count) {
        auto v = edge_sup[gtid+1] - edge_sup[gtid];
        in_bucket[gtid] = (v >= low && v < high);
    }
}

template<typename T, typename PeelT>
__global__
void filter_with_random_append(T* bucket_buf, T count, PeelT* EdgeSupport, bool* in_curr, T* curr, T* curr_cnt,
    T ref) 
{
    auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gtid < count) {
        auto edge_off = bucket_buf[gtid];
        if (EdgeSupport[edge_off] == ref) {
            in_curr[edge_off] = true;
            auto insert_idx = atomicAdd(curr_cnt, 1);
            curr[insert_idx] = edge_off;
        }
    }
}

template<typename T, typename PeelT>
__global__
void filter_with_random_append(T* bucket_buf, T count, PeelT* EdgeSupport, bool* in_curr, T* curr, T* curr_cnt,
    T ref, T span)
{
    auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gtid < count) {
        auto edge_off = bucket_buf[gtid];
        if (EdgeSupport[edge_off] >= ref && EdgeSupport[edge_off] < ref + span) {
            in_curr[edge_off] = true;
            auto insert_idx = atomicAdd(curr_cnt, 1);
            curr[insert_idx] = edge_off;
        }
    }
}

template<typename T, typename PeelT>
__global__
void filter_with_random_append_pointer(T* bucket_buf, T count, PeelT* EdgeSupport, bool* in_curr, T* curr, T* curr_cnt,
    int ref, int span)
{
    auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gtid < count) {
        auto edge_off = bucket_buf[gtid];
        auto v = EdgeSupport[edge_off + 1] - EdgeSupport[edge_off];

        if (v >= ref && v < ref + span) {
            in_curr[edge_off] = true;
            auto insert_idx = atomicAdd(curr_cnt, 1);
            curr[insert_idx] = edge_off;
        }
    }
}



template<typename T>
__device__ void add_to_queue_1(graph::GraphQueue_d<T, bool>& q, T element)
{
    auto insert_idx = atomicAdd(q.count, 1);
    q.queue[insert_idx] = element;
    q.mark[element] = true;
}

template<typename T>
__device__ void add_to_queue_1_no_dup(graph::GraphQueue_d<T, bool>& q, T element)
{
    auto old_token = atomicCASBool(q.mark + element, InBucketFalse, InBucketTrue);
    if (!old_token) {
        auto insert_idx = atomicAdd(q.count, 1);
        q.queue[insert_idx] = element;
    }
}

template<typename T, typename PeelT>
__inline__ __device__
void process_support2(
    T edge_idx, T level, PeelT* EdgeSupport,
    graph::GraphQueue_d<T, bool>& next,
    graph::GraphQueue_d<T, bool>& bucket,
    T bucket_level_end_)
{
    auto cur = atomicSub(&EdgeSupport[edge_idx], 1);
    if (cur == (level + 1)) {
        add_to_queue_1(next, edge_idx);
    }
    if (cur <= level) {
        atomicAdd(&EdgeSupport[edge_idx], 1);
    }

    // Update the Bucket.
    auto latest = cur - 1;
    if (latest > level && latest < bucket_level_end_) {
        add_to_queue_1_no_dup<T>(bucket, edge_idx);
    }

}


template <typename T, typename PeelT>
__device__ inline void addNexBucket(T e1, T e2, T e3, bool* processed,
    PeelT* edgeSupport, int level,
    bool* inCurr,
    graph::GraphQueue_d<T, bool>& next,
    graph::GraphQueue_d<T, bool>& bucket, 
    T bucket_upper_level)
{
    bool is_peel_e2 = !inCurr[e2];
    bool is_peel_e3 = !inCurr[e3];
    if (is_peel_e2 || is_peel_e3)
    {
        if ((!processed[e2]) && (!processed[e3]))
        {
            if (is_peel_e2 && is_peel_e3)
            {

                process_support2<T, PeelT>(e2, level, edgeSupport, next, bucket, bucket_upper_level);
                process_support2<T>(e3, level, edgeSupport, next, bucket, bucket_upper_level);



            }
            else if (is_peel_e2)
            {
                if (e1 < e3) {
                    process_support2<T, PeelT>(e2, level, edgeSupport, next, bucket, bucket_upper_level);
                }
            }
            else
            {
                if (e1 < e2)
                {
                    process_support2<T, PeelT>(e3, level, edgeSupport, next, bucket, bucket_upper_level);
                }
            }
        }
    }
}



__global__ void bmp_bsr_update_next(uint32_t* d_offsets, uint32_t* d_dsts,
    uint32_t* d_bitmaps, uint32_t* d_bitmap_states,
    uint32_t* vertex_count, uint32_t conc_blocks_per_SM,
    uint* eid, int32_t* d_intersection_count_GPU,
    int val_size_bitmap, int val_size_bitmap_indexes,
    uint32_t* bmp_offs, uint* bmp_word_indices, uint* bmp_words,
    int level, int* next, int* next_cnt, bool* inNext,
    InBucketWinType* in_bucket_window_, int* bucket_buf_, int* window_bucket_buf_size_,
    int bucket_level_end_
) {
    const uint32_t tid = threadIdx.x + blockDim.x * threadIdx.y; /*threads in a warp are with continuous threadIdx.x */
    const uint32_t num_threads = blockDim.x * blockDim.y;
    const uint32_t elem_bits = sizeof(uint32_t) * 8; /*#bits in a bitmap element*/

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
        if (dv > du || ((du == dv) && u > v))continue;

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
        if (threadIdx.x == 0) {
            auto edge_idx = eid[idx];
            auto prev = d_intersection_count_GPU[edge_idx];
            if (prev > level) {
                if (private_count < level) {
                    private_count = level;
                }
                d_intersection_count_GPU[edge_idx] = private_count;
                if (private_count == level) {
                    auto insert_idx = atomicAdd(next_cnt, 1);
                    next[insert_idx] = edge_idx;
                    inNext[edge_idx] = true;
                }
#ifndef LEGACY_SCAN
                // Update the Bucket.
                auto latest = private_count;
                if (latest > level && latest < bucket_level_end_) {
                    auto old_token = atomicCASBool(in_bucket_window_ + edge_idx, InBucketFalse, InBucketTrue);
                    if (!old_token) {
                        auto insert_idx = atomicAdd(window_bucket_buf_size_, 1);
                        bucket_buf_[insert_idx] = edge_idx;
                    }
                }
#endif
            }
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

