#pragma once
#include "../config.cuh"
#include "../../include/Timer.h"
#include "common_utils.cuh"

#define fundef template <typename T, uint BLOCK_DIM_X> \
__device__ __forceinline__
const uint DEPTH = 10;

__constant__ uint MINLEVEL;
__constant__ uint LUNMAT;

__constant__ uint QEDGE[DEPTH * (DEPTH - 1) / 2];
__constant__ uint QEDGE_PTR[DEPTH + 1];
__constant__ uint QDEG[DEPTH];

__constant__ uint SYMNODE[DEPTH * (DEPTH - 1) / 2];
__constant__ uint SYMNODE_PTR[DEPTH + 1];

__constant__ uint QREUSE[DEPTH];
__constant__ bool QREUSABLE[DEPTH];
__constant__ uint REUSE_PTR[DEPTH];

template <typename T>
struct mapping
{
    T src;
    T srcHead;
};

template <typename T>
struct GLOBAL_HANDLE
{
    uint64 *counter;
    graph::COOCSRGraph_d<T> g;
    mapping<T> *srcList;
    T *current_level;
    T *reuse_stats;
    T *levelStats;
    T *adj_enc;
    T *work_list_head;
    T work_list_tail;
};

template <typename T, uint BLOCK_DIM_X>
struct SHARED_HANDLE
{
    T level_index[DEPTH];
    T level_count[DEPTH];
    T level_prev_index[DEPTH];
    uint64 sg_count;
    T lvl, src, srcStart, srcLen, srcSplit, dstIdx, num_divs_local;
    T *cl, *reuse_offset, *encode;
    T to[BLOCK_DIM_X], newIndex;

    // For Worker Queue
    T state;
};

template <typename T>
struct LOCAL_HANDLE
{
    uint64 warpCount = 0;
    bool go_ahead = true;
};

fundef void init_sm(SHARED_HANDLE<T, BLOCK_DIM_X> &sh,
                    GLOBAL_HANDLE<T> &gh, const T head, const int *offset)
{
    __syncthreads();
    if (threadIdx.x == 0)
    {
        T i = atomicAdd(gh.work_list_head, 1);
        if (i < gh.work_list_tail)
        {
            sh.src = gh.srcList[i + head].src;
            sh.srcStart = gh.g.rowPtr[sh.src];
            sh.srcSplit = gh.g.splitPtr[sh.src];
            sh.srcLen = gh.g.rowPtr[sh.src + 1] - sh.srcStart;

            sh.dstIdx = i + head - gh.srcList[i + head].srcHead;

            sh.num_divs_local = (sh.srcLen + 32 - 1) / 32;
            sh.encode = &gh.adj_enc[(uint64)offset[sh.src] * NUMDIVS * MAXDEG];
            sh.cl = &gh.current_level[(uint64)(blockIdx.x * NUMDIVS * 1 * MAXLEVEL)];
            sh.state = 0;
        }
        else
            sh.state = 100;
    }
    __syncthreads();
}

fundef void compute_triangles(SHARED_HANDLE<T, BLOCK_DIM_X> &sh, GLOBAL_HANDLE<T> &gh, LOCAL_HANDLE<T> &lh)
{
    __syncthreads();
    if (SYMNODE_PTR[2] == 1 && sh.dstIdx < (sh.srcSplit - sh.srcStart))
    {
        lh.go_ahead = false;
        return;
    }
    lh.warpCount = 0;
    for (T k = threadIdx.x; k < sh.num_divs_local; k += BLOCK_DIM_X)
    {
        sh.cl[k] = get_mask(sh.srcLen, k) & unset_mask(sh.dstIdx, k);
        if (QEDGE_PTR[3] - QEDGE_PTR[2] == 2)
            sh.to[threadIdx.x] = sh.encode[sh.dstIdx * sh.num_divs_local + k];
        else
            sh.to[threadIdx.x] = sh.cl[k];

        // Remove Redundancies
        for (T sym_idx = SYMNODE_PTR[2]; sym_idx < SYMNODE_PTR[3]; sym_idx++)
        {
            if (SYMNODE[sym_idx] > 0)
                sh.to[threadIdx.x] &= ~(sh.cl[k] & get_mask(sh.dstIdx, k));
        }
        sh.cl[sh.num_divs_local + k] = sh.to[threadIdx.x];
        lh.warpCount += __popc(sh.to[threadIdx.x]);
    }
    typedef cub::BlockReduce<uint64, BLOCK_DIM_X> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    lh.warpCount = BlockReduce(temp_storage).Sum(lh.warpCount);
    __syncthreads();
    if (KCCOUNT == 3)
    {
        if (threadIdx.x == 0 && lh.warpCount > 0)
            atomicAdd(gh.counter, lh.warpCount);
        lh.go_ahead = false;
        return;
    }
    else if (threadIdx.x == 0)
    {
        sh.sg_count = 0;
    }
}

fundef void init_stack(SHARED_HANDLE<T, BLOCK_DIM_X> &sh, GLOBAL_HANDLE<T> &gh, LOCAL_HANDLE<T> &lh, const T j)
{
    if ((sh.cl[sh.num_divs_local + (j / 32)] >> (j % 32)) % 2 == 0)
    {
        lh.go_ahead = false;
        return;
    }
    lh.go_ahead = true;
    for (T k = threadIdx.x; k < DEPTH; k += BLOCK_DIM_X)
    {
        sh.level_index[k] = 0;
        sh.level_count[k] = 0;
        sh.level_prev_index[k] = 0;
    }
    for (T k = threadIdx.x; k < sh.num_divs_local; k += BLOCK_DIM_X)
    {
        sh.cl[k] = get_mask(sh.srcLen, k) & unset_mask(sh.dstIdx, k) & unset_mask(j, k);
        sh.cl[sh.num_divs_local + k] = sh.cl[sh.num_divs_local + k];
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        sh.lvl = 3;
        sh.level_prev_index[1] = sh.dstIdx + 1;
        sh.level_prev_index[2] = j + 1;
    }
    __syncthreads();
}

fundef void check_terminate(SHARED_HANDLE<T, BLOCK_DIM_X> &sh, LOCAL_HANDLE<T> &lh)
{
    if (threadIdx.x == 0)
    {
        if (sh.lvl + 1 == KCCOUNT)
            sh.sg_count += lh.warpCount;
        else
        {
            sh.lvl++;
            sh.level_count[sh.lvl - 3] = lh.warpCount;
            sh.level_index[sh.lvl - 3] = 0;
            sh.level_prev_index[sh.lvl - 1] = 0;
        }
    }
    __syncthreads();
}

fundef void get_newIndex(SHARED_HANDLE<T, BLOCK_DIM_X> &sh, LOCAL_HANDLE<T> &lh)
{

    if (threadIdx.x == 0)
    {
        T *from = &(sh.cl[sh.num_divs_local * (sh.lvl - 2)]);
        T maskBlock = sh.level_prev_index[sh.lvl - 1] / 32;
        T maskIndex = ~((1 << (sh.level_prev_index[sh.lvl - 1] & 0x1F)) - 1);

        sh.newIndex = __ffs(from[maskBlock] & maskIndex);
        while (sh.newIndex == 0)
        {
            maskIndex = 0xFFFFFFFF;
            maskBlock++;
            sh.newIndex = __ffs(from[maskBlock] & maskIndex);
        }
        sh.newIndex = 32 * maskBlock + sh.newIndex - 1;

        sh.level_prev_index[sh.lvl - 1] = sh.newIndex + 1;
        sh.level_index[sh.lvl - 3]++;
    }
}

fundef void backtrack(SHARED_HANDLE<T, BLOCK_DIM_X> &sh, LOCAL_HANDLE<T> &lh)
{
    if (threadIdx.x == 0)
    {
        if (sh.lvl + 1 == KCCOUNT)
            sh.sg_count += lh.warpCount;
        else if (sh.lvl + 1 < KCCOUNT) //&& warpCount >= KCCOUNT - l[wx])
        {
            (sh.lvl)++;
            sh.level_count[sh.lvl - 3] = lh.warpCount;
            sh.level_index[sh.lvl - 3] = 0;
            sh.level_prev_index[sh.lvl - 1] = 0;
            T idx = sh.level_prev_index[sh.lvl - 2] - 1;
            sh.cl[idx / 32] &= ~(1 << (idx & 0x1F));
        }

        while (sh.lvl > 4 && sh.level_index[sh.lvl - 3] >= sh.level_count[sh.lvl - 3])
        {
            (sh.lvl)--;
            T idx = sh.level_prev_index[sh.lvl - 1] - 1;
            sh.cl[idx / 32] |= 1 << (idx & 0x1F);
        }
    }
}

fundef void finalize_count(SHARED_HANDLE<T, BLOCK_DIM_X> &sh, GLOBAL_HANDLE<T> &gh, LOCAL_HANDLE<T> &lh)
{
    __syncthreads();
    if (threadIdx.x == 0)
    {
        if (sh.sg_count > 0)
            atomicAdd(gh.counter, sh.sg_count);
    }
    __syncthreads();
}

// space
//
//
//
//
//
//
//
//
//
//
//
//
// space end

template <typename T>
struct triplet
{
    T colind;
    T partition;
    T priority;
};

template <typename T, bool ASCENDING>
struct comparePriority
{
    __host__ __device__ bool operator()(const triplet<T> i, const triplet<T> j) const
    {
        if (ASCENDING)
            return (i.priority < j.priority);
        else
            return (i.priority > j.priority);
    }
};

template <typename T>
struct comparePartition
{
    __host__ __device__ bool operator()(const triplet<T> i, const triplet<T> j) const
    {
        return (i.partition < j.partition);
    }
};

template <typename T>
__host__ void get_max_blocks(const T Degree, T &blocks, const uint num_divs, const uint depth)
{
    cudaDeviceSynchronize();
    size_t total, free;
    cudaMemGetInfo(&free, &total);
    size_t encode = (size_t)Degree * num_divs * sizeof(T);
    size_t mask = (size_t)num_divs * sizeof(T);
    size_t level = 80 * (2048 / BLOCK_SIZE_LD) * depth * num_divs * (BLOCK_SIZE_LD / PARTITION_SIZE_LD) * sizeof(T);
    size_t mem_per_block = encode + mask;
    blocks = (T)((free * 0.90 - level) / mem_per_block);
}

template <typename T, uint CPARTSIZE, bool MAT>
__device__ __forceinline__ void compute_intersection(
    uint64 &wc,
    const size_t lx, const T partMask,
    const T num_divs_local, const T maskIdx, const T lvl,
    T *to, T *cl, const T *level_prev_index, const T *encode)
{
    wc = 0;
    for (T k = lx; k < num_divs_local; k += CPARTSIZE)
    {
        to[threadIdx.x] = cl[k] & unset_mask(maskIdx, k);
        // Compute Intersection
        for (T q_idx = QEDGE_PTR[lvl] + 1; q_idx < QEDGE_PTR[lvl + 1]; q_idx++)
        {
            to[threadIdx.x] &= encode[(level_prev_index[QEDGE[q_idx]] - 1) * num_divs_local + k];
        }

        // Remove Redundancies
        for (T sym_idx = SYMNODE_PTR[lvl]; sym_idx < SYMNODE_PTR[lvl + 1]; sym_idx++)
        {
            if (!MAT && SYMNODE[sym_idx] == lvl - 1)
                continue;
            if (SYMNODE[sym_idx] > 0)
                to[threadIdx.x] &= ~(get_mask(level_prev_index[SYMNODE[sym_idx]] - 1, k));
        }

        wc += __popc(to[threadIdx.x]);                        // counts number of set bits
        cl[(lvl - 1) * num_divs_local + k] = to[threadIdx.x]; // saves candidates list in cl
    }
    reduce_part<T, CPARTSIZE>(partMask, wc);
}

template <typename T, uint BLOCK_DIM_X, bool MAT>
__device__ __forceinline__ void compute_intersection_block(
    uint64 &wc, const T tx, const T num_divs_local, const T maskIdx,
    const T lvl, T *to, T *cl, const T *level_prev_index, const T *encode)
{
    wc = 0;
    for (T k = tx; k < num_divs_local; k += BLOCK_DIM_X)
    {
        to[threadIdx.x] = cl[k] & unset_mask(maskIdx, k);

        // Compute Intersection
        for (T q_idx = QEDGE_PTR[lvl] + 1; q_idx < QEDGE_PTR[lvl + 1]; q_idx++)
        {
            to[threadIdx.x] &= encode[(level_prev_index[QEDGE[q_idx]] - 1) * num_divs_local + k];
        }

        // Remove Redundancies
        for (T sym_idx = SYMNODE_PTR[lvl]; sym_idx < SYMNODE_PTR[lvl + 1]; sym_idx++)
        {
            if (!MAT && SYMNODE[sym_idx] == lvl - 1)
                continue;
            if (SYMNODE[sym_idx] > 0)
                to[threadIdx.x] &= ~(get_mask(level_prev_index[SYMNODE[sym_idx]] - 1, k));
        }
        wc += __popc(to[threadIdx.x]);
        cl[(lvl - 1) * num_divs_local + k] = to[threadIdx.x];
    }
    typedef cub::BlockReduce<uint64, BLOCK_DIM_X> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    wc = BlockReduce(temp_storage).Sum(wc); // whole block performs reduction here
    __syncthreads();
}

template <typename T>
__global__ void map_and_gen_triplet_array(triplet<T> *book, const graph::COOCSRGraph_d<T> g, const T *nodePriority)
{
    T gtid = threadIdx.x + blockIdx.x * blockDim.x;

    if (gtid < g.numEdges)
    {
        T src = g.rowInd[gtid];
        T dst = g.colInd[gtid];
        book[gtid].priority = nodePriority[dst];
        book[gtid].colind = dst;
        book[gtid].partition = src;
    }
}

template <typename T>
__global__ void map_back(const triplet<T> *book, graph::COOCSRGraph_d<T> g)
{
    T gtid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gtid < g.numEdges)
    {
        g.oriented_colInd[gtid] = book[gtid].colind;
    }
}

template <typename T>
void key_sort_ascending(T *mapping, T *queue, T len)
{
    // T *mapping;

    T *aux_mapping;
    T *aux_queue;
    // CUDA_RUNTIME(cudaMalloc((void **)&mapping, len * sizeof(T)));
    CUDA_RUNTIME(cudaMalloc((void **)&aux_mapping, len * sizeof(T)));
    CUDA_RUNTIME(cudaMalloc((void **)&aux_queue, len * sizeof(T)));
    CUDA_RUNTIME(cudaMalloc((void **)&queue, len * sizeof(T)));

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DoubleBuffer<T> d_keys(mapping, aux_mapping);
    cub::DoubleBuffer<T> d_values(queue, aux_queue);
    CUDA_RUNTIME(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, len));
    CUDA_RUNTIME(cudaMalloc((void **)&d_temp_storage, temp_storage_bytes));
    CUDA_RUNTIME(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, len));
    CUDA_RUNTIME(cudaMemcpy(queue, d_values.Current(), len * sizeof(T), cudaMemcpyDeviceToDevice));
    // cudaFree(mapping);

    cudaFree(aux_mapping);
    cudaFree(aux_queue);
    cudaFree(d_temp_storage);
    cudaDeviceSynchronize();
}

template <typename T, bool ASCENDING>
__global__ void set_priority(graph::COOCSRGraph_d<T> g, T *priority)
{
    auto tx = threadIdx.x, bx = blockIdx.x, ptx = tx + bx * blockDim.x;

    if (ptx < g.numEdges)
    {
        const T src = g.rowInd[ptx], dst = g.colInd[ptx];
        bool keep = (priority[src] == priority[dst]) ? (src < dst) : (priority[src] < priority[dst]);
        if (!ASCENDING)
            keep = !keep;
        if (!keep)
        {
            atomicAdd(&g.splitPtr[src], 1);
        }
    }
}

template <typename T>
__global__ void set_priority_l(graph::COOCSRGraph_d<T> g)
{
    auto tx = threadIdx.x, bx = blockIdx.x, ptx = tx + bx * blockDim.x;

    if (ptx < g.numEdges)
    {
        const T src = g.rowInd[ptx], dst = g.colInd[ptx];
        bool keep = src < dst;
        if (!keep)
        {
            atomicAdd(&g.splitPtr[src], 1);
        }
    }
}

template <typename T>
__global__ void map_src(mapping<T> *mapped,
                        const graph::GraphQueue_d<T, bool> current,
                        // const T *queue,
                        const T *scan, const T *degree)
{
    __shared__ T src, srcLen, start;
    if (threadIdx.x == 0)
    {
        src = current.queue[blockIdx.x];
        // src = queue[blockIdx.x];
        srcLen = degree[src];
        start = (blockIdx.x > 0) ? scan[blockIdx.x - 1] : 0;
    }
    __syncthreads();
    for (T i = threadIdx.x; i < srcLen; i += blockDim.x)
    {
        mapped[start + i].src = src;
        mapped[start + i].srcHead = start;
    }
}