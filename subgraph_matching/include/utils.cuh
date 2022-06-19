#pragma once
#include "config.cuh"
#include "../include/Timer.h"
#include "include/common_utils.cuh"

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
struct triplet
{
    T colind;
    T partition;
    T priority;
};

template <typename T>
struct mapping
{
    T src;
    T srcHead;
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

template <typename T>
__device__ __forceinline__ T get_mask(T idx, T partition)
{
    if (idx / 32 > partition)
        return 0xFFFFFFFF;
    if (idx / 32 < partition)
        return 0;
    return (0xFFFFFFFF >> (32 - (idx - partition * 32)));
}

template <typename T>
__device__ __forceinline__ T unset_mask(T idx, T partition)
{
    // Use with BITWISE AND. All bits 1 except at idx.
    if (idx / 32 == partition)
        return (~(1 << (idx - partition * 32)));
    else
        return 0xFFFFFFFF;
}

template <typename T>
__device__ __forceinline__ T set_mask(T idx, T partition)
{
    // Use with BITWISE OR. All bits 0 except at idx
    if (idx / 32 == partition)
        return (1 << (idx - partition * 32));
    else
        return 0;
}

template <typename T, uint CPARTSIZE, bool MAT>
__device__ __forceinline__ void compute_intersection(
    uint64 &wc, T &offset,
    const size_t lx, const T partMask,
    const T num_divs_local, const T maskIdx, const T lvl,
    T *to, T *cl, const T *level_prev_index, const T *encode)
{
    wc = 0;
#ifdef SYMOPT
    if (lx == 0)
    {
        offset = 0;
        for (T sym_idx = SYMNODE_PTR[lvl]; sym_idx < SYMNODE_PTR[lvl + 1]; sym_idx++)
        {
            if (SYMNODE[sym_idx] > 0)
                offset = max(offset, level_prev_index[SYMNODE[sym_idx]] - 1);
        }
    }
    __syncwarp(partMask);

#endif
    for (T k = lx; k < num_divs_local; k += CPARTSIZE)
    {
        to[threadIdx.x] = cl[k] & unset_mask(maskIdx, k);
        // Compute Intersection
        for (T q_idx = QEDGE_PTR[lvl] + 1; q_idx < QEDGE_PTR[lvl + 1]; q_idx++)
        {
            to[threadIdx.x] &= encode[(level_prev_index[QEDGE[q_idx]] - 1) * num_divs_local + k];
        }

// Remove Redundancies
#ifdef SYMOPT
        to[threadIdx.x] &= ~get_mask(offset, k);
#else
        for (T sym_idx = SYMNODE_PTR[lvl]; sym_idx < SYMNODE_PTR[lvl + 1]; sym_idx++)
        {
            if (!MAT && SYMNODE[sym_idx] == lvl - 1)
                continue;
            if (SYMNODE[sym_idx] > 0)
                to[threadIdx.x] &= ~(get_mask(level_prev_index[SYMNODE[sym_idx]] - 1, k));
        }
#endif
        wc += __popc(to[threadIdx.x]);                        // counts number of set bits
        cl[(lvl - 1) * num_divs_local + k] = to[threadIdx.x]; // saves candidates list in cl
    }

    reduce_part<T, CPARTSIZE>(partMask, wc);
}

template <typename T, uint CPARTSIZE, bool MAT>
__device__ __forceinline__ void compute_intersection_reuse(
    uint64 &wc, T &offset, const size_t lx, const T partMask,
    const T num_divs_local, const T maskIdx, const T lvl,
    T *to, T *cl, T *reuse, const T *level_prev_index, const T *encode)
{
    wc = 0;
#ifdef SYMOPT
    if (lx == 1)
    {
        offset = 0;
        for (T sym_idx = SYMNODE_PTR[lvl]; sym_idx < SYMNODE_PTR[lvl + 1]; sym_idx++)
        {
            if (SYMNODE[sym_idx] > 0)
                offset = max(offset, level_prev_index[SYMNODE[sym_idx]] - 1);
        }
    }
    __syncwarp(partMask);

    if (offset > 0)
    {
#endif
        for (T k = lx; k < num_divs_local; k += CPARTSIZE)
        {
            to[threadIdx.x] = cl[k] & unset_mask(maskIdx, k);
            if (QREUSE[lvl] > 0)
                to[threadIdx.x] &= reuse[QREUSE[lvl] * num_divs_local + k];

            // Compute Intersection
            for (T q_idx = REUSE_PTR[lvl]; q_idx < QEDGE_PTR[lvl + 1]; q_idx++)
                to[threadIdx.x] &= encode[(level_prev_index[QEDGE[q_idx]] - 1) * num_divs_local + k];

            if (QREUSABLE[lvl])
                reuse[lvl * num_divs_local + k] = to[threadIdx.x];

// Remove Redundancies
#ifdef SYMOPT
            to[threadIdx.x] &= ~get_mask(offset, k);
#else
        for (T sym_idx = SYMNODE_PTR[lvl]; sym_idx < SYMNODE_PTR[lvl + 1]; sym_idx++)
        {
            if (!MAT && SYMNODE[sym_idx] == lvl - 1)
                continue;
            if (SYMNODE[sym_idx] > 0)
                to[threadIdx.x] &= ~(get_mask(level_prev_index[SYMNODE[sym_idx]] - 1, k));
        }
#endif
            wc += __popc(to[threadIdx.x]);                        // counts number of set bits
            cl[(lvl - 1) * num_divs_local + k] = to[threadIdx.x]; // saves candidates list in cl
        }
#ifdef SYMOPT
    }
    else
    {
        for (T k = lx; k < num_divs_local; k += CPARTSIZE)
        {

            to[threadIdx.x] = cl[k] & unset_mask(maskIdx, k);

            if (QREUSE[lvl] > 0)
                to[threadIdx.x] &= reuse[QREUSE[lvl] * num_divs_local + k];

            // Compute Intersection
            for (T q_idx = REUSE_PTR[lvl]; q_idx < QEDGE_PTR[lvl + 1]; q_idx++)
                to[threadIdx.x] &= encode[(level_prev_index[QEDGE[q_idx]] - 1) * num_divs_local + k];

            if (QREUSABLE[lvl])
                reuse[lvl * num_divs_local + k] = to[threadIdx.x];

            wc += __popc(to[threadIdx.x]);                        // counts number of set bits
            cl[(lvl - 1) * num_divs_local + k] = to[threadIdx.x]; // saves candidates list in cl
        }
    }
#endif
    reduce_part<T, CPARTSIZE>(partMask, wc);
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
__global__ void map_src(mapping<T> *mapped, const graph::GraphQueue_d<T, bool> current, const T *scan, const T *degree)
{
    __shared__ T src, srcLen, start;
    if (threadIdx.x == 0)
    {
        src = current.queue[blockIdx.x];
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