#pragma once
#include "../config.cuh"
#include "../../include/Timer.h"
#include "common_utils.cuh"

#define fundef template <typename T, uint BLOCK_DIM_X> \
__device__ __forceinline__

const uint N_RECEPIENTS = 1; // Don't change
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

//
//
//
//
//
//  ******************* Worker Queue Functions *****************
//
//
//
//
//
__device__ __forceinline__ void wait_for_donor(
    cuda::atomic<uint32_t, cuda::thread_scope_device> &work_ready, uint32_t &shared_state,
    queue_callee(queue, tickets, head, tail))
{
    uint32_t ns = 8;
    do
    {
        if (work_ready.load(cuda::memory_order_relaxed))
        {
            if (work_ready.load(cuda::memory_order_acquire))
            {
                shared_state = 2;
                work_ready.store(0, cuda::memory_order_relaxed);
                break;
            }
        }
        else if (queue_full(queue, tickets, head, tail, CB))
        {
            shared_state = 100;
            break;
        }
        // else
        // {
        //     printf("Block %u is waiting\n", blockIdx.x);
        // }
    } while (ns = my_sleep(ns));
}

//
//
//
//
// *********************** Host Functions ***********************
//
//
//
//
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
        // g.rowInd[gtid] = book[gtid].partition;
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

template <typename T>
void quicksort1_triplet(triplet<T> *data, T start, T end)
{
    if ((end - start + 1) > 1)
    {
        T left = start, right = end;
        T pivot = data[right].priority;
        while (left <= right)
        {
            while (data[left].priority < pivot)
            {
                left = left + 1;
            }
            while (data[right].priority > pivot)
            {
                right = right - 1;
            }
            if (left <= right)
            {
                // swap data
                triplet<T> tmpData = data[left];
                data[left] = data[right];
                data[right] = tmpData;

                left = left + 1;
                right = right - 1;
            }
        }
        quicksort1_triplet(data, start, right);
        quicksort1_triplet(data, left, end);
    }
}

template <typename T>
void quicksort2_triplet(triplet<T> *data, T start, T end)
{
    if ((end - start + 1) > 1)
    {
        T left = start, right = end;
        T pivot = data[right].partition;
        while (left <= right)
        {
            while (data[left].partition < pivot)
            {
                left = left + 1;
            }
            while (data[right].partition > pivot)
            {
                right = right - 1;
            }
            if (left <= right)
            {
                // swap data
                triplet<T> tmpData = data[left];
                data[left] = data[right];
                data[right] = tmpData;

                left = left + 1;
                right = right - 1;
            }
        }
        quicksort2_triplet(data, start, right);
        quicksort2_triplet(data, left, end);
    }
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
                        const T *scan, const T *degree)
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

template <size_t WPB, typename T, bool reduce = true, uint CPARTSIZE>
__device__ __forceinline__ void warp_sorted_count_and_encode_full_undirected(
    const T *const A, //!< [in] array A
    const T aSz,      //!< [in] the number of elements in A
    const T *const B, //!< [in] array B
    const T bSz,      //!< [in] the number of elements in B
    const T j,
    const T num_divs_local,
    T *encode)
{
    const int laneIdx = threadIdx.x % CPARTSIZE; // which thread in warp
    // cover entirety of A with warp
    for (T i = laneIdx + j; i < aSz; i += CPARTSIZE) //+j since the graph is undirected
    {
        const T searchVal = A[i];
        bool found = false;
        const T lb = binary_search<T>(B, 0, bSz, searchVal, found);

        if (found)
        {

            //////////////////////////////Device function ///////////////////////
            T chunk_index = i / 32; // 32 here is the division size of the encode
            T inChunkIndex = i % 32;
            atomicOr(&encode[j * num_divs_local + chunk_index], 1 << inChunkIndex);

            T chunk_index1 = j / 32; // 32 here is the division size of the encode
            T inChunkIndex1 = j % 32;
            atomicOr(&encode[i * num_divs_local + chunk_index1], 1 << inChunkIndex1);
            /////////////////////////////////////////////////////////////////////
        }
    }
}

__global__ void final_counter(uint64 *glob, uint64 *dev)
{
    atomicAdd(glob, dev[0]);
}