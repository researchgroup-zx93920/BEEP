#pragma once
#include "../../include/GraphDataStructure.cuh"
#include "../../include/queue.cuh"

__constant__ uint KCCOUNT;
__constant__ uint MAXDEG;
__constant__ uint PARTSIZE;
__constant__ uint NUMPART;
__constant__ uint MAXLEVEL;
__constant__ uint NUMDIVS;
__constant__ uint CBPSM;
__constant__ uint ORIENTED_MAXDEG;
__constant__ uint FIRST_SYM_LEVEL;
__constant__ uint CB;

using InBucketWinType = bool;

struct MessageBlock
{
    unsigned int src_;
    unsigned int dstIdx_;
    unsigned int *encode_;
    unsigned int root_sm_block_id_;
    unsigned int level_;
    unsigned int level_prev_index_[10];
};

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

template <typename T, uint CPARTSIZE>
__device__ __forceinline__ void reduce_part(T partMask, uint64 &warpCount)
{
    for (int i = CPARTSIZE / 2; i >= 1; i /= 2)
        warpCount += __shfl_down_sync(partMask, warpCount, i);
}

template <typename T>
__host__ __device__ T binary_search(const T *arr, //!< [in] array to search
                                    const T lt,
                                    const T rt,        //!< [in] size of array
                                    const T searchVal, //!< [in] value to search for
                                    bool &found)
{
    T left = lt;
    T right = rt;
    found = false;
    while (left < right)
    {
        const T mid = (left + right) / 2;
        T val = arr[mid];
        if (val == searchVal)
        {
            found = true;
            return mid;
        }
        bool pred = val < searchVal;
        if (pred)
        {
            left = mid + 1;
        }
        else
        {
            right = mid;
        }
    }
    return left;
}

template <typename T>
__global__ void warp_detect_deleted_edges(
    T *rowPtr, T numRows,
    bool *keep,
    T *histogram)
{

    __shared__ uint32_t cnts[WARPS_PER_BLOCK];

    auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
    auto gtnum = blockDim.x * gridDim.x;
    auto gwid = gtid >> WARP_BITS;
    auto gwnum = gtnum >> WARP_BITS;
    auto lane = threadIdx.x & WARP_MASK;
    auto lwid = threadIdx.x >> WARP_BITS;

    for (auto u = gwid; u < numRows; u += gwnum)
    {
        if (0 == lane)
            cnts[lwid] = 0;
        __syncwarp();

        auto start = rowPtr[u];
        auto end = rowPtr[u + 1];
        for (auto v_idx = start + lane; v_idx < end; v_idx += WARP_SIZE)
        {
            if (keep[v_idx])
                atomicAdd(&cnts[lwid], 1);
        }
        __syncwarp();

        if (0 == lane)
            histogram[u] = cnts[lwid];
    }
}

template <typename T, int BLOCK_DIM_X>
__global__ void get_max_degree(graph::COOCSRGraph_d<T> g, T *edgePtr, T *maxDegree)
{
    const T gtid = (BLOCK_DIM_X * blockIdx.x + threadIdx.x);
    typedef cub::BlockReduce<T, BLOCK_DIM_X> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    T degree = 0;

    if (gtid < g.numEdges)
    {
        T src = g.rowInd[gtid];
        T dst = g.colInd[gtid];

        T srcDeg = g.rowPtr[src + 1] - g.rowPtr[src];
        T dstDeg = g.rowPtr[dst + 1] - g.rowPtr[dst];

        // degree = srcDeg > dstDeg ? srcDeg : dstDeg;
        degree = srcDeg;
        edgePtr[gtid] = degree;
    }

    __syncthreads();
    T aggregate = BlockReduce(temp_storage).Reduce(degree, cub::Max());

    if (threadIdx.x == 0)
        atomicMax(maxDegree, aggregate);
}

template <typename T, bool reduce = true, uint CPARTSIZE = 32>
__device__ __forceinline__ uint64 warp_sorted_count_and_encode_full(const T *const A, //!< [in] array A
                                                                    const size_t aSz, //!< [in] the number of elements in A
                                                                    T *B,             //!< [in] array B
                                                                    T bSz,            //!< [in] the number of elements in B

                                                                    T j,
                                                                    T num_divs_local,
                                                                    T *encode)
{
    // if (threadIdx.x == 0)
    // {
    // printf("CPARTSIZE: %u\n", CPARTSIZE);
    // }
    const int warpIdx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const int laneIdx = threadIdx.x % CPARTSIZE; // which thread in warp
    // cover entirety of A with warp
    for (T i = laneIdx; i < aSz; i += CPARTSIZE)
    {
        const T searchVal = A[i];
        bool found = false;
        const T lb = binary_search<T>(B, 0, bSz, searchVal, found);

        if (found)
        {
            // printf("\033[0;32m Found %u, in adjacency of: %u\033[0;37m\n", searchVal, A[j]);
            //////////////////////////////Device function ///////////////////////
            T chunk_index = i / 32; // 32 here is the division size of the encode
            T inChunkIndex = i % 32;
            atomicOr(&encode[j * num_divs_local + chunk_index], 1 << inChunkIndex);

            T chunk_index1 = j / 32; // 32 here is the division size of the encode
            T inChunkIndex1 = j % 32;
            atomicOr(&encode[i * num_divs_local + chunk_index1], 1 << inChunkIndex1);

            /////////////////////////////////////////////////////////////////////
        }
        // else
        //     printf("\033[0;31m Not found %u in adjacency of: %u\033[0;37m\n", searchVal, A[j]);
    }
    return 0;
}

template <typename T, int BLOCK_DIM_X>
__global__ void getNodeDegree_kernel(T *nodeDegree, graph::COOCSRGraph_d<T> g, T *maxDegree)
{
    T gtid = threadIdx.x + blockIdx.x * blockDim.x;
    typedef cub::BlockReduce<T, BLOCK_DIM_X> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    T degree = 0;
    if (gtid < g.numNodes)
    {
        degree = g.rowPtr[gtid + 1] - g.rowPtr[gtid];
        nodeDegree[gtid] = degree;
    }

    T aggregate = BlockReduce(temp_storage).Reduce(degree, cub::Max());
    if (threadIdx.x == 0)
        atomicMax(maxDegree, aggregate);
}

template <typename DataType, typename CntType>
__global__ void init_asc(DataType *data, CntType count)
{
    auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gtid < count)
        data[gtid] = (DataType)gtid;
}

template <typename T, typename PeelT>
__global__ void filter_window(PeelT *edge_sup, T count, InBucketWinType *in_bucket, T low, T high)
{
    auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gtid < count)
    {
        auto v = edge_sup[gtid];
        in_bucket[gtid] = (v >= low && v < high);
    }
}

template <typename T, typename PeelT>
__global__ void filter_with_random_append(T *bucket_buf, T count, PeelT *EdgeSupport, bool *in_curr, T *curr, T *curr_cnt,
                                          T ref, T span)
{
    auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gtid < count)
    {
        auto edge_off = bucket_buf[gtid];
        if (EdgeSupport[edge_off] >= ref && EdgeSupport[edge_off] < ref + span)
        {
            in_curr[edge_off] = true;
            auto insert_idx = atomicAdd(curr_cnt, 1);
            curr[insert_idx] = edge_off;
        }
    }
}

template <typename T>
__global__ void remove_edges_connected_to_node(
    const graph::COOCSRGraph_d<T> g,
    const graph::GraphQueue_d<T, bool> node_queue,
    bool *keep)
{
    const int partition = 1;
    auto lx = threadIdx.x % partition;
    auto wx = threadIdx.x / partition;
    auto numPart = blockDim.x / partition;
    for (auto i = wx + blockIdx.x * numPart; i < node_queue.count[0]; i += numPart * gridDim.x)
    {
        T src = node_queue.queue[i];
        T srcStart = g.rowPtr[src];
        T srcEnd = g.rowPtr[src + 1];
        for (T j = srcStart + lx; j < srcEnd; j += partition)
        {
            keep[j] = false;
            T dst = g.colInd[j];
            for (T k = g.rowPtr[dst]; k < g.rowPtr[dst + 1]; k++)
            {
                if (g.colInd[k] == src)
                {
                    keep[k] = false;
                    break;
                }
            }
        }
    }
}

template <typename T, int BLOCK_DIM_X>
__global__ void getNodeDegree_split_kernel(T *nodeDegree, graph::COOCSRGraph_d<T> g, T *maxDegree)
{
    T gtid = threadIdx.x + blockIdx.x * blockDim.x;
    typedef cub::BlockReduce<T, BLOCK_DIM_X> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    T degree = 0;
    if (gtid < g.numNodes)
    {
        degree = g.rowPtr[gtid + 1] - g.splitPtr[gtid];
        nodeDegree[gtid] = degree;
        // printf("id %u: %u\n", gtid, degree);
    }

    T aggregate = BlockReduce(temp_storage).Reduce(degree, cub::Max());
    if (threadIdx.x == 0)
        atomicMax(maxDegree, aggregate);
}

__device__ __inline__ uint32_t __mysmid()
{
    unsigned int r;
    asm("mov.u32 %0, %%smid;"
        : "=r"(r));
    return r;
}
