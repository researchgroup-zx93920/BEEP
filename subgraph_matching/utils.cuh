#pragma once

// template <typename T>
// __device__ __forceinline__ T get_mask(T idx, T partition)
// {
//     if (idx / 32 > partition)
//         return 0xFFFFFFFF;
//     if (idx / 32 < partition)
//         return 0;
//     return (0xFFFFFFFF >> (32 - (idx - partition * 32)));
// }

// template <typename T>
// __device__ __forceinline__ T unset_mask(T idx, T partition)
// {
//     // Use with BITWISE AND. All bits 1 except at idx.
//     if (idx / 32 == partition)
//         return (~(1 << (idx - partition * 32)));
//     else
//         return 0xFFFFFFFF;
// }

// template <typename T>
// __device__ __forceinline__ T set_mask(T idx, T partition)
// {
//     // Use with BITWISE OR. All bits 0 except at idx
//     if (idx / 32 == partition)
//         return (1 << (idx - partition * 32));
//     else
//         return 0;
// }

// template <typename T, uint CPARTSIZE>
// __device__ __forceinline__ void reduce_part(T partMask, uint64 &warpCount)
// {
//     for (int i = CPARTSIZE / 2; i >= 1; i /= 2)
//         warpCount += __shfl_down_sync(partMask, warpCount, i);
// }

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