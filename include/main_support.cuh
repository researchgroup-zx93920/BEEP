#pragma once
#include <cuda_runtime.h>
#include "../include/utils_cuda.cuh"
#include "../include/defs.cuh"
#include "../include/GraphDataStructure.cuh"

__host__ __device__ uint hash1(uint val, uint div) { return (val / 11) % div; };

template <typename T, typename PeelT>
__global__ void initAsc(T *asc, T count)
{
    auto tx = threadIdx.x;
    auto bx = blockIdx.x;

    auto ptx = tx + bx * blockDim.x;

    for (auto i = ptx; i < count; i += blockDim.x * gridDim.x)
    {

        asc[i] = i;
    }
}

template <typename T>
__global__ void InitEid(T numEdges, T *asc, T *newSrc, T *newDst, T *rowPtr, T *colInd, T *eid)
{
    uint tx = threadIdx.x;
    uint bx = blockIdx.x;

    uint ptx = tx + bx * blockDim.x;

    for (uint i = ptx; i < numEdges; i += blockDim.x * gridDim.x)
    {
        // i : is the new index of the edge !!
        T srcnode = newSrc[i];
        T dstnode = newDst[i];

        T olduV = asc[i];
        T oldUv = getEdgeId(rowPtr, colInd, dstnode, srcnode); // Search for it please !!

        eid[olduV] = i;
        eid[oldUv] = i;
    }
}

struct Node
{
    uint val;
    int i;
    int r;
    int l;
    int p;
};

template <typename T>
__global__ void split_inverse(bool *keep, T m)
{
    auto tx = threadIdx.x;
    auto bx = blockIdx.x;

    auto ptx = tx + bx * blockDim.x;

    for (auto i = ptx; i < m; i += blockDim.x * gridDim.x)
    {
        keep[i] ^= 1;
    }
}

template <typename T>
__global__ void split_acc(graph::COOCSRGraph_d<T> g, T *split_ptr)
{
    auto tx = threadIdx.x;
    auto bx = blockIdx.x;

    auto ptx = tx + bx * blockDim.x;

    for (auto i = ptx; i < g.numNodes; i += blockDim.x * gridDim.x)
    {
        split_ptr[i] += g.rowPtr[i];
    }
}

template <typename T>
__global__ void split_child(graph::COOCSRGraph_d<T> g, T *tmp_row, T *tmp_col, T *split_col, T *split_ptr)
{
    auto tx = threadIdx.x;
    auto bx = blockIdx.x;

    auto ptx = tx + bx * blockDim.x;

    for (auto i = ptx; i < g.numEdges / 2; i += blockDim.x * gridDim.x)
    {
        const T src = tmp_row[i];
        const T dst = tmp_col[i];
        split_col[g.rowPtr[src + 1] - (split_ptr[src + 1] - i)] = dst;
    }
}

template <typename T>
__global__ void split_parent(graph::COOCSRGraph_d<T> g, T *tmp_row, T *tmp_col, T *split_col, T *split_ptr)
{
    auto tx = threadIdx.x;
    auto bx = blockIdx.x;

    auto ptx = tx + bx * blockDim.x;

    for (auto i = ptx; i < g.numEdges / 2; i += blockDim.x * gridDim.x)
    {
        const T src = tmp_row[i];
        const T dst = tmp_col[i];
        split_col[g.rowPtr[src] + (i - split_ptr[src])] = dst;
    }
}

__global__ void print_array(uint size, uint *data)
{
    for (int i = 0; i < size; i++)
    {
        printf("%u,\t", data[i]);
    }
    printf("\n");
}

__global__ void print_graph(graph::COOCSRGraph_d<uint> dataGraph)
{
    printf("Printing graph from device\n");
    printf("testing rowInd: %u\n", dataGraph.rowInd[0]);
    for (uint src = dataGraph.rowInd[0]; src < dataGraph.numNodes; src++)
    {
        uint srcStart = dataGraph.rowPtr[src];
        uint srcEnd = dataGraph.rowPtr[src + 1];
        printf("%u\t: ", src);
        for (uint j = srcStart; j < srcEnd; j++)
        {
            printf("%u, ", dataGraph.colInd[j]);
        }
        printf("\n");
    }
    printf("\n");
}
