#pragma once
#include <cuda_runtime.h>
#include "../include/utils_cuda.cuh"
#include "../include/defs.cuh"

template<typename T>
__global__ void init(T numEdges, T* asc, bool* keep, T* src, T* dst)
{
	uint tx = threadIdx.x;
	uint bx = blockIdx.x;

	uint ptx = tx + bx * blockDim.x;

	for (uint i = ptx; i < numEdges; i += blockDim.x * gridDim.x)
	{
		keep[i] = src[i] < dst[i];
		asc[i] = i;
	}
}

template<typename T>
__global__ void InitEid(T numEdges, T* asc, T*newSrc, T* newDst, T* rowPtr, T* colInd, T* eid)
{
	uint tx = threadIdx.x;
	uint bx = blockIdx.x;

	uint ptx = tx + bx * blockDim.x;

	for (uint i = ptx; i < numEdges; i += blockDim.x * gridDim.x)
	{
		//i : is the new index of the edge !!
		T srcnode = newSrc[i];
		T dstnode = newDst[i];

		if (srcnode >= dstnode)
		{
			printf("Wrong \n");
		}

		T olduV = asc[i];
		T oldUv = getEdgeId(rowPtr, colInd, dstnode, srcnode); //Search for it please !!


		eid[olduV] = i;
		eid[oldUv] = i;
	}
}

