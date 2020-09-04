#pragma once
#include <cuda_runtime.h>
#include "../include/utils_cuda.cuh"
#include "../include/defs.cuh"
#include "../include/GraphDataStructure.cuh"




template <typename T, int BLOCK_DIM_X>
__global__ void count_triangles_kernel(
	uint64 *count,
	T numNodes,
	T numEdges,
	T tilesPerDim,
	T tileSize,
	T capacity,
	T* tileRowPtr,
	T* rowInd,
	T* colInd)
{
	unsigned int e = blockIdx.x * blockDim.x + threadIdx.x;
	T numTriangles_e = 0;


	if (e < numEdges) 
	{
		T* srcIdx = rowInd;
		T* dstIdx = colInd;
		T* tileSrcPtr = tileRowPtr;
		T dst = dstIdx[e];
		T src1 = srcIdx[e];
		T src2 = dst;
		T src1Tile = src1 / tileSize;
		T src2Tile = src2 / tileSize;
		unsigned int numTileSrcPtrs = tilesPerDim * tilesPerDim * tileSize + 1;

		for (T xTile = blockIdx.y; xTile < tilesPerDim; xTile += gridDim.y)
		{
			T tileSrc1 = (src1Tile * tilesPerDim + xTile) * tileSize + src1 % tileSize;
			T tileSrc2 = (src2Tile * tilesPerDim + xTile) * tileSize + src2 % tileSize;
			T e1 = tileSrcPtr[tileSrc1];
			T e2 = tileSrcPtr[tileSrc2];
			T end1 = tileSrcPtr[tileSrc1 + 1];
			T end2 = tileSrcPtr[tileSrc2 + 1];
			while (e1 < end1 && e2 < end2)
			{
				T dst1 = dstIdx[e1];
				T dst2 = dstIdx[e2];

				if (dst1 < dst2) {
					++e1;
				}
				else if (dst1 > dst2) {
					++e2;
				}
				else 
				{ // dst1 == dst2
					++e1;
					++e2;
					++numTriangles_e;

				}
			}
		}
		/*if (gridDim.y == 1) {
			info.numTriangles[e] = numTriangles_e;
		}
		else {
			atomicAdd(&info.numTriangles[e], numTriangles_e);
		}*/


	}

	typedef cub::BlockReduce<uint64, BLOCK_DIM_X> BlockReduce;
	__shared__ typename BlockReduce::TempStorage tempStorage;
	uint64 aggregate = BlockReduce(tempStorage).Sum(numTriangles_e);

	// Add to total count
	if (0 == threadIdx.x) {
		atomicAdd(count, aggregate);
	}
}


template<typename T>
__global__ void init(graph::COOCSRGraph_d<T> g, T* asc, bool* keep)
{
	uint tx = threadIdx.x;
	uint bx = blockIdx.x;

	uint ptx = tx + bx * blockDim.x;

	for (uint i = ptx; i < g.numEdges; i += blockDim.x * gridDim.x)
	{
		const T src = g.rowInd[i];
		const T dst = g.colInd[i];

		const T srcStart = g.rowPtr[src];
		const T srcStop = g.rowPtr[src + 1];

		const T dstStart = g.rowPtr[dst];
		const T dstStop = g.rowPtr[dst + 1];

		const T dstLen = dstStop - dstStart;
		const T srcLen = srcStop - srcStart;


		keep[i] = (dstLen < srcLen || ((dstLen == srcLen) && src < dst));// Some simple graph orientation
		//src[i] < dst[i];
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

		

		T olduV = asc[i];
		T oldUv = getEdgeId(rowPtr, colInd, dstnode, srcnode); //Search for it please !!


		eid[olduV] = i;
		eid[oldUv] = i;
	}
}



template<typename T>
uint64 CountTriangles(std::string message, graph::TcBase<T>* tc, graph::COOCSRGraph_d<T> *g,
	const size_t numEdges_upto, const size_t edgeOffset = 0, ProcessingElementEnum kernelType = Thread, int increasing = 0)
{
	tc->count_async(g, numEdges_upto, edgeOffset, kernelType, increasing);
	tc->sync();
	CUDA_RUNTIME(cudaGetLastError());
	printf("TC = %u\n", tc->count());
	double secs = tc->kernel_time();
	int dev = tc->device();
	Log(LogPriorityEnum::info, "Kernel [%s]: gpu %d kernel time %f (%f teps) \n", message.c_str(), dev, secs, (numEdges_upto-edgeOffset) / secs);
	cudaDeviceSynchronize();


	return tc->count();
}

template<typename T>
void CountTrianglesHash(const int divideConstant, graph::TcBase<T>* tc, graph::GPUArray<T> rowPtr, graph::GPUArray<T> rowInd, graph::GPUArray<T> colInd,
	const size_t numEdges, const size_t numRows, const size_t edgeOffset = 0, ProcessingElementEnum kernelType = Thread, int increasing = 0)
{

	const int minRowLen = 8 * 1024;
	const int maxRowLen = 32 * 1024;
	const int cNumBins = 512; //not used

	//Construct
	auto hash1 = [](uint val, uint div) { return (val / 11) % div; };
	graph::GPUArray<uint> htp("hash table pointer", AllocationTypeEnum::gpu, numRows + 1, 0);
	graph::GPUArray<uint> htd("hash table", AllocationTypeEnum::gpu, numEdges - edgeOffset, 0);

	htp.cdata()[0] = 0;
	for (int i = 0; i < numRows; i++)
	{
		uint s = rowPtr.cdata()[i];
		uint e = rowPtr.cdata()[i + 1];

		if ((e - s) >= minRowLen && (e - s) < maxRowLen)
			htp.cdata()[i + 1] = (e - s) / divideConstant + 1;
		else
			htp.cdata()[i + 1] = 0;
	}

	//reduce
	for (int i = 0; i < numRows + 1; i++)
	{
		htp.cdata()[i + 1] += htp.cdata()[i];//will implement on GPU, do not worry
	}

	uint totalBins = htp.cdata()[numRows];
	graph::GPUArray<uint> hts("bins start per row", AllocationTypeEnum::gpu, totalBins, 0);
	hts.setAll(0, true);
	hts.copytocpu(0);
	//Foreach row count
	for (int i = 0; i < numRows; i++)
	{
		uint s = rowPtr.cdata()[i];
		uint e = rowPtr.cdata()[i + 1];

		uint bin_start = htp.cdata()[i];
		uint bin_end = htp.cdata()[i + 1];

		if (bin_end > bin_start)
		{
			uint numBins = (e - s) / divideConstant;
			for (int j = s; j < e; j++)
			{
				uint val = colInd.cdata()[j];
				uint bin = hash1(val, numBins);

				hts.cdata()[bin_start + bin + 1] += 1;
			}
		}
	}

	//now reduce per row
	for (int i = 0; i < numRows; i++)
	{
		uint s = rowPtr.cdata()[i];
		uint e = rowPtr.cdata()[i + 1];

		uint bin_start = htp.cdata()[i];
		uint bin_end = htp.cdata()[i + 1];

		uint numBins = (e - s) / divideConstant;
		if (bin_end - bin_start > 0)
		{
			for (int j = bin_start; j < bin_end - 1; j++)
				hts.cdata()[j + 1] += hts.cdata()[j];
		}
	}

	//Mode data to hash tables
	for (int i = 0; i < numRows; i++)
	{
		const uint s = rowPtr.cdata()[i];
		const uint e = rowPtr.cdata()[i + 1];

		uint bin_start = htp.cdata()[i];
		uint bin_end = htp.cdata()[i + 1];
		const uint numBins = (e - s) / divideConstant;
		if (bin_end > bin_start)
		{
			uint* binCounter = new uint[numBins];
			for (int n = 0; n < numBins; n++)
				binCounter[n] = 0;
			for (int j = s; j < e; j++)
			{
				uint val = colInd.cdata()[j];
				uint bin = hash1(val, numBins);
				uint elementBinStart = hts.cdata()[bin_start + bin];
				uint nextBinStart = hts.cdata()[bin_start + bin + 1];
				htd.cdata()[s + elementBinStart + binCounter[bin]] = val;
				binCounter[bin]++;
			}
		}
		else
		{
			for (int j = s; j < e; j++)
			{
				htd.cdata()[j] = colInd.cdata()[j];
			}
		}
	}


	htp.switch_to_gpu(0);
	hts.switch_to_gpu(0);
	htd.switch_to_gpu(0);



	tc->count_hash_async(divideConstant, rowPtr, rowInd, htd, htp, hts, numEdges, edgeOffset, kernelType, increasing);
	tc->sync();
	CUDA_RUNTIME(cudaGetLastError());
	printf("TC Hash = %d\n", tc->count());
	double secs = tc->kernel_time();
	int dev = tc->device();
	Log(LogPriorityEnum::info, "gpu %d kernel time %f (%f teps) \n", dev, secs, numEdges / secs);
	cudaDeviceSynchronize();
}


template<typename T>
void ConstructTriList(graph::GPUArray<T>& triIndex, graph::GPUArray<T>& triPointer, graph::TcBase<T>* tc, graph::GPUArray<T> rowPtr, graph::GPUArray<T> rowInd, graph::GPUArray<T> colInd,
	const size_t numEdges, const size_t numRows, const size_t edgeOffset = 0, ProcessingElementEnum kernelType = Thread)
{
	//Count triangle
	tc->count_async(rowPtr, rowInd, colInd, numEdges, edgeOffset, kernelType, 0);
	tc->sync();
	uint tcount = tc->count();

	//Create memory just for reduction
	graph::GPUArray<uint> temp = graph::GPUArray<uint>("temp reduction", unified, numEdges, 0);

	tc->count_per_edge_async(temp, rowPtr, rowInd, colInd, numEdges, edgeOffset, kernelType, 0);
	tc->sync();


	//Scan
	triPointer = graph::GPUArray<uint>("TriPointer", unified, numEdges + 1, 0);
	void* d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, temp.gdata(), triPointer.gdata(), numEdges);
	CUDA_RUNTIME(cudaMalloc(&d_temp_storage, temp_storage_bytes));
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, temp.gdata(), triPointer.gdata(), numEdges);
	cudaDeviceSynchronize();

	triPointer.set(numEdges, tcount, true);

	triIndex = graph::GPUArray<uint>("TriIndex", unified, tcount, 0);
	tc->set_per_edge_async(triIndex, triPointer, rowPtr, rowInd, colInd, numEdges, edgeOffset, kernelType, 0);
	tc->sync();

	/*for (int j = 0; j < 20; j++)
	{
		int te_start = triPointer.cdata()[j];
		int te_end = triPointer.cdata()[j + 1];
		if (te_end - te_start != temp.cdata()[j])
			printf("Wrong!\n");
		else
		{
			if (te_end - te_start > 0)
				printf("For edge(%u,%u): \n", rowInd.cdata()[j], colInd.cdata()[j]);

			for (int i = te_start; i < te_end; i++)
			{
				printf("%u,", triIndex.cdata()[i]);
			}
		}
		if(te_end - te_start > 0)
			printf("\n");
	}*/

	//Extra Check
	/*temp.copytocpu(0);
	triPointer.copytocpu(0);
	printf("%u\n", triPointer.cdata()[numEdges - 1]);
	printf("%u\n", triPointer.cdata()[numEdges]);*/
}


struct Node
{
	uint val;
	int i;
	int r;
	int l;
	int p;
};