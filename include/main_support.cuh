#pragma once
#include <cuda_runtime.h>
#include "../include/utils_cuda.cuh"
#include "../include/defs.cuh"
#include "../include/GraphDataStructure.cuh"

__host__ __device__ uint hash1(uint val, uint div) { return (val / 11) % div; };


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
__global__ void init(graph::COOCSRGraph_d<T> g, T* asc, bool* keep, int* degeneracy)
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

		keep[i] = false;
		if (degeneracy[src] < degeneracy[dst])
			keep[i] = true;
		else if (degeneracy[src] == degeneracy[dst] && dstLen < srcLen)
			keep[i] = true;
		else if (degeneracy[src] == degeneracy[dst] && dstLen == srcLen && src < dst)
			keep[i] = true;


		asc[i] = i;
	}
}



template<typename T>
__global__ void createHashPointer2(graph::COOCSRGraph_d<T> g, T* hashPointer, int minLen, int maxLen, int divideConstant)
{
	uint tx = threadIdx.x;
	uint bx = blockIdx.x;

	uint ptx = tx + bx * blockDim.x;

	for (uint i = ptx; i < g.numNodes; i += blockDim.x * gridDim.x)
	{
		const T srcStart = g.rowPtr[i];
		const T srcStop = g.rowPtr[i + 1];
		const T srcLen = srcStop - srcStart;
		if (srcLen >= minLen && srcLen < maxLen)
			hashPointer[i] = srcLen / divideConstant + 1;
		else
			hashPointer[i] = 0;
	}
}

template<typename T>
__global__ void createHashStartBin(graph::COOCSRGraph_d<T> g, T* hashPointer, T *hashStart, int minLen, int maxLen, int divideConstant)
{
	uint tx = threadIdx.x;
	uint bx = blockIdx.x;

	uint ptx = tx + bx * blockDim.x;

	for (uint i = ptx; i < g.numNodes; i += blockDim.x * gridDim.x)
	{
		const T srcStart = g.rowPtr[i];
		const T srcStop = g.rowPtr[i + 1];
		const T srcLen = srcStop - srcStart;

		uint bin_start = hashPointer[i];
		uint bin_end = hashPointer[i + 1];

		if (bin_end > bin_start)
		{
			uint numBins = srcLen / divideConstant;


			for (int j = srcStart; j < srcStop; j++)
			{
				uint val = g.colInd[j];
				uint bin = hash1(val, numBins);

				hashStart[bin_start + bin + 1] += 1;
			}


			for (int j = bin_start; j < bin_end - 1; j++)
				hashStart[j + 1] += hashStart[j];


			/*if (hashStart[bin_end] != srcLen)
				printf("%u, %u, at %d\n", hashStart[bin_end], srcLen, i);*/
		}
		
	}
}




//Overloaded form Ktruss
__global__
void warp_detect_deleted_edges(
	uint* rowPtr, uint32_t numRows,
	bool* keep,
	uint* histogram)
{

	__shared__ uint32_t cnts[WARPS_PER_BLOCK];

	auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
	auto gtnum = blockDim.x * gridDim.x;
	auto gwid = gtid >> WARP_BITS;
	auto gwnum = gtnum >> WARP_BITS;
	auto lane = threadIdx.x & WARP_MASK;
	auto lwid = threadIdx.x >> WARP_BITS;

	for (auto u = gwid; u < numRows; u += gwnum) {
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
	
	std::cout.imbue(std::locale(""));
	cout << "TC = " << tc->count() << "\n";

	double secs = tc->kernel_time();
	int dev = tc->device();
	Log(LogPriorityEnum::info, "Kernel [%s]: gpu %d kernel time %f (%f teps) \n", message.c_str(), dev, secs, (numEdges_upto-edgeOffset) / secs);
	cudaDeviceSynchronize();


	return tc->count();
}

template<typename T>
void CountTrianglesHash(const int divideConstant, graph::TcBase<T>* tc, graph::COOCSRGraph_d<T>* g,
	const size_t numEdges, const size_t edgeOffset = 0, ProcessingElementEnum kernelType = Thread, int increasing = 0)
{

	const int minRowLen = 16*1024;
	const int maxRowLen = 32 * 1024;
	const int cNumBins = 512; //not used

	//Construct
	
	graph::GPUArray<uint> htp("hash table pointer", AllocationTypeEnum::unified, g->numNodes + 1, 0);
	graph::GPUArray<uint> htd("hash table", AllocationTypeEnum::gpu, numEdges - edgeOffset, 0);

	execKernel(createHashPointer2, (g->numNodes + 128 - 1) / 128, 128, false, *g, htp.gdata(), minRowLen, maxRowLen, divideConstant);


	//for (int i = 0; i < 100; i++)
	//	printf("%u, ", htp.gdata()[i]);
	//printf("\n");


	T total = CUBScanExclusive(htp.gdata(), htp.gdata(), g->numNodes);
	htp.gdata()[g->numNodes] = total;

	uint totalBins = htp.gdata()[g->numNodes];



	//for (int i = 0; i < 100; i++)
	//	printf("%u, ", htp.gdata()[i]);
	//printf("\n");


	graph::GPUArray<uint> hts("bins start per row", AllocationTypeEnum::unified, totalBins, 0);
	hts.setAll(0, true);
	execKernel(createHashStartBin, (g->numNodes + 128 - 1) / 128, 128, false, *g, htp.gdata(), hts.gdata(), minRowLen, maxRowLen, divideConstant);



	//Mode data to hash tables
	for (int i = 0; i < g->numNodes; i++)
	{
		const uint s = g->rowPtr[i];
		const uint e = g->rowPtr[i + 1];

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
				uint val = g->colInd[j];
				uint bin = hash1(val, numBins);
				uint elementBinStart = hts.cdata()[bin_start + bin];
				uint nextBinStart = hts.cdata()[bin_start + bin + 1];
				htd.cdata()[s + elementBinStart + binCounter[bin]] = val;

				if (s + elementBinStart + binCounter[bin] > e)
					printf("Shit\n");

				binCounter[bin]++;
			}
		}
		else
		{
			for (int j = s; j < e; j++)
			{
				htd.cdata()[j] = g->colInd[j];
			}
		}
	}

	htd.switch_to_gpu(0);



	tc->count_hash_async(divideConstant, g, htd, htp, hts, numEdges, edgeOffset, kernelType, increasing);
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