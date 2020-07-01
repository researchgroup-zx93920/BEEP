#include <cuda_runtime.h>
#include <iostream>
#include<string>
#include <fstream>
#include <map>

#include "omp.h"
#include<vector>

#include "../include/Logger.cuh"
#include "../include/FIleReader.cuh"
#include "../include/CGArray.cuh"
#include "../include/TriCountPrim.cuh"
#include "../triangle_counting/TcBase.cuh"
#include "../triangle_counting/TcSerial.cuh"
#include "../triangle_counting/TcBinary.cuh"
#include "../include/CSRCOO.cuh"

using namespace std;

__global__ void add(int *data, int count)
{
	int thread = threadIdx.x;
	
	if (thread < count)
	{
		printf("Hello\n");
	}
}


template<typename T>
void CountTriangles(graph::TcBase<T> *tc, graph::GPUArray<T> rowPtr, graph::GPUArray<T> rowInd, graph::GPUArray<T> colInd,
	const size_t numEdges, const size_t numRows, const size_t edgeOffset = 0, ProcessingElementEnum kernelType = Thread, int increasing = 0)
{
	tc->count_async(rowPtr, rowInd, colInd, numEdges, edgeOffset, kernelType, increasing);
	tc->sync();
	CUDA_RUNTIME(cudaGetLastError());
	printf("TC = %d\n", tc->count());
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


 timepoint stime()
{
	 return std::chrono::system_clock::now();
}

 double elapsedSec(timepoint start)
 {
	 return (std::chrono::system_clock::now() - start).count() / 1e9;
 }
int main(int argc, char **argv){

	CUDA_RUNTIME(cudaDeviceReset());

	printf("\033[0m");


	//1) Read File to EdgeList
	char matr[] = "D:\\graphs\\Theory-16-25-81-B1k.bel";

	graph::EdgeListFile f(matr);

	std::vector<EdgeTy<uint>> edges;
	std::vector<EdgeTy<uint>> fileEdges;
	auto lowerTriangular = [](const Edge& e) { return e.first > e.second; };
	while (f.get_edges(fileEdges, 100)) 
	{
		edges.insert(edges.end(), fileEdges.begin(), fileEdges.end());
	}
	graph::CSRCOO<uint> csrcoo = graph::CSRCOO<uint>::from_edgelist(edges, lowerTriangular);

	//2) Move to GPU
	graph::GPUArray<uint> sl("source", AllocationTypeEnum::cpuonly), 
		dl("destination", AllocationTypeEnum::cpuonly), 
		neil("row pointer", AllocationTypeEnum::cpuonly);

	//2.a CPU
	sl.cdata() = csrcoo.row_ind();
	dl.cdata() = csrcoo.col_ind();
	neil.cdata() = csrcoo.row_ptr();

	//2.b GPU
	sl.switch_to_gpu(0, csrcoo.nnz());
	dl.switch_to_gpu(0, csrcoo.nnz());
	neil.switch_to_gpu(0, csrcoo.num_rows()+1);


	for (int i = 10; i < csrcoo.num_rows(); i++)
	{
		uint s = neil.cdata()[i];
		uint e = neil.cdata()[i+1];
		bool exists = false;
		for (int j = s; j < e; j++)
		{
			
			if (dl.cdata()[j] < i)
			{
				printf("%u,", dl.cdata()[j]);
				exists = true;
			}
		}
		if(exists)
			printf(" ---> Row %u\n", i);
	}

	
	//Count triangle serially
	graph::TcBase<uint> *tc = new graph::TcSerial<uint>(0, csrcoo.nnz(), csrcoo.num_rows());
	CountTriangles<uint>(tc, neil, sl, dl, csrcoo.nnz(), csrcoo.num_rows(), 0, ProcessingElementEnum::Thread, 0);



	//Count traingles binary-search: Thread or Warp
	int ee = csrcoo.nnz();
	graph::TcBase<uint> *tcb = new graph::TcBinary<uint>(0, ee, csrcoo.num_rows());
	CountTriangles<uint>(tcb, neil, sl, dl, ee, csrcoo.num_rows(), 0, ProcessingElementEnum::Thread, 0);

	//Takes either serial or binary triangle Counter
	graph::GPUArray<uint> triPointer("tri Pointer", cpuonly);
	graph::GPUArray<uint> triIndex("tri Index", cpuonly);
	ConstructTriList(triIndex, triPointer, tcb, neil, sl, dl, csrcoo.nnz(), csrcoo.num_rows(), 0, ProcessingElementEnum::Warp);



	//Hashing tests
	//More tests on GPUArray

	graph::GPUArray<uint> A("Test A: In", AllocationTypeEnum::cpuonly);
	graph::GPUArray<uint> B("Test B: In", AllocationTypeEnum::cpuonly);



	const uint inputSize = 1000000;
	A.cdata() = new uint[inputSize];
	B.cdata() = new uint[inputSize];

	srand(20);
	
	A.cdata()[0] = 1;
	B.cdata()[0] = 1;
	for (uint i = 1; i < inputSize; i++)
	{
		A.cdata()[i] = A.cdata()[i - 1] + (rand() % 15) + 1;
		B.cdata()[i] = B.cdata()[i - 1] + (rand() % 15) + 1;
	}

	//Sanity check
	uint ap = 0;
	uint bp = 0;
	uint a, b;
	int count = 0;
	while (ap < inputSize && bp < inputSize)
	{
		a = A.cdata()[ap];
		b = B.cdata()[bp];
		if (a == b) {
			++count;
			//printf("%u, ", a);
			++ap;
			++bp;
		}
		else if (a < b) {
			++ap;

		}
		else {
			++bp;
		}
	}
	printf("\nTrue Count = %d\n", count);


	A.switch_to_gpu(0, inputSize);
	B.switch_to_gpu(0, inputSize);


	graph::GPUArray<uint> BH("Hashing test B", gpu, inputSize + inputSize / 2, 0);
	const int binSize = 4;
	const uint numBins = (inputSize + binSize-1) / binSize;
	const uint stashStart = binSize * numBins;
	const uint stashLimit = inputSize / 2;
	int binOcc[numBins];
	int stashSize = 0;
	for (int i = 0; i < numBins; i++)
	{
		binOcc[i] = 0;
		BH.cdata()[i * binSize] = 0xFFFFFFFF;
	}

	for (int i = 0; i < inputSize; i++)
	{
		uint v = B.cdata()[i];
		uint b = v % numBins;

		if (binOcc[b] < binSize)
		{
			BH.cdata()[b * binSize + binOcc[b]] = v;
			binOcc[b]++;
			if (binOcc[b] < binSize)
				BH.cdata()[b * binSize + binOcc[b]] = 0xFFFFFFFF;
		}
		else if (stashSize < stashLimit)
		{
			BH.cdata()[stashStart + stashSize] = v;
			stashSize++;
		}
		else
		{
			printf("Shit\n");
		}
	}

	BH.switch_to_gpu(0);

	const int dimBlock = 512;
	const int dimGrid = (inputSize + (dimBlock)-1) / (dimBlock);

	const auto startHash = stime();
	graph::GPUArray<uint> countHash("Test hash search: Out", AllocationTypeEnum::unified, 1, 0);
	graph::hash_search_g<uint, dimBlock> << <dimGrid, dimBlock >> > (countHash.gdata(), A.gdata(), inputSize, BH.gdata(), inputSize, binSize, stashSize);
	cudaDeviceSynchronize();
	double elapsedHash = elapsedSec(startHash);
	printf("Hash Elapsed time = %f, Count = %u Stash Size=%d\n", elapsedHash, countHash.cdata()[0], stashSize);


	const auto start = stime();
	graph::GPUArray<uint> countBinary("Test binary search: Out", AllocationTypeEnum::unified, 1, 0);
	graph::binary_search_2arr_g<uint, dimBlock> << <dimGrid, dimBlock >> > (countBinary.gdata(), A.gdata(), inputSize, B.gdata(), inputSize);
	cudaDeviceSynchronize();
	double elapsed = elapsedSec(start);
	printf("Binary Elapsed time = %f, Count = %u\n", elapsed, countBinary.cdata()[0]);


	//Binary tree as array
	//1: Restructure







	//For weighted edges
	//std::vector<WEdgeTy<uint, wtype>> wedges;
	//std::vector<WEdgeTy<uint,wtype>> wfileEdges;
	//while (f.get_weighted_edges(wfileEdges, 10)) {
	//	wedges.insert(wedges.end(), wfileEdges.begin(), wfileEdges.end());
	//}


    printf("Done ....\n");
	//sl.free();
	//dl.free();
	////neil.free();
	A.free();
	B.free();
	BH.free();
    return 0;
}


