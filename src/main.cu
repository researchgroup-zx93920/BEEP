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
#include "../triangle_counting/testHashing.cuh"

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


 
struct Node
{
	uint val;
	int i;
	int r;
	int l;
	int p;
};

int main(int argc, char **argv){

	//CUDA_RUNTIME(cudaDeviceReset());

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

	cudaDeviceSynchronize();

	//2.b GPU
	sl.switch_to_gpu(0, csrcoo.nnz());
	dl.switch_to_gpu(0, csrcoo.nnz());
	neil.switch_to_gpu(0, csrcoo.num_rows()+1);
	cudaDeviceSynchronize();

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



	const uint inputSize = pow<int,int>(2, 18) - 1;
	
	
	A.cdata() = new uint[inputSize];
	B.cdata() = new uint[inputSize];

	const int dimBlock = 512;
	const int dimGrid = (inputSize + (dimBlock)-1) / (dimBlock);

	srand(220);

	A.cdata()[0] = 1;
	B.cdata()[0] = 1;
	for (uint i = 1; i < inputSize; i++)
	{
		A.cdata()[i] = A.cdata()[i - 1] + (rand() % 13) + 1;
		B.cdata()[i] = B.cdata()[i - 1] + (rand() % 13) + 1;
	}

	graph::Hashing::goldenSerialIntersectionCPU<uint>(A, B, inputSize);

	A.switch_to_gpu(0, inputSize);
	B.switch_to_gpu(0, inputSize);

	graph::Hashing::goldenBinaryGPU<uint>(A, B, inputSize);

	//1-level hash with binary search for stash
	graph::Hashing::test1levelHashing<uint>(A, B, inputSize, 4);

	//Non-stash hashing
	graph::Hashing::testHashNosStash<uint>(A, B, inputSize, 5);


	//Store binary tree traversal: Now assume full binary tree
	vector<uint> treeSizes;
	
	int maxInputSize;
	int levels = 1;
	for (int i = 1; i < 19; i++)
	{
		int v = pow<int, int>(2, i) - 1;

		if (inputSize >= v)
		{
			maxInputSize = v;
			levels = i;
		}
		else
			break;
	}

	graph::GPUArray<Node> BT("Binary Traverasl", gpu, maxInputSize, 0);
	graph::GPUArray<uint> BTV("Binary Traverasl", gpu, maxInputSize, 0);
	
	int cl = 0;
	int totalElements = 1;

	int element0 = maxInputSize / 2;
	BT.cdata()[0].i = element0;
	BTV.cdata()[0] = B.cdata()[element0];
	BT.cdata()[0].l = 0;
	BT.cdata()[0].r = maxInputSize;

	while (totalElements < maxInputSize)
	{
		int num_elem_lev = pow<int, int>(2, cl);
		int levelStartIndex = pow<int, int>(2, cl) - 1;
		for (int i = levelStartIndex; i < num_elem_lev + levelStartIndex; i++)
		{
			Node parent = BT.cdata()[i];
			
			//left 
			int leftIndex = 2 * i + 1; //New Index
			int leftVal = (parent.l + parent.i) / 2; //PrevIndex

			BT.cdata()[leftIndex].i = leftVal;
			BT.cdata()[leftIndex].l = parent.l;
			BT.cdata()[leftIndex].r = parent.i;
			BTV.cdata()[leftIndex] = B.cdata()[leftVal];
			BT.cdata()[leftIndex].p = i;


			//right
			int rightIndex = 2 * i + 2; //New Index
			int rightVal = (parent.i + 1 + parent.r) / 2; //PrevIndex

			BT.cdata()[rightIndex].i = rightVal;
			BT.cdata()[rightIndex].l = parent.i + 1;
			BT.cdata()[rightIndex].r = parent.r;
			BTV.cdata()[rightIndex] = B.cdata()[rightVal];
			BT.cdata()[rightIndex].p = i;

			totalElements += 2;
		}
		cl++;
	}

	BTV.switch_to_gpu(0);

	const auto startBST = stime();
	graph::GPUArray<uint> countBST("Test Binary Search Tree search: Out", AllocationTypeEnum::unified, 1, 0);
	graph::binary_search_bst_g<uint, 32> << <1, 32 >> > (countBST.gdata(), A.gdata(), inputSize, BTV.gdata(), inputSize);
	cudaDeviceSynchronize();
	double elapsedBST = elapsedSec(startBST);
	printf("BST Elapsed time = %f, Count = %u \n", elapsedBST, countBST.cdata()[0]);

	BTV.freeCPU();
	BTV.freeGPU();

	BT.freeCPU();
	BT.freeGPU();


	//For weighted edges
	//std::vector<WEdgeTy<uint, wtype>> wedges;
	//std::vector<WEdgeTy<uint,wtype>> wfileEdges;
	//while (f.get_weighted_edges(wfileEdges, 10)) {
	//	wedges.insert(wedges.end(), wfileEdges.begin(), wfileEdges.end());
	//}


    printf("Done ....\n");
	sl.freeGPU();
	dl.freeGPU();
	neil.freeGPU();
	A.freeGPU();
	B.freeGPU();
    return 0;
}


