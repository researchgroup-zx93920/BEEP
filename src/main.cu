
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
#include "../triangle_counting/TcVariablehash.cuh"
#include "../triangle_counting/TcNvgraph.cuh"
#include "../include/CSRCOO.cuh"
#include "../triangle_counting/testHashing.cuh"
#include "../triangle_counting/TcBmp.cuh"
#include "../truss/cudaKtruss.cuh"
#include "../truss19/ourtruss19.cuh"
#include "../truss19/newTruss.cuh"

using namespace std;


template<typename T>
int CountTriangles(graph::TcBase<T> *tc, graph::GPUArray<T> rowPtr, graph::GPUArray<T> rowInd, graph::GPUArray<T> colInd,
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


	return tc->count();
}

template<typename T>
void CountTrianglesHash(const int divideConstant, graph::TcBase<T>* tc, graph::GPUArray<T> rowPtr, graph::GPUArray<T> rowInd, graph::GPUArray<T> colInd,
	const size_t numEdges, const size_t numRows, const size_t edgeOffset = 0, ProcessingElementEnum kernelType = Thread, int increasing = 0)
{

	const int minRowLen = 8*1024;
	const int maxRowLen = 32 * 1024;
	const int cNumBins = 512; //not used

	//Construct
	auto hash1 = [](uint val, uint div) { return (val / 11) % div; };
	graph::GPUArray<uint> htp("hash table pointer", AllocationTypeEnum::gpu, numRows + 1, 0);
	graph::GPUArray<uint> htd("hash table", AllocationTypeEnum::gpu, numEdges-edgeOffset, 0);

	htp.cdata()[0] = 0;
	for (int i = 0; i < numRows; i++)
	{
		uint s = rowPtr.cdata()[i];
		uint e = rowPtr.cdata()[i + 1];

		if ((e - s) >= minRowLen && (e-s) < maxRowLen)
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

#define __DEBUG__

int main(int argc, char **argv){

	//CUDA_RUNTIME(cudaDeviceReset());

	printf("\033[0m");


	//1) Read File to EdgeList
	
	char* matr2, *matr1;
	matr2 = "D:\\graphs\\amazon0601_new.bel";
	matr1 = "D:\\graphs\\amazon0601_adj.bel";
	//matr = "D:\\graphs\\cit-Patents\\cit-Patents.bel";






	#ifndef __DEBUG__
	if(argc > 1)
		matr = argv[1];
	#endif

	graph::EdgeListFile f(matr1);

	std::vector<EdgeTy<uint>> edges;
	std::vector<EdgeTy<uint>> fileEdges;
	auto lowerTriangular = [](const Edge& e) { return e.first > e.second; };
	auto upperTriangular = [](const Edge& e) { return e.first < e.second; };
	auto full = [](const Edge& e) { return false; };
	while (f.get_edges(fileEdges, 100)) 
	{
		edges.insert(edges.end(), fileEdges.begin(), fileEdges.end());
	}




	graph::EdgeListFile f2(matr2);

	std::vector<EdgeTy<uint>> edges2;
	std::vector<EdgeTy<uint>> fileEdges2;
	while (f2.get_edges(fileEdges2, 100))
	{
		edges2.insert(edges2.end(), fileEdges2.begin(), fileEdges2.end());
	}

	graph::MtB_Writer m;
	m.write_market_bel<uint, int>("D:\\graphs\\amazon0601.mtx", "D:\\graphs\\amazon0601_new.bel", false);

	graph::CSRCOO<int> csrcoo = graph::CSRCOO<int>::from_edgelist(edges, upperTriangular);
	//graph::CSRCOO<uint> csrcooFull = graph::CSRCOO<uint>::from_edgelist(edges, full);

#pragma region TCTEST
	//2) Move to GPU
	graph::GPUArray<int> sl("source", AllocationTypeEnum::cpuonly), 
		dl("destination", AllocationTypeEnum::cpuonly), 
		rowPtr("row pointer", AllocationTypeEnum::cpuonly);

	//2.a CPU
	sl.cdata() = csrcoo.row_ind();
	dl.cdata() = csrcoo.col_ind();
	rowPtr.cdata() = csrcoo.row_ptr();

	cudaDeviceSynchronize();

	//2.b GPU
	sl.switch_to_gpu(0, csrcoo.nnz());
	dl.switch_to_gpu(0, csrcoo.nnz());
	rowPtr.switch_to_gpu(0, csrcoo.num_rows()+1);
	cudaDeviceSynchronize();

	
	//Count triangle serially
	/*graph::TcBase<uint> *tc = new graph::TcSerial<uint>(0, csrcoo.nnz(), csrcoo.num_rows());
	CountTriangles<uint>(tc, rowPtr, sl, dl, csrcoo.nnz(), csrcoo.num_rows(), 0, ProcessingElementEnum::Thread, 0);*/


	////Count traingles binary-search: Thread or Warp
	int st = 0;
	int ee = csrcoo.nnz(); // st + 2;
	graph::TcBase<int> *tcb = new graph::TcBinary<int>(0, ee, csrcoo.num_rows());

	graph::TcBase<int>* tcNV = new graph::TcNvgraph<int>(0, ee, csrcoo.num_rows());

	while (true)
	{

		printf("Edge = %d\n", st);
		int trueVal = CountTriangles<int>(tcb, rowPtr, sl, dl, ee, csrcoo.num_rows(), st, ProcessingElementEnum::Warp, 0);
		int testVal = CountTriangles<int>(tcb, rowPtr, sl, dl, ee, csrcoo.num_rows(), st, ProcessingElementEnum::Test, 0);

		CountTriangles<int>(tcNV, rowPtr, sl, dl, ee, csrcoo.num_rows());

		if (trueVal != testVal)
			break;
		st += 2;
		ee += 2;

		printf("------------------------------\n");
		break;
	}

	////Takes either serial or binary triangle Counter
	//graph::GPUArray<uint> triPointer("tri Pointer", cpuonly);
	//graph::GPUArray<uint> triIndex("tri Index", cpuonly);
	//ConstructTriList(triIndex, triPointer, tcb, rowPtr, sl, dl, csrcoo.nnz(), csrcoo.num_rows(), 0, ProcessingElementEnum::Warp);


	////Now I will start TC with hash
	//
	//const int divideConstant = 1;
	//graph::TcBase<uint>* tchash = new graph::TcVariableHash<uint>(0, ee, csrcoo.num_rows());
	//CountTrianglesHash<uint>(divideConstant,tchash, rowPtr, sl, dl, ee, csrcoo.num_rows(), 0, ProcessingElementEnum::Warp, 0);

	//int n = csrcoo.num_rows();
	//int m = csrcoo.nnz();
	//graph::BmpGpu<uint> bmp;
	//bmp.InitBMP(n, m, rowPtr, dl);
	//bmp.bmpConstruct(n, m, rowPtr, dl);
	//double tc_time = bmp.Count(n, rowPtr, dl);


#pragma endregion


	//Now bmp (binary packing)
	//Here we need the full graph --> Ktruss
	//graph::GPUArray<uint> slFull("source Full ", AllocationTypeEnum::cpuonly),
	//	dlFull("destination Full ", AllocationTypeEnum::cpuonly),
	//	rowPtrFull("row pointer Full ", AllocationTypeEnum::cpuonly);

	////2.a CPU
	//slFull.cdata() = csrcooFull.row_ind();
	//dlFull.cdata() = csrcooFull.col_ind();
	//rowPtrFull.cdata() = csrcooFull.row_ptr();

	//////2.b GPU
	//slFull.switch_to_gpu(0, csrcooFull.nnz());
	//dlFull.switch_to_gpu(0, csrcooFull.nnz());
	//rowPtrFull.switch_to_unified(0, csrcooFull.num_rows() + 1);
	//cudaDeviceSynchronize();
	//
	//
	//graph::BmpGpu<uint> bmp;
	//int n = csrcooFull.num_rows();
	//int m = csrcooFull.nnz();

	//bmp.InitBMP(n, m, rowPtrFull, dlFull);
	////bmp.getEidAndEdgeList(n, m, rowPtrFull, dlFull);
	//bmp.bmpConstruct(n, m, rowPtrFull, dlFull);
	//double tc_time = bmp.Count(n, rowPtrFull, dlFull);



	//graph::IterHelper<uint> h(n,m);
	//auto process_functor = [&h](int level) {
	///*	PKT_processSubLevel_intersection_handling_skew(iter_helper.g, iter_helper.curr_, iter_helper.in_curr_,
	//		iter_helper.curr_tail_,
	//		*iter_helper.edge_sup_ptr_, level, iter_helper.next_,
	//		iter_helper.in_next_, &iter_helper.next_tail_,
	//		iter_helper.processed_, *iter_helper.edge_lst_ptr_,
	//		iter_helper.off_end_,
	//		iter_helper.is_vertex_updated_, iter_helper);*/
	//};

	//graph::AbstractBKT(n, m, rowPtrFull, dlFull, bmp, &h, process_functor);


	//GPU Ktruss BMP
	//graph::GPUArray<int> output("KT Output", AllocationTypeEnum::gpu, m / 2, 0);


	//#define MAX_LEVEL  (20000)
	//auto level_start_pos = (uint*)calloc(MAX_LEVEL, sizeof(uint));
	//
	//graph::PKT_cuda(
	//	n, m,
	//	rowPtrFull, dlFull, bmp,
	//	nullptr,
	//	100, output, level_start_pos, 0, tc_time);



	//graph::SingleGPU_Ktruss<uint> mohatruss(0);

	//Timer t;
	//mohatruss.findKtrussIncremental_sync(3, 1000, rowPtrFull, slFull, dlFull,
	//	n, m, nullptr, nullptr, 0, 0);
	//mohatruss.sync();
	//double time = t.elapsed();
	//

	//Log(info, "count time %.f s", time);
	//Log(info, "MOHA %d ktruss (%f teps)", mohatruss.count(), m / time);
	//




	//graph::SingleGPU_KtrussMod<uint> mohatrussM(0);

	//Timer t;
	//graph::TcBase<uint>* tcb = new graph::TcSerial<uint>(0, csrcooFull.nnz(), csrcooFull.num_rows(), mohatrussM.stream());
	//mohatrussM.findKtrussIncremental_sync(3, 1000, tcb, rowPtrFull, slFull, dlFull,
	//	n, m, nullptr, nullptr, 0, 0);
	//mohatrussM.sync();
	//double time = t.elapsed();


	//Log(info, "count time %.f s", time);
	//Log(info, "MOHA %d ktruss (%f teps)", mohatrussM.count(), m / time);



#pragma region MyRegion
	////Hashing tests
////More tests on GPUArray
//graph::GPUArray<uint> A("Test A: In", AllocationTypeEnum::cpuonly);
//graph::GPUArray<uint> B("Test B: In", AllocationTypeEnum::cpuonly);



//const uint inputSize = pow<int,int>(2, 18) - 1;
//
//
//A.cdata() = new uint[inputSize];
//B.cdata() = new uint[inputSize];

//const int dimBlock = 512;
//const int dimGrid = (inputSize + (dimBlock)-1) / (dimBlock);

//srand(220);

//A.cdata()[0] = 1;
//B.cdata()[0] = 1;
//for (uint i = 1; i < inputSize; i++)
//{
//	A.cdata()[i] = A.cdata()[i - 1] + (rand() % 13) + 1;
//	B.cdata()[i] = B.cdata()[i - 1] + (rand() % 13) + 1;
//}

//graph::Hashing::goldenSerialIntersectionCPU<uint>(A, B, inputSize);

//A.switch_to_gpu(0, inputSize);
//B.switch_to_gpu(0, inputSize);

//graph::Hashing::goldenBinaryGPU<uint>(A, B, inputSize);

////1-level hash with binary search for stash
//graph::Hashing::test1levelHashing<uint>(A, B, inputSize, 4);

////Non-stash hashing
//graph::Hashing::testHashNosStash<uint>(A, B, inputSize, 5);


////Store binary tree traversal: Now assume full binary tree
//vector<uint> treeSizes;
//
//int maxInputSize;
//int levels = 1;
//for (int i = 1; i < 19; i++)
//{
//	int v = pow<int, int>(2, i) - 1;

//	if (inputSize >= v)
//	{
//		maxInputSize = v;
//		levels = i;
//	}
//	else
//		break;
//}

//graph::GPUArray<Node> BT("Binary Traverasl", gpu, maxInputSize, 0);
//graph::GPUArray<uint> BTV("Binary Traverasl", gpu, maxInputSize, 0);
//
//int cl = 0;
//int totalElements = 1;

//int element0 = maxInputSize / 2;
//BT.cdata()[0].i = element0;
//BTV.cdata()[0] = B.cdata()[element0];
//BT.cdata()[0].l = 0;
//BT.cdata()[0].r = maxInputSize;

//while (totalElements < maxInputSize)
//{
//	int num_elem_lev = pow<int, int>(2, cl);
//	int levelStartIndex = pow<int, int>(2, cl) - 1;
//	for (int i = levelStartIndex; i < num_elem_lev + levelStartIndex; i++)
//	{
//		Node parent = BT.cdata()[i];
//		
//		//left 
//		int leftIndex = 2 * i + 1; //New Index
//		int leftVal = (parent.l + parent.i) / 2; //PrevIndex

//		BT.cdata()[leftIndex].i = leftVal;
//		BT.cdata()[leftIndex].l = parent.l;
//		BT.cdata()[leftIndex].r = parent.i;
//		BTV.cdata()[leftIndex] = B.cdata()[leftVal];
//		BT.cdata()[leftIndex].p = i;


//		//right
//		int rightIndex = 2 * i + 2; //New Index
//		int rightVal = (parent.i + 1 + parent.r) / 2; //PrevIndex

//		BT.cdata()[rightIndex].i = rightVal;
//		BT.cdata()[rightIndex].l = parent.i + 1;
//		BT.cdata()[rightIndex].r = parent.r;
//		BTV.cdata()[rightIndex] = B.cdata()[rightVal];
//		BT.cdata()[rightIndex].p = i;

//		totalElements += 2;
//	}
//	cl++;
//}

//BTV.switch_to_gpu(0);

//const auto startBST = stime();
//graph::GPUArray<uint> countBST("Test Binary Search Tree search: Out", AllocationTypeEnum::unified, 1, 0);
//graph::binary_search_bst_g<uint, 32> << <1, 32 >> > (countBST.gdata(), A.gdata(), inputSize, BTV.gdata(), inputSize);
//cudaDeviceSynchronize();
//double elapsedBST = elapsedSec(startBST);
//printf("BST Elapsed time = %f, Count = %u \n", elapsedBST, countBST.cdata()[0]);

//BTV.freeCPU();
//BTV.freeGPU();

//BT.freeCPU();
//BT.freeGPU();


////For weighted edges
////std::vector<WEdgeTy<uint, wtype>> wedges;
////std::vector<WEdgeTy<uint,wtype>> wfileEdges;
////while (f.get_weighted_edges(wfileEdges, 10)) {
////	wedges.insert(wedges.end(), wfileEdges.begin(), wfileEdges.end());
////}

#pragma endregion



    printf("Done ....\n");
	/*slFull.freeGPU();
	dlFull.freeGPU();
	rowPtrFull.freeGPU();*/
	//A.freeGPU();
	//B.freeGPU();
    return 0;
}


