
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
#include "../triangle_counting/TcBinaryEncoding.cuh"

#include "../truss/cudaKtruss.cuh"
#include "../truss19/ourtruss19.cuh"
#include "../truss19/newTruss.cuh"
#include "../truss19/ourTruss19Warp.cuh"

#include "../graph_partition/cross_decmp.cuh"

#include "../include/main_support.cuh"

using namespace std;




#define __VS__ //visual studio debug

///*File Conversion */
//#define TSV_MARKET
//#define MARKET_BEL
//#define TSV_BEL
//#define BEL_MARKET


#define NORMAL
//#define Matrix_Stats
#define TC
//#define Cross_Decomposition
//#define TriListConstruct
//#define KTRUSS


int main(int argc, char** argv) {

	//CUDA_RUNTIME(cudaDeviceReset());

	printf("\033[0m");
	graph::MtB_Writer mwriter;
	auto fileSrc = argv[1];
	auto fileDst = argv[2];

#ifdef MARKET_BEL
	mwriter.write_market_bel<uint, int>(fileSrc, fileDst, false);
	return;
#endif

#ifdef TSV_BEL
	mwriter.write_tsv_bel<uint, uint>(fileSrc, fileDst);
	return;
#endif

#ifdef TSV_MARKET
	mwriter.write_tsv_market<uint, int>(fileSrc, fileDst);
	return;
#endif

#ifdef BEL_MARKET
	mwriter.write_bel_market<uint, int>(fileSrc, fileDst);
	return;
#endif

#ifndef NORMAL
	return;
#endif

//HERE is the normal program !!
	//1) Read File to EdgeList

	char* matr;

	matr = "D:\\graphs\\graph500-scale21-ef16_adj.bel";

#ifndef __VS__
	if (argc > 1)
		matr = argv[1];
#endif

	graph::EdgeListFile f(matr);
	std::vector<EdgeTy<uint>> edges;
	std::vector<EdgeTy<uint>> fileEdges;
	auto lowerTriangular = [](const Edge& e) { return e.first > e.second; };
	auto upperTriangular = [](const Edge& e) { return e.first < e.second; };
	auto full = [](const Edge& e) { return false; };
	while (f.get_edges(fileEdges, 100))
	{
		edges.insert(edges.end(), fileEdges.begin(), fileEdges.end());
	}

	bool sort = false;
	if (sort)
	{
		f.sort_edges(edges);
	}

	graph::CSRCOO<uint> csrcoo = graph::CSRCOO<uint>::from_edgelist(edges, full);
	int n = csrcoo.num_rows();
	int m = csrcoo.nnz();
	//2) Move to GPU
	graph::GPUArray<uint> sl("source", AllocationTypeEnum::cpuonly),
		dl("destination", AllocationTypeEnum::cpuonly),
		rowPtr("row pointer", AllocationTypeEnum::cpuonly);

	//2.a CPU
	sl.cdata() = csrcoo.row_ind();
	dl.cdata() = csrcoo.col_ind();
	rowPtr.cdata() = csrcoo.row_ptr();

#ifdef Matrix_Stats
	MatrixStats(csrcoo.nnz(), csrcoo.num_rows(), csrcoo.num_rows(), rowPtr.cdata(), dl.cdata());
	PrintMtarixStruct(csrcoo.nnz(), csrcoo.num_rows(), csrcoo.num_rows(), rowPtr.cdata(), dl.cdata());
#endif


#ifdef TC

	sl.switch_to_gpu(0, csrcoo.nnz());
	dl.switch_to_gpu(0, csrcoo.nnz());
	rowPtr.switch_to_gpu(0, csrcoo.num_rows() + 1);
	cudaDeviceSynchronize();

	//Count traingles binary-search: Thread or Warp
	uint step = csrcoo.nnz();
	uint st = 0;
	uint ee = st + step; // st + 2;
	graph::TcBase<uint>* tcb = new graph::TcBinary<uint>(0, ee, csrcoo.num_rows());
	graph::TcBase<uint>* tcNV = new graph::TcNvgraph<uint>(0, ee, csrcoo.num_rows());
	graph::TcBase<uint>* tcBE = new graph::TcBinaryEncoding<uint>(0, ee, csrcoo.num_rows());
	graph::TcBase<uint>* tc = new graph::TcSerial<uint>(0, ee, csrcoo.num_rows());

	const int divideConstant = 1;
	graph::TcBase<uint>* tchash = new graph::TcVariableHash<uint>(0, ee, csrcoo.num_rows());


	graph::BmpGpu<uint> bmp;
	bmp.InitBMP(csrcoo.num_rows(), csrcoo.nnz(), rowPtr, dl);
	bmp.bmpConstruct(csrcoo.num_rows(), csrcoo.nnz(), rowPtr, dl);

	while (st < csrcoo.nnz())
	{
		printf("Edge = %d\n", st);
		if (step == 1)
		{
			uint s = sl.cdata()[st];
			uint d = dl.cdata()[st];
			const uint srcStart = rowPtr.cdata()[s];
			const uint srcStop = rowPtr.cdata()[s + 1];

			const uint dstStart = rowPtr.cdata()[d];
			const uint dstStop = rowPtr.cdata()[d + 1];

			const uint dstLen = dstStop - dstStart;
			const uint srcLen = srcStop - srcStart;

			printf("S = (%u, %u, %u, %u) / D = (%u, %u, %u, %u)\n", s, srcStart, srcStop, srcLen, d, dstStart, dstStop, dstLen);

			printf("Source col ind = {");
			for (int i = 0; i < srcLen; i++)
				printf("%u,", dl.cdata()[srcStart + i]);
			printf("}\n");


			printf("Destenation col ind = {");
			for (int i = 0; i < dstLen; i++)
				printf("%u,", dl.cdata()[dstStart + i]);
			printf("}\n");
		}

		//uint64  serialTc = CountTriangles<uint>("Serial Thread", tc, rowPtr, sl, dl, ee, csrcoo.num_rows(), st, ProcessingElementEnum::Thread, 0);

		//CountTriangles<uint>("Serial Warp", tc, rowPtr, sl, dl, ee, csrcoo.num_rows(), st, ProcessingElementEnum::Warp, 0);
		uint64  binaryTc = CountTriangles<uint>("Binary Warp", tcb, rowPtr, sl, dl, ee, csrcoo.num_rows(), st, ProcessingElementEnum::Warp, 0);
		uint64  binarySharedTc = CountTriangles<uint>("Binary Warp Shared", tcb, rowPtr, sl, dl, ee, csrcoo.num_rows(), st, ProcessingElementEnum::WarpShared, 0);
		uint64  binarySharedCoalbTc = CountTriangles<uint>("Binary Warp Shared", tcb, rowPtr, sl, dl, ee, csrcoo.num_rows(), st, ProcessingElementEnum::Test, 0);

		uint64  binaryQueueTc = CountTriangles<uint>("Binary Queue", tcb, rowPtr, sl, dl, ee, csrcoo.num_rows(), st, ProcessingElementEnum::Queue, 0);

		uint64 binaryEncodingTc = CountTriangles<uint>("Binary Encoding", tcBE, rowPtr, sl, dl, ee, csrcoo.num_rows(), st, ProcessingElementEnum::Warp, 0);
		CountTrianglesHash<uint>(divideConstant, tchash, rowPtr, sl, dl, ee, csrcoo.num_rows(), 0, ProcessingElementEnum::Warp, 0);
		bmp.Count(csrcoo.num_rows(), rowPtr, dl);

		CountTriangles<uint>("NVGRAPH", tcNV, rowPtr, sl, dl, ee, csrcoo.num_rows());

		/*if (serialTc != binaryTc)
			break;*/
		st += step;
		ee += step;

		printf("------------------------------\n");

		break;
	}
#endif
#ifdef Cross_Decomposition
	sl.switch_to_gpu(0, csrcoo.nnz());
	dl.switch_to_gpu(0, csrcoo.nnz());
	rowPtr.switch_to_gpu(0, csrcoo.num_rows() + 1);
	Thanos<uint> t(rowPtr, sl, dl, csrcoo.nnz(), csrcoo.num_rows());
#endif
#ifdef TriListConstruct
	sl.switch_to_gpu(0, csrcoo.nnz());
	dl.switch_to_gpu(0, csrcoo.nnz());
	rowPtr.switch_to_gpu(0, csrcoo.num_rows() + 1);
	////Takes either serial or binary triangle Counter
	graph::TcBase<uint>* tcb = new graph::TcBinary<uint>(0, csrcoo.nnz(), csrcoo.num_rows());
	graph::GPUArray<uint> triPointer("tri Pointer", cpuonly);
	graph::GPUArray<uint> triIndex("tri Index", cpuonly);
	ConstructTriList(triIndex, triPointer, tcb, rowPtr, sl, dl, csrcoo.nnz(), csrcoo.num_rows(), 0, ProcessingElementEnum::Warp);
#endif
#ifdef KTRUSS

	//The problem with Ktruss that it physically changes the graph structure due to stream compaction !!
	sl.switch_to_unified(0, csrcoo.nnz());
	dl.switch_to_unified(0, csrcoo.nnz());
	rowPtr.switch_to_unified(0, csrcoo.num_rows() + 1);

//#define VLDB2020
#ifdef VLDB2020
	//We need unified to do stream compaction
	graph::GPUArray<int> output("KT Output", AllocationTypeEnum::unified, m / 2, 0);
	graph::BmpGpu<uint> bmp;
	bmp.getEidAndEdgeList(n, m, rowPtr, dl);// EID creation
	bmp.InitBMP(csrcoo.num_rows(), csrcoo.nnz(), rowPtr, dl);
	bmp.bmpConstruct(csrcoo.num_rows(), csrcoo.nnz(), rowPtr, dl);
	
	double tc_time = bmp.Count_Set(n, m, rowPtr, dl);
	#define MAX_LEVEL  (20000)
	auto level_start_pos = (uint*)calloc(MAX_LEVEL, sizeof(uint));

	graph::PKT_cuda(
		n, m,
		rowPtr, dl, bmp,
		nullptr,
		100, output, level_start_pos, 0, tc_time);
#endif

//#define OUR2019
#ifdef OUR2019
	graph::SingleGPU_Ktruss<uint> mohatruss(0);
	Timer t;
	mohatruss.findKtrussIncremental_sync(3, 1000, rowPtr, sl, dl,
		n, m, nullptr, nullptr, 0, 0);
	mohatruss.sync();
	double time = t.elapsed();
	Log(info, "count time %f s", time);
	Log(info, "MOHA %d ktruss (%f teps)", mohatruss.count(), m / time);*/
#endif	

#define OUR_NEW_KTRUSS
#ifdef OUR_NEW_KTRUSS
	//We need to change the graph representation
	graph::GPUArray<uint> rowIndex("Half Row Index", AllocationTypeEnum::unified, csrcoo.nnz() / 2, 0),
		colIndex("Half Col Index", AllocationTypeEnum::unified, csrcoo.nnz() / 2, 0),
		EID("EID", AllocationTypeEnum::unified, csrcoo.nnz(), 0),
		asc("ASC temp", AllocationTypeEnum::unified, csrcoo.nnz(), 0);

	Timer t_init;
	graph::GPUArray<bool> keep("Keep temp", AllocationTypeEnum::unified, csrcoo.nnz(), 0);


	execKernel(init, (csrcoo.nnz() + 512 - 1) / 512, 512, false, (uint)csrcoo.nnz(), asc.gdata(), keep.gdata(), rowPtr.gdata(), sl.gdata(), dl.gdata());

	CUBSelect(asc.gdata(), asc.gdata(), keep.gdata(), m);
	CUBSelect(sl.gdata(), rowIndex.gdata(), keep.gdata(), m);
	uint newNumEdges = CUBSelect(dl.gdata(), colIndex.gdata(), keep.gdata(), m);
	execKernel(InitEid, (newNumEdges + 512 - 1) / 512, 512, false, newNumEdges, asc.gdata(), rowIndex.gdata(), colIndex.gdata(), rowPtr.gdata(), dl.gdata(), EID.gdata());
	asc.freeGPU();
	keep.freeGPU();
	double time_init = t_init.elapsed();
	Log(info, "Create EID (by malmasri): %f s", time_init);


	graph::SingleGPU_KtrussMod<uint> mohatrussM(0);

	Timer t;
	graph::TcBase<uint>* tcb = new graph::TcBinary<uint>(0, csrcoo.nnz(), csrcoo.num_rows(), mohatrussM.stream());

	mohatrussM.findKtrussIncremental_sync(3, 1000, tcb, rowPtr,  dl,
		rowIndex, colIndex, EID,
		n, m, nullptr, nullptr, 0, 0);
	mohatrussM.sync();
	double time = t.elapsed();


	Log(info, "count time %f s", time);
	Log(info, "MOHA %d ktruss (%f teps)", mohatrussM.count(), m / time);
#endif

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


#endif
	printf("Done ....\n");
	sl.freeGPU();
	dl.freeGPU();
	rowPtr.freeGPU();
	//A.freeGPU();
	//B.freeGPU();
	return 0;
}


