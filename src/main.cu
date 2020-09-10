
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

#include "../kcore/kcore.cuh"






using namespace std;




#define __VS__ //visual studio debug

///*File Conversion */
//#define TSV_MARKET
//#define MARKET_BEL
//#define TSV_BEL
//#define BEL_MARKET


#define NORMAL
//#define Matrix_Stats
//#define TC
//#define Cross_Decomposition
//#define TriListConstruct
//#define KTRUSS
#define KCORE


int main(int argc, char** argv) {

	//CUDA_RUNTIME(cudaDeviceReset());
	
	printf("\033[0m");
	printf("Welcome ---------------------\n");
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

	matr = "D:\\graphs\\amazon0601_adj.bel";

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
	graph::COOCSRGraph<uint> g;
	g.capacity = m;
	g.numEdges = m;
	g.numNodes = n;
	
	g.rowPtr = new graph::GPUArray<uint>("Row pointer", AllocationTypeEnum::cpuonly);
	g.rowInd= new graph::GPUArray<uint>("Src Index", AllocationTypeEnum::cpuonly);
	g.colInd = new graph::GPUArray<uint>("Dst Index", AllocationTypeEnum::cpuonly);
	g.rowPtr->cdata() = csrcoo.row_ptr();
	g.rowInd->cdata() = csrcoo.row_ind();
	g.colInd->cdata() = csrcoo.col_ind();



	//No try the ew storage format
	graph::TiledCOOCSRGraph<uint>* gtiled;
	coo2tiledcoocsrOnDevice(g, 32,  gtiled, unified);
	//graph::TiledCOOCSRGraph_d<uint> gtiled_d;
	/*gtiled_d.numNodes = gtiled->numNodes;
	gtiled_d.numNodes = gtiled->numNodes;
	gtiled_d.tileSize = gtiled->tileSize;
	gtiled_d.tilesPerDim = gtiled->tilesPerDim;
	gtiled_d.tileRowPtr = gtiled->tileRowPtr->gdata();
	gtiled_d.rowInd = gtiled->rowInd->gdata();
	gtiled_d.colInd = gtiled->colInd->gdata();*/

	unsigned int numThreadsPerBlock = 128;
	unsigned int numBlocks = (m + numThreadsPerBlock - 1) / numThreadsPerBlock;

	graph::GPUArray<uint64> c("C", unified, 1, 0);
	c.setSingle(0, 0, true);


	cudaDeviceSynchronize();

	cudaEvent_t kernelStart_;
	cudaEvent_t kernelStop_;
	CUDA_RUNTIME(cudaEventCreate(&kernelStart_));
	CUDA_RUNTIME(cudaEventCreate(&kernelStop_));


	CUDA_RUNTIME(cudaEventRecord(kernelStart_));
	count_triangles_kernel<uint, 128> << < dim3(numBlocks, 2), numThreadsPerBlock >> > (c.gdata(), gtiled->numNodes,
	gtiled->numEdges,
	gtiled->tilesPerDim,
	gtiled->tileSize,
	gtiled->capacity,
	gtiled->tileRowPtr->gdata(),
	gtiled->rowInd->gdata(),
	gtiled->colInd->gdata());
	CUDA_RUNTIME(cudaEventRecord(kernelStop_));

	float ms;
	CUDA_RUNTIME(cudaEventSynchronize(kernelStop_));
	CUDA_RUNTIME(cudaEventElapsedTime(&ms, kernelStart_, kernelStop_));
	cudaDeviceSynchronize();
	cudaGetLastError();

	printf("Tiled Count = %lu, time = %f\n", *c.gdata(), ms/1e3);
	

#ifdef Matrix_Stats
	MatrixStats(csrcoo.nnz(), csrcoo.num_rows(), csrcoo.num_rows(), rowPtr.cdata(), dl.cdata());
	PrintMtarixStruct(csrcoo.nnz(), csrcoo.num_rows(), csrcoo.num_rows(), rowPtr.cdata(), dl.cdata());
#endif


#ifdef TC

	graph::COOCSRGraph_d<uint>* gd;
	to_csrcoo_device(g, gd); //got to device !!
	cudaDeviceSynchronize();

	//Count traingles binary-search: Thread or Warp
	uint step = csrcoo.nnz();
	uint st = 0;
	uint ee = st + step; // st + 2;
	graph::TcBase<uint>* tcb = new graph::TcBinary<uint>(0, ee, n);
	graph::TcBase<uint>* tcNV = new graph::TcNvgraph<uint>(0, ee, n);
	graph::TcBase<uint>* tcBE = new graph::TcBinaryEncoding<uint>(0, ee, n);
	graph::TcBase<uint>* tc = new graph::TcSerial<uint>(0, ee, n);

	const int divideConstant = 1;
	graph::TcBase<uint>* tchash = new graph::TcVariableHash<uint>(0, ee, n);


	graph::BmpGpu<uint> bmp;
	bmp.InitBMP(*gd);
	bmp.bmpConstruct(*gd);

	while (st < csrcoo.nnz())
	{
		printf("Edge = %d\n", st);
		if (step == 1)
		{
			uint s = g.rowInd->cdata()[st];
			uint d = g.colInd->cdata()[st];
			const uint srcStart = g.rowPtr->cdata()[s];
			const uint srcStop = g.rowPtr->cdata()[s + 1];

			const uint dstStart = g.rowPtr->cdata()[d];
			const uint dstStop = g.rowPtr->cdata()[d + 1];

			const uint dstLen = dstStop - dstStart;
			const uint srcLen = srcStop - srcStart;

			printf("S = (%u, %u, %u, %u) / D = (%u, %u, %u, %u)\n", s, srcStart, srcStop, srcLen, d, dstStart, dstStop, dstLen);

			printf("Source col ind = {");
			for (int i = 0; i < srcLen; i++)
				printf("%u,", g.colInd->cdata()[srcStart + i]);
			printf("}\n");


			printf("Destenation col ind = {");
			for (int i = 0; i < dstLen; i++)
				printf("%u,", g.colInd->cdata()[dstStart + i]);
			printf("}\n");
		}

		//bmp.Count(*gd);

		//uint64  serialTc = CountTriangles<uint>("Serial Thread", tc, gd, ee, st, ProcessingElementEnum::Thread, 0);

		//CountTriangles<uint>("Serial Warp", tc, rowPtr, sl, dl, ee, csrcoo.num_rows(), st, ProcessingElementEnum::Warp, 0);
		uint64  binaryTc = CountTriangles<uint>("Binary Warp", tcb, gd, ee, st, ProcessingElementEnum::Warp, 0);
		uint64  binarySharedTc = CountTriangles<uint>("Binary Warp Shared", tcb, gd, ee,  st, ProcessingElementEnum::WarpShared, 0);
		uint64  binarySharedCoalbTc = CountTriangles<uint>("Binary Warp Shared", tcb,gd,  ee,  st, ProcessingElementEnum::Test, 0);

		//

		////uint64 binaryEncodingTc = CountTriangles<uint>("Binary Encoding", tcBE, rowPtr, sl, dl, ee, csrcoo.num_rows(), st, ProcessingElementEnum::Warp, 0);
		////CountTrianglesHash<uint>(divideConstant, tchash, rowPtr, sl, dl, ee, csrcoo.num_rows(), 0, ProcessingElementEnum::Warp, 0);
	
		uint64  binaryQueueTc = CountTriangles<uint>("Binary Queue", tcb, gd, ee, st, ProcessingElementEnum::Queue, 0);
	
		//CountTriangles<uint>("NVGRAPH", tcNV, rowPtr, sl, dl, ee, csrcoo.num_rows());

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


#ifdef KCORE
	graph::COOCSRGraph_d<uint>* gd;
	to_csrcoo_device(g, gd); //got to device !!
	cudaDeviceSynchronize();

	graph::SingleGPU_Kcore<uint> mohacore(0);
	Timer t;
	mohacore.findKcoreIncremental_async(3, 1000, *gd, 0, 0);
	mohacore.sync();
	double time = t.elapsed();
	Log(info, "count time %f s", time);
	Log(info, "MOHA %d kcore (%f teps)", mohacore.count(), m / time);
#endif

#ifdef KTRUSS

	//The problem with Ktruss that it physically changes the graph structure due to stream compaction !!
	graph::COOCSRGraph_d<uint>* gd;
	to_csrcoo_device(g, gd); //got to device !!

//#define VLDB2020
#ifdef VLDB2020
	//We need unified to do stream compaction
	graph::GPUArray<int> output("KT Output", AllocationTypeEnum::unified, m / 2, 0);
	graph::BmpGpu<uint> bmp;
	bmp.getEidAndEdgeList(g);// CPU
	bmp.InitBMP(*gd);
	bmp.bmpConstruct(*gd);
	
	double tc_time = bmp.Count_Set(*gd);
	#define MAX_LEVEL  (20000)
	auto level_start_pos = (uint*)calloc(MAX_LEVEL, sizeof(uint));

	graph::PKT_cuda(
		n, m,
		*g.rowPtr, *g.colInd, bmp,
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
	graph::GPUArray<uint> rowIndex("Half Row Index", AllocationTypeEnum::unified, m / 2, 0),
		colIndex("Half Col Index", AllocationTypeEnum::unified, m / 2, 0),
		EID("EID", AllocationTypeEnum::unified, m, 0),
		asc("ASC temp", AllocationTypeEnum::unified, m, 0);

	Timer t_init;
	graph::GPUArray<bool> keep("Keep temp", AllocationTypeEnum::unified, m, 0);


	execKernel(init, (m + 512 - 1) / 512, 512, false, *gd, asc.gdata(), keep.gdata());

	CUBSelect(asc.gdata(), asc.gdata(), keep.gdata(), m);
	CUBSelect(gd->rowInd, rowIndex.gdata(), keep.gdata(), m);
	uint newNumEdges = CUBSelect(gd->colInd, colIndex.gdata(), keep.gdata(), m);
	execKernel(InitEid, (newNumEdges + 512 - 1) / 512, 512, false, newNumEdges, asc.gdata(), rowIndex.gdata(), colIndex.gdata(), gd->rowPtr, gd->colInd, EID.gdata());
	asc.freeGPU();
	keep.freeGPU();
	double time_init = t_init.elapsed();
	Log(info, "Create EID (by malmasri): %f s", time_init);

	graph::EidGraph_d<uint> geid;
	geid.numNodes = n;
	geid.capacity = m;
	geid.numEdges = m;
	geid.rowPtr_csr = gd->rowPtr;
	geid.colInd_csr = gd->colInd;
	geid.rowInd = rowIndex.gdata();
	geid.colInd = colIndex.gdata();
	geid.eid = EID.gdata();

	graph::SingleGPU_KtrussMod<uint> mohatrussM(0);

	Timer t;
	graph::TcBase<uint>* tcb = new graph::TcBinary<uint>(0, m, n, mohatrussM.stream());

	mohatrussM.findKtrussIncremental_sync(3, 1000, tcb, geid, nullptr, nullptr, 0, 0);
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
	
	//A.freeGPU();
	//B.freeGPU();
	return 0;
}


