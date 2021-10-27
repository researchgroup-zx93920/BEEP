
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
#include "../kclique/kclique.cuh"
#include "../kclique/kclique_local.cuh"
#include "../kclique/mclique.cuh"


#include "../include/Config.h"
#include "../include/ScanLarge.cuh"


using namespace std;
//#define TriListConstruct

int main(int argc, char** argv)
{

    //CUDA_RUNTIME(cudaDeviceReset());
    Config config = parseArgs(argc, argv);

    printf("\033[0m");
    printf("Welcome ---------------------\n");
    printConfig(config);

    graph::MtB_Writer mwriter;
    auto fileSrc = config.srcGraph;
    auto fileDst = config.dstGraph;
    if (config.mt == CONV_MTX_BEL) {
        mwriter.write_market_bel<uint, int>(fileSrc, fileDst, false);
        return;
    }

    if (config.mt == CONV_TSV_BEL) {
        mwriter.write_tsv_bel<uint64, uint64>(fileSrc, fileDst);
        return;
    }

    if (config.mt == CONV_TSV_MTX) {
        mwriter.write_tsv_market<uint, int>(fileSrc, fileDst);
        return;
    }

    if (config.mt == CONV_BEL_MTX) {
        mwriter.write_bel_market<uint, int>(fileSrc, fileDst);
        return;
    }

    if (config.mt == CONV_TXT_BEL) {
        mwriter.write_txt_bel<uint, uint>(fileSrc, fileDst, true, 2, 0);
        return;
    }




    //test Scan
    /*const int size = 256;

    graph::CubLarge<uint> s;
    graph::GPUArray<uint>
        asc("ASC temp", AllocationTypeEnum::unified, size, 0);
    graph::GPUArray<bool> keep("Keep temp", AllocationTypeEnum::unified, size, 0);
    keep.setAll(true, true);

    keep.gdata()[1] = false;
    keep.gdata()[2] = false;
    keep.gdata()[3] = false;

    keep.gdata()[128] = false;
    keep.gdata()[255] = false;


    execKernel((initAsc<uint, PeelType>), (size + 512 - 1) / 512, 512, false, asc.gdata(), size);

    uint t = s.Select(asc.gdata(), keep.gdata(), asc.gdata(), size);*/


    //HERE is the normal program !!
        //1) Read File to EdgeList

    Timer read_graph_timer;

    const char* matr = config.srcGraph;
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

    if (config.sortEdges)
    {
        f.sort_edges(edges);
    }


    //Importatnt
    graph::CSRCOO<uint> csrcoo;
    if (config.orient == Upper)
        csrcoo = graph::CSRCOO<uint>::from_edgelist(edges, lowerTriangular);
    else if (config.orient == Lower)
        csrcoo = graph::CSRCOO<uint>::from_edgelist(edges, upperTriangular);
    else
        csrcoo = graph::CSRCOO<uint>::from_edgelist(edges, full);

    uint n = csrcoo.num_rows();
    uint m = csrcoo.nnz();


    graph::COOCSRGraph<uint> g;
    g.capacity = m;
    g.numEdges = m;
    g.numNodes = n;

    g.rowPtr = new graph::GPUArray<uint>("Row pointer", AllocationTypeEnum::noalloc, n+1, config.deviceId, true );
    g.rowInd = new graph::GPUArray<uint>("Src Index", AllocationTypeEnum::noalloc,  m, config.deviceId, true );
    g.colInd = new graph::GPUArray<uint>("Dst Index", AllocationTypeEnum::noalloc,  m, config.deviceId, true);
    uint *rp, *ri, *ci;
    cudaMallocHost((void**)&rp, (n+1)*sizeof(uint));
    cudaMallocHost((void**)&ri, (m)*sizeof(uint));
    cudaMallocHost((void**)&ci, (m)*sizeof(uint));
    CUDA_RUNTIME(cudaMemcpy(rp, csrcoo.row_ptr(), (n+1)*sizeof(uint), cudaMemcpyKind::cudaMemcpyHostToHost));
    CUDA_RUNTIME(cudaMemcpy(ri, csrcoo.row_ind(), (m)*sizeof(uint) , cudaMemcpyKind::cudaMemcpyHostToHost));
    CUDA_RUNTIME(cudaMemcpy(ci, csrcoo.col_ind(), (m)*sizeof(uint), cudaMemcpyKind::cudaMemcpyHostToHost));

    g.rowPtr->cdata() = rp; g.rowPtr->setAlloc(cpuonly);
    g.rowInd->cdata() = ri; g.rowInd->setAlloc(cpuonly);
    g.colInd->cdata() = ci; g.colInd->setAlloc(cpuonly);

    Log(info, "Read graph time: %f s", read_graph_timer.elapsed());

    ///Now we need to orient the graph
    Timer total_timer;

    graph::COOCSRGraph_d<uint>* gd = (graph::COOCSRGraph_d<uint>*)malloc(sizeof(graph::COOCSRGraph_d<uint>));
    g.rowPtr->switch_to_gpu(config.deviceId);

    gd->numNodes = g.numNodes;
    gd->numEdges = g.numEdges;
    gd->capacity = g.capacity;
    gd->rowPtr = g.rowPtr->gdata();

    if(!config.isSmall || g.numEdges > 500000000)
    {
        gd->rowInd = g.rowInd->cdata();
        gd->colInd = g.colInd->cdata();

    }
    else
    {
        g.rowInd->switch_to_gpu(config.deviceId);
        g.colInd->switch_to_gpu(config.deviceId);
        gd->rowInd = g.rowInd->gdata();
        gd->colInd = g.colInd->gdata();
    }
    double total = total_timer.elapsed();
    Log(info, "Transfer Time: %f s", total);

    Timer t;
    graph::SingleGPU_Kcore<uint, PeelType> mohacore(config.deviceId);
    if (config.orient == Degree || config.orient == Degeneracy)
    {
        if (config.orient == Degeneracy)
            mohacore.findKcoreIncremental_async(3, 1000, *gd, 0, 0);
        else if (config.orient == Degree)
            mohacore.getNodeDegree(*gd);

        graph::GPUArray<uint> rowInd_half("Half Row Index", config.allocation, m / 2, config.deviceId),
            colInd_half("Half Col Index", config.allocation, m / 2, config.deviceId),
            new_rowPtr("New Row Pointer", config.allocation, n + 1, config.deviceId),
            asc("ASC temp", AllocationTypeEnum::unified, m, config.deviceId);
        graph::GPUArray<bool> keep("Keep temp", AllocationTypeEnum::unified, m, config.deviceId);

        if (config.orient == Degree)
        {
            execKernel((init<uint, PeelType>), ((m - 1) / 51200) + 1, 512, config.deviceId, false, *gd, asc.gdata(), keep.gdata(), mohacore.nodeDegree.gdata());
        }
        else if (config.orient == Degeneracy)
        {
            execKernel((init<uint, PeelType>), ((m - 1) / 51200) + 1, 512, config.deviceId, false, *gd, asc.gdata(), keep.gdata(), mohacore.nodeDegree.gdata(), mohacore.nodePriority.gdata());
        }

        graph::CubLarge<uint> s(config.deviceId);
        uint newNumEdges;
        if (m < INT_MAX)
        {
            //CUBSelect(asc.gdata(), asc.gdata(), keep.gdata(), m, config.deviceId);
            CUBSelect(gd->rowInd, rowInd_half.gdata(), keep.gdata(), m, config.deviceId);
            newNumEdges = CUBSelect(gd->colInd, colInd_half.gdata(), keep.gdata(), m, config.deviceId);
        }
        else
        {
            //s.Select(asc.gdata(), asc.gdata(), keep.gdata(), m);
            // s.Select(gd->rowInd, rowInd_half.gdata(), keep.gdata(), m);
            // newNumEdges = s.Select(gd->colInd, colInd_half.gdata(), keep.gdata(), m);

            newNumEdges = s.Select2(gd->rowInd,  gd->colInd,  rowInd_half.gdata(), colInd_half.gdata(), keep.gdata(), m);
        }


        execKernel((warp_detect_deleted_edges<uint>), (32 * n + 128 - 1) / 128, 128, config.deviceId, false, gd->rowPtr, n, keep.gdata(), new_rowPtr.gdata());
        uint total = CUBScanExclusive<uint, uint>(new_rowPtr.gdata(), new_rowPtr.gdata(), n, config.deviceId, 0, config.allocation);
        new_rowPtr.setSingle(n, total, false);
        //assert(total == new_edge_num * 2);
        cudaDeviceSynchronize();
        asc.freeGPU();
        keep.freeGPU();
        free_csrcoo_device(g);

        m = m / 2;

        g.capacity = m;
        g.numEdges = m;
        g.numNodes = n;

        g.rowPtr = &new_rowPtr;
        g.rowInd = &rowInd_half;
        g.colInd = &colInd_half;
        //to_csrcoo_device(g, gd, config.deviceId, config.allocation); //got to device !!

        gd->numNodes = g.numNodes;
        gd->numEdges = g.numEdges;
        gd->capacity = g.capacity;
        gd->rowPtr = new_rowPtr.gdata();
        gd->rowInd = g.rowInd->gdata();
        gd->colInd = g.colInd->gdata();


    }

    // cudaFreeHost(rp);
    // cudaFreeHost(ri);
    // cudaFreeHost(ci);

    double time_init = t.elapsed();
    if (config.orient == Degree || config.orient == Degeneracy)
    {
        Log(info, "Preprocess time: %f s", time_init);
    }


    //Just need to verify some new storage format
    //graph::GPUArray<uint> countCont("Half Row Index", AllocationTypeEnum::unified, m, 0);
    //for (int i = 0; i < g.numNodes; i++)
    //{
    //	uint s = g.rowPtr->cdata()[i];
    //	uint e = g.rowPtr->cdata()[i + 1];
    //	for (int j = s; j < e; j++)
    //	{
    //		uint v = g.colInd->cdata()[j];

    //		uint s2 = g.rowPtr->cdata()[v];
    //		uint e2 = g.rowPtr->cdata()[v + 1];

    //		if (e2 - s2 < e - s)
    //			printf("What !!!!\n");

    //	}
    //}



    /*double time = t.elapsed();
    Log(info, "count time %f s", time);
    Log(info, "MOHA %d kcore (%f teps)", mohacore.count(), m / time);*/

    uint dv = 32;
    typedef unsigned int ttt;
    if (config.printStats) {
        MatrixStats(m, n, n, g.rowPtr->cdata(), g.colInd->cdata());
        PrintMtarixStruct(m, n, n, g.rowPtr->cdata(), g.colInd->cdata());


        ////////////////// intersection !!
        printf("Now # of bytes we need to make this matrix binary encoded !!\n");

        uint64 sum = 0;
        uint64 sumc = 0;
        for (uint i = 0; i < n; i++)
        {
            uint s = g.rowPtr->cdata()[i];
            uint d = g.rowPtr->cdata()[i + 1];
            uint deg = d - s;

            // if(i >=37 && i<44)
            // {
            // 	printf("For %u, %u, %u, %u\n", i, d-s, g.colInd->cdata()[s], g.colInd->cdata()[s + 1]);
            // }


            //if (deg > 128)
            {
                uint64 v = deg * (deg + dv - 1) / dv;
                sum += v;

                //now the compressed one :D
                uint64 nelem8 = deg / dv;
                uint64 rem = deg - nelem8 * dv;

                sumc += dv * nelem8 * (1 + nelem8) / 2;
                sumc += rem * (1 + nelem8);
            }
        }

        printf("n = %u, m = %u, elements = %llu\n", n, m, sum);
        printf("n = %u, m = %u, elements = %llu\n", n, m, sumc);

        uint src = 3541; // index id
        uint s = g.rowPtr->cdata()[src];
        uint d = g.rowPtr->cdata()[src + 1];
        uint degree = d - s;
        while (degree < 50)
        {
            src++;
            s = g.rowPtr->cdata()[src];
            d = g.rowPtr->cdata()[src + 1];
            degree = d - s;
        }

        uint divisions = (degree + dv - 1) / dv;
        graph::GPUArray<ttt> node_be("BE", unified, divisions * degree, 0);
        node_be.setAll(0, true);
        for (uint i = 0; i < degree; i++)
        {
            uint dst = g.colInd->cdata()[i + s];
            uint dstStart = g.rowPtr->cdata()[dst];
            uint dstEnd = g.rowPtr->cdata()[dst + 1];
            uint dstDegree = dstEnd - dstStart;

            //Intersect Src, Dst
            uint s1 = 0, s2 = 0;
            bool loadA = true, loadB = true;
            uint a, b;
            uint rsi = 0;
            uint offset = 0;
            while (s1 < degree && s2 < dstDegree)
            {

                if (loadA) {
                    a = g.colInd->cdata()[s1 + s];
                    loadA = false;
                }
                if (loadB) {
                    b = g.colInd->cdata()[s2 + dstStart];
                    loadB = false;
                }

                if (a == b) {
                    uint startIndex = i * divisions;
                    uint divIndex = s1 / dv;
                    uint inDivIndex = s1 % dv;
                    node_be.cdata()[startIndex + divIndex] |= (1 << inDivIndex);


                    //i and s1
                    //if (i > 0)
                    //{
                    //	if (i > s1)
                    //	{
                    //		uint ss = i / dv;
                    //		uint sum = dv * ss * (ss + 1) / 2;
                    //		uint sr = i % dv;
                    //		uint sumr = sr * ((i + dv - 1) / dv) - 1;

                    //		rsi = sum + sumr;
                    //		offset = s1 / dv;
                    //		uint numBytes = (i + dv - 1) / dv;
                    //		uint byteIndex = s1 % dv;

                    //		//Encode
                    //		node_be.cdata()[rsi + offset] |= (1 << byteIndex);


                    //	}
                    //	else
                    //	{
                    //		uint ss = s1 / dv;
                    //		uint sum = dv * ss * (ss + 1) / 2;
                    //		uint sr = s1 % dv;
                    //		uint sumr = sr * ((s1 + dv - 1) / dv) - 1;

                    //		rsi = sum + sumr;
                    //		offset = i / dv;
                    //		uint numBytes = (s1 + dv - 1) / dv;
                    //		uint byteIndex = i % dv;
                    //		node_be.cdata()[rsi + offset] |= (1 << byteIndex);
                    //	}


                    ++s1;
                    ++s2;
                    loadA = true;
                    loadB = true;
                }
                else if (a < b) {
                    ++s1;
                    loadA = true;
                }
                else {
                    ++s2;
                    loadB = true;
                }
            }

        }


    }


    if (config.mt == TC)
    {
        /*CUDA_RUNTIME(cudaSetDevice(config.deviceId));
        graph::TiledCOOCSRGraph<uint>* gtiled;
        coo2tiledcoocsrOnDevice(g, 16, gtiled, unified);
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
        count_triangles_kernel<uint, 128> << < dim3(numBlocks, 16), numThreadsPerBlock >> > (c.gdata(), gtiled->numNodes,
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

        printf("Tiled Count = %lu, time = %f\n", *c.gdata(), ms / 1e3);*/


        //Count traingles binary-search: Thread or Warp
        uint step = m;
        uint st = 0;
        uint ee = st + step; // st + 2;
        graph::TcBase<uint>* tcb = new graph::TcBinary<uint>(config.deviceId, ee, n);
        graph::TcBase<uint>* tcNV = new graph::TcNvgraph<uint>(config.deviceId, ee, n);
        graph::TcBase<uint>* tcBE = new graph::TcBinaryEncoding<uint>(config.deviceId, ee, n);
        graph::TcBase<uint>* tc = new graph::TcSerial<uint>(config.deviceId, ee, n);

        const int divideConstant = 10;
        graph::TcBase<uint>* tchash = new graph::TcVariableHash<uint>(config.deviceId, ee, n);


        graph::BmpGpu<uint> bmp(config.deviceId);
        bmp.InitBMP(*gd);
        bmp.bmpConstruct(*gd, config.allocation);

        while (st < m)
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

            bmp.Count(*gd);

            uint64  serialTc = CountTriangles<uint>("Serial Thread", config.deviceId, config.allocation, tc, gd, ee, st, ProcessingElementEnum::Thread, 0);

            ////CountTriangles<uint>("Serial Warp", tc, rowPtr, sl, dl, ee, csrcoo.num_rows(), st, ProcessingElementEnum::Warp, 0);
            uint64  binaryTc = CountTriangles<uint>("Binary Warp", config.deviceId, config.allocation, tcb, gd, ee, st, ProcessingElementEnum::Block, 0);
            uint64  binarySharedTc = CountTriangles<uint>("Binary Warp Shared", config.deviceId, config.allocation, tcb, gd, ee, st, ProcessingElementEnum::WarpShared, 0);
            uint64  binarySharedCoalbTc = CountTriangles<uint>("Binary Warp Shared", config.deviceId, config.allocation, tcb, gd, ee, st, ProcessingElementEnum::Test, 0);

            uint64 binaryEncodingTc = CountTriangles<uint>("Binary Encoding", config.deviceId, config.allocation, tcBE, gd, ee, st, ProcessingElementEnum::Warp, 0);
            CountTrianglesHash<uint>(config.deviceId, divideConstant, tchash, g, gd, ee, 0, ProcessingElementEnum::Warp, 0);

            uint64  binaryQueueTc = CountTriangles<uint>("Binary Queue", config.deviceId, config.allocation, tcb, gd, ee, st, ProcessingElementEnum::Queue, 0);

            CountTriangles<uint>("NVGRAPH", config.deviceId, config.allocation, tcNV, gd, ee);

            /*if (serialTc != binaryTc)
                break;*/
            st += step;
            ee += step;

            printf("------------------------------\n");

            break;
        }
    }
    else if (config.mt == CROSSDECOMP)
    {
        //Update Please
    /*	sl.switch_to_gpu(0, csrcoo.nnz());
        dl.switch_to_gpu(0, csrcoo.nnz());
        rowPtr.switch_to_gpu(0, csrcoo.num_rows() + 1);
        Thanos<uint> t(rowPtr, sl, dl, csrcoo.nnz(), csrcoo.num_rows());*/
    }

    //Not needed anymore
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


    if (config.mt == KCORE)
    {
        graph::COOCSRGraph_d<uint>* gd;
        to_csrcoo_device(g, gd, config.deviceId, config.allocation); //got to device !!
        cudaDeviceSynchronize();

        graph::SingleGPU_Kcore<uint, PeelType> mohacore(config.deviceId);
        Timer t;
        mohacore.findKcoreIncremental_async(3, 1000, *gd, 0, 0);
        mohacore.sync();
        double time = t.elapsed();
        Log(info, "count time %f s", time);
        Log(info, "MOHA %d kcore (%f teps)", mohacore.count(), m / time);
    }

    if (config.mt == KCLIQUE)
    {
        if (config.orient == None)
            Log(warn, "Redundunt K-cliques, Please orient the graph\n");



        // if(config.processElement == BlockWarp)
        // {
        // 	graph::SingleGPU_Kclique_NoOutQueue<uint> mohaclique(config.deviceId, *gd);
        // 	for (int i = 0; i < 3; i++)
        // 	{
        // 		Timer t;
        // 		if (config.processBy == ByNode)
        // 			mohaclique.findKclqueIncremental_node_async(config.k, *gd, config.processElement);
        // 		else if (config.processBy == ByEdge)
        // 			mohaclique.findKclqueIncremental_edge_async(config.k, *gd, config.processElement);
        // 		mohaclique.sync();
        // 		double time = t.elapsed();
        // 		Log(info, "count time %f s", time);
        // 		Log(info, "MOHA %d k-clique (%f teps)", mohaclique.count(), m / time);
        // 	}
        // 	mohaclique.free();
        // }
        // else
        {

            //read the nCr values, to be saved in (Constant or global or )

            graph::SingleGPU_Kclique<uint> mohaclique(config.deviceId, *gd);


            KcliqueConfig kcc = config.kcConfig;

            // int k = 4;
            // while ( k < 11)
            {
                //printf("------------------ K=%d ----------------------\n", k);
                for (int i = 0; i < 1; i++)
                {
                    Timer t;
                    if (config.processBy == ByNode)
                    {
                        if(kcc.Algo == GraphOrient)
                        {

                            if(kcc.BinaryEncode)
                            {

                                switch(kcc.PartSize)
                                {
                                    case 32:
                                    mohaclique.findKclqueIncremental_node_binary_async<32>(config.k, *gd, config.processElement);
                                    break;
                                    case 16:
                                    mohaclique.findKclqueIncremental_node_binary_async<16>(config.k, *gd, config.processElement);
                                    break;
                                    case 8:
                                    mohaclique.findKclqueIncremental_node_binary_async<8>(config.k, *gd, config.processElement);
                                    break;
                                    case 4:
                                    mohaclique.findKclqueIncremental_node_binary_async<4>(config.k, *gd, config.processElement);
                                    break;
                                    case 2:
                                    mohaclique.findKclqueIncremental_node_binary_async<2>(config.k, *gd, config.processElement);
                                    break;
                                    case 1:
                                    mohaclique.findKclqueIncremental_node_binary_async<1>(config.k, *gd, config.processElement);
                                    break;
                                    default:
                                        Log(error, "WRONG PARTITION SIZE SELECTED\n");


                                }


                            }
                            else
                            {


                                switch(kcc.PartSize)
                                {
                                    case 32:
                                    mohaclique.findKclqueIncremental_node_async<32>(config.k, *gd, config.processElement);
                                    break;
                                    case 16:
                                    mohaclique.findKclqueIncremental_node_async<16>(config.k, *gd, config.processElement);
                                    break;
                                    case 8:
                                    mohaclique.findKclqueIncremental_node_async<8>(config.k, *gd, config.processElement);
                                    break;
                                    case 4:
                                    mohaclique.findKclqueIncremental_node_async<4>(config.k, *gd, config.processElement);
                                    break;
                                    case 2:
                                    mohaclique.findKclqueIncremental_node_async<2>(config.k, *gd, config.processElement);
                                    break;
                                    case 1:
                                    mohaclique.findKclqueIncremental_node_async<1>(config.k, *gd, config.processElement);
                                    break;
                                    default:
                                        Log(error, "WRONG PARTITION SIZE SELECTED\n");


                                }

                                //mohaclique.findKclqueIncremental_node_async(config.k, *gd, config.processElement);
                            }

                        }
                        else // Pivoting
                        {
                            if(kcc.BinaryEncode)
                            {


                                switch(kcc.PartSize)
                                {
                                    case 32:
                                    mohaclique.findKclqueIncremental_node_pivot_async<32>(config.k, *gd, config.processElement);
                                    break;
                                    case 16:
                                    mohaclique.findKclqueIncremental_node_pivot_async<16>(config.k, *gd, config.processElement);
                                    break;
                                    case 8:
                                    mohaclique.findKclqueIncremental_node_pivot_async<8>(config.k, *gd, config.processElement);
                                    break;
                                    case 4:
                                    mohaclique.findKclqueIncremental_node_pivot_async<4>(config.k, *gd, config.processElement);
                                    break;
                                    case 2:
                                    mohaclique.findKclqueIncremental_node_pivot_async<2>(config.k, *gd, config.processElement);
                                    break;
                                    case 1:
                                    mohaclique.findKclqueIncremental_node_pivot_async<1>(config.k, *gd, config.processElement);
                                    break;
                                    default:
                                        Log(error, "WRONG PARTITION SIZE SELECTED\n");


                                }

                                //mohaclique.findKclqueIncremental_node_pivot_async(config.k, *gd, config.processElement);
                            }
                            else
                            {

                                switch(kcc.PartSize)
                                {
                                    case 32:
                                    mohaclique.findKclqueIncremental_node_nobin_pivot_async<32>(config.k, *gd, config.processElement);
                                    break;
                                    case 16:
                                    mohaclique.findKclqueIncremental_node_nobin_pivot_async<16>(config.k, *gd, config.processElement);
                                    break;
                                    case 8:
                                    mohaclique.findKclqueIncremental_node_nobin_pivot_async<8>(config.k, *gd, config.processElement);
                                    break;
                                    case 4:
                                    mohaclique.findKclqueIncremental_node_nobin_pivot_async<4>(config.k, *gd, config.processElement);
                                    break;
                                    case 2:
                                    mohaclique.findKclqueIncremental_node_nobin_pivot_async<2>(config.k, *gd, config.processElement);
                                    break;
                                    case 1:
                                    mohaclique.findKclqueIncremental_node_nobin_pivot_async<1>(config.k, *gd, config.processElement);
                                    break;
                                    default:
                                        Log(error, "WRONG PARTITION SIZE SELECTED\n");


                                }

                                //mohaclique.findKclqueIncremental_node_nobin_pivot_async(config.k, *gd, config.processElement);
                            }

                        }
                    }
                    else if (config.processBy == ByEdge)
                    {
                        if(kcc.Algo == GraphOrient)
                        {

                            if(kcc.BinaryEncode)
                            {

                                switch(kcc.PartSize)
                                {
                                    case 32:
                                    mohaclique.findKclqueIncremental_edge_binary_async<32>(config.k, *gd, config.processElement);
                                    break;
                                    case 16:
                                    mohaclique.findKclqueIncremental_edge_binary_async<16>(config.k, *gd, config.processElement);
                                    break;
                                    case 8:
                                    mohaclique.findKclqueIncremental_edge_binary_async<8>(config.k, *gd, config.processElement);
                                    break;
                                    case 4:
                                    mohaclique.findKclqueIncremental_edge_binary_async<4>(config.k, *gd, config.processElement);
                                    break;
                                    case 2:
                                    mohaclique.findKclqueIncremental_edge_binary_async<2>(config.k, *gd, config.processElement);
                                    break;
                                    case 1:
                                    mohaclique.findKclqueIncremental_edge_binary_async<1>(config.k, *gd, config.processElement);
                                    break;
                                    default:
                                        Log(error, "WRONG PARTITION SIZE SELECTED\n");


                                }

                                //mohaclique.findKclqueIncremental_edge_binary_async(config.k, *gd, config.processElement);
                            }
                            else
                            {

                                switch(kcc.PartSize)
                                {
                                    case 32:
                                    mohaclique.findKclqueIncremental_edge_async<32>(config.k, *gd, config.processElement);
                                    break;
                                    case 16:
                                    mohaclique.findKclqueIncremental_edge_async<16>(config.k, *gd, config.processElement);
                                    break;
                                    case 8:
                                    mohaclique.findKclqueIncremental_edge_async<8>(config.k, *gd, config.processElement);
                                    break;
                                    case 4:
                                    mohaclique.findKclqueIncremental_edge_async<4>(config.k, *gd, config.processElement);
                                    break;
                                    case 2:
                                    mohaclique.findKclqueIncremental_edge_async<2>(config.k, *gd, config.processElement);
                                    break;
                                    case 1:
                                    mohaclique.findKclqueIncremental_edge_async<1>(config.k, *gd, config.processElement);
                                    break;
                                    default:
                                        Log(error, "WRONG PARTITION SIZE SELECTED\n");


                                }

                                //mohaclique.findKclqueIncremental_edge_async(config.k, *gd, config.processElement);
                            }

                        }
                        else //Pivoting
                        {
                            if(kcc.BinaryEncode)
                            {

                                switch(kcc.PartSize)
                                {
                                    case 32:
                                    mohaclique.findKclqueIncremental_edge_pivot_async<32>(config.k, *gd, config.processElement);
                                    break;
                                    case 16:
                                    mohaclique.findKclqueIncremental_edge_pivot_async<16>(config.k, *gd, config.processElement);
                                    break;
                                    case 8:
                                    mohaclique.findKclqueIncremental_edge_pivot_async<8>(config.k, *gd, config.processElement);
                                    break;
                                    case 4:
                                    mohaclique.findKclqueIncremental_edge_pivot_async<4>(config.k, *gd, config.processElement);
                                    break;
                                    case 2:
                                    mohaclique.findKclqueIncremental_edge_pivot_async<2>(config.k, *gd, config.processElement);
                                    break;
                                    case 1:
                                    mohaclique.findKclqueIncremental_edge_pivot_async<1>(config.k, *gd, config.processElement);
                                    break;
                                    default:
                                        Log(error, "WRONG PARTITION SIZE SELECTED\n");


                                }

                                //mohaclique.findKclqueIncremental_edge_pivot_async(config.k, *gd, config.processElement);
                            }
                            else
                            {

                                switch(kcc.PartSize)
                                {
                                    case 32:
                                    mohaclique.findKclqueIncremental_edge_nobin_pivot_async<32>(config.k, *gd, config.processElement);
                                    break;
                                    case 16:
                                    mohaclique.findKclqueIncremental_edge_nobin_pivot_async<16>(config.k, *gd, config.processElement);
                                    break;
                                    case 8:
                                    mohaclique.findKclqueIncremental_edge_nobin_pivot_async<8>(config.k, *gd, config.processElement);
                                    break;
                                    case 4:
                                    mohaclique.findKclqueIncremental_edge_nobin_pivot_async<4>(config.k, *gd, config.processElement);
                                    break;
                                    case 2:
                                    mohaclique.findKclqueIncremental_edge_nobin_pivot_async<2>(config.k, *gd, config.processElement);
                                    break;
                                    case 1:
                                    mohaclique.findKclqueIncremental_edge_nobin_pivot_async<1>(config.k, *gd, config.processElement);
                                    break;
                                    default:
                                        Log(error, "WRONG PARTITION SIZE SELECTED\n");

                                }

                                //mohaclique.findKclqueIncremental_edge_nobin_pivot_async(config.k, *gd, config.processElement);
                            }

                        }
                    }
                    mohaclique.sync();
                    double time = t.elapsed();
                    Log(info, "count time %f s", time);
                    Log(info, "MOHA %d k-clique (%f teps)", mohaclique.count(), m / time);
                }

                //k++;
            }


        }
    }

    if (config.mt == KCLIQUE_LOCAL)
    {
        if (config.orient == None)
            Log(warn, "Redundunt K-cliques, Please orient the graph\n");

        // read the nCr values, to be saved in (Constant or global or )
        graph::SingleGPU_Kclique_Local<uint> localclique(config.deviceId, *gd);

        KcliqueConfig kcc = config.kcConfig;

        Timer t;
        if (config.processBy == ByNode)
        {
            if(kcc.Algo == GraphOrient)
            {
                if(kcc.BinaryEncode)
                {
                    switch(kcc.PartSize)
                    {
                        case 32:
                        localclique.findKclqueIncremental_node_binary_async_local<32>(config.k, *gd);
                        break;
                        case 16:
                        localclique.findKclqueIncremental_node_binary_async_local<16>(config.k, *gd);
                        break;
                        case 8:
                        localclique.findKclqueIncremental_node_binary_async_local<8>(config.k, *gd);
                        break;
                        case 4:
                        localclique.findKclqueIncremental_node_binary_async_local<4>(config.k, *gd);
                        break;
                        case 2:
                        localclique.findKclqueIncremental_node_binary_async_local<2>(config.k, *gd);
                        break;
                        case 1:
                        localclique.findKclqueIncremental_node_binary_async_local<1>(config.k, *gd);
                        break;
                        default:
                            Log(error, "WRONG PARTITION SIZE SELECTED\n");
                    }
                }
                else
                {
                    Log(error, "LOCAL CLIQUE COUNTING IS NOT IMPLEMENTED FOR NON-BINARY ENCODING\n");
                }
            }
            else // Pivoting
            {
                if(kcc.BinaryEncode)
                {
                    switch(kcc.PartSize)
                    {
                        case 32:
                        localclique.findKclqueIncremental_node_pivot_async_local<32>(config.k, *gd);
                        break;
                        case 16:
                        localclique.findKclqueIncremental_node_pivot_async_local<16>(config.k, *gd);
                        break;
                        case 8:
                        localclique.findKclqueIncremental_node_pivot_async_local<8>(config.k, *gd);
                        break;
                        case 4:
                        localclique.findKclqueIncremental_node_pivot_async_local<4>(config.k, *gd);
                        break;
                        case 2:
                        localclique.findKclqueIncremental_node_pivot_async_local<2>(config.k, *gd);
                        break;
                        case 1:
                        localclique.findKclqueIncremental_node_pivot_async_local<1>(config.k, *gd);
                        break;
                        default:
                            Log(error, "WRONG PARTITION SIZE SELECTED\n");
                    }
                }
                else
                {
                    Log(error, "LOCAL CLIQUE COUNTING IS NOT IMPLEMENTED FOR NON-BINARY ENCODING\n");
                }
            }
        }
        else if (config.processBy == ByEdge)
        {
            if(kcc.Algo == GraphOrient)
            {
                if(kcc.BinaryEncode)
                {
                    switch(kcc.PartSize)
                    {
                        case 32:
                        localclique.findKclqueIncremental_edge_binary_async_local<32>(config.k, *gd);
                        break;
                        case 16:
                        localclique.findKclqueIncremental_edge_binary_async_local<16>(config.k, *gd);
                        break;
                        case 8:
                        localclique.findKclqueIncremental_edge_binary_async_local<8>(config.k, *gd);
                        break;
                        case 4:
                        localclique.findKclqueIncremental_edge_binary_async_local<4>(config.k, *gd);
                        break;
                        case 2:
                        localclique.findKclqueIncremental_edge_binary_async_local<2>(config.k, *gd);
                        break;
                        case 1:
                        localclique.findKclqueIncremental_edge_binary_async_local<1>(config.k, *gd);
                        break;
                        default:
                            Log(error, "WRONG PARTITION SIZE SELECTED\n");
                    }
                }
                else
                {
                    Log(error, "LOCAL CLIQUE COUNTING IS NOT IMPLEMENTED FOR NON-BINARY ENCODING\n");
                }
            }
            else // Pivoting
            {
                if(kcc.BinaryEncode)
                {
                    switch(kcc.PartSize)
                    {
                        case 32:
                        localclique.findKclqueIncremental_edge_pivot_async_local<32>(config.k, *gd);
                        break;
                        case 16:
                        localclique.findKclqueIncremental_edge_pivot_async_local<16>(config.k, *gd);
                        break;
                        case 8:
                        localclique.findKclqueIncremental_edge_pivot_async_local<8>(config.k, *gd);
                        break;
                        case 4:
                        localclique.findKclqueIncremental_edge_pivot_async_local<4>(config.k, *gd);
                        break;
                        case 2:
                        localclique.findKclqueIncremental_edge_pivot_async_local<2>(config.k, *gd);
                        break;
                        case 1:
                        localclique.findKclqueIncremental_edge_pivot_async_local<1>(config.k, *gd);
                        break;
                        default:
                            Log(error, "WRONG PARTITION SIZE SELECTED\n");
                    }
                }
                else
                {
                    Log(error, "LOCAL CLIQUE COUNTING IS NOT IMPLEMENTED FOR NON-BINARY ENCODING\n");
                }
            }
        }
        localclique.sync();
        double time = t.elapsed();
        // (teps: traversed edges per second)
        Log(info, "count time %f s (%f teps)", time, m / time);
        localclique.show(n);
    }

    if (config.mt == MAXIMAL_CLIQUE)
    {
        if (config.orient != None)
        {
            Log(error, "Full graph is needed for maximal cliques.\n");
            return 0;
        }
        
        Timer degeneracy_time;

        mohacore.findKcoreIncremental_async(3, 1000, *gd, 0, 0);

        Log(info, "Degeneracy ordering time: %f s", degeneracy_time.elapsed());

        Timer csr_recreation_time;

        graph::GPUArray<uint> split_col("Split Column", config.allocation, m, config.deviceId),
            tmp_row("Temp Row", config.allocation, m / 2, config.deviceId),
            tmp_col("Temp Column", config.allocation, m / 2, config.deviceId),
            split_ptr("Split Pointer", config.allocation, n + 1, config.deviceId),
            asc("ASC temp", AllocationTypeEnum::unified, m, config.deviceId);

        graph::GPUArray<bool> keep("Keep temp", AllocationTypeEnum::unified, m, config.deviceId);

        execKernel((init<uint, PeelType>), ((m - 1) / 51200) + 1, 512, config.deviceId, false, *gd, asc.gdata(), keep.gdata(), mohacore.nodeDegree.gdata(), mohacore.nodePriority.gdata());
        
        graph::CubLarge<uint> s(config.deviceId);
        if (m < INT_MAX)
        {
            CUBSelect(gd->rowInd, tmp_row.gdata(), keep.gdata(), m, config.deviceId);
            CUBSelect(gd->colInd, tmp_col.gdata(), keep.gdata(), m, config.deviceId);
        }
        else
        {
            s.Select2(gd->rowInd, gd->colInd, tmp_row.gdata(), tmp_col.gdata(), keep.gdata(), m);
        }

        execKernel((warp_detect_deleted_edges<uint>), (32 * n + 128 - 1) / 128, 128, config.deviceId, false, gd->rowPtr, n, keep.gdata(), split_ptr.gdata());
        total = CUBScanExclusive<uint, uint>(split_ptr.gdata(), split_ptr.gdata(), n, config.deviceId, 0, config.allocation);
        split_ptr.setSingle(n, total, true);
        execKernel((split_child<uint>), ((m - 1) / 51200) + 1, 512, config.deviceId, false, *gd, tmp_row.gdata(), tmp_col.gdata(), split_col.gdata(), split_ptr.gdata());
        execKernel((split_inverse<uint>), ((m - 1) / 51200) + 1, 512, config.deviceId, false, keep.gdata(), m);

        if (m < INT_MAX)
        {
            CUBSelect(gd->rowInd, tmp_row.gdata(), keep.gdata(), m, config.deviceId);
            CUBSelect(gd->colInd, tmp_col.gdata(), keep.gdata(), m, config.deviceId);
        }
        else
        {
            s.Select2(gd->rowInd, gd->colInd, tmp_row.gdata(), tmp_col.gdata(), keep.gdata(), m);
        }

        execKernel((warp_detect_deleted_edges<uint>), (32 * n + 128 - 1) / 128, 128, config.deviceId, false, gd->rowPtr, n, keep.gdata(), split_ptr.gdata());
        uint total = CUBScanExclusive<uint, uint>(split_ptr.gdata(), split_ptr.gdata(), n, config.deviceId, 0, config.allocation);
        split_ptr.setSingle(n, total, true);
        execKernel((split_parent<uint>), ((m - 1) / 51200) + 1, 512, config.deviceId, false, *gd, tmp_row.gdata(), tmp_col.gdata(), split_col.gdata(), split_ptr.gdata());
        execKernel((warp_detect_deleted_edges<uint>), (32 * n + 128 - 1) / 128, 128, config.deviceId, false, gd->rowPtr, n, keep.gdata(), split_ptr.gdata());
        execKernel((split_acc<uint>), ((n - 1) / 51200) + 1, 512, config.deviceId, false, *gd, split_ptr.gdata());

        cudaDeviceSynchronize();
        asc.freeGPU();
        keep.freeGPU();
        tmp_row.freeGPU();
        tmp_col.freeGPU();

        graph::COOCSRGraph_d<uint>* gsplit = (graph::COOCSRGraph_d<uint>*)malloc(sizeof(graph::COOCSRGraph_d<uint>));

        gsplit->numNodes = n;
        gsplit->numEdges = m;
        gsplit->capacity = m;
        gsplit->rowPtr = gd->rowPtr;
        gsplit->rowInd = gd->rowInd;
        gsplit->colInd = split_col.gdata();
        gsplit->splitPtr = split_ptr.gdata();

        Log(info, "CSR Recreation time: %f s", csr_recreation_time.elapsed());

        graph::SingleGPU_Maximal_Clique<uint> maximalclique(config.deviceId, *gsplit);

        KcliqueConfig kcc = config.kcConfig;

        Timer t;
        if (config.processBy == ByNode)
        {
            if(kcc.Algo == GraphOrient)
            {
                Log(error, "Not Implemented for graph orientation method.\n");
            }
            else // Pivoting
            {
                if(kcc.EncodeHalf)
                {
                    switch(kcc.PartSize)
                    {
                        case 32:
                        maximalclique.find_maximal_clique_node_pivot_encode_half<32>(*gsplit);
                        break;
                        case 16:
                        maximalclique.find_maximal_clique_node_pivot_encode_half<16>(*gsplit);
                        break;
                        case 8:
                        maximalclique.find_maximal_clique_node_pivot_encode_half<8>(*gsplit);
                        break;
                        case 4:
                        maximalclique.find_maximal_clique_node_pivot_encode_half<4>(*gsplit);
                        break;
                        case 2:
                        maximalclique.find_maximal_clique_node_pivot_encode_half<2>(*gsplit);
                        break;
                        case 1:
                        maximalclique.find_maximal_clique_node_pivot_encode_half<1>(*gsplit);
                        break;
                        default:
                            Log(error, "WRONG PARTITION SIZE SELECTED\n");
                    }
                }
                else
                {
                    switch(kcc.PartSize)
                    {
                        case 32:
                        maximalclique.find_maximal_clique_node_pivot_encode_induced<32>(*gsplit);
                        break;
                        case 16:
                        maximalclique.find_maximal_clique_node_pivot_encode_induced<16>(*gsplit);
                        break;
                        case 8:
                        maximalclique.find_maximal_clique_node_pivot_encode_induced<8>(*gsplit);
                        break;
                        case 4:
                        maximalclique.find_maximal_clique_node_pivot_encode_induced<4>(*gsplit);
                        break;
                        case 2:
                        maximalclique.find_maximal_clique_node_pivot_encode_induced<2>(*gsplit);
                        break;
                        case 1:
                        maximalclique.find_maximal_clique_node_pivot_encode_induced<1>(*gsplit);
                        break;
                        default:
                            Log(error, "WRONG PARTITION SIZE SELECTED\n");
                    }
                }
            }
        }
        else if (config.processBy == ByEdge)
        {
            if(kcc.Algo == GraphOrient)
            {
                Log(error, "Not Implemented for graph orientation method.\n");
            }
            else // Pivoting
            {
                if(kcc.EncodeHalf)
                {
                    switch(kcc.PartSize)
                    {
                        case 32:
                        maximalclique.find_maximal_clique_edge_pivot_encode_half<32>(*gsplit);
                        break;
                        case 16:
                        maximalclique.find_maximal_clique_edge_pivot_encode_half<16>(*gsplit);
                        break;
                        case 8:
                        maximalclique.find_maximal_clique_edge_pivot_encode_half<8>(*gsplit);
                        break;
                        case 4:
                        maximalclique.find_maximal_clique_edge_pivot_encode_half<4>(*gsplit);
                        break;
                        case 2:
                        maximalclique.find_maximal_clique_edge_pivot_encode_half<2>(*gsplit);
                        break;
                        case 1:
                        maximalclique.find_maximal_clique_edge_pivot_encode_half<1>(*gsplit);
                        break;
                        default:
                            Log(error, "WRONG PARTITION SIZE SELECTED\n");
                    }
                }
                else
                {
                    switch(kcc.PartSize)
                    {
                        case 32:
                        maximalclique.find_maximal_clique_edge_pivot_encode_induced<32>(*gsplit);
                        break;
                        case 16:
                        maximalclique.find_maximal_clique_edge_pivot_encode_induced<16>(*gsplit);
                        break;
                        case 8:
                        maximalclique.find_maximal_clique_edge_pivot_encode_induced<8>(*gsplit);
                        break;
                        case 4:
                        maximalclique.find_maximal_clique_edge_pivot_encode_induced<4>(*gsplit);
                        break;
                        case 2:
                        maximalclique.find_maximal_clique_edge_pivot_encode_induced<2>(*gsplit);
                        break;
                        case 1:
                        maximalclique.find_maximal_clique_edge_pivot_encode_induced<1>(*gsplit);
                        break;
                        default:
                            Log(error, "WRONG PARTITION SIZE SELECTED\n");
                    }
                }
            }
        }
        maximalclique.sync();
        double time = t.elapsed();
        // (teps: traversed edges per second)
        Log(info, "count time %f s (%f teps)", time, m / time);
        maximalclique.show(n);
    }

    if (config.mt == KTRUSS)
    {
        //The problem with Ktruss that it physically changes the graph structure due to stream compaction !!
        graph::COOCSRGraph_d<uint>* gd;
        to_csrcoo_device(g, gd, config.deviceId, unified); //got to device !!

    //#define VLDB2020
#ifdef VLDB2020
    //We need unified to do stream compaction
        graph::GPUArray<int> output("KT Output", AllocationTypeEnum::unified, m / 2, config.deviceId);
        graph::BmpGpu<uint> bmp(config.deviceId);
        bmp.getEidAndEdgeList(g);// CPU
        bmp.InitBMP(*gd);
        bmp.bmpConstruct(*gd);

        double tc_time = bmp.Count_Set(*gd);
#define MAX_LEVEL  (20000)
        auto level_start_pos = (uint*)calloc(MAX_LEVEL, sizeof(uint));

        graph::PKT_cuda(
            config.deviceId,
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
        Log(info, "MOHA %d ktruss (%f teps)", mohatruss.count(), m / time);
#endif

#define OUR_NEW_KTRUSS
#ifdef OUR_NEW_KTRUSS
            //We need to change the graph representation
            graph::GPUArray<uint> rowIndex("Half Row Index", AllocationTypeEnum::unified, m / 2, config.deviceId),
            colIndex("Half Col Index", AllocationTypeEnum::unified, m / 2, config.deviceId),
            EID("EID", AllocationTypeEnum::unified, m, config.deviceId),
            asc("ASC temp", AllocationTypeEnum::unified, m, config.deviceId);

        Timer t_init;
        graph::GPUArray<bool> keep("Keep temp", AllocationTypeEnum::unified, m, config.deviceId);


        execKernel(init, (m + 512 - 1) / 512, 512, config.deviceId, false, *gd, asc.gdata(), keep.gdata());

        CUBSelect(asc.gdata(), asc.gdata(), keep.gdata(), m, config.deviceId);
        CUBSelect(gd->rowInd, rowIndex.gdata(), keep.gdata(), m, config.deviceId);
        uint newNumEdges = CUBSelect(gd->colInd, colIndex.gdata(), keep.gdata(), m, config.deviceId);
        execKernel(InitEid, (newNumEdges + 512 - 1) / 512, 512, config.deviceId, false, newNumEdges, asc.gdata(), rowIndex.gdata(), colIndex.gdata(), gd->rowPtr, gd->colInd, EID.gdata());
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

        graph::SingleGPU_KtrussMod<uint, PeelType> mohatrussM(config.deviceId, config.allocation);

        Timer t;
        graph::TcBase<uint>* tcb = new graph::TcBinary<uint>(config.deviceId, m, n, mohatrussM.stream());

        mohatrussM.findKtrussIncremental_sync(3, 1000, tcb, geid, nullptr, nullptr, 0, 0);
        mohatrussM.sync();
        double time = t.elapsed();


        Log(info, "count time %f s", time);
        Log(info, "MOHA %d ktruss (%f teps)", mohatrussM.count(), m / time);
#endif


    }

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

    //A.freeGPU();
    //B.freeGPU();
    return 0;
}


