
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <fstream>
#include <map>

#include "omp.h"
#include <vector>

#include "../include/Logger.cuh"
#include "../include/FIleReader.cuh"
#include "../include/CGArray.cuh"
#include "../include/TriCountPrim.cuh"
#include "../triangle_counting/TcBase.cuh"
#include "../triangle_counting/TcSerial.cuh"
#include "../triangle_counting/TcBinary.cuh"
#include "../triangle_counting/TcVariablehash.cuh"
// #include "../triangle_counting/TcNvgraph.cuh"
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
// #include "../kclique/kclique.cuh"
// #include "../kclique/kclique_local.cuh"
// #include "../kclique/mclique.cuh"

#include "../include/Config.h"
#include "../include/ScanLarge.cuh"

#include "../subgraph_matching/subgraph_matching.cuh"

using namespace std;
//#define TriListConstruct

int main(int argc, char **argv)
{

    // CUDA_RUNTIME(cudaDeviceReset());
    Config config = parseArgs(argc, argv);
    setbuf(stdout, NULL);
    printf("\033[0m");
    printf("Welcome ---------------------\n");
    printConfig(config);

    graph::MtB_Writer mwriter;
    auto fileSrc = config.srcGraph;
    auto fileDst = config.dstGraph;
    if (config.mt == CONV_MTX_BEL)
    {
        mwriter.write_market_bel<uint, int>(fileSrc, fileDst, false);
        return;
    }

    if (config.mt == CONV_TSV_BEL)
    {
        mwriter.write_tsv_bel<uint64, uint64>(fileSrc, fileDst);
        return;
    }

    if (config.mt == CONV_TSV_MTX)
    {
        mwriter.write_tsv_market<uint, int>(fileSrc, fileDst);
        return;
    }

    if (config.mt == CONV_BEL_MTX)
    {
        mwriter.write_bel_market<uint, int>(fileSrc, fileDst);
        return;
    }

    if (config.mt == CONV_TXT_BEL)
    {
        mwriter.write_txt_bel<uint, uint>(fileSrc, fileDst, true, 2, 0);
        return;
    }

    Timer read_graph_timer;

    const char *matr = config.srcGraph;
    graph::EdgeListFile f(matr);
    std::vector<EdgeTy<uint>> edges;
    std::vector<EdgeTy<uint>> fileEdges;
    auto lowerTriangular = [](const Edge &e)
    { return e.first > e.second; };
    auto upperTriangular = [](const Edge &e)
    { return e.first < e.second; };
    auto full = [](const Edge &e)
    { return false; };

    while (f.get_edges(fileEdges, 100))
    {
        edges.insert(edges.end(), fileEdges.begin(), fileEdges.end());
    }

    if (config.sortEdges)
    {
        f.sort_edges(edges);
    }

    graph::CSRCOO<uint> csrcoo;
    if (config.orient == Upper)
        csrcoo = graph::CSRCOO<uint>::from_edgelist(edges, lowerTriangular);
    else if (config.orient == Lower)
        csrcoo = graph::CSRCOO<uint>::from_edgelist(edges, upperTriangular);
    else
        csrcoo = graph::CSRCOO<uint>::from_edgelist(edges, full);

    uint n = csrcoo.num_rows();
    uint m = csrcoo.nnz();
    Log(debug, "value of n: %u\n", n);
    Log(debug, "value of m: %u\n", m);

    graph::COOCSRGraph<uint> g;
    g.capacity = m;
    g.numEdges = m;
    g.numNodes = n;

    g.rowPtr = new graph::GPUArray<uint>("Row pointer", AllocationTypeEnum::gpu, n + 1, config.deviceId, true);
    g.rowInd = new graph::GPUArray<uint>("Src Index", AllocationTypeEnum::gpu, m, config.deviceId, true);
    g.colInd = new graph::GPUArray<uint>("Dst Index", AllocationTypeEnum::gpu, m, config.deviceId, true);

    uint *rp, *ri, *ci;
    cudaMallocHost((void **)&rp, (n + 1) * sizeof(uint));
    cudaMallocHost((void **)&ri, (m) * sizeof(uint));
    cudaMallocHost((void **)&ci, (m) * sizeof(uint));

    CUDA_RUNTIME(cudaMemcpy(rp, csrcoo.row_ptr(), (n + 1) * sizeof(uint), cudaMemcpyKind::cudaMemcpyHostToHost));
    CUDA_RUNTIME(cudaMemcpy(ri, csrcoo.row_ind(), (m) * sizeof(uint), cudaMemcpyKind::cudaMemcpyHostToHost));
    CUDA_RUNTIME(cudaMemcpy(ci, csrcoo.col_ind(), (m) * sizeof(uint), cudaMemcpyKind::cudaMemcpyHostToHost));

    g.rowPtr->cdata() = rp;
    g.rowInd->cdata() = ri;
    g.colInd->cdata() = ci;

    Log(info, "Read graph time: %f s", read_graph_timer.elapsed());

    /// Now we need to orient the graph
    Timer total_timer;

    graph::COOCSRGraph_d<uint> *gd = (graph::COOCSRGraph_d<uint> *)malloc(sizeof(graph::COOCSRGraph_d<uint>));

    gd->numNodes = g.numNodes;
    gd->numEdges = g.numEdges;
    gd->capacity = g.capacity;

    size_t mf, ma;
    g.rowPtr->switch_to_gpu(config.deviceId, g.numNodes + 1);
    cudaDeviceSynchronize();
    Log(debug, "Moved rowPtr to device memory");
    cudaMemGetInfo(&mf, &ma);
    // std::cout << "free: " << mf << " total: " << ma << std::endl;
    gd->rowPtr = g.rowPtr->gdata();
    if ((!config.isSmall || g.numEdges > 5E08) && !(config.mt == GRAPH_MATCH || config.mt == GRAPH_COUNT))
    {
        Log(debug, "code Reached in big graphs!");
        gd->rowInd = g.rowInd->cdata();
        gd->colInd = g.colInd->cdata();
    }
    else
    {
        g.rowInd->switch_to_gpu(config.deviceId, g.numEdges);
        cudaDeviceSynchronize();
        Log(debug, "Moved rowIndices to device memory");
        cudaMemGetInfo(&mf, &ma);
        // std::cout << "free: " << mf << " total: " << ma << std::endl;
        g.colInd->switch_to_gpu(config.deviceId, g.numEdges);
        cudaDeviceSynchronize();
        Log(debug, "Moved colIndices to device memory");
        cudaMemGetInfo(&mf, &ma);
        // std::cout << "free: " << mf << " total: " << ma << std::endl;
        gd->rowInd = g.rowInd->gdata();
        gd->colInd = g.colInd->gdata();
    }
    Log(debug, "Moved graph to device memory");
    cudaFreeHost(rp);
    cudaFreeHost(ri);
    cudaFreeHost(ci);
    // double total = total_timer.elapsed();
    Log(info, "Transfer Time: %f s", total_timer.elapsed());
    // printf("value at 97: %u\n", g.colInd->cdata()[97]);
    // execKernel(print_array, 1, 1, config.deviceId, false, 116, gd->rowInd);
    // execKernel(print_graph, 1, 1, config.deviceId, false, *gd);

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
            CUBSelect(gd->rowInd, rowInd_half.gdata(), keep.gdata(), m, config.deviceId);
            newNumEdges = CUBSelect(gd->colInd, colInd_half.gdata(), keep.gdata(), m, config.deviceId);
        }
        else
        {
            newNumEdges = s.Select2(gd->rowInd, gd->colInd, rowInd_half.gdata(), colInd_half.gdata(), keep.gdata(), m);
        }

        execKernel((warp_detect_deleted_edges<uint>), (32 * n + 128 - 1) / 128, 128, config.deviceId, false, gd->rowPtr, n, keep.gdata(), new_rowPtr.gdata());
        uint total = CUBScanExclusive<uint, uint>(new_rowPtr.gdata(), new_rowPtr.gdata(), n, config.deviceId, 0, config.allocation);
        new_rowPtr.setSingle(n, total, false);
        // assert(total == new_edge_num * 2);
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

        // cudaFreeHost(rp);
        // cudaFreeHost(ri);
        // cudaFreeHost(ci);

        gd->numNodes = g.numNodes;
        gd->numEdges = g.numEdges;
        gd->capacity = g.capacity;
        gd->rowPtr = new_rowPtr.gdata();
        gd->rowInd = g.rowInd->gdata();
        gd->colInd = g.colInd->gdata();
    }

    double time_init = t.elapsed();
    if (config.orient == Degree || config.orient == Degeneracy)
    {
        Log(info, "Preprocess time: %f s", time_init);
    }

    uint dv = 32;
    typedef unsigned int ttt;
    if (config.printStats)
    {
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

            // if (deg > 128)
            {
                uint64 v = deg * (deg + dv - 1) / dv;
                sum += v;

                // now the compressed one :D
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

            // Intersect Src, Dst
            uint s1 = 0, s2 = 0;
            bool loadA = true, loadB = true;
            uint a, b;
            uint rsi = 0;
            uint offset = 0;
            while (s1 < degree && s2 < dstDegree)
            {

                if (loadA)
                {
                    a = g.colInd->cdata()[s1 + s];
                    loadA = false;
                }
                if (loadB)
                {
                    b = g.colInd->cdata()[s2 + dstStart];
                    loadB = false;
                }

                if (a == b)
                {
                    uint startIndex = i * divisions;
                    uint divIndex = s1 / dv;
                    uint inDivIndex = s1 % dv;
                    node_be.cdata()[startIndex + divIndex] |= (1 << inDivIndex);

                    // i and s1
                    // if (i > 0)
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
                else if (a < b)
                {
                    ++s1;
                    loadA = true;
                }
                else
                {
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

        // Count traingles binary-search: Thread or Warp
        /*uint step = m;
        uint st = 0;
        uint ee = st + step; // st + 2;
        graph::TcBase<uint> *tcb = new graph::TcBinary<uint>(config.deviceId, ee, n);
        graph::TcBase<uint> *tcNV = new graph::TcNvgraph<uint>(config.deviceId, ee, n);
        graph::TcBase<uint> *tcBE = new graph::TcBinaryEncoding<uint>(config.deviceId, ee, n);
        graph::TcBase<uint> *tc = new graph::TcSerial<uint>(config.deviceId, ee, n);

        const int divideConstant = 10;
        graph::TcBase<uint> *tchash = new graph::TcVariableHash<uint>(config.deviceId, ee, n);

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

            uint64 serialTc = CountTriangles<uint>("Serial Thread", config.deviceId, config.allocation, tc, gd, ee, st, ProcessingElementEnum::Thread, 0);

            ////CountTriangles<uint>("Serial Warp", tc, rowPtr, sl, dl, ee, csrcoo.num_rows(), st, ProcessingElementEnum::Warp, 0);
            uint64 binaryTc = CountTriangles<uint>("Binary Warp", config.deviceId, config.allocation, tcb, gd, ee, st, ProcessingElementEnum::Block, 0);
            uint64 binarySharedTc = CountTriangles<uint>("Binary Warp Shared", config.deviceId, config.allocation, tcb, gd, ee, st, ProcessingElementEnum::WarpShared, 0);
            uint64 binarySharedCoalbTc = CountTriangles<uint>("Binary Warp Shared", config.deviceId, config.allocation, tcb, gd, ee, st, ProcessingElementEnum::Test, 0);

            uint64 binaryEncodingTc = CountTriangles<uint>("Binary Encoding", config.deviceId, config.allocation, tcBE, gd, ee, st, ProcessingElementEnum::Warp, 0);
            CountTrianglesHash<uint>(config.deviceId, divideConstant, tchash, g, gd, ee, 0, ProcessingElementEnum::Warp, 0);

            uint64 binaryQueueTc = CountTriangles<uint>("Binary Queue", config.deviceId, config.allocation, tcb, gd, ee, st, ProcessingElementEnum::Queue, 0);

            CountTriangles<uint>("NVGRAPH", config.deviceId, config.allocation, tcNV, gd, ee);

            if (serialTc != binaryTc)
                break;
            st += step;
            ee += step;

            printf("------------------------------\n");

            break;
        }*/
    }
    
    else if (config.mt == CROSSDECOMP)
    {
        // Update Please
        /*	sl.switch_to_gpu(0, csrcoo.nnz());
        dl.switch_to_gpu(0, csrcoo.nnz());
        rowPtr.switch_to_gpu(0, csrcoo.num_rows() + 1);
        Thanos<uint> t(rowPtr, sl, dl, csrcoo.nnz(), csrcoo.num_rows());*/
    }

    // Not needed anymore
#ifdef TriListConstruct
    sl.switch_to_gpu(0, csrcoo.nnz());
    dl.switch_to_gpu(0, csrcoo.nnz());
    rowPtr.switch_to_gpu(0, csrcoo.num_rows() + 1);
    ////Takes either serial or binary triangle Counter
    graph::TcBase<uint> *tcb = new graph::TcBinary<uint>(0, csrcoo.nnz(), csrcoo.num_rows());
    graph::GPUArray<uint> triPointer("tri Pointer", cpuonly);
    graph::GPUArray<uint> triIndex("tri Index", cpuonly);
    ConstructTriList(triIndex, triPointer, tcb, rowPtr, sl, dl, csrcoo.nnz(), csrcoo.num_rows(), 0, ProcessingElementEnum::Warp);
#endif

    if (config.mt == KCORE)
    {
        graph::COOCSRGraph_d<uint> *gd;
        to_csrcoo_device(g, gd, config.deviceId, config.allocation); // got to device !!
        cudaDeviceSynchronize();

        graph::SingleGPU_Kcore<uint, PeelType> mohacore(config.deviceId);
        Timer t;
        mohacore.findKcoreIncremental_async(3, 1000, *gd, 0, 0);
        mohacore.sync();
        double time = t.elapsed();
        Log(info, "count time %f s", time);
        Log(info, "MOHA %d kcore (%f teps)", mohacore.count(), m / time);
    }

    if (config.mt == GRAPH_MATCH || config.mt == GRAPH_COUNT)
    {
        // Read Template graph from file
        graph::EdgeListFile patFile(config.patGraph);
        std::vector<EdgeTy<uint>> patEdges;
        std::vector<EdgeTy<uint>> patFileEdges;

        while (patFile.get_edges(patFileEdges, 10 * 10))
        {
            patEdges.insert(patEdges.end(), patFileEdges.begin(), patFileEdges.end());
        }

        graph::CSRCOO<uint> patCsrcoo = graph::CSRCOO<uint>::from_edgelist(patEdges);
        graph::COOCSRGraph<uint> patG;
        patG.capacity = patCsrcoo.nnz();
        patG.numEdges = patCsrcoo.nnz();
        patG.numNodes = patCsrcoo.num_rows();

        patG.rowPtr = new graph::GPUArray<uint>("Row pointer", AllocationTypeEnum::cpuonly);
        patG.rowInd = new graph::GPUArray<uint>("Src index", AllocationTypeEnum::cpuonly);
        patG.colInd = new graph::GPUArray<uint>("Dst index", AllocationTypeEnum::cpuonly);

        patG.rowPtr->cdata() = patCsrcoo.row_ptr();
        patG.rowInd->cdata() = patCsrcoo.row_ind();
        patG.colInd->cdata() = patCsrcoo.col_ind();

        // if (true)
        // Degree or degeneracy based orientation

        // Initialise subgraph matcher class
        // graph::SG_Match<uint> *sgm = new graph::SG_Match<uint>(config.mt, config.processBy, config.deviceId);

        // Process template and count subgraphs

        graph::SG_Match<uint> *sgm = new graph::SG_Match<uint>(config.mt, config.processBy, config.deviceId);
        // sgm->run(*gd, patG);
        printf("code initiates sgm->run\n");
        sgm->run(*gd, patG);

        // Clean up
        // delete sgm;
    }

#pragma region MyRegion
    ////Hashing tests
    ////More tests on GPUArray
    // graph::GPUArray<uint> A("Test A: In", AllocationTypeEnum::cpuonly);
    // graph::GPUArray<uint> B("Test B: In", AllocationTypeEnum::cpuonly);

    // const uint inputSize = pow<int,int>(2, 18) - 1;
    //
    //
    // A.cdata() = new uint[inputSize];
    // B.cdata() = new uint[inputSize];

    // const int dimBlock = 512;
    // const int dimGrid = (inputSize + (dimBlock)-1) / (dimBlock);

    // srand(220);

    // A.cdata()[0] = 1;
    // B.cdata()[0] = 1;
    // for (uint i = 1; i < inputSize; i++)
    //{
    //	A.cdata()[i] = A.cdata()[i - 1] + (rand() % 13) + 1;
    //	B.cdata()[i] = B.cdata()[i - 1] + (rand() % 13) + 1;
    // }

    // graph::Hashing::goldenSerialIntersectionCPU<uint>(A, B, inputSize);

    // A.switch_to_gpu(0, inputSize);
    // B.switch_to_gpu(0, inputSize);

    // graph::Hashing::goldenBinaryGPU<uint>(A, B, inputSize);

    ////1-level hash with binary search for stash
    // graph::Hashing::test1levelHashing<uint>(A, B, inputSize, 4);

    ////Non-stash hashing
    // graph::Hashing::testHashNosStash<uint>(A, B, inputSize, 5);

    ////Store binary tree traversal: Now assume full binary tree
    // vector<uint> treeSizes;
    //
    // int maxInputSize;
    // int levels = 1;
    // for (int i = 1; i < 19; i++)
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

    // graph::GPUArray<Node> BT("Binary Traverasl", gpu, maxInputSize, 0);
    // graph::GPUArray<uint> BTV("Binary Traverasl", gpu, maxInputSize, 0);
    //
    // int cl = 0;
    // int totalElements = 1;

    // int element0 = maxInputSize / 2;
    // BT.cdata()[0].i = element0;
    // BTV.cdata()[0] = B.cdata()[element0];
    // BT.cdata()[0].l = 0;
    // BT.cdata()[0].r = maxInputSize;

    // while (totalElements < maxInputSize)
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

    // BTV.switch_to_gpu(0);

    // const auto startBST = stime();
    // graph::GPUArray<uint> countBST("Test Binary Search Tree search: Out", AllocationTypeEnum::unified, 1, 0);
    // graph::binary_search_bst_g<uint, 32> << <1, 32 >> > (countBST.gdata(), A.gdata(), inputSize, BTV.gdata(), inputSize);
    // cudaDeviceSynchronize();
    // double elapsedBST = elapsedSec(startBST);
    // printf("BST Elapsed time = %f, Count = %u \n", elapsedBST, countBST.cdata()[0]);

    // BTV.freeCPU();
    // BTV.freeGPU();

    // BT.freeCPU();
    // BT.freeGPU();

    ////For weighted edges
    ////std::vector<WEdgeTy<uint, wtype>> wedges;
    ////std::vector<WEdgeTy<uint,wtype>> wfileEdges;
    ////while (f.get_weighted_edges(wfileEdges, 10)) {
    ////	wedges.insert(wedges.end(), wfileEdges.begin(), wfileEdges.end());
    ////}

#pragma endregion

    // A.freeGPU();
    // B.freeGPU();
    return 0;
}
