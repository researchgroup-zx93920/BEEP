
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

#include "../include/CSRCOO.cuh"

#include "../include/main_support.cuh"

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

    uint dv = 32;
    typedef unsigned int ttt;
    config.cutoff = get_stats(m, n, n, g.rowPtr->cdata(), g.colInd->cdata());
    Log(debug, "cutoff: %u", config.cutoff);

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

    g.rowInd->switch_to_gpu(config.deviceId, g.numEdges);
    cudaDeviceSynchronize();
    Log(debug, "Moved rowIndices to device memory");
    cudaMemGetInfo(&mf, &ma);
    // std::cout << "free: " << mf << " total: " << ma << std::endl;)
    g.colInd->switch_to_gpu(config.deviceId, g.numEdges);
    cudaDeviceSynchronize();
    Log(debug, "Moved colIndices to device memory");
    cudaMemGetInfo(&mf, &ma);
    // std::cout << "free: " << mf << " total: " << ma << std::endl;
    gd->rowInd = g.rowInd->gdata();
    gd->colInd = g.colInd->gdata();

    Log(debug, "Moved graph to device memory");
    cudaFreeHost(rp);
    cudaFreeHost(ri);
    cudaFreeHost(ci);

    Log(info, "Transfer Time: %f s", total_timer.elapsed());

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

        graph::SG_Match<uint> *sgm = new graph::SG_Match<uint>(config.mt, config.processBy, config.deviceId, config.cutoff, config.ndev);
        sgm->run(*gd, patG);

        // Clean up
        delete sgm;
    }

    return 0;
}
