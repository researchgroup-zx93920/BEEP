#pragma once
#include "TcBase.cuh"
#include "nvgraph.h"
namespace graph {

    template<typename T>
    class TcNvgraph : public TcBase<T>
    {
    public:

        TcNvgraph(int dev, uint64 ne, uint64 nn, cudaStream_t stream = 0) :TcBase<T>(dev, ne, nn, stream)
        {}

        void count_async(COOCSRGraph_d<T>* g, const size_t numEdges, const size_t edgeOffset = 0, ProcessingElementEnum kernelType = Thread, int increasing = 0)
        {
         
           int* rp = (int*)g->rowPtr;
            int* ri = (int*)g->rowInd;
            int* ci = (int*)g->colInd;

          /* CUDA_RUNTIME(cudaMemset(TcBase<T>::count_, 0, sizeof(*TcBase<T>::count_)));



            nvgraphHandle_t handle;
            nvgraphGraphDescr_t graph;
            nvgraphCSRTopology32I_t CSR_input;
            CSR_input = (nvgraphCSRTopology32I_t)malloc(sizeof(struct nvgraphCSRTopology32I_st));


           
            nvgraphCreate(&handle);
            nvgraphCreateGraphDescr(handle, &graph);
            CSR_input->nvertices = TcBase<T>::numNodes;
            CSR_input->nedges = numEdges;
            CSR_input->source_offsets = rp;
            CSR_input->destination_indices = ci;
            nvgraphSetGraphStructure(handle, graph, (void*)CSR_input, NVGRAPH_CSR_32);

            CUDA_RUNTIME(cudaEventRecord(TcBase<T>::kernelStart_, TcBase<T>::stream_));
            uint64_t trcount = 0;
            nvgraphTriangleCount(handle, graph, &trcount);
            CUDA_RUNTIME(cudaEventRecord(TcBase<T>::kernelStop_, TcBase<T>::stream_));

            *TcBase<T>::count_ = trcount;*/

        }


    };


}