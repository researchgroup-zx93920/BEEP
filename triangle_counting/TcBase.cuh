#pragma once

#include "cuda_runtime.h"
//#include "../cub/cub.cuh"
#include "../include/TriCountPrim.cuh"
#include "../include/CGArray.cuh"

struct Edge1
{
    uint32_t s;
    uint32_t d;
};

namespace graph {
    template<typename T>
    class TcBase {
    public:
        int dev_;
        cudaStream_t stream_;
        uint64* count_;
        uint64 numEdges;
        uint64 numNodes;

        // events for measuring time
        cudaEvent_t kernelStart_;
        cudaEvent_t kernelStop_;

    public:
        /*! Device constructor

            Create a counter on device dev
        */
        TcBase(int dev, uint64 ne, uint64 nn, cudaStream_t stream = 0) : dev_(dev), numEdges(ne), numNodes(nn), stream_(stream), count_(nullptr) {
            CUDA_RUNTIME(cudaSetDevice(dev_));
            CUDA_RUNTIME(cudaMallocManaged(&count_, sizeof(*count_)));
            CUDA_RUNTIME(cudaMemset(count_, 0, 1 * sizeof(uint64)));
            CUDA_RUNTIME(cudaGetLastError());

            CUDA_RUNTIME(cudaEventCreate(&kernelStart_));
            CUDA_RUNTIME(cudaEventCreate(&kernelStop_));
        }

        /*! default ctor - counter on device 0
         */
        TcBase() : TcBase(0, 0, 0) {}

        /*! copy ctor - create a new counter on the same device

        All fields are reset
         */
        TcBase(const TcBase& other) : TcBase(other.dev_, other.numEdges, other.numNodes, other.stream_) {}

        ~TcBase() {
            CUDA_RUNTIME(cudaEventDestroy(kernelStart_));
            CUDA_RUNTIME(cudaEventDestroy(kernelStop_));
        }

        TcBase& operator=(TcBase&& other) noexcept {

            /* We just swap other and this, which has the following benefits:
               Don't call delete on other (maybe faster)
               Opportunity for data to be reused since it was not deleted
               No exceptions thrown.
            */

            other.swap(*this);
            return *this;
        }

        void swap(TcBase& other) noexcept {
            std::swap(other.dev_, dev_);
            std::swap(other.kernelStart_, kernelStart_);
            std::swap(other.kernelStop_, kernelStop_);
            std::swap(other.stream_, stream_);
        }

        //CSRCOO based counting and setting
        virtual void count_async(GPUArray<T> rowPtr, GPUArray<T> rowInd, GPUArray<T> colInd, const size_t numEdges, const size_t edgeOffset = 0, ProcessingElementEnum kernelType = Thread, int increasing = 0)
        {}

        virtual void count_hash_async(const int divideConstant, GPUArray<T> rowPtr, GPUArray<T> rowInd, GPUArray<T> colInd, GPUArray<T> hp, GPUArray<T> hps, const size_t numEdges, const size_t edgeOffset = 0, ProcessingElementEnum kernelType = Thread, int increasing = 0)
        {}

        virtual void count_per_edge_async(GPUArray<T>& tcpt, GPUArray<T> rowPtr, GPUArray<T> rowInd, GPUArray<T> colInd, const size_t numEdges, const size_t edgeOffset = 0, ProcessingElementEnum kernelType = Thread, int increasing = 0)
        {}


        virtual void set_per_edge_async(GPUArray<T>& tcs, GPUArray<T> triPointer, GPUArray<T> rowPtr, GPUArray<T> rowInd, GPUArray<T> colInd, const size_t numEdges, const size_t edgeOffset = 0, ProcessingElementEnum kernelType = Thread, int increasing = 0)
        {}



        void sync() { CUDA_RUNTIME(cudaStreamSynchronize(stream_)); }

        uint64 count() const { return *count_; }
        int device() const { return dev_; }



        /*! return the number of ms the GPU spent in the triangle counting kernel

          After this call, the kernel will have been completed, though the count may not be available.
         */
        float kernel_time() {
            float ms;
            CUDA_RUNTIME(cudaEventSynchronize(kernelStop_));
            CUDA_RUNTIME(cudaEventElapsedTime(&ms, kernelStart_, kernelStop_));
            return ms / 1e3;
        }
    };
}

