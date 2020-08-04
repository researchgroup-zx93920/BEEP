#pragma once
#include "TcBase.cuh"
#include "../truss/kernels.cuh"




template <typename T, size_t BLOCK_DIM_X>
__global__ void __launch_bounds__(BLOCK_DIM_X)
kernel_serial_arrays(int* count, //!< [inout] the count, caller should zero
    T* rowPtr, T* rowInd, T* colInd, const size_t numEdges, const size_t edgeStart, int increasing = 0) {

    size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
    uint64 threadCount = 0;

    for (size_t i = gx + edgeStart; i < numEdges; i += BLOCK_DIM_X * gridDim.x) {
        const T src = rowInd[i];
        const T dst = colInd[i];

        //assert(src < dst);

        const T* srcBegin = &colInd[rowPtr[src]];
        const T* srcEnd = &colInd[rowPtr[src + 1]];
        const T* dstBegin = &colInd[rowPtr[dst]];
        const T* dstEnd = &colInd[rowPtr[dst + 1]];

        T min = increasing == 0 ? 0 : dst;

        threadCount += graph::serial_sorted_count_linear<T>(min, srcBegin, srcEnd, dstBegin, dstEnd);
    }

    // Block-wide reduction of threadCount
    typedef cub::BlockReduce<uint64, BLOCK_DIM_X> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    uint64 aggregate = BlockReduce(tempStorage).Sum(threadCount);

    // Add to total count
    if (0 == threadIdx.x) {
        atomicAdd(count, aggregate);
    }
}

template <typename T, size_t BLOCK_DIM_X>
__global__ void __launch_bounds__(BLOCK_DIM_X)
kernel_serial_pe_arrays(int* count, //!< [inout] the count, caller should zero
    T* rowPtr, T* rowInd, T* colInd, const size_t numEdges, const size_t edgeStart, int increasing = 0) {

    size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
    uint64 threadCount = 0;

    for (size_t i = gx + edgeStart; i < numEdges; i += BLOCK_DIM_X * gridDim.x) {
        const T src = rowInd[i];
        const T dst = colInd[i];

        const T* srcBegin = &colInd[rowPtr[src]];
        const T* srcEnd = &colInd[rowPtr[src + 1]];
        const T* dstBegin = &colInd[rowPtr[dst]];
        const T* dstEnd = &colInd[rowPtr[dst + 1]];

        T min = increasing == 0 ? 0 : dst;
        threadCount += graph::serial_sorted_count_linear(min, srcBegin, srcEnd, dstBegin, dstEnd);
    }

    count[gx] = threadCount;
}


template <typename T, size_t BLOCK_DIM_X>
__global__ void __launch_bounds__(BLOCK_DIM_X)
kernel_serial_pe_upto_arrays(int upto, bool *mask,
    int* count, //!< [inout] the count, caller should zero
    T* rowPtr, T* rowInd, T* colInd, const size_t numEdges, const size_t edgeStart, int increasing = 0) {

    size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
    uint64 threadCount = 0;

    for (size_t i = gx + edgeStart; i < numEdges; i += BLOCK_DIM_X * gridDim.x) 
    {
        const T src = rowInd[i];
        const T dst = colInd[i];
        if (!mask[i])
        {
            T min = 0; // increasing == 0 ? 0 : dst;
            threadCount += graph::serial_sorted_count_upto_linear(upto, mask, min, colInd, rowPtr[src], rowPtr[src + 1], rowPtr[dst], rowPtr[dst + 1]);
        }
    }

    count[gx] = threadCount;
}


template <typename T, size_t BLOCK_DIM_X>
__global__ void __launch_bounds__(BLOCK_DIM_X)
kernel_serial_pe_level_q_arrays(
    int* count,
    T* rowPtr, T* rowInd, T* colInd, const size_t numEdges,
    int level, bool* processed,
    int* affected, int& affected_cnt,
    int* next, bool* inNext, int& next_cnt,
    bool* in_bucket_window_, T* bucket_buf_, T& window_bucket_buf_size_, int bucket_level_end_
) 
{

    size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
    

    for (size_t i = gx; i < affected_cnt; i += BLOCK_DIM_X * gridDim.x)
    {
        int edgeId = affected[i];

        uint64 threadCount = 0;
        const T src = rowInd[edgeId];
        const T dst = colInd[edgeId];

        T min = 0; // increasing == 0 ? 0 : dst;
        T ap = rowPtr[src];
        T bp = rowPtr[dst];

        T aEnd = rowPtr[src + 1];
        T bEnd = rowPtr[dst + 1];

        bool loadA = true;
        bool loadB = true;

        T a, b;
        while (ap < aEnd && bp < bEnd) {

            if (loadA) {
                a = colInd[ap];
                loadA = false;
            }
            if (loadB) {
                b = colInd[bp];
                loadB = false;
            }

            if (a == b) {
                if (!processed[ap] && !processed[bp])
                    ++threadCount;

                ++ap;
                ++bp;
                loadA = true;
                loadB = true;
            }
            else if (a < b)
            {
                ++ap;
                loadA = true;
            }
            else {
                ++bp;
                loadB = true;
            }
        }

        //Queue and bucket
        auto prev = count[edgeId];
        if (prev > level)
        {
            if (threadCount < level)
            {
                threadCount = level;
            }

            count[edgeId] = threadCount;
            if (threadCount == level) {
                auto insert_idx = atomicAdd(&next_cnt, 1);
                next[insert_idx] = edgeId;
                inNext[edgeId] = true;
            }
            // Update the Bucket.
            auto latest = threadCount;
            if (latest > level && latest < bucket_level_end_)
            {
                auto old_token = atomicCASBool(in_bucket_window_ + edgeId, false, true);
                if (!old_token) {
                    auto insert_idx = atomicAdd(&window_bucket_buf_size_, 1);
                    bucket_buf_[insert_idx] = edgeId;
                }
            }

        }
    }
}



template <typename T, size_t BLOCK_DIM_X>
__global__ void __launch_bounds__(BLOCK_DIM_X)
kernel_serial_pe_level_affected(
    T* rowPtr, T* rowInd, T* colInd, const size_t numEdges,
    int level, bool* processed,
    int *curr, bool* inCurr, int curr_cnt,
    int* affected, bool* inAffected, int& affected_cnt
)
{

    size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;


    for (size_t i = gx; i < curr_cnt; i += BLOCK_DIM_X * gridDim.x)
    {
        int edgeId = curr[i];

        const T src = rowInd[edgeId];
        const T dst = colInd[edgeId];
        T ap = rowPtr[src];
        T bp = rowPtr[dst];

        T aEnd = rowPtr[src + 1];
        T bEnd = rowPtr[dst + 1];

        bool loadA = true;
        bool loadB = true;

        T a, b;
        while (ap < aEnd && bp < bEnd) {

            if (loadA) {
                a = colInd[ap];
                loadA = false;
            }
            if (loadB) {
                b = colInd[bp];
                loadB = false;
            }

            if (a == b) 
            {
                if (!processed[ap] && !inCurr[ap])
                {
                    //Add to affected queue
                    auto old_token = atomicCASBool(inAffected + ap, false, true);
                    if (!old_token)
                    {
                        auto insert_idx = atomicAdd(&affected_cnt, 1);
                        affected[insert_idx] = ap;
                    }
                }

                if (!processed[bp] && !inCurr[bp])
                {

                    auto old_token = atomicCASBool(inAffected + bp, false, true);
                    if (!old_token)
                    {
                        auto insert_idx = atomicAdd(&affected_cnt, 1);
                        affected[insert_idx] = bp;
                    }
                }
                
                ++ap;
                ++bp;
                loadA = true;
                loadB = true;
            }
            else if (a < b)
            {
                ++ap;
                loadA = true;
            }
            else {
                ++bp;
                loadB = true;
            }
        }

    }
}





template <typename T, size_t BLOCK_DIM_X>
__global__ void __launch_bounds__(BLOCK_DIM_X)
kernel_serial_set_arrays(T* count, //!< [inout] the count, caller should zero
    T* triPointer,
    T* rowPtr, T* rowInd, T* colInd, const size_t numEdges, const size_t edgeStart, int increasing = 0) {
    size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
    uint64 threadCount = 0;

    for (size_t i = gx + edgeStart; i < numEdges; i += BLOCK_DIM_X * gridDim.x) {
        const T src = rowInd[i];
        const T dst = colInd[i];

        const T* srcBegin = &colInd[rowPtr[src]];
        const T* srcEnd = &colInd[rowPtr[src + 1]];
        const T* dstBegin = &colInd[rowPtr[dst]];
        const T* dstEnd = &colInd[rowPtr[dst + 1]];

        const T startTriStorage = triPointer[i];


        threadCount += graph::serial_sorted_set_linear<T>(&count[startTriStorage], srcBegin, srcEnd, dstBegin, dstEnd);
    }

}

namespace graph {

    template<typename T>
    class TcSerial : public TcBase<T>
    {
    public:

        TcSerial(int dev, uint64 ne, uint64 nn, cudaStream_t stream = 0) :TcBase<T>(dev, ne, nn, stream)
        {}

        void count_async(GPUArray<T> rowPtr, GPUArray<T> rowInd, GPUArray<T> colInd, const size_t numEdges, const size_t edgeOffset = 0, ProcessingElementEnum kernelType = Thread, int increasing = 0)
        {
            const size_t dimBlock = 512;
            const size_t ne = numEdges;
            T* rp = rowPtr.gdata();
            T* ri = rowInd.gdata();
            T* ci = colInd.gdata();

            CUDA_RUNTIME(cudaMemset(TcBase<T>::count_, 0, sizeof(*TcBase<T>::count_)));

            // create one warp per edge
            const int dimGrid = (numEdges - edgeOffset + (dimBlock)-1) / (dimBlock);
            const int dimGridWarp = (32 * numEdges + (dimBlock)-1) / (dimBlock);
            const int dimGridBlock = (dimBlock * numEdges + (dimBlock)-1) / (dimBlock);

            assert(TcBase<T>::count_);
            Log(LogPriorityEnum::info, "device = %d, blocks = %d, threads = %d", TcBase<T>::dev_, dimGrid, dimBlock);
            CUDA_RUNTIME(cudaSetDevice(TcBase<T>::dev_));


            CUDA_RUNTIME(cudaEventRecord(TcBase<T>::kernelStart_, TcBase<T>::stream_));
            kernel_serial_arrays<T, dimBlock> << <dimGrid, dimBlock, 0, TcBase<T>::stream_ >> > (TcBase<T>::count_, rp, ri, ci, ne, edgeOffset);
            CUDA_RUNTIME(cudaEventRecord(TcBase<T>::kernelStop_, TcBase<T>::stream_));
        }


        void count_per_edge_async(GPUArray<int>& tcpt, GPUArray<T> rowPtr, GPUArray<T> rowInd, GPUArray<T> colInd, const size_t numEdges, const size_t edgeOffset = 0, ProcessingElementEnum kernelType = Thread, int increasing = 0)
        {
            const size_t dimBlock = 512;
            const size_t ne = numEdges;
            T* rp = rowPtr.gdata();
            T* ri = rowInd.gdata();
            T* ci = colInd.gdata();

            CUDA_RUNTIME(cudaMemset(TcBase<T>::count_, 0, sizeof(*TcBase<T>::count_)));

            // create one warp per edge
            const int dimGrid = (numEdges - edgeOffset + (dimBlock)-1) / (dimBlock);
            const int dimGridWarp = (32 * numEdges + (dimBlock)-1) / (dimBlock);
            const int dimGridBlock = (dimBlock * numEdges + (dimBlock)-1) / (dimBlock);

            assert(TcBase<T>::count_);
            Log(LogPriorityEnum::info, "device = %d, blocks = %d, threads = %d\n", TcBase<T>::dev_, dimGrid, dimBlock);
            CUDA_RUNTIME(cudaSetDevice(TcBase<T>::dev_));


            CUDA_RUNTIME(cudaEventRecord(TcBase<T>::kernelStart_, TcBase<T>::stream_));
            kernel_serial_pe_arrays<T, dimBlock> << <dimGrid, dimBlock, 0, TcBase<T>::stream_ >> > (tcpt.gdata(), rp, ri, ci, ne, edgeOffset, increasing);
            CUDA_RUNTIME(cudaEventRecord(TcBase<T>::kernelStop_, TcBase<T>::stream_));
        }


        void count_per_edge_upto_async(int upto, GPUArray<bool> mask, GPUArray<int>& tcpt, GPUArray<T> rowPtr, GPUArray<T> rowInd, GPUArray<T> colInd, const size_t numEdges, const size_t edgeOffset = 0, ProcessingElementEnum kernelType = Thread, int increasing = 0)
        {
            const size_t dimBlock = 32;
            const size_t ne = numEdges;
            T* rp = rowPtr.gdata();
            T* ri = rowInd.gdata();
            T* ci = colInd.gdata();
            const int dimGrid = (numEdges - edgeOffset + (dimBlock)-1) / (dimBlock);
            CUDA_RUNTIME(cudaSetDevice(TcBase<T>::dev_));
            kernel_serial_pe_upto_arrays<T, dimBlock> << <dimGrid, dimBlock, 0, TcBase<T>::stream_ >> > (upto, mask.gdata(), tcpt.gdata(), rp, ri, ci, ne, edgeOffset, increasing);
        }


        void count_per_edge_level_q_async(
            GPUArray<int>& tcpt, GPUArray<T> rowPtr, GPUArray<T> rowInd, GPUArray<T> colInd, const size_t numEdges,
            int level, GPUArray<bool> processed,
            GPUArray<int>& curr, int curr_cnt,
            GPUArray<int>& affected, GPUArray<bool>& inAffected, GPUArray<int>& affected_cnt, //next queue
            GPUArray<int>& next, GPUArray<bool>& inNext, GPUArray<int> next_cnt, //next queue
            GPUArray<bool>& in_bucket_window_, GPUArray<uint>& bucket_buf_, uint*& window_bucket_buf_size_, int bucket_level_end_,
            const size_t edgeOffset = 0, ProcessingElementEnum kernelType = Thread, int increasing = 0)
        {
        
            const size_t dimBlock = 32;
            const size_t ne = numEdges;
            T* rp = rowPtr.gdata();
            T* ri = rowInd.gdata();
            T* ci = colInd.gdata();
            const int dimGrid = (*affected_cnt.gdata() - edgeOffset + (dimBlock)-1) / (dimBlock);
            CUDA_RUNTIME(cudaSetDevice(TcBase<T>::dev_));


            //process affetced
            kernel_serial_pe_level_q_arrays<T, dimBlock> << <dimGrid, dimBlock, 0, TcBase<T>::stream_ >> > (
                tcpt.gdata(), rp, ri, ci, ne,
                level, processed.gdata(),
                affected.gdata(), *affected_cnt.gdata(),
                next.gdata(), inNext.gdata(), *next_cnt.gdata(),
                in_bucket_window_.gdata(), bucket_buf_.gdata(), *window_bucket_buf_size_, bucket_level_end_);

        }


        void affect_per_edge_level_q_async(
            GPUArray<T> rowPtr, GPUArray<T> rowInd, GPUArray<T> colInd, const size_t numEdges,
            int level, GPUArray<bool> processed,
            GPUArray<int>& curr, GPUArray<bool> inCurr, int curr_cnt,
            GPUArray<int>& affected, GPUArray<bool>& inAffected, GPUArray<int>& affected_cnt, //next queue
            const size_t edgeOffset = 0, ProcessingElementEnum kernelType = Thread, int increasing = 0)
        {
        
            const size_t dimBlock = 32;
            const size_t ne = numEdges;
            T* rp = rowPtr.gdata();
            T* ri = rowInd.gdata();
            T* ci = colInd.gdata();
            const int dimGrid = (curr_cnt - edgeOffset + (dimBlock)-1) / (dimBlock);
            CUDA_RUNTIME(cudaSetDevice(TcBase<T>::dev_));
            
            kernel_serial_pe_level_affected<T, dimBlock> << <dimGrid, dimBlock, 0, TcBase<T>::stream_ >> > (
                rp, ri, ci, ne,
                level, processed.gdata(),
                curr.gdata(), inCurr.gdata(), curr_cnt,
                affected.gdata(), inAffected.gdata(), *affected_cnt.gdata());
        
        }




        void set_per_edge_async(GPUArray<T>& tcs, GPUArray<T> triPointer, GPUArray<T> rowPtr, GPUArray<T> rowInd, GPUArray<T> colInd, const size_t numEdges, const size_t edgeOffset = 0, ProcessingElementEnum kernelType = Thread, int increasing = 0)
        {

            const size_t dimBlock = 512;
            const size_t ne = numEdges;
            T* rp = rowPtr.gdata();
            T* ri = rowInd.gdata();
            T* ci = colInd.gdata();

            CUDA_RUNTIME(cudaMemset(TcBase<T>::count_, 0, sizeof(*TcBase<T>::count_)));

            // create one warp per edge
            const int dimGrid = (numEdges - edgeOffset + (dimBlock)-1) / (dimBlock);
            const int dimGridWarp = (32 * numEdges + (dimBlock)-1) / (dimBlock);
            const int dimGridBlock = (dimBlock * numEdges + (dimBlock)-1) / (dimBlock);

            assert(TcBase<T>::count_);
            Log(LogPriorityEnum::info, "device = %d, blocks = %d, threads = %d", TcBase<T>::dev_, dimGrid, dimBlock);
            CUDA_RUNTIME(cudaSetDevice(TcBase<T>::dev_));


            CUDA_RUNTIME(cudaEventRecord(TcBase<T>::kernelStart_, TcBase<T>::stream_));
            kernel_serial_set_arrays<T, dimBlock> << <dimGrid, dimBlock, 0, TcBase<T>::stream_ >> > (tcs.gdata(), triPointer.gdata(), rp, ri, ci, ne, edgeOffset, increasing);
            CUDA_RUNTIME(cudaEventRecord(TcBase<T>::kernelStop_, TcBase<T>::stream_));

        }


        uint64 count_sync(uint32_t* rowPtr, uint32_t* rowInd, uint32_t* colInd, const size_t edgeOffset, const size_t n) {
            count_async(rowPtr, rowInd, colInd, edgeOffset, n);
            TcBase<T>::sync();
            return TcBase<T>::count();
        }


    };


}