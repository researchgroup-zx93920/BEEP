#pragma once
#include "TcBase.cuh"


template <typename T, size_t BLOCK_DIM_X>
__global__ void __launch_bounds__(BLOCK_DIM_X)
kernel_binary_thread_arrays(uint64_t* count, //!< [inout] the count, caller should zero
    T* rowPtr, T* rowInd, T* colInd, const size_t numEdges, const size_t edgeStart) {

    size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
    uint64_t threadCount = 0;

    for (size_t i = gx + edgeStart; i < numEdges; i += BLOCK_DIM_X * gridDim.x) {
        const T src = rowInd[i];
        const T dst = colInd[i];

        const T srcStart = rowPtr[src];
        const T srcStop = rowPtr[src + 1];

        const T dstStart = rowPtr[dst];
        const T dstStop = rowPtr[dst + 1];

        const T dstLen = dstStop - dstStart;
        const T srcLen = srcStop - srcStart;

       //printf("%u,%u,%u,%u,%u,%u,%u,%u\n", src, dst, srcStart, srcStop, dstStart, dstStop, srcLen, dstLen);

        if (dstLen > srcLen) {
            threadCount += graph::thread_sorted_count_binary<T>(&(colInd[srcStart]), srcLen,
                &(colInd[dstStart]), dstLen);
        }
        else {
            threadCount += graph::thread_sorted_count_binary<T>(&(colInd[dstStart]), dstLen,
                &(colInd[srcStart]), srcLen);
        }
    }

    // Block-wide reduction of threadCount
    typedef cub::BlockReduce<uint64_t, BLOCK_DIM_X> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    uint64_t aggregate = BlockReduce(tempStorage).Sum(threadCount);

    // Add to total count
    if (0 == threadIdx.x) {
        atomicAdd(count, aggregate);
    }
}

template <typename T, size_t BLOCK_DIM_X>
__global__ void __launch_bounds__(BLOCK_DIM_X)
kernel_binary_thread_pe_arrays(T* count, //!< [inout] the count, caller should zero
    T* rowPtr, T* rowInd, T* colInd, const size_t numEdges, const size_t edgeStart) {

    size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
    uint64_t threadCount = 0;

    for (size_t i = gx + edgeStart; i < numEdges; i += BLOCK_DIM_X * gridDim.x) {
        const T src = rowInd[i];
        const T dst = colInd[i];

        const T srcStart = rowPtr[src];
        const T srcStop = rowPtr[src + 1];

        const T dstStart = rowPtr[dst];
        const T dstStop = rowPtr[dst + 1];

        const T dstLen = dstStop - dstStart;
        const T srcLen = srcStop - srcStart;

        //printf("%u,%u,%u,%u,%u,%u,%u,%u\n", src, dst, srcStart, srcStop, dstStart, dstStop, srcLen, dstLen);

        if (dstLen > srcLen) {
            threadCount += graph::thread_sorted_count_binary<T>(&(colInd[srcStart]), srcLen,
                &(colInd[dstStart]), dstLen);
        }
        else {
            threadCount += graph::thread_sorted_count_binary<T>(&(colInd[dstStart]), dstLen,
                &(colInd[srcStart]), srcLen);
        }
    }
    
    count[gx + edgeStart] = threadCount;
}


template <typename T, size_t BLOCK_DIM_X>
__global__ void __launch_bounds__(BLOCK_DIM_X)
kernel_binary_thread_set_arrays(T* count, //!< [inout] the count, caller should zero
    T* triPointer,
    T* rowPtr, T* rowInd, T* colInd, const size_t numEdges, const size_t edgeStart) {

    size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
    uint64_t threadCount = 0;

    for (size_t i = gx + edgeStart; i < numEdges; i += BLOCK_DIM_X * gridDim.x) {
        const T src = rowInd[i];
        const T dst = colInd[i];

        const T srcStart = rowPtr[src];
        const T srcStop = rowPtr[src + 1];

        const T dstStart = rowPtr[dst];
        const T dstStop = rowPtr[dst + 1];

        const T dstLen = dstStop - dstStart;
        const T srcLen = srcStop - srcStart;

        const T startTriStorage = triPointer[i];

        if (dstLen > srcLen) {
            threadCount += graph::thread_sorted_set_binary<T>(&count[startTriStorage],
                &(colInd[srcStart]), srcLen,
                &(colInd[dstStart]), dstLen);
        }
        else {
            threadCount += graph::thread_sorted_set_binary<T>(&count[startTriStorage],
                &(colInd[dstStart]), dstLen,
                &(colInd[srcStart]), srcLen);
        }
    }
}




template <typename T, size_t BLOCK_DIM_X>
__global__ void __launch_bounds__(BLOCK_DIM_X)
kernel_binary_warp_arrays(uint64_t* count,                //!< [inout] the count, caller should zero
    T* rowPtr, T* rowInd, T* colInd, const size_t numEdges, //!< the number of edges this kernel will count
    const size_t edgeStart                       //!< the edge this kernel will start counting at
) {
    constexpr size_t warpsPerBlock = BLOCK_DIM_X / 32;

    const size_t lx = threadIdx.x % 32;
    const size_t gwx = (BLOCK_DIM_X * blockIdx.x + threadIdx.x) / 32;
    uint64_t warpCount = 0;

    for (size_t i = gwx + edgeStart; i < numEdges; i += BLOCK_DIM_X * gridDim.x / 32) {
        const T src = rowInd[i];
        const T dst = colInd[i];

        const T srcStart = rowPtr[src];
        const T srcStop = rowPtr[src + 1];

        const T dstStart = rowPtr[dst];
        const T dstStop = rowPtr[dst + 1];

        const T dstLen = dstStop - dstStart;
        const T srcLen = srcStop - srcStart;

        // FIXME: remove warp reduction from this function call
        if (dstLen > srcLen) {
            warpCount += graph::warp_sorted_count_binary<warpsPerBlock>(&colInd[srcStart], srcLen,
                &colInd[dstStart], dstLen);
        }
        else {
            warpCount += graph::warp_sorted_count_binary<warpsPerBlock>(&colInd[dstStart], dstLen,
                &colInd[srcStart], srcLen);
        }
    }

    // if(lx > 0)
    //   warpCount = 0;


    //   __syncthreads();
    //   typedef cub::BlockReduce<uint64_t, BLOCK_DIM_X> BlockReduce;
    //   __shared__ typename BlockReduce::TempStorage tempStorage;
    //   uint64_t aggregate = BlockReduce(tempStorage).Sum(warpCount);


    //   if(threadIdx.x == 0)
    //     atomicAdd(count, aggregate);


    if (lx == 0)
        atomicAdd(count, warpCount);


}


template <typename T, size_t BLOCK_DIM_X>
__global__ void __launch_bounds__(BLOCK_DIM_X)
kernel_binary_warp_pe_arrays(T* count,                //!< [inout] the count, caller should zero
    T* rowPtr, T* rowInd, T* colInd, const size_t numEdges, //!< the number of edges this kernel will count
    const size_t edgeStart                       //!< the edge this kernel will start counting at
) {
    constexpr size_t warpsPerBlock = BLOCK_DIM_X / 32;

    const size_t lx = threadIdx.x % 32;
    const size_t gwx = (BLOCK_DIM_X * blockIdx.x + threadIdx.x) / 32;
    uint64_t warpCount = 0;

    for (size_t i = gwx + edgeStart; i < numEdges; i += BLOCK_DIM_X * gridDim.x / 32) {
        const T src = rowInd[i];
        const T dst = colInd[i];

        const T srcStart = rowPtr[src];
        const T srcStop = rowPtr[src + 1];

        const T dstStart = rowPtr[dst];
        const T dstStop = rowPtr[dst + 1];

        const T dstLen = dstStop - dstStart;
        const T srcLen = srcStop - srcStart;

        // FIXME: remove warp reduction from this function call
        if (dstLen > srcLen) {
            warpCount += graph::warp_sorted_count_binary<warpsPerBlock>(&colInd[srcStart], srcLen,
                &colInd[dstStart], dstLen);
        }
        else {
            warpCount += graph::warp_sorted_count_binary<warpsPerBlock>(&colInd[dstStart], dstLen,
                &colInd[srcStart], srcLen);
        }
    }
    if (lx == 0)
        count[gwx + edgeStart] = warpCount;
}


template <typename T, size_t BLOCK_DIM_X>
__global__ void __launch_bounds__(BLOCK_DIM_X)
kernel_binary_warp_set_arrays(T* count,
    T* triPointer,//!< [inout] the count, caller should zero
    T* rowPtr, T* rowInd, T* colInd, const size_t numEdges, //!< the number of edges this kernel will count
    const size_t edgeStart                       //!< the edge this kernel will start counting at
) {
    constexpr size_t warpsPerBlock = BLOCK_DIM_X / 32;

    __shared__ T index[warpsPerBlock];

    const size_t wid = threadIdx.x / 32;
    const size_t lx = threadIdx.x % 32;
    if (lx == 0)
        index[wid] = 0;

    const size_t gwx = (BLOCK_DIM_X * blockIdx.x + threadIdx.x) / 32;
    uint64_t warpCount = 0;

    for (size_t i = gwx + edgeStart; i < numEdges; i += BLOCK_DIM_X * gridDim.x / 32) {
        const T src = rowInd[i];
        const T dst = colInd[i];

        const T srcStart = rowPtr[src];
        const T srcStop = rowPtr[src + 1];

        const T dstStart = rowPtr[dst];
        const T dstStop = rowPtr[dst + 1];

        const T dstLen = dstStop - dstStart;
        const T srcLen = srcStop - srcStart;

        const T startTriStorage = triPointer[i];

        // FIXME: remove warp reduction from this function call
        if (dstLen > srcLen) {
            warpCount += graph::warp_sorted_set_binary<warpsPerBlock, T>(&index[wid], &count[startTriStorage], &colInd[srcStart], srcLen,
                &colInd[dstStart], dstLen);
        }
        else {
            warpCount += graph::warp_sorted_set_binary<warpsPerBlock, T>(&index[wid], &count[startTriStorage], &colInd[dstStart], dstLen,
                &colInd[srcStart], srcLen);
        }
    }
}


namespace graph {

    
    template<typename T>
    class TcBinary : public TcBase<T>
    {
    public:

        TcBinary(int dev, uint64_t ne, uint64_t nn, cudaStream_t stream = 0) :TcBase(dev, ne, nn, stream)
        {}

        void count_async(GPUArray<T> rowPtr, GPUArray<T> rowInd, GPUArray<T> colInd, const size_t numEdges, const size_t edgeOffset = 0, ProcessingElementEnum kernelType = Thread, int limit = 0)
        {
            const size_t dimBlock = 512;
            const size_t ne = numEdges;
            T* rp = rowPtr.gdata();
            T* ri = rowInd.gdata();
            T* ci = colInd.gdata();

            CUDA_RUNTIME(cudaMemset(count_, 0, sizeof(*count_)));

            // create one warp per edge
            const int dimGrid = (numEdges - edgeOffset + (dimBlock)-1) / (dimBlock);
            const int dimGridWarp = (32 * (numEdges-edgeOffset) + (dimBlock)-1) / (dimBlock);
            const int dimGridBlock = (dimBlock * numEdges + (dimBlock)-1) / (dimBlock);

            assert(count_);
            Log(LogPriorityEnum::info, "device = %d, blocks = %d, threads = %d\n", dev_, dimGrid, dimBlock);
            CUDA_RUNTIME(cudaSetDevice(dev_));


            CUDA_RUNTIME(cudaEventRecord(kernelStart_, stream_));
            if(kernelType == ProcessingElementEnum::Thread)
                kernel_binary_thread_arrays<T, dimBlock> << <dimGrid, dimBlock, 0, stream_ >> > (count_, rp, ri, ci, ne, edgeOffset);
            else if(kernelType == ProcessingElementEnum::Warp)
                kernel_binary_warp_arrays<T, dimBlock> << <dimGridWarp, dimBlock, 0, stream_ >> > (count_, rp, ri, ci, ne, edgeOffset);
            CUDA_RUNTIME(cudaEventRecord(kernelStop_, stream_));
        }

        void count_per_edge_async(GPUArray<T>& tcpt, GPUArray<T> rowPtr, GPUArray<T> rowInd, GPUArray<T> colInd, const size_t numEdges, const size_t edgeOffset = 0, ProcessingElementEnum kernelType = Thread, int limit = 0)
        {

            const size_t dimBlock = 512;
            const size_t ne = numEdges;
            T* rp = rowPtr.gdata();
            T* ri = rowInd.gdata();
            T* ci = colInd.gdata();

            // create one warp per edge
            const int dimGrid = (numEdges - edgeOffset + (dimBlock)-1) / (dimBlock);
            const int dimGridWarp = (32 * numEdges + (dimBlock)-1) / (dimBlock);
            const int dimGridBlock = (dimBlock * numEdges + (dimBlock)-1) / (dimBlock);

            assert(count_);
            Log(LogPriorityEnum::info, "device = %d, blocks = %d, threads = %d\n", dev_, dimGrid, dimBlock);
            CUDA_RUNTIME(cudaSetDevice(dev_));


            
            CUDA_RUNTIME(cudaEventRecord(kernelStart_, stream_));
            if (kernelType == ProcessingElementEnum::Thread)
                kernel_binary_thread_pe_arrays<T, dimBlock> << <dimGrid, dimBlock, 0, stream_ >> > (tcpt.gdata(), rp, ri, ci, ne, edgeOffset);
            else if (kernelType == ProcessingElementEnum::Warp)
                kernel_binary_warp_pe_arrays<T, dimBlock> << <dimGridWarp, dimBlock, 0, stream_ >> > (tcpt.gdata(), rp, ri, ci, ne, edgeOffset);
            CUDA_RUNTIME(cudaEventRecord(kernelStop_, stream_));
        }



        void set_per_edge_async(GPUArray<T>& tcpt, GPUArray<T> triPointer, GPUArray<T> rowPtr, GPUArray<T> rowInd, GPUArray<T> colInd, const size_t numEdges, const size_t edgeOffset = 0, ProcessingElementEnum kernelType = Thread, int increasing = 0)
        {

            const size_t dimBlock = 512;
            const size_t ne = numEdges;
            T* rp = rowPtr.gdata();
            T* ri = rowInd.gdata();
            T* ci = colInd.gdata();

            // create one warp per edge
            const int dimGrid = (numEdges - edgeOffset + (dimBlock)-1) / (dimBlock);
            const int dimGridWarp = (32 * numEdges + (dimBlock)-1) / (dimBlock);
            const int dimGridBlock = (dimBlock * numEdges + (dimBlock)-1) / (dimBlock);

            assert(count_);
            Log(LogPriorityEnum::info, "device = %d, blocks = %d, threads = %d\n", dev_, dimGrid, dimBlock);
            CUDA_RUNTIME(cudaSetDevice(dev_));


            CUDA_RUNTIME(cudaEventRecord(kernelStart_, stream_));
            if (kernelType == ProcessingElementEnum::Thread)
                kernel_binary_thread_set_arrays<T, dimBlock> << <dimGrid, dimBlock, 0, stream_ >> > (tcpt.gdata(), triPointer.gdata(), rp, ri, ci, ne, edgeOffset);
            else if (kernelType == ProcessingElementEnum::Warp)
                kernel_binary_warp_set_arrays<T, dimBlock> << <dimGridWarp, dimBlock, 0, stream_ >> > (tcpt.gdata(), triPointer.gdata(), rp, ri, ci, ne, edgeOffset);
            CUDA_RUNTIME(cudaEventRecord(kernelStop_, stream_));

        }


        uint64_t count_sync(T* rowPtr, T* rowInd, T* colInd, const size_t edgeOffset, const size_t n) {
            count_async(rowPtr, rowInd, colInd, edgeOffset, n);
            sync();
            return count();
        }


    };


}