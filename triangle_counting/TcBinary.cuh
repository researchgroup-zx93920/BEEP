#pragma once
#include "TcBase.cuh"


template <typename T, size_t BLOCK_DIM_X>
__global__ void __launch_bounds__(BLOCK_DIM_X)
kernel_binary_thread_arrays(uint64* count, //!< [inout] the count, caller should zero
    T* rowPtr, T* rowInd, T* colInd, const size_t numEdges, const size_t edgeStart) {

    size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
    uint64 threadCount = 0;

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
kernel_binary_thread_pe_arrays(int* count, //!< [inout] the count, caller should zero
    T* rowPtr, T* rowInd, T* colInd, const size_t numEdges, const size_t edgeStart) {

    size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
    uint64 threadCount = 0;

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
kernel_binary_thread_pe_upto_arrays(int upto, bool* mask, int* count, //!< [inout] the count, caller should zero
    T* rowPtr, T* rowInd, T* colInd, const size_t numEdges, const size_t edgeStart) {

    size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
    uint64 threadCount = 0;

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
            threadCount += graph::thread_sorted_count_upto_binary<T>(upto, mask, srcStart, dstStart, colInd, srcLen,
                 dstLen);
        }
        /*else {
            threadCount += graph::thread_sorted_count_upto_binary<T>(upto, mask, dstStart, srcStart, colInd, dstLen,
                srcLen);
        }*/
    }

    count[gx + edgeStart] = threadCount;
}



template <typename T, size_t BLOCK_DIM_X>
__global__ void __launch_bounds__(BLOCK_DIM_X)
kernel_binary_thread_set_arrays(int* count, //!< [inout] the count, caller should zero
    T* triPointer,
    T* rowPtr, T* rowInd, T* colInd, const size_t numEdges, const size_t edgeStart) {

    size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
    uint64 threadCount = 0;

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
kernel_binary_warp_arrays(uint64* count,                //!< [inout] the count, caller should zero
    T* rowPtr, T* rowInd, T* colInd, const size_t numEdges, //!< the number of edges this kernel will count
    const size_t edgeStart                       //!< the edge this kernel will start counting at
) {
    constexpr size_t warpsPerBlock = BLOCK_DIM_X / 32;

    const size_t lx = threadIdx.x % 32;
    const size_t gwx = (BLOCK_DIM_X * blockIdx.x + threadIdx.x) / 32;
    uint64 warpCount = 0;

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
    //   typedef cub::BlockReduce<uint64, BLOCK_DIM_X> BlockReduce;
    //   __shared__ typename BlockReduce::TempStorage tempStorage;
    //   uint64 aggregate = BlockReduce(tempStorage).Sum(warpCount);


    //   if(threadIdx.x == 0)
    //     atomicAdd(count, aggregate);


    if (lx == 0)
        atomicAdd(count, warpCount);


}


template <typename T, size_t BLOCK_DIM_X>
__global__ void __launch_bounds__(BLOCK_DIM_X)
kernel_binary_warp_pe_arrays( int* count,                //!< [inout] the count, caller should zero
    T* rowPtr, T* rowInd, T* colInd, const size_t numEdges, //!< the number of edges this kernel will count
    const size_t edgeStart                       //!< the edge this kernel will start counting at
) {
    constexpr size_t warpsPerBlock = BLOCK_DIM_X / 32;

    const size_t lx = threadIdx.x % 32;
    const size_t gwx = (BLOCK_DIM_X * blockIdx.x + threadIdx.x) / 32;
    uint64 warpCount = 0;

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
kernel_binary_warp_pe_arrays(bool* mask, int* count,                //!< [inout] the count, caller should zero
    T* rowPtr, T* rowInd, T* colInd, const size_t numEdges, //!< the number of edges this kernel will count
    const size_t edgeStart                       //!< the edge this kernel will start counting at
) {
    constexpr size_t warpsPerBlock = BLOCK_DIM_X / 32;

    const size_t lx = threadIdx.x % 32;
    const size_t gwx = (BLOCK_DIM_X * blockIdx.x + threadIdx.x) / 32;
    uint64 warpCount = 0;

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
            warpCount += graph::warp_sorted_count_mask_binary<warpsPerBlock>(mask, srcStart, dstStart, colInd, srcLen,
                dstLen);
        }
       /* else {
            warpCount += graph::warp_sorted_count_mask_binary<warpsPerBlock>(mask, dstStart, srcStart, colInd, dstLen,
                 srcLen);
        }*/
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
    uint64 warpCount = 0;

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




template <typename T, size_t BLOCK_DIM_X>
__global__ void __launch_bounds__(BLOCK_DIM_X)
kernel_binary_block_arrays(uint64* count,                //!< [inout] the count, caller should zero
    T* rowPtr, T* rowInd, T* colInd, const size_t numEdges, //!< the number of edges this kernel will count
    const size_t edgeStart                       //!< the edge this kernel will start counting at
) {
    constexpr size_t warpsPerBlock = BLOCK_DIM_X / 32;

    const size_t lx = threadIdx.x;
    const size_t gwx = blockIdx.x;
    uint64 blockCount = 0;

    for (size_t i = gwx + edgeStart; i < numEdges; i += gridDim.x) {
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
            blockCount += graph::block_sorted_count_binary<BLOCK_DIM_X>(&colInd[srcStart], srcLen,
                &colInd[dstStart], dstLen);
        }
        else {
            blockCount += graph::block_sorted_count_binary<BLOCK_DIM_X>(&colInd[dstStart], dstLen,
                &colInd[srcStart], srcLen);
        }
    }

    if (threadIdx.x == 0)
        atomicAdd(count, blockCount);


}


template <typename T, size_t BLOCK_DIM_X>
__global__ void __launch_bounds__(BLOCK_DIM_X)
kernel_binary_warp_shared_arrays(uint64* count,                //!< [inout] the count, caller should zero
    T* rowPtr, T* rowInd, T* colInd, const size_t numEdges, //!< the number of edges this kernel will count
    const size_t edgeStart                       //!< the edge this kernel will start counting at
) {
    constexpr size_t warpsPerBlock = BLOCK_DIM_X / 32;
    const size_t lx = threadIdx.x % 32;
    const size_t wx = threadIdx.x / 32;
    const size_t gwx = (BLOCK_DIM_X * blockIdx.x + threadIdx.x) / 32;

    const int num32perWarp = 4;
    const int pwMaxSize = num32perWarp * 32;
   
    __shared__ T first[warpsPerBlock * pwMaxSize];


    uint64 warpCount = 0;

    for (size_t i = gwx + edgeStart; i < numEdges; i += BLOCK_DIM_X * gridDim.x / 32) {
        T src = rowInd[i];
        T dst = colInd[i];

        T srcStart = rowPtr[src];
        T srcStop = rowPtr[src + 1];

        T dstStart = rowPtr[dst];
        T dstStop = rowPtr[dst + 1];

        T dstLen = dstStop - dstStart;
        T srcLen = srcStop - srcStart;


        if (srcLen >dstLen) 
        {
            swap_ele(srcStart, dstStart);
            swap_ele(srcStop, dstStop);
            swap_ele(srcLen, dstLen);
        }
        
        int startIndex = wx * pwMaxSize;
        const T par = (dstLen + pwMaxSize - 1) / (pwMaxSize);
        const T numElements = dstLen < pwMaxSize ? dstLen : (dstLen + par - 1) / par;

        for (int i = 0; i < num32perWarp; i++)
        {
            int sharedIndex = startIndex + 32 * i + lx;
            int realIndex = dstStart + (lx + i * 32 ) * par;

            first[sharedIndex] = (lx + 32 * i) <= numElements ? colInd[realIndex] : 0;
        }

      
        warpCount += graph::warp_sorted_count_binary_s<warpsPerBlock>(&colInd[srcStart], srcLen,
                &colInd[dstStart], dstLen, &(first[startIndex]), par, numElements, pwMaxSize);

    }

    if (lx == 0)
        atomicAdd(count, warpCount);


}

template <typename T, size_t BLOCK_DIM_X>
__global__ void __launch_bounds__(BLOCK_DIM_X)
kernel_binary_warp_shared_colab_arrays(uint64* count,                //!< [inout] the count, caller should zero
    T* rowPtr, T* rowInd, T* colInd, const size_t numEdges, //!< the number of edges this kernel will count
    const size_t edgeStart                       //!< the edge this kernel will start counting at
) {
    constexpr size_t warpsPerBlock = BLOCK_DIM_X / 32;
    const unsigned int lx = threadIdx.x % 32;
    const unsigned int wx = threadIdx.x / 32;
    const size_t gwx = (BLOCK_DIM_X * blockIdx.x + threadIdx.x) / 32;


    const int num32perWarp = 4;
    const int pwMaxSize = num32perWarp * 32;
    __shared__ T comm[warpsPerBlock];
    __shared__ T first[warpsPerBlock * pwMaxSize];


    uint64 warpCount = 0;

    for (size_t i = gwx + edgeStart; i < numEdges; i += BLOCK_DIM_X * gridDim.x / 32) {
        T src = rowInd[i];
        T dst = colInd[i];

        T srcStart = rowPtr[src];
        T srcStop = rowPtr[src + 1];

        T dstStart = rowPtr[dst];
        T dstStop = rowPtr[dst + 1];

        T dstLen = dstStop - dstStart;
        T srcLen = srcStop - srcStart;


        if (srcLen > dstLen)
        {
            swap_ele(src, dst);
            swap_ele(srcStart, dstStart);
            swap_ele(srcStop, dstStop);
            swap_ele(srcLen, dstLen);
        }


        if (lx == 0)
            comm[wx] = dst;
        __syncthreads();

        bool colab = false;
        bool oddWarp = wx & 0x1 == 1;
        if (!oddWarp  && comm[wx + 1] == dst)
            colab = true;
        else if (oddWarp && comm[wx - 1] == dst)
            colab = true;

        oddWarp = colab && oddWarp;

        int colabFactor = 1;
        int startIndex = wx * pwMaxSize;
        int colabMaxSize = pwMaxSize;

        if (colab && !oddWarp)
        {
            colabMaxSize = 2 * pwMaxSize;
        }
        else if (colab && oddWarp)
        {
            colabMaxSize = 2 * pwMaxSize;
            startIndex = (wx-1) * pwMaxSize;
        }


        
        const T par = (dstLen + colabMaxSize - 1) / (colabMaxSize);
        const T numElements = dstLen < colabMaxSize ? dstLen : (dstLen + par - 1) / par;


        //if (lx == 0)
        /*{
            int sharedIndex = startIndex + oddWarp * pwMaxSize +  lx;
            int realIndex = dstStart + (lx + oddWarp * pwMaxSize) * par;


            printf("th=%u, lx = %u, dst = %u, %u, isOdd = %d, is Colab = %d, startIndex = %u, %u, %u, %d\n", threadIdx.x, lx, dst, wx, oddWarp ? 1 : 0, colab ? 1 : 0, startIndex, par, numElements, sharedIndex);
        }*/


        for (int i = 0; i < num32perWarp; i++)
        {
            int sharedIndex = startIndex + oddWarp * pwMaxSize + 32 * i + lx;
            int realIndex = dstStart + (lx + i * 32 + oddWarp * pwMaxSize) * par;

            first[sharedIndex] = (lx + 32*i + oddWarp * pwMaxSize) <= numElements ? colInd[realIndex] : 0;
        }
     
        __syncthreads();


        warpCount += graph::warp_sorted_count_binary_s<warpsPerBlock>(&colInd[srcStart], srcLen,
            &colInd[dstStart], dstLen, &(first[startIndex]), par, numElements, colabMaxSize);

    }

    if (lx == 0)
        atomicAdd(count, warpCount);
}




namespace graph {


    template<typename T>
    class TcBinary : public TcBase<T>
    {
    public:

        TcBinary(int dev, uint64 ne, uint64 nn, cudaStream_t stream = 0) :TcBase<T>(dev, ne, nn, stream)
        {}

        void count_async(GPUArray<T> rowPtr, GPUArray<T> rowInd, GPUArray<T> colInd, const size_t numEdges, const size_t edgeOffset = 0, ProcessingElementEnum kernelType = Thread, int limit = 0)
        {
            const size_t dimBlock = 128;
            const size_t ne = numEdges;
            T* rp = rowPtr.gdata();
            T* ri = rowInd.gdata();
            T* ci = colInd.gdata();

            CUDA_RUNTIME(cudaMemset(TcBase<T>::count_, 0, sizeof(*TcBase<T>::count_)));

            // create one warp per edge
            const int dimGrid = (numEdges - edgeOffset + (dimBlock)-1) / (dimBlock);
            const int dimGridWarp = (32 * (numEdges - edgeOffset) + (dimBlock)-1) / (dimBlock);
            const int dimGridBlock = (dimBlock * numEdges + (dimBlock)-1) / (dimBlock);

            assert(TcBase<T>::count_);
            Log(LogPriorityEnum::info, "device = %d, blocks = %d, threads = %d\n", TcBase<T>::dev_, dimGrid, dimBlock);
            CUDA_RUNTIME(cudaSetDevice(TcBase<T>::dev_));


            CUDA_RUNTIME(cudaEventRecord(TcBase<T>::kernelStart_, TcBase<T>::stream_));
            if (kernelType == ProcessingElementEnum::Thread)
                kernel_binary_thread_arrays<T, dimBlock> << <dimGrid, dimBlock, 0, TcBase<T>::stream_ >> > (TcBase<T>::count_, rp, ri, ci, ne, edgeOffset);
            else if (kernelType == ProcessingElementEnum::Warp)
                kernel_binary_warp_arrays<T, dimBlock> << <dimGridWarp, dimBlock, 0, TcBase<T>::stream_ >> > (TcBase<T>::count_, rp, ri, ci, ne, edgeOffset);
            else if (kernelType == ProcessingElementEnum::Test)
                kernel_binary_warp_shared_arrays<T, dimBlock> << <dimGridWarp, dimBlock, 0, TcBase<T>::stream_ >> > (TcBase<T>::count_, rp, ri, ci, ne, edgeOffset);
            else if (kernelType == ProcessingElementEnum::Block)
                kernel_binary_block_arrays<T, dimBlock> << <dimGridBlock, dimBlock, 0, TcBase<T>::stream_ >> > (TcBase<T>::count_, rp, ri, ci, ne, edgeOffset);
            CUDA_RUNTIME(cudaEventRecord(TcBase<T>::kernelStop_, TcBase<T>::stream_));
        }

        void count_per_edge_async(GPUArray<int>& tcpt, GPUArray<T> rowPtr, GPUArray<T> rowInd, GPUArray<T> colInd, const size_t numEdges, const size_t edgeOffset = 0, ProcessingElementEnum kernelType = Thread, int limit = 0)
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

            assert(TcBase<T>::count_);
            Log(LogPriorityEnum::info, "device = %d, blocks = %d, threads = %d\n", TcBase<T>::dev_, dimGrid, dimBlock);
            CUDA_RUNTIME(cudaSetDevice(TcBase<T>::dev_));



            CUDA_RUNTIME(cudaEventRecord(TcBase<T>::kernelStart_, TcBase<T>::stream_));
            if (kernelType == ProcessingElementEnum::Thread)
                kernel_binary_thread_pe_arrays<T, dimBlock> << <dimGrid, dimBlock, 0, TcBase<T>::stream_ >> > (tcpt.gdata(), rp, ri, ci, ne, edgeOffset);
            else if (kernelType == ProcessingElementEnum::Warp)
                kernel_binary_warp_pe_arrays<T, dimBlock> << <dimGridWarp, dimBlock, 0, TcBase<T>::stream_ >> > (tcpt.gdata(), rp, ri, ci, ne, edgeOffset);
            CUDA_RUNTIME(cudaEventRecord(TcBase<T>::kernelStop_, TcBase<T>::stream_));
        }

        void count_per_edge_upto_async(int upto, GPUArray<bool> mask, GPUArray<int>& tcpt, GPUArray<T> rowPtr, GPUArray<T> rowInd, GPUArray<T> colInd, const size_t numEdges, const size_t edgeOffset = 0, ProcessingElementEnum kernelType = Thread, int increasing = 0)
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

            assert(TcBase<T>::count_);
            //Log(LogPriorityEnum::info, "device = %d, blocks = %d, threads = %d\n", TcBase<T>::dev_, dimGrid, dimBlock);
            CUDA_RUNTIME(cudaSetDevice(TcBase<T>::dev_));



            CUDA_RUNTIME(cudaEventRecord(TcBase<T>::kernelStart_, TcBase<T>::stream_));
            if (kernelType == ProcessingElementEnum::Thread)
                kernel_binary_thread_pe_upto_arrays<T, dimBlock> << <dimGrid, dimBlock, 0, TcBase<T>::stream_ >> > (upto, mask.gdata(), tcpt.gdata(), rp, ri, ci, ne, edgeOffset);
            else if (kernelType == ProcessingElementEnum::Warp)
                kernel_binary_warp_pe_arrays<T, dimBlock> << <dimGridWarp, dimBlock, 0, TcBase<T>::stream_ >> > (mask.gdata(), tcpt.gdata(), rp, ri, ci, ne, edgeOffset);
            CUDA_RUNTIME(cudaEventRecord(TcBase<T>::kernelStop_, TcBase<T>::stream_));
        }



        void set_per_edge_async(GPUArray<int>& tcpt, GPUArray<T> triPointer, GPUArray<T> rowPtr, GPUArray<T> rowInd, GPUArray<T> colInd, const size_t numEdges, const size_t edgeOffset = 0, ProcessingElementEnum kernelType = Thread, int increasing = 0)
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

            assert(TcBase<T>::count_);
            Log(LogPriorityEnum::info, "device = %d, blocks = %d, threads = %d\n", TcBase<T>::dev_, dimGrid, dimBlock);
            CUDA_RUNTIME(cudaSetDevice(TcBase<T>::dev_));


            CUDA_RUNTIME(cudaEventRecord(TcBase<T>::kernelStart_, TcBase<T>::stream_));
            if (kernelType == ProcessingElementEnum::Thread)
                kernel_binary_thread_set_arrays<T, dimBlock> << <dimGrid, dimBlock, 0, TcBase<T>::stream_ >> > (tcpt.gdata(), triPointer.gdata(), rp, ri, ci, ne, edgeOffset);
            else if (kernelType == ProcessingElementEnum::Warp)
                kernel_binary_warp_set_arrays<T, dimBlock> << <dimGridWarp, dimBlock, 0, TcBase<T>::stream_ >> > (tcpt.gdata(), triPointer.gdata(), rp, ri, ci, ne, edgeOffset);
            CUDA_RUNTIME(cudaEventRecord(TcBase<T>::kernelStop_, TcBase<T>::stream_));

        }


        uint64 count_sync(T* rowPtr, T* rowInd, T* colInd, const size_t edgeOffset, const size_t n) {
            count_async(rowPtr, rowInd, colInd, edgeOffset, n);
            TcBase<T>::sync();
            return TcBase<T>::count();
        }


    };


}