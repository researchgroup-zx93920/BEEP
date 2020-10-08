#pragma once
#include "TcBase.cuh"


template <typename T, size_t BLOCK_DIM_X>
__global__ void __launch_bounds__(BLOCK_DIM_X)
kernel_hash_thread_arrays(uint64* count, //!< [inout] the count, caller should zero
    T* rowPtr, T* rowInd, T* colInd,
    T* hp, T* hbs, const size_t hashConstant,
    const size_t numEdges, const size_t edgeStart, int increasing = 0) {

    size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
    uint64 threadCount = 0;

    for (size_t i = gx + edgeStart; i < numEdges; i += BLOCK_DIM_X * gridDim.x) {
        const T src = rowInd[i];
        const T dst = colInd[i];

        // assert(src < dst);

        bool srcHashed = hp[src + 1] - hp[src] > 0;
        bool dstHashed = hp[dst + 1] - hp[dst] > 0;
        bool bothHashed = srcHashed && dstHashed;

        const T srcStart = rowPtr[src];
        const T srcStop = rowPtr[src + 1];

        const T dstStart = rowPtr[dst];
        const T dstStop = rowPtr[dst + 1];

        const T dstLen = dstStop - dstStart;
        const T srcLen = srcStop - srcStart;

        //printf("SrcLen = %u, Dst Len = %u\n", srcLen, dstLen);

        if (!srcHashed && !dstHashed)
        {

            //printf("GPU: Binary Search\n");
            if (dstLen > srcLen) {
                threadCount += graph::thread_sorted_count_binary<T>(&(colInd[srcStart]), srcLen,
                    &(colInd[dstStart]), dstLen);
            }
            else {
                threadCount += graph::thread_sorted_count_binary<T>(&(colInd[dstStart]), dstLen,
                    &(colInd[srcStart]), srcLen);
            }
        }
        else if (!srcHashed || (bothHashed && dstLen >= srcLen)) //only dest is hashed
        {
            T binStart = hp[dst];
            const uint numBins = dstLen / hashConstant;

            T min = increasing == 0 ? 0 : dst;

            threadCount += graph::hash_search_nostash_thread_d<T, BLOCK_DIM_X>(&colInd[srcStart], srcLen,
                &hbs[binStart],
                &colInd[dstStart], numBins);
        }
        else if (!dstHashed || (bothHashed && srcLen > dstLen))
        {
            const uint numBins = srcLen / hashConstant;
            T binStart = hp[src];


            T min = increasing == 0 ? 0 : dst;

            threadCount += graph::hash_search_nostash_thread_d<T, BLOCK_DIM_X>(&colInd[dstStart], dstLen,
                &hbs[binStart],
                &colInd[srcStart], numBins);
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
kernel_hash_warp_arrays(uint64* count, //!< [inout] the count, caller should zero
    T* rowPtr, T* rowInd, T* colInd,
    T* hp, T* hbs, const size_t hashConstant,
    const size_t numEdges, const size_t edgeStart, int increasing = 0) {

    constexpr size_t warpsPerBlock = BLOCK_DIM_X / 32;
    const size_t lx = threadIdx.x % 32;

    const size_t gwx = (BLOCK_DIM_X * blockIdx.x + threadIdx.x) / 32;
    uint64 warpCount = 0;

    for (size_t i = gwx + edgeStart; i < numEdges; i += BLOCK_DIM_X * gridDim.x / 32) {
        const T src = rowInd[i];
        const T dst = colInd[i];

        //assert(src < dst);

        bool srcHashed = hp[src + 1] - hp[src] > 0;
        bool dstHashed = hp[dst + 1] - hp[dst] > 0;
        bool bothHashed = srcHashed && dstHashed;

        const T srcStart = rowPtr[src];
        const T srcStop = rowPtr[src + 1];

        const T dstStart = rowPtr[dst];
        const T dstStop = rowPtr[dst + 1];

        const T dstLen = dstStop - dstStart;
        const T srcLen = srcStop - srcStart;

        //printf("SrcLen = %u, Dst Len = %u\n", srcLen, dstLen);

        if (!(srcHashed || dstHashed))
        {
            if (dstLen > srcLen) {
                warpCount += graph::warp_sorted_count_binary<warpsPerBlock, T>(&(colInd[srcStart]), srcLen,
                    &(colInd[dstStart]), dstLen);
            }
            else {
                warpCount += graph::warp_sorted_count_binary<warpsPerBlock, T>(&(colInd[dstStart]), dstLen,
                    &(colInd[srcStart]), srcLen);
            }
        }
        else if (!srcHashed || (bothHashed && srcLen <= dstLen)) //only dest is hashed
        {
            //printf("DST HASH\n");

            T binStart = hp[dst];
            const uint numBins = dstLen / hashConstant;

            T min = increasing == 0 ? 0 : dst;

            warpCount += graph::hash_search_nostash_warp_d<warpsPerBlock, T>(&colInd[srcStart], srcLen,
                &hbs[binStart],
                &colInd[dstStart], numBins);
        }
        else if (!dstHashed || (bothHashed && srcLen > dstLen))
        {

            //printf("SRC HASH\n");

            T binStart = hp[src];
            const uint numBins = srcLen / hashConstant;

            T min = increasing == 0 ? 0 : dst;

            warpCount += graph::hash_search_nostash_warp_d<warpsPerBlock, T>(&colInd[dstStart], dstLen,
                &hbs[binStart],
                &colInd[srcStart], numBins);
        }
        //else
        //{
        //    const uint srcNumBins = srcLen / hashConstant;

        //}

    }

    if (lx == 0)
        atomicAdd(count, warpCount);
}




namespace graph {

    template<typename T>
    class TcVariableHash : public TcBase<T>
    {
    public:

        TcVariableHash(int dev, uint64 ne, uint64 nn, cudaStream_t stream = 0) :TcBase<T>(dev, ne, nn, stream)
        {}

        void count_hash_async(const int divideConstant, COOCSRGraph_d<T>* g,
            GPUArray<T> hashedColInd,
            GPUArray<T> hashPointer, GPUArray<T> hashBinStart,
            const size_t numEdges, const size_t edgeOffset = 0, ProcessingElementEnum kernelType = Thread, int increasing = 0)
        {
            const size_t dimBlock = 512;
            const size_t ne = numEdges;
            T* rp = g->rowPtr;
            T* ri = g->rowInd;

            T* hci = hashedColInd.gdata();
            T* hp = hashPointer.gdata();
            T* hbs = hashBinStart.gdata();

            CUDA_RUNTIME(cudaMemset(TcBase<T>::count_, 0, sizeof(*TcBase<T>::count_)));

            // create one warp per edge
            const int dimGrid = (numEdges - edgeOffset + (dimBlock)-1) / (dimBlock);
            const int dimGridWarp = (32 * numEdges + (dimBlock)-1) / (dimBlock);
            const int dimGridBlock = (dimBlock * numEdges + (dimBlock)-1) / (dimBlock);

            assert(TcBase<T>::count_);
            Log(LogPriorityEnum::debug, "device = %d, blocks = %d, threads = %d\n", TcBase<T>::dev_, dimGrid, dimBlock);
            CUDA_RUNTIME(cudaSetDevice(TcBase<T>::dev_));



            CUDA_RUNTIME(cudaEventRecord(TcBase<T>::kernelStart_, TcBase<T>::stream_));
            if (kernelType == ProcessingElementEnum::Thread)
                kernel_hash_thread_arrays<T, dimBlock> << <dimGrid, dimBlock, 0, TcBase<T>::stream_ >> > (TcBase<T>::count_, rp, ri, hci, hp, hbs, divideConstant, ne, edgeOffset);
            else if (kernelType == ProcessingElementEnum::Warp)
                kernel_hash_warp_arrays<T, dimBlock> << <dimGridWarp, dimBlock, 0, TcBase<T>::stream_ >> > (TcBase<T>::count_, rp, ri, hci, hp, hbs, divideConstant, ne, edgeOffset);
            CUDA_RUNTIME(cudaEventRecord(TcBase<T>::kernelStop_, TcBase<T>::stream_));


        }


        void count_per_edge_async(GPUArray<T>& tcpt, GPUArray<T> rowPtr, GPUArray<T> rowInd, GPUArray<T> colInd, const size_t numEdges, const size_t edgeOffset = 0, ProcessingElementEnum kernelType = Thread, int increasing = 0)
        {
            //const size_t dimBlock = 512;
            //const size_t ne = numEdges;
            //T* rp = rowPtr.gdata();
            //T* ri = rowInd.gdata();
            //T* ci = colInd.gdata();

            //CUDA_RUNTIME(cudaMemset(TcBase<T>::count_, 0, sizeof(*TcBase<T>::count_)));

            //// create one warp per edge
            //const int dimGrid = (numEdges - edgeOffset + (dimBlock)-1) / (dimBlock);
            //const int dimGridWarp = (32 * numEdges + (dimBlock)-1) / (dimBlock);
            //const int dimGridBlock = (dimBlock * numEdges + (dimBlock)-1) / (dimBlock);

            //assert(TcBase<T>::count_);
            //Log(LogPriorityEnum::info, "device = %d, blocks = %d, threads = %d\n", TcBase<T>::dev_, dimGrid, dimBlock);
            //CUDA_RUNTIME(cudaSetDevice(TcBase<T>::dev_));


            //CUDA_RUNTIME(cudaEventRecord(TcBase<T>::kernelStart_, TcBase<T>::stream_));
            //kernel_serial_pe_arrays<T, dimBlock> << <dimGrid, dimBlock, 0, TcBase<T>::stream_ >> > (tcpt.gdata(), rp, ri, ci, ne, edgeOffset, increasing);
            //CUDA_RUNTIME(cudaEventRecord(TcBase<T>::kernelStop_, TcBase<T>::stream_));
        }


        void set_per_edge_async(GPUArray<T>& tcs, GPUArray<T> triPointer, GPUArray<T> rowPtr, GPUArray<T> rowInd, GPUArray<T> colInd, const size_t numEdges, const size_t edgeOffset = 0, ProcessingElementEnum kernelType = Thread, int increasing = 0)
        {

            //const size_t dimBlock = 512;
            //const size_t ne = numEdges;
            //T* rp = rowPtr.gdata();
            //T* ri = rowInd.gdata();
            //T* ci = colInd.gdata();

            //CUDA_RUNTIME(cudaMemset(TcBase<T>::count_, 0, sizeof(*TcBase<T>::count_)));

            //// create one warp per edge
            //const int dimGrid = (numEdges - edgeOffset + (dimBlock)-1) / (dimBlock);
            //const int dimGridWarp = (32 * numEdges + (dimBlock)-1) / (dimBlock);
            //const int dimGridBlock = (dimBlock * numEdges + (dimBlock)-1) / (dimBlock);

            //assert(TcBase<T>::count_);
            //Log(LogPriorityEnum::info, "device = %d, blocks = %d, threads = %d\n", TcBase<T>::dev_, dimGrid, dimBlock);
            //CUDA_RUNTIME(cudaSetDevice(TcBase<T>::dev_));


            //CUDA_RUNTIME(cudaEventRecord(TcBase<T>::kernelStart_, TcBase<T>::stream_));
            //kernel_serial_set_arrays<T, dimBlock> << <dimGrid, dimBlock, 0, TcBase<T>::stream_ >> > (tcs.gdata(), triPointer.gdata(), rp, ri, ci, ne, edgeOffset, increasing);
            //CUDA_RUNTIME(cudaEventRecord(TcBase<T>::kernelStop_, TcBase<T>::stream_));

        }



    };


}