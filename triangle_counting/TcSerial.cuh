#pragma once
#include "TcBase.cuh"


template <typename T, size_t BLOCK_DIM_X>
__global__ void __launch_bounds__(BLOCK_DIM_X)
kernel_serial_arrays(uint64_t* count, //!< [inout] the count, caller should zero
    T* rowPtr, T* rowInd, T* colInd, const size_t numEdges, const size_t edgeStart) {

    size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
    uint64_t threadCount = 0;

    for (size_t i = gx + edgeStart; i < numEdges; i += BLOCK_DIM_X * gridDim.x) {
        const T src = rowInd[i];
        const T dst = colInd[i];

        const T* srcBegin = &colInd[rowPtr[src]];
        const T* srcEnd = &colInd[rowPtr[src + 1]];
        const T* dstBegin = &colInd[rowPtr[dst]];
        const T* dstEnd = &colInd[rowPtr[dst + 1]];

        threadCount += graph::serial_sorted_count_linear(srcBegin, srcEnd, dstBegin, dstEnd);
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


namespace graph {

    template<typename T>
	class TcSerial : public TcBase<T>
	{
	public:

        TcSerial(int dev, uint64_t ne, uint64_t nn, cudaStream_t stream = 0) :TcBase(dev, ne, nn, stream)
        {}

        void count_async(GPUArray<T> rowPtr, GPUArray<T> rowInd, GPUArray<T> colInd, const size_t numEdges, const size_t edgeOffset = 0, char kernelType = 1, int limit = 0)
        {
            const size_t dimBlock = 512;
            const size_t ne = numEdges;
            T* rp = rowPtr.gdata();
            T* ri = rowInd.gdata();
            T* ci = colInd.gdata();

            CUDA_RUNTIME(cudaMemset(count_, 0, sizeof(*count_)));

            // create one warp per edge
            const int dimGrid = (numEdges-edgeOffset + (dimBlock) - 1) / (dimBlock);
            const int dimGridWarp = (32 * numEdges + (dimBlock) - 1) / (dimBlock);
            const int dimGridBlock = (dimBlock * numEdges + (dimBlock) - 1) / (dimBlock);

            assert(count_);
            Log(LogPriorityEnum::info, "device = %d, blocks = %d, threads = %d\n", dev_, dimGrid, dimBlock);
            CUDA_RUNTIME(cudaSetDevice(dev_));


            CUDA_RUNTIME(cudaEventRecord(kernelStart_, stream_));
            kernel_serial_arrays<T, dimBlock><<<dimGrid, dimBlock, 0, stream_>>>(count_, rp, ri, ci, ne, edgeOffset);
            CUDA_RUNTIME(cudaEventRecord(kernelStop_, stream_));
        }

        uint64_t count_sync(uint32_t* rowPtr, uint32_t* rowInd, uint32_t* colInd, const size_t edgeOffset, const size_t n) {
            count_async(rowPtr, rowInd, colInd, edgeOffset, n);
            sync();
            return count();
        }
		

	};


}