#pragma once
#include "TcBase.cuh"



using namespace std;

EncodeDataType* encodeLargeRows(int nr, int* csrRowPointers, int* csrColumns, unsigned short* reverseIndex, int* numberOfLongRows)
{
    const int bitArraySize = sizeof(EncodeDataType) * 8;
    const int numEntries = nr / bitArraySize + 1;

    Log(LogPriorityEnum::debug, "bitArraySize = %d\n", bitArraySize);

    //Get largest rows
    vector<int> longRowIndex;
    for (int i = 0; i < nr; i++)
    {
        int start = csrRowPointers[i];
        int end = csrRowPointers[i + 1];
        if ((end - start) > nr / bitArraySize)
        {
            longRowIndex.push_back(i);
        }

        reverseIndex[i] = 0;
    }
    const int numLongRows = longRowIndex.size();
    EncodeDataType* bitmap = 0;
    CUDA_RUNTIME(cudaMallocManaged((void**)&bitmap, sizeof(EncodeDataType) * (numLongRows == 0 ? 1 : numLongRows) * numEntries));


    for (int k = 0; k < numLongRows; k++)
    {
        int i = longRowIndex[k];
        int stratIndex = k * numEntries;

        reverseIndex[i] = k;
        for (int z = 0; z < numEntries; z++)
        {
            bitmap[stratIndex + z] = 0;
        }

        int start = csrRowPointers[i];
        int end = csrRowPointers[i + 1];
        for (int j = start; j < end; j++)
        {
            int colIndex = csrColumns[j];

            int baseIndex = colIndex / bitArraySize;

            EncodeDataType mask = 1ULL << (colIndex % bitArraySize);
            bitmap[stratIndex + baseIndex] |= mask;
        }
    }
    *numberOfLongRows = numLongRows;
    return bitmap;
}


template <typename T, size_t BLOCK_DIM_X>
__global__ void __launch_bounds__(BLOCK_DIM_X)
kernel_binaryEncoding_thread_arrays(int* count, //!< [inout] the count, caller should zero
    T* rowPtr, T* rowInd, T* colInd, const size_t numEdges, const size_t edgeStart, 
    unsigned short* reverseIndex, EncodeDataType* bitMap, int numNodes) {

    size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
    uint64 threadCount = 0;
    const EncodeDataType bitLength = sizeof(EncodeDataType) * 8;
    const int numEntries = numNodes / bitLength + 1;

    for (size_t i = gx + edgeStart; i < numEdges; i += BLOCK_DIM_X * gridDim.x) {
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


        bool sn_lr = (srcLen) > numNodes / (bitLength);
        bool dn_lr = (dstLen) > numNodes / (bitLength);
        if (dn_lr)
        {

            
            int index = reverseIndex[dst];
            for (int v = srcStart; v < srcStop; v++)
            {
                T val = colInd[v];
                int base = val / bitLength;
                int rem = val % bitLength; // val & 0x3F;
                EncodeDataType bm = bitMap[index * numEntries + base];
                if (bm & (1ULL << rem))
                    threadCount++;
            }
        }
        else
        {
            threadCount += graph::thread_sorted_count_binary<T>(&(colInd[srcStart]), srcLen,
                &(colInd[dstStart]), dstLen);
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


namespace graph {


    template<typename T>
    class TcBinaryEncoding : public TcBase<T>
    {
    public:

        TcBinaryEncoding(int dev, uint64 ne, uint64 nn, cudaStream_t stream = 0) :TcBase<T>(dev, ne, nn, stream)
        {}


        void count_async(GPUArray<T> rowPtr, GPUArray<T> rowInd, GPUArray<T> colInd, const size_t numEdges, const size_t edgeOffset = 0, ProcessingElementEnum kernelType = Thread, int limit = 0)
        {
            const size_t dimBlock = 128;
            const size_t ne = numEdges;
            T* rp = rowPtr.gdata();
            T* ri = rowInd.gdata();
            T* ci = colInd.gdata();

            unsigned short* reversed;
            CUDA_RUNTIME(cudaMallocManaged((void**)&reversed, sizeof(unsigned short) * TcBase<T>::numNodes));

            int longRowCount = 0;
            EncodeDataType *bitMap = encodeLargeRows(TcBase<T>::numNodes, rowPtr.cdata(), colInd.cdata(), reversed, &longRowCount);


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
                kernel_binaryEncoding_thread_arrays<T, dimBlock> << <dimGrid, dimBlock, 0, TcBase<T>::stream_ >> > (TcBase<T>::count_, rp, ri, ci, ne, edgeOffset, reversed, bitMap, TcBase<T>::numNodes);
            else if (kernelType == ProcessingElementEnum::Warp)
                kernel_binary_warp_arrays<T, dimBlock> << <dimGridWarp, dimBlock, 0, TcBase<T>::stream_ >> > (TcBase<T>::count_, rp, ri, ci, ne, edgeOffset);
           
            CUDA_RUNTIME(cudaEventRecord(TcBase<T>::kernelStop_, TcBase<T>::stream_));
        }

       


        uint64 count_sync(T* rowPtr, T* rowInd, T* colInd, const size_t edgeOffset, const size_t n) {
            count_async(rowPtr, rowInd, colInd, edgeOffset, n);
            TcBase<T>::sync();
            return TcBase<T>::count();
        }


    };


}