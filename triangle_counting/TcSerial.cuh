#pragma once
#include "TcBase.cuh"
#include "../truss/kernels.cuh"




template <typename T, size_t BLOCK_DIM_X>
__global__ void __launch_bounds__(BLOCK_DIM_X)
kernel_serial_arrays(uint64* count, //!< [inout] the count, caller should zero
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
kernel_serial_warp_arrays(uint64* count, //!< [inout] the count, caller should zero
    T* rowPtr, T* rowInd, T* colInd, const size_t numEdges, const size_t edgeStart, int increasing = 0) {

   
    constexpr size_t warpsPerBlock = BLOCK_DIM_X / 32;
    const size_t lx = threadIdx.x % 32;
    const size_t wx = threadIdx.x / 32;
    const size_t gwx = (BLOCK_DIM_X * blockIdx.x + threadIdx.x) / 32;

    uint64 threadCount = 0;

    __shared__ T srcSpace[warpsPerBlock][32];
    __shared__ T dstSpace[warpsPerBlock][32];

    for (size_t i = gwx + edgeStart; i < numEdges; i += BLOCK_DIM_X * gridDim.x/32) 
    {
        const T src = rowInd[i];
        const T dst = colInd[i];
        T ap = rowPtr[src];
        T bp = rowPtr[dst];
        const T aEnd = rowPtr[src + 1];
        const T bEnd = rowPtr[dst + 1];

        bool loadA = true;
        bool loadB = true;

        bool loadAShared = true;
        bool loadBShared = true;

        T a, b, numA, numB;
        while (ap < aEnd && bp < bEnd) 
        {
            numA = (aEnd - ap) > 32 ? 32 : (aEnd - ap);
            numB = (bEnd - bp) > 32 ? 32 : (bEnd - bp);

            if (loadAShared)
            {
                //All threads in the warp will load
                srcSpace[wx][lx] = (lx < numA) ? colInd[ap + lx] : 0;
                loadAShared = false;
            }
            if (loadBShared)
            {
                //All threads in the warp will load
                dstSpace[wx][lx] = (lx < numB) ? colInd[bp + lx] : 0;
                loadBShared = false;
            }

            //Ufortunately process by single thread
            if (lx == 0)
            {
                T sap = 0;
                T sbp = 0;

                while (sap < numA && sbp < numB)
                {
                    a = srcSpace[wx][sap];
                    b = dstSpace[wx][sbp];
                      
                    if (a == b) {
                        ++threadCount;
                        ++sap;
                        ++sbp;
                    }
                    else if (a < b) {
                        ++sap;
                    }
                    else {
                        ++sbp;
                    }
                }
                if (sap == numA)
                {
                    ap += numA;
                    loadAShared = true;
                }
                if (sbp == numB)
                {
                    bp += numB;
                    loadBShared = true;
                }
            }    
           unsigned int writemask_deq = __activemask();
           
            ap = __shfl_sync(writemask_deq, ap, 0);
            loadBShared = __shfl_sync(writemask_deq, loadBShared, 0);
            loadAShared = __shfl_sync(writemask_deq, loadAShared, 0);
            bp = __shfl_sync(writemask_deq, bp, 0);
        }
    }

    // Add to total count
    if (0 == lx) {
        atomicAdd(count, threadCount);
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


        count[i] = threadCount;
    }

}



template <typename T, size_t BLOCK_DIM_X>
__global__ void __launch_bounds__(BLOCK_DIM_X)
kernel_serial_pe_eid_arrays(int* count, //!< [inout] the count, caller should zero
    T* rowPtr_csr, T* colIndex_csr,
    T* rowInd, T* colInd, const size_t numEdges, const size_t edgeStart, int increasing = 0) {

    size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
    uint64 threadCount = 0;

    for (size_t i = gx + edgeStart; i < numEdges; i += BLOCK_DIM_X * gridDim.x) {
        const T src = rowInd[i];
        const T dst = colInd[i];

        const T* srcBegin = &colIndex_csr[rowPtr_csr[src]];
        const T* srcEnd = &colIndex_csr[rowPtr_csr[src + 1]];
        const T* dstBegin = &colIndex_csr[rowPtr_csr[dst]];
        const T* dstEnd = &colIndex_csr[rowPtr_csr[dst + 1]];

        T min = increasing == 0 ? 0 : dst;
        threadCount += graph::serial_sorted_count_linear(min, srcBegin, srcEnd, dstBegin, dstEnd);
    
    
        count[i] = threadCount;
    }

    
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


        count[i] = threadCount;
    }

  
}


template <typename T, size_t BLOCK_DIM_X>
__global__ void __launch_bounds__(BLOCK_DIM_X)
kernel_serial_pe_level_q_arrays(
    int* count,
    T* rowPtr_csr, T* colIndex_csr,
    T* rowInd, T* colInd, T* eid,
    const size_t numEdges,
    int level, bool* processed,
    int* affected, int& affected_cnt
) 
{

    size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
    

    for (size_t i = gx; i < numEdges; i += BLOCK_DIM_X * gridDim.x)
    {
        int edgeId = i; // affected[i];

        uint64 threadCount = 0;
        const T src = rowInd[edgeId];
        const T dst = colInd[edgeId];

        T min = 0; // increasing == 0 ? 0 : dst;
        T ap = rowPtr_csr[src];
        T bp = rowPtr_csr[dst];

        T aEnd = rowPtr_csr[src + 1];
        T bEnd = rowPtr_csr[dst + 1];

        bool loadA = true;
        bool loadB = true;

        T a, b;
        while (ap < aEnd && bp < bEnd)
        {

            if (loadA) {
                a = colIndex_csr[ap];
                loadA = false;
            }
            if (loadB) {
                b = colIndex_csr[bp];
                loadB = false;
            }

            if (a == b) {

                T ap_e = eid[ap];
                T bp_e = eid[bp];

                if (!processed[ap_e] && !processed[bp_e])
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
        count[edgeId] = threadCount;
           
    }
}



template <typename T, size_t BLOCK_DIM_X>
__global__ void __launch_bounds__(BLOCK_DIM_X)
kernel_serial_pe_level_affected(
    T* rowPtr_csr, T* colIndex_csr,
    T* rowInd, T* colInd, T* eid,
    const size_t numEdges,
    int level, bool* processed,
    int *curr, bool* inCurr, int curr_cnt,
    int* affected, int* inAffected, int& affected_cnt, T *reversed
)
{

    size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
    for (size_t i = gx; i < curr_cnt; i += BLOCK_DIM_X * gridDim.x)
    {
        int edgeId = curr[i];

        const T src = rowInd[edgeId];
        const T dst = colInd[edgeId];
        T ap = rowPtr_csr[src];
        T bp = rowPtr_csr[dst];

        T aEnd = rowPtr_csr[src + 1];
        T bEnd = rowPtr_csr[dst + 1];

        bool loadA = true;
        bool loadB = true;

        T a, b;
        while (ap < aEnd && bp < bEnd) {

            if (loadA) {
                a = colIndex_csr[ap];
                loadA = false;
            }
            if (loadB) {
                b = colIndex_csr[bp];
                loadB = false;
            }

            if (a == b) 
            {

                T ap_e = eid[ap];
                T bp_e = eid[bp];

                bool is_peel_e2 = !inCurr[ap_e];
                bool is_peel_e3 = !inCurr[bp_e];
                if (is_peel_e2 || is_peel_e3) 
                {
                    if ((!processed[ap_e]) && (!processed[bp_e]))
                    {
                        if (is_peel_e2 && is_peel_e3) {
                            auto old_token = atomicAdd(inAffected + ap_e, 1);
                            if (old_token == 0)
                            {
                                auto insert_idx = atomicAdd(&affected_cnt, 1);
                                affected[insert_idx] = ap_e;
                            }

                            old_token = atomicAdd(inAffected + bp_e, 1);
                            if (old_token == 0)
                            {
                                auto insert_idx = atomicAdd(&affected_cnt, 1);
                                affected[insert_idx] = bp_e;
                            }
                        }
                        else if (is_peel_e2)
                        {
                            if (edgeId < bp_e) {
                                auto old_token = atomicAdd(inAffected + ap_e, 1);
                                if (old_token == 0)
                                {
                                    auto insert_idx = atomicAdd(&affected_cnt, 1);
                                    affected[insert_idx] = ap_e;
                                }
                            }
                        }
                        else
                        {
                            if (edgeId < ap_e)
                            {
                                auto old_token = atomicAdd(inAffected + bp_e, 1);
                                if (old_token == 0)
                                {
                                    auto insert_idx = atomicAdd(&affected_cnt, 1);
                                    affected[insert_idx] = bp_e;
                                }
                            }
                        }
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
            Log(LogPriorityEnum::debug, "device = %d, blocks = %d, threads = %d", TcBase<T>::dev_, dimGrid, dimBlock);
            CUDA_RUNTIME(cudaSetDevice(TcBase<T>::dev_));


            CUDA_RUNTIME(cudaEventRecord(TcBase<T>::kernelStart_, TcBase<T>::stream_));
            if(kernelType == Thread)
                kernel_serial_arrays<T, dimBlock> << <dimGrid, dimBlock, 0, TcBase<T>::stream_ >> > (TcBase<T>::count_, rp, ri, ci, ne, edgeOffset);
            else if(kernelType == Warp)
                kernel_serial_warp_arrays<T, dimBlock> << <dimGridWarp, dimBlock, 0, TcBase<T>::stream_ >> > (TcBase<T>::count_, rp, ri, ci, ne, edgeOffset);
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


        void count_per_edge_eid_async(GPUArray<int>& tcpt, GPUArray<T> rowPtr_csr, GPUArray<T> colIndex_csr,  GPUArray<T> rowInd, GPUArray<T> colInd, const size_t numEdges, const size_t edgeOffset = 0, ProcessingElementEnum kernelType = Thread, int increasing = 0)
        {
            const size_t dimBlock = 512;
            const size_t ne = numEdges;
            T* rp_csr = rowPtr_csr.gdata();
            T* ci_csr = colIndex_csr.gdata();

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
            kernel_serial_pe_eid_arrays<T, dimBlock> << <dimGrid, dimBlock, 0, TcBase<T>::stream_ >> > (tcpt.gdata(), rp_csr, ci_csr, ri, ci, ne, edgeOffset, increasing);
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
            GPUArray<int>& tcpt,
            GPUArray<T> rowPtr_csr, GPUArray<T> colIndex_csr,
            GPUArray<T> rowInd, GPUArray<T> colInd, GPUArray<T> eid, const size_t numEdges,
            int level, GPUArray<bool> processed, GPUArray<int>& edgeSupport,
            GPUArray<int>& curr, int curr_cnt,
            GPUArray<int>& affected, GPUArray<int>& inAffected, GPUArray<int>& affected_cnt, //next queue
            GPUArray<int>& next, GPUArray<int>& inNext, GPUArray<int>& next_cnt, //next queue
            GPUArray <bool>& in_bucket_window_, GPUArray<uint>& bucket_buf_, GPUArray<uint>& window_bucket_buf_size_, int bucket_level_end_,
            const size_t edgeOffset = 0, ProcessingElementEnum kernelType = Thread, int increasing = 0)
        {
        
            const size_t dimBlock = 32;
            const size_t ne = numEdges;
            T* rp_csr = rowPtr_csr.gdata();
            T* ci_csr = colIndex_csr.gdata();

            T* ri = rowInd.gdata();
            T* ci = colInd.gdata();
            const int dimGrid = (numEdges - edgeOffset + (dimBlock)-1) / (dimBlock);
            CUDA_RUNTIME(cudaSetDevice(TcBase<T>::dev_));


            //process affetced
           kernel_serial_pe_level_q_arrays<T, dimBlock> << <dimGrid, dimBlock, 0, TcBase<T>::stream_ >> > (
                tcpt.gdata(), 
                rp_csr,ci_csr,  ri, ci, eid.gdata(), ne,
                level, processed.gdata(),
                affected.gdata(), *affected_cnt.gdata());

        }


        void affect_per_edge_level_q_async(
            GPUArray<T> rowPtr_csr, GPUArray<T> colIndex_csr,
            GPUArray<T> rowInd, GPUArray<T> colInd, GPUArray<T> eid, const size_t numEdges,
            int level, GPUArray<bool> processed,
            GPUArray<int>& curr, GPUArray<bool> inCurr, int curr_cnt,
            GPUArray<int>& affected, GPUArray<int>& inAffected, GPUArray<int>& affected_cnt, //next queue
            GPUArray<uint> reversed,
            const size_t edgeOffset = 0, ProcessingElementEnum kernelType = Thread, int increasing = 0)
        {
        
            const size_t dimBlock = 32;
            const size_t ne = numEdges;
            T* rp_csr = rowPtr_csr.gdata();
            T* ci_csr = colIndex_csr.gdata();

            T* ri = rowInd.gdata();
            T* ci = colInd.gdata();

            const int dimGrid = (curr_cnt - edgeOffset + (dimBlock)-1) / (dimBlock);
            CUDA_RUNTIME(cudaSetDevice(TcBase<T>::dev_));
            
            kernel_serial_pe_level_affected<T, dimBlock> << <dimGrid, dimBlock, 0, TcBase<T>::stream_ >> > (
                rp_csr, ci_csr, ri, ci, eid.gdata(), ne,
                level, processed.gdata(),
                curr.gdata(), inCurr.gdata(), curr_cnt,
                affected.gdata(), inAffected.gdata(), *affected_cnt.gdata(), reversed.gdata());
        
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