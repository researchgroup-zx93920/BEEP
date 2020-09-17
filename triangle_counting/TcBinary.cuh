#pragma once
#include "TcBase.cuh"

#include <cooperative_groups.h>
using namespace cooperative_groups;
namespace cg = cooperative_groups;

template <typename T, size_t BLOCK_DIM_X>
__global__ void __launch_bounds__(BLOCK_DIM_X)
kernel_binary_thread_arrays(uint64* count, //!< [inout] the count, caller should zero
    graph::COOCSRGraph_d<T> g, const size_t numEdges, const size_t edgeStart) {

    size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
    uint64 threadCount = 0;

    for (size_t i = gx + edgeStart; i < numEdges; i += BLOCK_DIM_X * gridDim.x) {
        const T src = g.rowInd[i];
        const T dst = g.colInd[i];

        const T srcStart = g.rowPtr[src];
        const T srcStop = g.rowPtr[src + 1];

        const T dstStart = g.rowPtr[dst];
        const T dstStop = g.rowPtr[dst + 1];

        const T dstLen = dstStop - dstStart;
        const T srcLen = srcStop - srcStart;

        //printf("%u,%u,%u,%u,%u,%u,%u,%u\n", src, dst, srcStart, srcStop, dstStart, dstStop, srcLen, dstLen);

        if (dstLen > srcLen) {
            threadCount += graph::thread_sorted_count_binary<T>(&(g.colInd[srcStart]), srcLen,
                &(g.colInd[dstStart]), dstLen);
        }
        else {
            threadCount += graph::thread_sorted_count_binary<T>(&(g.colInd[dstStart]), dstLen,
                &(g.colInd[srcStart]), srcLen);
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
kernel_binary_enqueue_arrays(uint64* count, //!< [inout] the count, caller should zero
    graph::COOCSRGraph_d<T> g, const size_t numEdges, const size_t edgeStart,
    T* wq, T* wc, T* bq, T* bc
    
    ) {

    size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
    const size_t gbx = (BLOCK_DIM_X * blockIdx.x + threadIdx.x) / BLOCK_DIM_X;

    int tid = threadIdx.x;
    int lx = threadIdx.x % 32;
    uint64 threadCount = 0;

    __shared__ T sharedWarpQ[2*BLOCK_DIM_X];
    __shared__ T sharedBlockQ[2*BLOCK_DIM_X];

    __shared__ int swc, sbc;
    __shared__ int startWarpQueue, startBlockQueue, global;


    if (gx == 0)
        *wc = 0;
    if (gx == 1)
        *bc = 0;

    if (tid == 0)
        swc = 0;
    if (tid == 1)
        sbc = 0;
    __syncthreads();

    for (size_t b = gbx; b < (numEdges + BLOCK_DIM_X-1)/ BLOCK_DIM_X; b += gridDim.x)
    {
        int i = b * BLOCK_DIM_X + tid;

        if (i < numEdges)
        {
            T src = g.rowInd[i];
            T dst = g.colInd[i];

            T srcStart = g.rowPtr[src];
            T srcStop = g.rowPtr[src + 1];

            T dstStart = g.rowPtr[dst];
            T dstStop = g.rowPtr[dst + 1];

            T dstLen = dstStop - dstStart;
            T srcLen = srcStop - srcStart;

            if (srcLen > dstLen)
            {
                swap_ele(srcStart, dstStart);
                swap_ele(srcStop, dstStop);
                swap_ele(srcLen, dstLen);
            }

            if (srcLen < 4)
            {
                threadCount += graph::thread_sorted_count_binary<T>(&(g.colInd[srcStart]), srcLen,
                    &(g.colInd[dstStart]), dstLen);
            }
            else if (srcLen >= 4 && srcLen < 128)
            {
                auto prev = atomicAdd(&swc, 1);
                sharedWarpQ[prev] = i;
            }
            else if (srcLen >= 128)
            {
                auto prev = atomicAdd(&sbc, 1);
                sharedBlockQ[prev] = i;
            }
        }

        __syncthreads();
        if (tid == 0)
        {
            if (swc > 0)
            {
                startWarpQueue = atomicAdd(wc, swc);
            }
        }
        if (tid == 1)
        {
            if (sbc > 0)
                startBlockQueue = atomicAdd(bc, sbc);
        }
        __syncthreads();
        for (int q = tid; q < swc; q += BLOCK_DIM_X)
            wq[startWarpQueue + q] = sharedWarpQ[q];

        for (int q = tid; q < sbc; q += BLOCK_DIM_X)
            bq[startBlockQueue + q] = sharedBlockQ[q];

        __syncthreads();
        if (tid == 0)
        {
            swc = 0;
        }
        if (tid == 1)
        {
            sbc = 0;
        }

    }

    typedef cub::BlockReduce<uint64_t, BLOCK_DIM_X> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    uint64_t aggregate = BlockReduce(tempStorage).Sum(threadCount);
    if (threadIdx.x == 0)
        atomicAdd(count, aggregate);
}




template <typename T, size_t BLOCK_DIM_X>
__global__ void __launch_bounds__(BLOCK_DIM_X)
kernel_binary_dequeue_arrays(uint64* count, //!< [inout] the count, caller should zero
    graph::COOCSRGraph_d<T> g, const size_t numEdges, const size_t edgeStart,
    T* wq, T* wc, T* bq, T* bc

) {

    size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    int lx = threadIdx.x % 32;
    uint64 threadCount = 0;
    //Now :1 Warp
    constexpr size_t warpsPerBlock = BLOCK_DIM_X / 32;
    gx = gx / 32;
    for (size_t i = gx; i < *wc; i += BLOCK_DIM_X * gridDim.x / 32) {

        T edgeId = wq[i];

        const T src = g.rowInd[edgeId];
        const T dst = g.colInd[edgeId];

        const T srcStart = g.rowPtr[src];
        const T srcStop = g.rowPtr[src + 1];

        const T dstStart = g.rowPtr[dst];
        const T dstStop = g.rowPtr[dst + 1];

        const T dstLen = dstStop - dstStart;
        const T srcLen = srcStop - srcStart;

        // FIXME: remove warp reduction from this function call
        if (dstLen > srcLen) {
            threadCount += graph::warp_sorted_count_binary<warpsPerBlock, T, false>(&(g.colInd[srcStart]), srcLen,
                &(g.colInd[dstStart]), dstLen);
        }
        else {
            threadCount += graph::warp_sorted_count_binary<warpsPerBlock, T, false>(&(g.colInd[dstStart]), dstLen,
                &(g.colInd[srcStart]), srcLen);
        }
    }


    //Now: 2 Block
    uint64 blockCount = 0;
    for (size_t i = blockIdx.x; i < *bc; i += gridDim.x)
    {
        T edgeId = bq[i];

        const T src = g.rowInd[edgeId];
        const T dst = g.colInd[edgeId];

        const T srcStart = g.rowPtr[src];
        const T srcStop = g.rowPtr[src + 1];

        const T dstStart = g.rowPtr[dst];
        const T dstStop = g.rowPtr[dst + 1];

        const T dstLen = dstStop - dstStart;
        const T srcLen = srcStop - srcStart;

        // FIXME: remove warp reduction from this function call
        if (dstLen > srcLen) {
            threadCount += graph::block_sorted_count_binary<BLOCK_DIM_X, T, false>(&(g.colInd[srcStart]), srcLen,
                &(g.colInd[dstStart]), dstLen);
        }
        else {
            threadCount += graph::block_sorted_count_binary<BLOCK_DIM_X, T, false>(&(g.colInd[dstStart]), dstLen,
                &(g.colInd[srcStart]), srcLen);
        }
    }



   typedef cub::BlockReduce<uint64_t, BLOCK_DIM_X> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    uint64_t aggregate = BlockReduce(tempStorage).Sum(threadCount);
    if (threadIdx.x == 0)
        atomicAdd(count, aggregate);
}


template <typename T, size_t BLOCK_DIM_X>
__global__ void __launch_bounds__(BLOCK_DIM_X)
kernel_binary_thread_pe_arrays(int* count, //!< [inout] the count, caller should zero
    T* rowPtr, T* rowInd, T* colInd, const size_t numEdges, const size_t edgeStart) {

    size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
    uint64 threadCount = 0;

    for (size_t i = gx + edgeStart; i < numEdges; i += BLOCK_DIM_X * gridDim.x) {
        T src = rowInd[i];
        T dst = colInd[i];

        T srcStart = rowPtr[src];
        T srcStop = rowPtr[src + 1];

        T dstStart = rowPtr[dst];
        T dstStop = rowPtr[dst + 1];

        T dstLen = dstStop - dstStart;
        T srcLen = srcStop - srcStart;

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
kernel_binary_thread_pe_eid_arrays(int* count, //!< [inout] the count, caller should zero
    T* rowPtr_csr, T* colIndex_csr,
    T* rowInd, T* colInd, const size_t numEdges, const size_t edgeStart) {

    size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
    uint64 threadCount = 0;

    for (size_t i = gx + edgeStart; i < numEdges; i += BLOCK_DIM_X * gridDim.x) {
        T src = rowInd[i];
        T dst = colInd[i];

        T srcStart = rowPtr_csr[src];
        T srcStop = rowPtr_csr[src + 1];

        T dstStart = rowPtr_csr[dst];
        T dstStop = rowPtr_csr[dst + 1];

        T dstLen = dstStop - dstStart;
        T srcLen = srcStop - srcStart;

        //printf("%u,%u,%u,%u,%u,%u,%u,%u\n", src, dst, srcStart, srcStop, dstStart, dstStop, srcLen, dstLen);


        if (srcLen > dstLen)
        {
            swap_ele(srcStart, dstStart);
            swap_ele(srcStop, dstStop);
            swap_ele(srcLen, dstLen);
        }


        threadCount += graph::thread_sorted_count_binary<T>(&(colIndex_csr[srcStart]), srcLen,
            &(colIndex_csr[dstStart]), dstLen);

        count[i + edgeStart] = threadCount;

    }


}

__device__ void add_to_queue_1(graph::GraphQueue_d<int, bool>& q, int element)
{
    auto insert_idx = atomicAdd(q.count, 1);
    q.queue[insert_idx] = element;
    q.mark[element] = true;
}

__device__ void add_to_queue_1_no_dup(graph::GraphQueue_d<int, bool>& q, int element)
{
    auto old_token = atomicCASBool(q.mark + element, InBucketFalse, InBucketTrue);
    if (!old_token) {
        auto insert_idx = atomicAdd(q.count, 1);
        q.queue[insert_idx] = element;
    }
}


__inline__ __device__
void process_support2(
    uint32_t edge_idx, int level, int* EdgeSupport,
    graph::GraphQueue_d<int, bool>& next,
    graph::GraphQueue_d<int, bool>& bucket,
    int bucket_level_end_)
{
    auto cur = atomicSub(&EdgeSupport[edge_idx], 1);
    if (cur == (level + 1)) {
        add_to_queue_1(next, edge_idx);
    }
    if (cur <= level) {
        atomicAdd(&EdgeSupport[edge_idx], 1);
    }

    // Update the Bucket.
    auto latest = cur - 1;
    if (latest > level && latest < bucket_level_end_) {
        add_to_queue_1_no_dup(bucket, edge_idx);
    }

}


template <typename T>
__device__ inline void addNexBucket(T e1, T e2, T e3, bool* processed,
    int* edgeSupport, int level,
    bool* inCurr,
    graph::GraphQueue_d<int, bool>& next,
    graph::GraphQueue_d<int, bool>& bucket, 
    int bucket_upper_level)
{
    bool is_peel_e2 = !inCurr[e2];
    bool is_peel_e3 = !inCurr[e3];
    if (is_peel_e2 || is_peel_e3)
    {
        if ((!processed[e2]) && (!processed[e3]))
        {
            if (is_peel_e2 && is_peel_e3)
            {

                process_support2(e2, level, edgeSupport, next, bucket, bucket_upper_level);
                process_support2(e3, level, edgeSupport, next, bucket, bucket_upper_level);



            }
            else if (is_peel_e2)
            {
                if (e1 < e3) {
                    process_support2(e2, level, edgeSupport, next, bucket, bucket_upper_level);
                }
            }
            else
            {
                if (e1 < e2)
                {
                    process_support2(e3, level, edgeSupport, next, bucket, bucket_upper_level);
                }
            }
        }
    }
}

template <typename T, size_t BLOCK_DIM_X>
__global__ void __launch_bounds__(BLOCK_DIM_X)
kernel_binary_thread_pe_level_next(
    T* rowPtr_csr, T* colIndex_csr,
    T* rowInd, T* colInd, T* eid,
    const size_t numEdges,
    int level, bool* processed, int* edgeSupport,
    graph::GraphQueue_d<int, bool> current,
    graph::GraphQueue_d<int, bool>& next,
    graph::GraphQueue_d<int, bool>& bucket, 
    int bucket_level_end_
)
{
    size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
    for (size_t i = gx; i < current.count[0]; i += BLOCK_DIM_X * gridDim.x)
    {
        T edgeId = current.queue[i];

        T src = rowInd[edgeId];
        T dst = colInd[edgeId];

        T srcStart = rowPtr_csr[src];
        T srcStop = rowPtr_csr[src + 1];

        T dstStart = rowPtr_csr[dst];
        T dstStop = rowPtr_csr[dst + 1];

        T dstLen = dstStop - dstStart;
        T srcLen = srcStop - srcStart;


        if (srcLen > dstLen)
        {
            swap_ele(srcStart, dstStart);
            swap_ele(srcStop, dstStop);
            swap_ele(srcLen, dstLen);
        }

        T lb = 0;

        T count = 0;
        // cover entirety of A with warp
        for (size_t j = 0; j < srcLen; j++) {
            // one element of A per thread, just search for A into B
            const T searchVal = colIndex_csr[srcStart + j];
            lb = graph::binary_search<T>(&colIndex_csr[dstStart], lb, dstLen, searchVal);
            if (lb < dstLen)
            {
                if (colIndex_csr[dstStart + lb] == searchVal)
                {
                    T ap_e = eid[srcStart + j];
                    T bp_e = eid[dstStart + lb];
                    addNexBucket(edgeId, ap_e, bp_e, processed, edgeSupport, level, current.mark, next, bucket, bucket_level_end_);
                }
            }
            else
            {
                break;
            }

        }

    }
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
    graph::COOCSRGraph_d<T> g, const size_t numEdges, //!< the number of edges this kernel will count
    const size_t edgeStart                       //!< the edge this kernel will start counting at
) {
    constexpr size_t warpsPerBlock = BLOCK_DIM_X / 32;

    const size_t lx = threadIdx.x % 32;
    const size_t gwx = (BLOCK_DIM_X * blockIdx.x + threadIdx.x) / 32;
    uint64 warpCount = 0;

    for (size_t i = gwx + edgeStart; i < numEdges; i += BLOCK_DIM_X * gridDim.x / 32) {
        const T src = g.rowInd[i];
        const T dst = g.colInd[i];

        const T srcStart = g.rowPtr[src];
        const T srcStop = g.rowPtr[src + 1];

        const T dstStart = g.rowPtr[dst];
        const T dstStop = g.rowPtr[dst + 1];

        const T dstLen = dstStop - dstStart;
        const T srcLen = srcStop - srcStart;



        // FIXME: remove warp reduction from this function call
        if (dstLen > srcLen) {
            warpCount += graph::warp_sorted_count_binary<warpsPerBlock>(&g.colInd[srcStart], srcLen,
                &g.colInd[dstStart], dstLen);
        }
        else {
            warpCount += graph::warp_sorted_count_binary<warpsPerBlock>(&g.colInd[dstStart], dstLen,
                &g.colInd[srcStart], srcLen);
        }
    }

    if (lx == 0)
        atomicAdd(count, warpCount);


}


template <typename T, size_t BLOCK_DIM_X>
__global__ void __launch_bounds__(BLOCK_DIM_X)
kernel_binary_warp_pe_arrays(int* count,                //!< [inout] the count, caller should zero
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
kernel_binary_warp_pe_eid_arrays(int* count,                //!< [inout] the count, caller should zero
    T* rowPtr_csr, T* colIndex_csr,
    T* rowInd, T* colInd, const size_t numEdges, //!< the number of edges this kernel will count
    const size_t edgeStart                       //!< the edge this kernel will start counting at
) {
    constexpr size_t warpsPerBlock = BLOCK_DIM_X / 32;

    const size_t lx = threadIdx.x % 32;
    const size_t gwx = (BLOCK_DIM_X * blockIdx.x + threadIdx.x) / 32;
    uint64 warpCount = 0;

    for (size_t i = gwx + edgeStart; i < numEdges; i += BLOCK_DIM_X * gridDim.x / 32)
    {
        T src = rowInd[i];
        T dst = colInd[i];

        T srcStart = rowPtr_csr[src];
        T srcStop = rowPtr_csr[src + 1];

        T dstStart = rowPtr_csr[dst];
        T dstStop = rowPtr_csr[dst + 1];

        T dstLen = dstStop - dstStart;
        T srcLen = srcStop - srcStart;


        // if (srcLen > dstLen)
        // {
        //     swap_ele(srcStart, dstStart);
        //     swap_ele(srcStop, dstStop);
        //     swap_ele(srcLen, dstLen);
        // }

        // FIXME: remove warp reduction from this function call
        warpCount += graph::warp_sorted_count_binary<warpsPerBlock>(&colIndex_csr[dstStart], dstLen,
            &colIndex_csr[srcStart], srcLen);

        if (lx == 0)
            count[i + edgeStart] = warpCount;
    }

}




template <typename T, size_t BLOCK_DIM_X>
__global__ void __launch_bounds__(BLOCK_DIM_X)
kernel_binary_warp_pe_level_next(
    T* rowPtr_csr, T* colIndex_csr,
    T* rowInd, T* colInd, T* eid,
    const size_t numEdges,
    int level, bool* processed, int* edgeSupport,
    graph::GraphQueue_d<int, bool> current,
    graph::GraphQueue_d<int, bool> next,
    graph::GraphQueue_d<int, bool> bucket, 
    int bucket_level_end_
)
{

    constexpr size_t warpsPerBlock = BLOCK_DIM_X / 32;

    const size_t lx = threadIdx.x % 32;
    const int warpIdx = threadIdx.x / 32; // which warp in thread block
    const size_t gwx = (BLOCK_DIM_X * blockIdx.x + threadIdx.x) / 32;


    __shared__ T q[warpsPerBlock][3 * 2 * 32];
    __shared__ int sizes[warpsPerBlock];
    T* e1_arr = &q[warpIdx][0];
    T* e2_arr = &q[warpIdx][1 * 32 * 2];
    T* e3_arr = &q[warpIdx][2 * 32 * 2];
    int* size = &sizes[warpIdx];

    if (lx == 0)
        *size = 0;

    for (size_t i = gwx; i < *current.count; i += BLOCK_DIM_X * gridDim.x / 32)
    {
        T edgeId = current.queue[i];

        T src = rowInd[edgeId];
        T dst = colInd[edgeId];

        T srcStart = rowPtr_csr[src];
        T srcStop = rowPtr_csr[src + 1];

        T dstStart = rowPtr_csr[dst];
        T dstStop = rowPtr_csr[dst + 1];

        T dstLen = dstStop - dstStart;
        T srcLen = srcStop - srcStart;


        if (srcLen > dstLen)
        {
            swap_ele(srcStart, dstStart);
            swap_ele(srcStop, dstStop);
            swap_ele(srcLen, dstLen);
        }

        // FIXME: remove warp reduction from this function call

        T lastIndex = 0;

        // cover entirety of A with warp
        for (size_t j = lx; j < (srcLen + 31) / 32 * 32; j += 32)
        {
            __syncwarp();
            if (*size >= 32)
            {
                for (int e = lx; e < *size; e += 32)
                {
                    T e1 = e1_arr[e];
                    T e2 = e2_arr[e];
                    T e3 = e3_arr[e];
                    addNexBucket(e1, e2, e3, processed, edgeSupport, level, current.mark, next, bucket, bucket_level_end_);


                }

                __syncwarp();
                if (lx == 0)
                    *size = 0;
                __syncwarp();
            }



            if (j < srcLen)
            {
                // one element of A per thread, just search for A into B
                const T searchVal = colIndex_csr[srcStart + j];
                const T leftValue = colIndex_csr[dstStart + lastIndex];

                if (searchVal >= leftValue)
                {

                    const T lb = graph::binary_search<T>(&colIndex_csr[dstStart], lastIndex, dstLen, searchVal);
                    if (lb < dstLen)
                    {
                        if (colIndex_csr[dstStart + lb] == searchVal)
                        {
                            T ap_e = eid[srcStart + j];
                            T bp_e = eid[dstStart + lb];
                            auto pos = atomicAdd(size, 1);
                            e1_arr[pos] = edgeId;
                            e2_arr[pos] = ap_e;
                            e3_arr[pos] = bp_e;
                        }
                    }

                    lastIndex = lb;
                }
            }

            unsigned int writemask_deq = __activemask();
            lastIndex = __shfl_sync(writemask_deq, lastIndex, 31);
        }

        __syncwarp();
        for (int e = lx; e < *size; e += 32)
        {
            T e1 = e1_arr[e];
            T e2 = e2_arr[e];
            T e3 = e3_arr[e];
            addNexBucket(e1, e2, e3, processed, edgeSupport, level, current.mark, next, bucket, bucket_level_end_);
        }
    }


}

template <typename T, size_t BLOCK_DIM_X>
__global__ void __launch_bounds__(BLOCK_DIM_X)
kernel_binary_block_pe_level_next(
    T* rowPtr_csr, T* colIndex_csr,
    T* rowInd, T* colInd, T* eid,
    const size_t numEdges,
    int level, bool* processed, int* edgeSupport,
    graph::GraphQueue_d<int, bool> current,
    graph::GraphQueue_d<int, bool> next,
    graph::GraphQueue_d<int, bool> bucket, int bucket_level_end_
)
{

    auto tid = threadIdx.x;
    const size_t gbx = (BLOCK_DIM_X * blockIdx.x + threadIdx.x) / BLOCK_DIM_X;


    __shared__ T first[BLOCK_DIM_X];
    __shared__ T q[3 * 2 * BLOCK_DIM_X];
    __shared__ int size;
    T* e1_arr = &q[0];
    T* e2_arr = &q[1 * BLOCK_DIM_X * 2];
    T* e3_arr = &q[2 * BLOCK_DIM_X * 2];

    if (tid == 0)
        size = 0;

    __syncthreads();

    for (size_t i = gbx; i < current.count[0]; i += gridDim.x)
    {
        T edgeId = current.queue[i];

        T src = rowInd[edgeId];
        T dst = colInd[edgeId];

        T srcStart = rowPtr_csr[src];
        T srcStop = rowPtr_csr[src + 1];

        T dstStart = rowPtr_csr[dst];
        T dstStop = rowPtr_csr[dst + 1];

        T dstLen = dstStop - dstStart;
        T srcLen = srcStop - srcStart;


        // if (srcLen > dstLen)
        // {
        //     swap_ele(srcStart, dstStart);
        //     swap_ele(srcStop, dstStop);
        //     swap_ele(srcLen, dstLen);
        // }

        int startIndex = 0;
        const T par = (srcLen + BLOCK_DIM_X - 1) / (BLOCK_DIM_X);
        const T numElements = srcLen < BLOCK_DIM_X ? srcLen : (srcLen + par - 1) / par;

        for (int i = 0; i < 1; i++)
        {
            int sharedIndex = startIndex + BLOCK_DIM_X * i + tid;
            int realIndex = srcStart + (tid + i * BLOCK_DIM_X) * par;

            first[sharedIndex] = (tid + BLOCK_DIM_X * i) <= numElements ? colIndex_csr[realIndex] : 0;
        }

        T lastIndex = 0;
        T fl = 0;
        // cover entirety of A with warp
        for (size_t j = tid; j < (dstLen + BLOCK_DIM_X - 1) / BLOCK_DIM_X * BLOCK_DIM_X; j += BLOCK_DIM_X)
        {
            __syncthreads();
            if (size >= BLOCK_DIM_X)
            {
                for (int e = tid; e < size; e += BLOCK_DIM_X)
                {
                    T e1 = e1_arr[e];
                    T e2 = e2_arr[e];
                    T e3 = e3_arr[e];
                    addNexBucket(e1, e2, e3, processed, edgeSupport, level, current.mark, next, bucket, bucket_level_end_);
                }

                __syncthreads();
                if (tid == 0)
                {
                    size = 0;
                }
                __syncthreads();
            }

            if (j < dstLen)
            {
                // one element of A per thread, just search for A into B
                const T searchVal = colIndex_csr[dstStart + j];
                fl = graph::binary_search_left<T>(first, fl, numElements, searchVal);
                lastIndex = par * fl;
                T right = 0;

                if (srcLen < BLOCK_DIM_X)
                {
                    if (searchVal == first[fl])
                    {
                        T ap_e = eid[dstStart + j];
                        T bp_e = eid[srcStart + lastIndex];
                        auto pos = atomicAdd(&size, 1);
                        e1_arr[pos] = edgeId;
                        e2_arr[pos] = ap_e;
                        e3_arr[pos] = bp_e;
                    }
                    continue;
                }
                else if (fl == numElements - 1)
                    right = srcLen;
                else
                    right = (fl + 1) * par;

                const T lb = graph::binary_search_left<T>(&colIndex_csr[srcStart], lastIndex, right, searchVal);
                if (lb < srcLen)
                {
                    if (colIndex_csr[srcStart + lb] == searchVal)
                    {
                        T ap_e = eid[dstStart + j];
                        T bp_e = eid[srcStart + lb];
                        auto pos = atomicAdd(&size, 1);
                        e1_arr[pos] = edgeId;
                        e2_arr[pos] = ap_e;
                        e3_arr[pos] = bp_e;
                    }
                }
            }
        }

        __syncthreads();
        for (int e = tid; e < size; e += BLOCK_DIM_X)
        {
            T e1 = e1_arr[e];
            T e2 = e2_arr[e];
            T e3 = e3_arr[e];
            addNexBucket(e1, e2, e3, processed, edgeSupport, level, current.mark, next, bucket, bucket_level_end_);
        }
    }


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
    graph::COOCSRGraph_d<T> g, const size_t numEdges, //!< the number of edges this kernel will count
    const size_t edgeStart                       //!< the edge this kernel will start counting at
) {
    constexpr size_t warpsPerBlock = BLOCK_DIM_X / 32;

    const size_t lx = threadIdx.x;
    const size_t gwx = blockIdx.x;
    uint64 blockCount = 0;

    for (size_t i = gwx + edgeStart; i < numEdges; i += gridDim.x) {
        const T src = g.rowInd[i];
        const T dst = g.colInd[i];

        const T srcStart = g.rowPtr[src];
        const T srcStop = g.rowPtr[src + 1];

        const T dstStart = g.rowPtr[dst];
        const T dstStop = g.rowPtr[dst + 1];

        const T dstLen = dstStop - dstStart;
        const T srcLen = srcStop - srcStart;

        // FIXME: remove warp reduction from this function call
        if (dstLen > srcLen) {
            blockCount += graph::block_sorted_count_binary<BLOCK_DIM_X>(&g.colInd[srcStart], srcLen,
                &g.colInd[dstStart], dstLen);
        }
        else {
            blockCount += graph::block_sorted_count_binary<BLOCK_DIM_X>(&g.colInd[dstStart], dstLen,
                &g.colInd[srcStart], srcLen);
        }
    }

    if (threadIdx.x == 0)
        atomicAdd(count, blockCount);


}


template <typename T, size_t BLOCK_DIM_X>
__global__ void __launch_bounds__(BLOCK_DIM_X)
kernel_binary_warp_shared_arrays(uint64* count,                //!< [inout] the count, caller should zero
    graph::COOCSRGraph_d<T> g, const size_t numEdges, //!< the number of edges this kernel will count
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
        T src = g.rowInd[i];
        T dst = g.colInd[i];

        T srcStart = g.rowPtr[src];
        T srcStop = g.rowPtr[src + 1];

        T dstStart = g.rowPtr[dst];
        T dstStop = g.rowPtr[dst + 1];

        T dstLen = dstStop - dstStart;
        T srcLen = srcStop - srcStart;


        if (srcLen > dstLen)
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
            int realIndex = dstStart + (lx + i * 32) * par;

            first[sharedIndex] = (lx + 32 * i) <= numElements ? g.colInd[realIndex] : 0;
        }


        warpCount += graph::warp_sorted_count_binary_s<warpsPerBlock>(&g.colInd[srcStart], srcLen,
            &g.colInd[dstStart], dstLen, &(first[startIndex]), par, numElements, pwMaxSize);

    }

    if (lx == 0)
        atomicAdd(count, warpCount);


}

template <typename T, size_t BLOCK_DIM_X>
__global__ void __launch_bounds__(BLOCK_DIM_X)
kernel_binary_warp_shared_colab_arrays(uint64* count,                //!< [inout] the count, caller should zero
    graph::COOCSRGraph_d<T> g, const size_t numEdges, //!< the number of edges this kernel will count
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
        T src = g.rowInd[i];
        T dst = g.colInd[i];

        T srcStart = g.rowPtr[src];
        T srcStop = g.rowPtr[src + 1];

        T dstStart = g.rowPtr[dst];
        T dstStop = g.rowPtr[dst + 1];

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
        if (!oddWarp && comm[wx + 1] == dst)
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
            startIndex = (wx - 1) * pwMaxSize;
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

            first[sharedIndex] = (lx + 32 * i + oddWarp * pwMaxSize) <= numElements ? g.colInd[realIndex] : 0;
        }

        __syncthreads();


        warpCount += graph::warp_sorted_count_binary_s<warpsPerBlock>(&g.colInd[srcStart], srcLen,
            &g.colInd[dstStart], dstLen, &(first[startIndex]), par, numElements, colabMaxSize);

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

        void count_async(COOCSRGraph_d<T>* g, const size_t numEdges, const size_t edgeOffset = 0, ProcessingElementEnum kernelType = Thread, int limit = 0)
        {
            const size_t dimBlock = 256;
            const size_t ne = numEdges;

            CUDA_RUNTIME(cudaMemset(TcBase<T>::count_, 0, sizeof(*TcBase<T>::count_)));

            // create one warp per edge
            const int dimGrid = (numEdges - edgeOffset + (dimBlock)-1) / (dimBlock);
            const int dimGridWarp = (32 * (numEdges - edgeOffset) + (dimBlock)-1) / (dimBlock);
            const int dimGridBlock = (dimBlock * numEdges + (dimBlock)-1) / (dimBlock);

            assert(TcBase<T>::count_);
            Log(LogPriorityEnum::debug, "device = %d, blocks = %d, threads = %d\n", TcBase<T>::dev_, dimGrid, dimBlock);
            CUDA_RUNTIME(cudaSetDevice(TcBase<T>::dev_));


            CUDA_RUNTIME(cudaEventRecord(TcBase<T>::kernelStart_, TcBase<T>::stream_));
            if (kernelType == ProcessingElementEnum::Thread)
                kernel_binary_thread_arrays<T, dimBlock> << <dimGrid, dimBlock, 0, TcBase<T>::stream_ >> > (TcBase<T>::count_, *g, ne, edgeOffset);
            else if (kernelType == ProcessingElementEnum::Warp)
                kernel_binary_warp_arrays<T, dimBlock> << <dimGridWarp, dimBlock, 0, TcBase<T>::stream_ >> > (TcBase<T>::count_, *g, ne, edgeOffset);
            else if (kernelType == ProcessingElementEnum::WarpShared)
                kernel_binary_warp_shared_arrays<T, dimBlock> << <dimGridWarp, dimBlock, 0, TcBase<T>::stream_ >> > (TcBase<T>::count_, *g, ne, edgeOffset);
            else if (kernelType == ProcessingElementEnum::Test)
                kernel_binary_warp_shared_colab_arrays<T, dimBlock> << <dimGridWarp, dimBlock, 0, TcBase<T>::stream_ >> > (TcBase<T>::count_, *g, ne, edgeOffset);
            else if (kernelType == ProcessingElementEnum::Block)
                kernel_binary_block_arrays<T, dimBlock> << <dimGridBlock, dimBlock, 0, TcBase<T>::stream_ >> > (TcBase<T>::count_, *g, ne, edgeOffset);
            else if (kernelType == Queue)
            {
                GPUArray<T> warpQueue("Warp Queue", AllocationTypeEnum::unified, numEdges, 0);
                GPUArray<T> blockQueue("Block Queue", AllocationTypeEnum::unified, numEdges, 0);
                GPUArray<T> warpCount("Warp Count", AllocationTypeEnum::unified, 1, 0);
                GPUArray<T> blockCount("block Count", AllocationTypeEnum::unified, 1, 0);

                /// This will launch a grid that can maximally fill the GPU, on the default stream with kernel arguments
                
                CUDAContext c;
                auto num_SMs = c.num_SMs;
                auto conc_blocks_per_SM = c.GetConCBlocks(dimBlock);
               

              kernel_binary_enqueue_arrays<T, dimBlock> << <dimGrid, dimBlock, 0, TcBase<T>::stream_ >> > (TcBase<T>::count_, *g, ne, edgeOffset,
                    warpQueue.gdata(), warpCount.gdata(), blockQueue.gdata(), blockCount.gdata()
                    );


                cudaDeviceSynchronize();

                printf("Warp Queue = %u elements\n", warpCount.gdata()[0]);
                printf("Block Queue = %u elements\n", blockCount.gdata()[0]);


               kernel_binary_dequeue_arrays<T, dimBlock> << < (32 * warpCount.gdata()[0] + dimBlock -1) / (dimBlock) , dimBlock, 0, TcBase<T>::stream_ >> > (TcBase<T>::count_, *g, ne, edgeOffset,

                    warpQueue.gdata(), warpCount.gdata(), blockQueue.gdata(), blockCount.gdata()
                    );


                warpQueue.freeGPU();
                warpCount.freeGPU();
                blockQueue.freeGPU();
                blockCount.freeGPU();

            }
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

        void count_per_edge_eid_async(GPUArray<int>& tcpt, EidGraph_d<T> g, const size_t numEdges, const size_t edgeOffset = 0, ProcessingElementEnum kernelType = Thread, int increasing = 0)
        {
            const size_t dimBlock = 32;
            const size_t ne = numEdges;
            T* rp_csr = g.rowPtr_csr;
            T* ci_csr = g.colInd_csr;

            T* ri = g.rowInd;
            T* ci = g.colInd;

            CUDA_RUNTIME(cudaMemset(TcBase<T>::count_, 0, sizeof(*TcBase<T>::count_)));

            // create one warp per edge
            const int dimGrid = (numEdges - edgeOffset + (dimBlock)-1) / (dimBlock);
            const int dimGridWarp = (32 * numEdges + (dimBlock)-1) / (dimBlock);
            const int dimGridBlock = (dimBlock * numEdges + (dimBlock)-1) / (dimBlock);

            assert(TcBase<T>::count_);
            Log(LogPriorityEnum::info, "device = %d, blocks = %d, threads = %d\n", TcBase<T>::dev_, dimGrid, dimBlock);
            CUDA_RUNTIME(cudaSetDevice(TcBase<T>::dev_));


            //CUDA_RUNTIME(cudaEventRecord(TcBase<T>::kernelStart_, TcBase<T>::stream_));
            //kernel_serial_pe_eid_arrays<T, dimBlock> << <dimGrid, dimBlock, 0, TcBase<T>::stream_ >> > (tcpt.gdata(), rp_csr, ci_csr, ri, ci, ne, edgeOffset, increasing);
            //CUDA_RUNTIME(cudaEventRecord(TcBase<T>::kernelStop_, TcBase<T>::stream_));



            CUDA_RUNTIME(cudaEventRecord(TcBase<T>::kernelStart_, TcBase<T>::stream_));
            if (kernelType == ProcessingElementEnum::Thread)
                kernel_binary_thread_pe_eid_arrays<T, dimBlock> << <dimGrid, dimBlock, 0, TcBase<T>::stream_ >> > (tcpt.gdata(), rp_csr, ci_csr, ri, ci, ne, edgeOffset);//need to be changed
            else if (kernelType == ProcessingElementEnum::Warp)
                kernel_binary_warp_pe_eid_arrays<T, dimBlock> << <dimGridWarp, dimBlock, 0, TcBase<T>::stream_ >> > (tcpt.gdata(), rp_csr, ci_csr, ri, ci, ne, edgeOffset);
            CUDA_RUNTIME(cudaEventRecord(TcBase<T>::kernelStop_, TcBase<T>::stream_));
        }


        void count_moveNext_per_edge_async(
            EidGraph_d<T>& g, const size_t numEdges,
            int level, GPUArray<bool> processed, GPUArray<int>& edgeSupport,
            GraphQueue<int, bool>& current, GraphQueue<int, bool>& next, GraphQueue<int, bool>& bucket, int bucket_level_end_,
            const size_t edgeOffset = 0, ProcessingElementEnum kernelType = Thread, int increasing = 0)
        {

            const size_t dimBlock = 256;
            const size_t ne = numEdges;
            T* rp_csr = g.rowPtr_csr;
            T* ci_csr = g.colInd_csr;

            T* ri = g.rowInd;
            T* ci = g.colInd;

            const int dimGrid = (current.count.gdata()[0] - edgeOffset + (dimBlock)-1) / (dimBlock);
            const int dimGridWarp = (32 * current.count.gdata()[0] + (dimBlock)-1) / (dimBlock);
            const int dimGridBlock = current.count.gdata()[0]; //(dimBlock * curr_cnt + (dimBlock)-1) / (dimBlock);
            CUDA_RUNTIME(cudaSetDevice(TcBase<T>::dev_));





            //CUDA_RUNTIME(cudaEventRecord(TcBase<T>::kernelStart_, TcBase<T>::stream_));
            if (kernelType == ProcessingElementEnum::Thread)
                kernel_binary_thread_pe_level_next<T, dimBlock> << <dimGrid, dimBlock, 0, TcBase<T>::stream_ >> > (
                    rp_csr, ci_csr, ri, ci, g.eid, ne,
                    level, processed.gdata(), edgeSupport.gdata(),
                    current.device_queue->gdata()[0],
                    next.device_queue->gdata()[0],
                    bucket.device_queue->gdata()[0],
                    bucket_level_end_);
            else if (kernelType == ProcessingElementEnum::Warp)
                kernel_binary_warp_pe_level_next<T, dimBlock> << <dimGridWarp, dimBlock, 0, TcBase<T>::stream_ >> > (
                    rp_csr, ci_csr, ri, ci, g.eid, ne,
                    level, processed.gdata(), edgeSupport.gdata(),
                    current.device_queue->gdata()[0],
                    next.device_queue->gdata()[0],
                    bucket.device_queue->gdata()[0],
                    bucket_level_end_);
            else if (kernelType == ProcessingElementEnum::Block)
                kernel_binary_block_pe_level_next<T, dimBlock> << <dimGridBlock, dimBlock, 0, TcBase<T>::stream_ >> > (
                    rp_csr, ci_csr, ri, ci, g.eid, ne,
                    level, processed.gdata(), edgeSupport.gdata(),
                    current.device_queue->gdata()[0],
                    next.device_queue->gdata()[0],
                    bucket.device_queue->gdata()[0],
                    bucket_level_end_);
          

            //CUDA_RUNTIME(cudaEventRecord(TcBase<T>::kernelStop_, TcBase<T>::stream_));
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