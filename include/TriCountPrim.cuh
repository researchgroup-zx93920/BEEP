#pragma once

#include "utils.cuh"

#include "cub/cub.cuh"
#include <cuda_runtime.h>

namespace graph
{
    template <typename T>
    __host__ __device__ T binary_search(const T *arr,         //!< [in] array to search
        const T lt,
        const T rt, //!< [in] size of array
        const T searchVal   //!< [in] value to search for
    ) {
        T left = lt;
        T right = rt;
        while (left < right) {
            const T mid = (left + right) / 2;
            T val = arr[mid];
            bool pred = val < searchVal;
            if (pred) {
                left = mid + 1;
            }
            else {
                right = mid;
            }
        }
        return left;
    }


    template <typename T>
    __host__ __device__ static uint64_t serial_sorted_count_linear(const T* const aBegin, //!< beginning of a
        const T* const aEnd,   //!< end of a
        const T* const bBegin, //!< beginning of b
        const T* const bEnd    //!< end of b
    ) {
        uint64_t count = 0;
        const T* ap = aBegin;
        const T* bp = bBegin;

        bool loadA = true;
        bool loadB = true;

        T a, b;


        while (ap < aEnd && bp < bEnd) {

            if (loadA) {
                a = *ap;
                loadA = false;
            }
            if (loadB) {
                b = *bp;
                loadB = false;
            }

            if (a == b) {
                ++count;
                ++ap;
                ++bp;
                loadA = true;
                loadB = true;
            }
            else if (a < b) {
                ++ap;
                loadA = true;
            }
            else {
                ++bp;
                loadB = true;
            }
        }
        return count;
    }

    /*! \brief return the number of common elements between sorted lists A and B
     */
    template <typename T>
    __host__ __device__ static size_t serial_sorted_count_linear(const T* const A, //!< beginning of a
        const size_t aSz,
        const T* const B, //!< beginning of b
        const size_t bSz) {
        return serial_sorted_count_linear(A, &A[aSz], B, &B[bSz]);
    }


    // Count per thread
    template <typename T>
    __device__ __forceinline__ uint64_t thread_sorted_count_binary(const T* A, //!< [in] array A
        const T aSz, //!< [in] the number of elements in A
        const T*  B, //!< [in] array B
        const T bSz  //!< [in] the number of elements in B
    ) {
        uint64_t threadCount = 0;
        T lb = 0;
        // cover entirety of A with warp
        for (size_t i = 0; i < aSz; i++) {
            // one element of A per thread, just search for A into B
            const T searchVal = A[i];
            lb = graph::binary_search<T>(B, lb, bSz, searchVal);
            if (lb < bSz) 
            {
                threadCount += (B[lb] == searchVal ? 1 : 0);   
            }
            else
            {
                break;
            }
        }
        return threadCount;
    }



    template <size_t WARPS_PER_BLOCK, typename T, bool reduce = true>
    __device__ __forceinline__ uint64_t warp_sorted_count_binary(const T* const A, //!< [in] array A
        const size_t aSz, //!< [in] the number of elements in A
        const T* const B, //!< [in] array B
        const size_t bSz  //!< [in] the number of elements in B

    ) {
        const int warpIdx = threadIdx.x / 32; // which warp in thread block
        const int laneIdx = threadIdx.x % 32; // which thread in warp

        uint64_t threadCount = 0;
        size_t lastIndex = 0;
        // cover entirety of A with warp
        for (size_t i = laneIdx; i < aSz; i += 32)
        {
            // one element of A per thread, just search for A into B
            const T searchVal = A[i];
            const T leftValue = B[lastIndex];

            if (searchVal >= leftValue)
            {
                const size_t lb = binary_search(B, lastIndex, bSz, searchVal);
                if (lb < bSz) {
                    threadCount += (B[lb] == searchVal ? 1 : 0);
                }

                lastIndex = (lb < 0) ? 0 : lb;

                if (B[lb] < searchVal)
                    break;

            }

            lastIndex = __shfl_sync(0xffffffff, lastIndex, 31);
        }

        if (reduce) {
            // give lane 0 the total count discovered by the warp
            typedef cub::WarpReduce<uint64_t> WarpReduce;
            __shared__ typename WarpReduce::TempStorage tempStorage[WARPS_PER_BLOCK];
            uint64_t aggregate = WarpReduce(tempStorage[warpIdx]).Sum(threadCount);
            return aggregate;
        }
        else {
            return threadCount;
        }
    }


    template <typename T>
    __host__ __device__ static uint8_t serial_sorted_count_binary(const T* const array, //!< [in] array to search through
        size_t left,          //!< [in] lower bound of search
        size_t right,         //!< [in] upper bound of search
        const T search_val    //!< [in] value to search for
    ) {
        while (left < right) {
            size_t mid = (left + right) / 2;
            T val = array[mid];
            if (val < search_val) {
                left = mid + 1;
            }
            else if (val > search_val) {
                right = mid;
            }
            else { // val == search_val
                return 1;
            }
        }
        return 0;
    }
    template <size_t BLOCK_DIM_X, typename T>
    __device__ uint64_t block_sorted_count_binary_s(const T* const A, //!< [in] array A
        const size_t aSz, //!< [in] the number of elements in A
        const T* const B, //!< [in] array B
        const size_t bSz,  //!< [in] the number of elements in B
        int* blockLeft
    ) {


        uint64_t threadCount = 0;
        const int warpIdx = threadIdx.x / 32; // which warp in thread block

        // cover entirety of A with block
        for (size_t i = threadIdx.x * C; i < aSz; i += BLOCK_DIM_X)
        {
            const T* aChunkBegin = &A[i];
            const T* aChunkEnd = &A[i + C];
            if (aChunkEnd > & A[aSz]) {
                aChunkEnd = &A[aSz];
            }

            // find the lower bound of the beginning of the A-chunk in B
            ulonglong2 uu = serial_sorted_search_binary(B, 0, bSz, *aChunkBegin);
            T lowerBound = uu.y;

            // Search for the A chunk in B, starting at the lower bound
            threadCount += serial_sorted_count_linear(aChunkBegin, aChunkEnd, &B[lowerBound], &B[bSz]);
        }

 
        blockLeft[threadIdx.x] = 0;

        __syncthreads();

        typedef cub::BlockReduce<uint64_t, BLOCK_DIM_X> BlockReduce;
        __shared__ typename BlockReduce::TempStorage tempStorage;
        uint64_t aggregate = BlockReduce(tempStorage).Sum(threadCount);


        return aggregate;
    }
}
