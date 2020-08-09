#pragma once

#include "utils.cuh"
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
            if (val == searchVal)
                return mid;
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
    __host__ __device__ T binary_search_left(const T* arr,         //!< [in] array to search
        const T lt,
        const T rt, //!< [in] size of array
        const T searchVal   //!< [in] value to search for
    ) {
        T left = lt;
        T right = rt;
        while (left < right) {
            const T mid = (left + right) / 2;
            T val = arr[mid];
            if (val == searchVal)
                return mid;
            bool pred = val < searchVal;
            if (pred) {
                left = mid + 1;
            }
            else {
                right = mid;
            }
        }
        return left>0?left-1:0;
    }




    template <typename T>
    __host__ __device__ T binary_search_full_bst(const T* arr,         //!< [in] array to search
        const T lt,
        const T rt, //!< [in] size of array
        const T searchVal   //!< [in] value to search for
    ) {
        T n = lt;
        while (n < rt)
        {
            if (arr[n] == searchVal)
                break;
            else if (arr[n] > searchVal)
                n = 2 * n + 1;
            else n = 2 * n + 2;
        }

        return n;

    }

    template <typename T>
    __global__ void binary_search_g(T *result,
        const T* arr,         //!< [in] array to search
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

        *result = left;
    }


    template <typename T>
    __device__ bool hash_search(
        const T* arr,         //!< [in] array to search
        const T binSize,
        const T size, //!< [in] size of array
        const T stashSize,
        const T searchVal   //!< [in] value to search for
    ) {
        const int numBins = (size + binSize - 1)/binSize;
        const int stashStart = binSize * numBins;
        T b = (searchVal/11) % numBins;

        
       for (int i = 0; i < binSize; i++)
       {
           T val = arr[b * binSize + i];
           if (searchVal == arr[b * binSize + i])
           {
               return true;
           }
           if (val == 0xFFFFFFFF)
           {
               return false;
           }
       }
        //for (int i = 0; i < stashSize; i++)
        //{
        //    if (arr[i + stashStart] == searchVal)
        //    {
        //        //printf("Hash - Bin: %u\n", searchVal);
        //        return true;
        //    }
        //}


       /*T left = graph::binary_search<T>(&arr[b*binSize], 0, binSize, searchVal);
       if (arr[b*binSize + left] == searchVal)
       {
           return true;
       }*/

        T left = graph::binary_search<T>(&arr[stashStart], 0, stashSize, searchVal);
        if (arr[stashStart + left] == searchVal)
        {
            return true;
        }

        return false;
    }

    template<typename T, size_t BLOCK_DIM_X>
    __global__ void hash_search_g(T* count,
        T* A, T sizeA, T* B, T sizeB, const T binSize, const T stashSize)
    {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        int threadCount = 0;
        for (int i = tid; i < sizeA; i += blockDim.x * gridDim.x)
        {
            T searchVal = A[i];
            threadCount += graph::hash_search<T>(B, binSize, sizeB, stashSize, searchVal) ? 1 : 0;
        }

        typedef cub::BlockReduce<uint64_t, BLOCK_DIM_X> BlockReduce;
        __shared__ typename BlockReduce::TempStorage tempStorage;
        uint64_t aggregate = BlockReduce(tempStorage).Sum(threadCount);

        // Add to total count
        if (0 == threadIdx.x) {
            atomicAdd(count, aggregate);
        }

    }

    template <typename T>
    __device__ bool hash_nostash_search(
        const T*arrPointer,
        const T* arr,         //!< [in] array to search
        const T numBins,
        const T searchVal   //!< [in] value to search for
    ) {
        T b = (searchVal / 11) % numBins;
        T start = arrPointer[b];
        T end = arrPointer[b + 1];

        if (end - start == 0)//empty bin
            return false;

        if (end - start < 128)
        {
            for (int i = start; i < end; i++)
            {
                if (searchVal == arr[i])
                {
                    return true;
                }
            }
        }
       else
        {
            T left = graph::binary_search<T>(&arr[start], 0, end-start, searchVal);
            if (arr[start + left] == searchVal)
            {
                return true;
            }

        }
        return false;
    }


    template<typename T, size_t BLOCK_DIM_X>
    __device__ int hash_search_nostash_thread_d(T* A, T sizeA, T* BP, T* BD, T numBins)
    {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        int threadCount = 0;
        for (int i = 0; i < sizeA; i++)
        {
            T searchVal = A[i];

            //printf("%d, %u\n", i, searchVal);
            threadCount += graph::hash_nostash_search<T>(BP, BD, numBins, searchVal) ? 1 : 0;
        }
        return threadCount;
    }

    template <size_t WARPS_PER_BLOCK, typename T, bool reduce = true>
    __device__ int hash_search_nostash_warp_d(T* A, T sizeA, T* BP, T* BD, T numBins)
    {
        const int warpIdx = threadIdx.x / 32; // which warp in thread block
        const int laneIdx = threadIdx.x % 32; // which thread in warp

        uint64_t threadCount = 0;
        T lastIndex = 0;

        // cover entirety of A with warp
        for (size_t i = laneIdx; i < sizeA; i += 32)
        {
            // one element of A per thread, just search for A into B
            const T searchVal = A[i];
            threadCount += graph::hash_nostash_search<T>(BP, BD, numBins, searchVal) ? 1 : 0;
         }

        if (reduce) {
            // give lane 0 the total count discovered by the warp
            typedef cub::WarpReduce<uint64_t> WarpReduce;
            __shared__ typename WarpReduce::TempStorage tempStorage[WARPS_PER_BLOCK];
            uint64_t aggregate = WarpReduce(tempStorage[warpIdx]).Sum(threadCount);
            return aggregate;
        }
        else
        {
            return threadCount;
        }
    }


    template<typename T, size_t BLOCK_DIM_X>
    __global__ void hash_search_nostash_g(T* count,
        T* A, T sizeA, T* BP, T* BD, T numBins)
    {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        int threadCount = 0;
        for (int i = tid; i < sizeA; i += blockDim.x * gridDim.x)
        {
            T searchVal = A[i];
            threadCount += graph::hash_nostash_search<T>(BP, BD, numBins, searchVal) ? 1 : 0;
        }

        typedef cub::BlockReduce<uint64_t, BLOCK_DIM_X> BlockReduce;
        __shared__ typename BlockReduce::TempStorage tempStorage;
        uint64_t aggregate = BlockReduce(tempStorage).Sum(threadCount);

        // Add to total count
        if (0 == threadIdx.x) {
            atomicAdd(count, aggregate);
        }

    }



    template<typename T, size_t BLOCK_DIM_X>
    __global__ void binary_search_2arr_g(T* count,
        T* A, T sizeA, T* B, T sizeB)
    {
        size_t gx = blockDim.x * blockIdx.x + threadIdx.x;
        uint64_t threadCount = 0;

        for (size_t i = gx; i < sizeA; i += blockDim.x * gridDim.x) 
        {
            T searchVal = A[i];
            const T lb = graph::binary_search<T>(B, 0, sizeB, searchVal);
            if (lb < sizeB)
            {
                threadCount += (B[lb] == searchVal ? 1 : 0);
            }

        }

        typedef cub::BlockReduce<uint64_t, BLOCK_DIM_X> BlockReduce;
        __shared__ typename BlockReduce::TempStorage tempStorage;
        uint64_t aggregate = BlockReduce(tempStorage).Sum(threadCount);

        // Add to total count
        if (0 == threadIdx.x) {
            atomicAdd(count, aggregate);
        }
    }

    template<typename T, size_t BLOCK_DIM_X>
    __global__ void binary_search_bst_g(T* count,
        T* A, T sizeA, T* B, T sizeB)
    {
        size_t gx = blockDim.x * blockIdx.x + threadIdx.x;
        uint64_t threadCount = 0;

        for (size_t i = gx; i < sizeA; i += blockDim.x * gridDim.x)
        {
            T searchVal = A[i];
            const T lb = graph::binary_search_full_bst<T>(B, 0, sizeB, searchVal);
            if (lb < sizeB)
            {
                threadCount += (B[lb] == searchVal ? 1 : 0);
            }

        }

        typedef cub::BlockReduce<uint64_t, BLOCK_DIM_X> BlockReduce;
        __shared__ typename BlockReduce::TempStorage tempStorage;
        uint64_t aggregate = BlockReduce(tempStorage).Sum(threadCount);

        // Add to total count
        if (0 == threadIdx.x) {
            atomicAdd(count, aggregate);
        }
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


    template <typename T>
    __device__ __forceinline__ uint64_t thread_sorted_count_upto_binary(int upto, bool* mask, T srcStart, T dstStart,  const T* arr, //!< [in] array A
        const T aSz, //!< [in] the number of elements in A
        const T bSz  //!< [in] the number of elements in B
    ) {
        uint64_t threadCount = 0;
        T lb = 0;
        // cover entirety of A with warp
        for (size_t i = 0; i < aSz; i++) 
        {

                // one element of A per thread, just search for A into B
                const T searchVal = arr[srcStart + i];

                lb = graph::binary_search<T>(&arr[dstStart], lb, bSz, searchVal);

                if (lb < bSz )
                {
                    if (mask[dstStart + lb] && mask[srcStart + i])
                    {
                        threadCount += (arr[dstStart + lb] == searchVal ? 1 : 0);
                    }
                    if (threadCount == upto)
                        return threadCount;
                }
                else
                {
                    break;
                }
            }

        return threadCount;
    }


    template <typename T>
    __device__ __forceinline__ uint64_t thread_sorted_set_binary(T* arr, const T* A, //!< [in] array A
        const T aSz, //!< [in] the number of elements in A
        const T* B, //!< [in] array B
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
                if (searchVal == B[lb])
                {
                    arr[threadCount] = searchVal;
                    threadCount++;
                }
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
        T* B, //!< [in] array B
        T bSz  //!< [in] the number of elements in B
    ) 
    {
        const int warpIdx = threadIdx.x / 32; // which warp in thread block
        const int laneIdx = threadIdx.x % 32; // which thread in warp

        uint64_t threadCount = 0;
        T lastIndex = 0;

        // cover entirety of A with warp
        for (size_t i = laneIdx; i < aSz; i += 32)
        {
            // one element of A per thread, just search for A into B
            const T searchVal = A[i];
            const T leftValue = B[lastIndex];

            if (searchVal >= leftValue)
            {

                const T lb = graph::binary_search<T>(B, lastIndex, bSz, searchVal);
                if (lb < bSz)
                {
                    if (B[lb] == searchVal)
                    {
                        //printf("At %u, SearchVal = %u\n", lb, searchVal);
                        threadCount++;
                    }


                }

                lastIndex = lb;
            }

            unsigned int writemask_deq = __activemask();
            lastIndex = __shfl_sync(writemask_deq, lastIndex, 31);
        }

        if (reduce) {
            // give lane 0 the total count discovered by the warp
            typedef cub::WarpReduce<uint64_t> WarpReduce;
            __shared__ typename WarpReduce::TempStorage tempStorage[WARPS_PER_BLOCK];
            uint64_t aggregate = WarpReduce(tempStorage[warpIdx]).Sum(threadCount);
            return aggregate;
        }
        else 
        {
            return threadCount;
        }
    }

    template <size_t WARPS_PER_BLOCK, typename T, bool reduce = true>
    __device__ __forceinline__ uint64_t warp_sorted_count_binary_s(const T* const A, //!< [in] array A
        const size_t aSz, //!< [in] the number of elements in A
        T* B, //!< [in] array B
        T bSz,  //!< [in] the number of elements in B
        T* first_level,
        T par,
        int numElements,
        int pwMaxSize
    )
    {
        const int warpIdx = threadIdx.x / 32; // which warp in thread block
        const int laneIdx = threadIdx.x % 32; // which thread in warp

        uint64_t threadCount = 0;
        T lastIndex = 0;
        T fl = 0;

        // cover entirety of A with warp
        for (size_t i = laneIdx; i < aSz; i += 32)
        {
            // one element of A per thread, just search for A into B
            const T searchVal = A[i];

            fl = graph::binary_search_left<T>(first_level, fl, numElements, searchVal);
            lastIndex = par * fl;
            T right = 0;

            if (bSz < pwMaxSize)
            {
                if (searchVal == first_level[fl])
                {
                    threadCount++;
                    //printf("At %u, searchVal = %u, direct\n", fl * par, searchVal);
                }
                continue;
            }
            else if (fl == numElements - 1)
                right = bSz;
            else
                right = (fl + 1) * par;

            const T lb = graph::binary_search_left<T>(B, lastIndex, right, searchVal);
            if (lb < bSz)
            {
                if (B[lb] == searchVal)
                {
                    //printf("At %u, searchVal = %u\n", lb, searchVal);
                    threadCount++;
                }
            }

            //lastIndex = lb;


           /* unsigned int writemask_deq = __activemask();
            fl = __shfl_sync(writemask_deq, fl, 31);*/
        }

        if (reduce) {
            // give lane 0 the total count discovered by the warp
            typedef cub::WarpReduce<uint64_t> WarpReduce;
            __shared__ typename WarpReduce::TempStorage tempStorage[WARPS_PER_BLOCK];
            uint64_t aggregate = WarpReduce(tempStorage[warpIdx]).Sum(threadCount);
            return aggregate;
        }
        else
        {
            return threadCount;
        }
    }


    template <size_t WARPS_PER_BLOCK, typename T, bool reduce = true>
    __device__ __forceinline__ uint64_t warp_sorted_count_mask_binary(bool* mask, T srcStart, T dstStart, const T* const arr, //!< [in] array A
        const size_t aSz, //!< [in] the number of elements in A
        T bSz  //!< [in] the number of elements in B
    )
    {
        const int warpIdx = threadIdx.x / 32; // which warp in thread block
        const int laneIdx = threadIdx.x % 32; // which thread in warp

        uint64_t threadCount = 0;
        T lastIndex = 0;

        // cover entirety of A with warp
        for (size_t i = laneIdx; i < aSz; i += 32)
        {
            // one element of A per thread, just search for A into B
            const T searchVal = arr[srcStart +  i];
            const T leftValue = arr[dstStart +  lastIndex];

            if (searchVal >= leftValue && mask[srcStart + i])
            {

                const T lb = graph::binary_search<T>(&arr[dstStart], lastIndex, bSz, searchVal);
                if (lb < bSz)
                {
                    threadCount += ((arr[dstStart + lb] == searchVal) && (mask[dstStart + lb]) ? 1 : 0);
                }

                lastIndex = lb;
            }

            unsigned int writemask_deq = __activemask();
            lastIndex = __shfl_sync(writemask_deq, lastIndex, 31);
        }

        if (reduce) {
            // give lane 0 the total count discovered by the warp
            typedef cub::WarpReduce<uint64_t> WarpReduce;
            __shared__ typename WarpReduce::TempStorage tempStorage[WARPS_PER_BLOCK];
            uint64_t aggregate = WarpReduce(tempStorage[warpIdx]).Sum(threadCount);
            return aggregate;
        }
        else
        {
            return threadCount;
        }
    }




    template <size_t WARPS_PER_BLOCK, typename T, bool reduce = true>
    __device__ __forceinline__ uint64_t warp_sorted_set_binary(T* indecies, T* arr, const T* A, //!< [in] array A
        const T aSz, //!< [in] the number of elements in A
        const T* B, //!< [in] array B
        const T bSz  //!< [in] the number of elements in B

    ) 
    {
        const int warpIdx = threadIdx.x / 32; // which warp in thread block
        const int laneIdx = threadIdx.x % 32; // which thread in warp

        uint64_t threadCount = 0;
        T lastIndex = 0;

        // cover entirety of A with warp
        for (size_t i = laneIdx; i < aSz; i += 32)
        {
            // one element of A per thread, just search for A into B
            const T searchVal = A[i];
            const T leftValue = B[lastIndex];

            if (searchVal >= leftValue)
            {

                const T lb = graph::binary_search<T>(B, 0, bSz, searchVal);
                if (lb < bSz)
                {
                    if (B[lb] == searchVal)
                    {
                        T index = atomicAdd(indecies, 1);
                        arr[index] = searchVal;
                        threadCount++;
                    }
                   
                }

                lastIndex = lb;
            }

            unsigned int writemask_deq = __activemask();
            lastIndex = __shfl_sync(writemask_deq, lastIndex, 31);
        }

        
        return threadCount;
        
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
    __device__ uint64_t block_sorted_count_binary(const T* const A, //!< [in] array A
        const size_t aSz, //!< [in] the number of elements in A
        const T* const B, //!< [in] array B
        const size_t bSz  //!< [in] the number of elements in B
    ) {


        uint64_t threadCount = 0;
        T lastIndex = 0;

        // cover entirety of A with block
        for (size_t i = threadIdx.x; i < aSz; i += BLOCK_DIM_X)
        {
            // one element of A per thread, just search for A into B
            const T searchVal = A[i];
            const T leftValue = B[lastIndex];

            if (searchVal >= leftValue)
            {
                const T lb = graph::binary_search<T>(B, lastIndex, bSz, searchVal);
                if (lb < bSz)
                {
                    if (B[lb] == searchVal)
                    {
                        threadCount++;
                    }

                }

                lastIndex = lb;
            }
        }

        __syncthreads();

        typedef cub::BlockReduce<uint64_t, BLOCK_DIM_X> BlockReduce;
        __shared__ typename BlockReduce::TempStorage tempStorage;
        uint64_t aggregate = BlockReduce(tempStorage).Sum(threadCount);


        return aggregate;
    }




    ///////////////////////////////////////SERIAL ENTERSECTION //////////////////////////////////////////////
    template <typename T>
    __host__ __device__ static size_t serial_sorted_count_linear(T min, const T* const A, //!< beginning of a
        const size_t aSz,
        const T* const B, //!< beginning of b
        const size_t bSz) {
        return serial_sorted_count_linear(min, A, &A[aSz], B, &B[bSz]);
    }


    template <typename T>
    __host__ __device__ static uint64_t serial_sorted_count_linear(const T min, const T* const aBegin, //!< beginning of a
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




    template <typename T>
    __host__ __device__ static uint64_t serial_sorted_count_upto_linear(int upto, bool *mask, const T min, T* arr,
        const T const aBegin, //!< beginning of a
        const T const aEnd,   //!< end of a
        const T const bBegin, //!< beginning of b
        const T const bEnd    //!< end of b
    ) {
        uint64_t count = 0;
        T ap = aBegin;
        T bp = bBegin;

        bool loadA = true;
        bool loadB = true;

        T a, b;


        while (ap < aEnd && bp < bEnd) {

            if (loadA) {
                a = arr[ap];
                loadA = false;
            }
            if (loadB) {
                b = arr[bp];
                loadB = false;
            }

            if (a == b) {
                if(!mask[ap] && !mask[bp])
                    ++count;
                if (count == upto)
                {
                    return count;
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
        return count;
    }


    //min is used to enforce increasing order !!
    template <typename T>
    __host__ __device__ static uint64_t serial_sorted_set_linear(T* arr, const T* aBegin, //!< beginning of a
        const T* aEnd,   //!< end of a
        const T* bBegin, //!< beginning of b
        const T* bEnd    //!< end of b
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
                arr[count] = a;
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

}
