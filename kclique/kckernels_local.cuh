#pragma once
#include "kckernels.cuh"

/* AtomicAdd to SharedMemory with accumulation through levels + Deep:
 *		where we accumulate the counter when we backtrack to the previous level,
 *		instead of incrementing all counters included by the clique by 1 when a k-clique is found.
 *      This kernel is built on top of baseline_deep.
 * Extra shared memory: __shared__ uint64 local_clique_count[1024];
 *						__shared__ uint64 root_count;
 *						__shared__ uint64 level_local_clique_count[numPartitions][9];
 * Difference from baseline_deep:
 * 		Clear local_clique counter in shared memory:
 *			for (unsigned int idx = threadIdx.x; idx < srcLen; idx += blockDim.x)
 *			{
 *				local_clique_count[idx] = 0;
 *			}
 *			if (threadIdx.x == 0)
 *			{
 *				root_count = 0;
 *			}
 *			__syncthreads();
 *		Clear the first level when starting from a new branch:
 * 			level_local_clique_count[wx][0] = 0;
 * 		Whenever we find a clique:
 *			atomicAdd(&local_clique_count[level_prev_index[wx][l[wx] - 3] - 1]], 1);
 *			level_local_clique_count[wx][l[wx] - 3] ++;
 * 		When backtracking:
 *			uint64 cur = level_local_clique_count[wx][l[wx] - 3];
 *			level_local_clique_count[wx][l[wx] - 4] += cur;
 *			atomicAdd(&local_clique_count[level_prev_index[wx][l[wx] - 4] - 1], cur);
 *		After exploring a branch from the source node:
 * 			atomicAdd(&root_count, clique_count[wx]);
 *			atomicAdd(&local_clique_count[j], clique_count[wx]);
 *		After exploring a source node:
 * 			for (unsigned int idx = threadIdx.x; idx < srcLen; idx += blockDim.x)
 *			{
 *				atomicAdd(&cpn[g.colInd[srcStart + idx]], local_clique_count[idx]);
 *			}
 *			if (threadIdx.x == 0)
 *			{
 *				atomicAdd(&cpn[src], root_count);
 *			}
 *			__syncthreads();
 */

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_block_warp_binary_count_local_sharedmem_lazy_deep(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;
    __shared__ unsigned short level_index[numPartitions][9];
    __shared__ unsigned short level_count[numPartitions][9];
    __shared__ unsigned short level_prev_index[numPartitions][9];

    __shared__ uint64 clique_count[numPartitions];
    __shared__ T l[numPartitions];
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen;

    __shared__ T num_divs_local, encode_offset, *encode;

    // shared memory counter for local_clique
    __shared__ uint64 local_clique_count[1024];
    __shared__ uint64 root_count;
    __shared__ uint64 level_local_clique_count[numPartitions][9];

    __syncthreads();

    if (threadIdx.x == 0)
    {
        sm_id = __mysmid();
        T temp = 0;
        while (atomicCAS(&(levelStats[(sm_id * CBPSM) + temp]), 0, 1) != 0)
        {
            temp++;
        }
        levelPtr = temp;
    }
    __syncthreads();

    for (unsigned long long i = blockIdx.x; i < (unsigned long long)current.count[0]; i += gridDim.x)
    {
        //block things
        if (threadIdx.x == 0)
        {
            src = current.queue[i];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;
        }
        __syncthreads();

        // Clear local_clique counter in shared memory
        for (unsigned int idx = threadIdx.x; idx < srcLen; idx += blockDim.x)
        {
            local_clique_count[idx] = 0;
        }
        if (threadIdx.x == 0)
        {
            root_count = 0;
        }
        __syncthreads();

        if (threadIdx.x == 0)
            num_divs_local = (srcLen + 32 - 1) / 32;
        else if (threadIdx.x == 1)
        {
            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];
        }
        __syncthreads();

        // Encode
        T partMask = (1 << CPARTSIZE) - 1;
        partMask = partMask << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
            __syncwarp(partMask);
            graph::warp_sorted_count_and_encode<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&g.colInd[srcStart], srcLen,
                &g.colInd[g.rowPtr[g.colInd[srcStart + j]]], g.rowPtr[g.colInd[srcStart + j] + 1] - g.rowPtr[g.colInd[srcStart + j]],
                &encode[j * num_divs_local]);
        }
        __syncthreads(); //Done encoding

        T* cl = &current_level[((sm_id * CBPSM + levelPtr) * numPartitions + wx) * NUMDIVS * 9];

        for (T j = wx; j < srcLen; j += numPartitions)
        {
            if (lx == 0)
            {
                level_count[wx][0] = 0;
                level_index[wx][0] = 0;
                level_prev_index[wx][0] = 0;
                level_local_clique_count[wx][0] = 0;
                l[wx] = 3;
                clique_count[wx] = 0;
            }

            uint64 warpCount = 0;
            for (unsigned long long k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask, warpCount);

            if (lx == 0)
            {
                if (KCCOUNT >= 3 && warpCount >= KCCOUNT - 2)
                {
                    level_count[wx][0] = warpCount;
                }
            }
            __syncwarp(partMask);

            while (level_count[wx][l[wx] - 3] > level_index[wx][l[wx] - 3])
            {
                //First Index
                T* from = l[wx] == 3 ? &(encode[num_divs_local * j]) : &(cl[num_divs_local * (l[wx] - 3)]);
                T maskBlock = level_prev_index[wx][l[wx] - 3] / 32;
                T maskIndex = ~((1 << (level_prev_index[wx][l[wx] - 3] & 0x1F)) -1);

                T newIndex = __ffs(from[maskBlock] & maskIndex);
                while(newIndex == 0)
                {
                    maskIndex = 0xFFFFFFFF;
                    maskBlock++;
                    newIndex = __ffs(from[maskBlock] & maskIndex);
                }
                newIndex =  32*maskBlock + newIndex - 1;

                if (lx == 0)
                {
                    level_prev_index[wx][l[wx] - 3] = newIndex + 1;
                    level_index[wx][l[wx] - 3]++;
                }
                __syncwarp(partMask);

                if (l[wx] < KCCOUNT)
                {
                    // Intersect
                    T* to = &(cl[num_divs_local * (l[wx] - 2)]);
                    warpCount = 0;
                    for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                    {
                        to[k] = from[k] & encode[newIndex * num_divs_local + k];
                        warpCount += __popc(to[k]);
                    }
                    reduce_part<T, CPARTSIZE>(partMask, warpCount);
                }

                if (lx == 0)
                {
                    if (l[wx] == KCCOUNT)
                    {
                        clique_count[wx] ++;
                        atomicAdd(&local_clique_count[level_prev_index[wx][l[wx] - 3] - 1], 1);
                        level_local_clique_count[wx][l[wx] - 3] ++;
                    }	
                    else if (l[wx] < KCCOUNT && warpCount >= KCCOUNT - l[wx])
                    {
                        (l[wx])++;
                        level_count[wx][l[wx] - 3] = warpCount;
                        level_index[wx][l[wx] - 3] = 0;
                        level_prev_index[wx][l[wx] - 3] = 0;
                        level_local_clique_count[wx][l[wx] - 3] = 0;
                    }
                
                    while (l[wx] > 3 && level_index[wx][l[wx] - 3] >= level_count[wx][l[wx] - 3])
                    {
                        uint64 cur = level_local_clique_count[wx][l[wx] - 3];
                        level_local_clique_count[wx][l[wx] - 4] += cur;
                         atomicAdd(&local_clique_count[level_prev_index[wx][l[wx] - 4] - 1], cur);
                        (l[wx])--;
                    }
                }
                __syncwarp(partMask);
            }

            if (lx == 0)
            {
                atomicAdd(counter, clique_count[wx]);
                atomicAdd(&root_count, clique_count[wx]);
                atomicAdd(&local_clique_count[j], clique_count[wx]);
            }
            __syncwarp(partMask);
        }
        __syncthreads();

        for (unsigned int idx = threadIdx.x; idx < srcLen; idx += blockDim.x)
        {
            atomicAdd(&cpn[g.colInd[srcStart + idx]], local_clique_count[idx]);
        }
        if (threadIdx.x == 0)
        {
            atomicAdd(&cpn[src], root_count);
        }
        __syncthreads();
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}

/* AtomicAdd to GlobalMemory with accumulation through levels + Deep:
 * 		Use atomicAdd to increase the local_clique counter in global memory,
 *		where we accumulate the counter when we backtrack to the previous level,
 *		instead of incrementing all counters included by the clique by 1 when a k-clique is found.
 *      This kernel is built on top of baseline_deep.
 * Extra shared memory:
 * 		__shared__ uint64 level_local_clique_count[numPartitions][9];
 * Difference from baseline_deep:
 * 		Clear the first level when starting from a new branch:
 * 			level_local_clique_count[wx][0] = 0;
 * 		Whenever we find a clique:
 *			atomicAdd(&cpn[g.colInd[srcStart + level_prev_index[wx][l[wx] - 3] - 1]], 1);
 *			level_local_clique_count[wx][l[wx] - 3] ++;
 *		Clear the new level When branching:
 *			level_local_clique_count[wx][l[wx] - 3] = 0;
 *		When backtracking:
 *			uint64 cur = level_local_clique_count[wx][l[wx] - 3];
 *			level_local_clique_count[wx][l[wx] - 4] += cur;
 *			atomicAdd(&cpn[g.colInd[srcStart + level_prev_index[wx][l[wx] - 4] - 1]], cur);
 *		After exploring a branch from the source node:
 * 			atomicAdd(&cpn[current.queue[i]], clique_count[wx]);
 *			atomicAdd(&cpn[g.colInd[srcStart + j]], clique_count[wx]);
 */

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_block_warp_binary_count_local_globalmem_lazy_deep(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;
    __shared__ unsigned short level_index[numPartitions][9];
    __shared__ unsigned short level_count[numPartitions][9];
    __shared__ unsigned short level_prev_index[numPartitions][9];

    __shared__ uint64 clique_count[numPartitions];
    __shared__ T l[numPartitions];
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen;

    __shared__ T num_divs_local, encode_offset, *encode;

    __shared__ uint64 level_local_clique_count[numPartitions][9];

    __syncthreads();

    if (threadIdx.x == 0)
    {
        sm_id = __mysmid();
        T temp = 0;
        while (atomicCAS(&(levelStats[(sm_id * CBPSM) + temp]), 0, 1) != 0)
        {
            temp++;
        }
        levelPtr = temp;
    }
    __syncthreads();

    for (unsigned long long i = blockIdx.x; i < (unsigned long long)current.count[0]; i += gridDim.x)
    {
        //block things
        if (threadIdx.x == 0)
        {
            src = current.queue[i];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;
        }
        __syncthreads();

        if (threadIdx.x == 0)
            num_divs_local = (srcLen + 32 - 1) / 32;
        else if (threadIdx.x == 1)
        {
            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];
        }
        __syncthreads();

        // Encode
        T partMask = (1 << CPARTSIZE) - 1;
        partMask = partMask << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
            __syncwarp(partMask);
            graph::warp_sorted_count_and_encode<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&g.colInd[srcStart], srcLen,
                &g.colInd[g.rowPtr[g.colInd[srcStart + j]]], g.rowPtr[g.colInd[srcStart + j] + 1] - g.rowPtr[g.colInd[srcStart + j]],
                &encode[j * num_divs_local]);
        }
        __syncthreads(); //Done encoding

        T* cl = &current_level[((sm_id * CBPSM + levelPtr) * numPartitions + wx) * NUMDIVS * 9];

        for (T j = wx; j < srcLen; j += numPartitions)
        {
            if (lx == 0)
            {
                level_count[wx][0] = 0;
                level_index[wx][0] = 0;
                level_prev_index[wx][0] = 0;
                level_local_clique_count[wx][0] = 0;
                l[wx] = 3;
                clique_count[wx] = 0;
            }

            uint64 warpCount = 0;
            for (unsigned long long k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask, warpCount);

            if (lx == 0)
            {
                if (KCCOUNT >= 3 && warpCount >= KCCOUNT - 2)
                {
                    level_count[wx][0] = warpCount;
                }
            }
            __syncwarp(partMask);

            while (level_count[wx][l[wx] - 3] > level_index[wx][l[wx] - 3])
            {
                //First Index
                T* from = l[wx] == 3 ? &(encode[num_divs_local * j]) : &(cl[num_divs_local * (l[wx] - 3)]);
                T maskBlock = level_prev_index[wx][l[wx] - 3] / 32;
                T maskIndex = ~((1 << (level_prev_index[wx][l[wx] - 3] & 0x1F)) -1);

                T newIndex = __ffs(from[maskBlock] & maskIndex);
                while(newIndex == 0)
                {
                    maskIndex = 0xFFFFFFFF;
                    maskBlock++;
                    newIndex = __ffs(from[maskBlock] & maskIndex);
                }
                newIndex =  32*maskBlock + newIndex - 1;

                if (lx == 0)
                {
                    level_prev_index[wx][l[wx] - 3] = newIndex + 1;
                    level_index[wx][l[wx] - 3]++;
                }
                __syncwarp(partMask);

                if (l[wx] < KCCOUNT)
                {
                    // Intersect
                    T* to = &(cl[num_divs_local * (l[wx] - 2)]);
                    warpCount = 0;
                    for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                    {
                        to[k] = from[k] & encode[newIndex * num_divs_local + k];
                        warpCount += __popc(to[k]);
                    }
                    reduce_part<T, CPARTSIZE>(partMask, warpCount);
                }

                if (lx == 0)
                {
                    if (l[wx] == KCCOUNT)
                    {
                        clique_count[wx] ++;
                        atomicAdd(&cpn[g.colInd[srcStart + level_prev_index[wx][l[wx] - 3] - 1]], 1);
                        level_local_clique_count[wx][l[wx] - 3] ++;
                    }
                    else if (l[wx] < KCCOUNT && warpCount >= KCCOUNT - l[wx])
                    {
                        (l[wx])++;
                        level_count[wx][l[wx] - 3] = warpCount;
                        level_index[wx][l[wx] - 3] = 0;
                        level_prev_index[wx][l[wx] - 3] = 0;
                        level_local_clique_count[wx][l[wx] - 3] = 0;
                    }
                
                    while (l[wx] > 3 && level_index[wx][l[wx] - 3] >= level_count[wx][l[wx] - 3])
                    {
                        uint64 cur = level_local_clique_count[wx][l[wx] - 3];
                        level_local_clique_count[wx][l[wx] - 4] += cur;
                        atomicAdd(&cpn[g.colInd[srcStart + level_prev_index[wx][l[wx] - 4] - 1]], cur);
                        (l[wx])--;
                    }
                }
                __syncwarp(partMask);
            }

            if (lx == 0)
            {
                atomicAdd(counter, clique_count[wx]);
                atomicAdd(&cpn[current.queue[i]], clique_count[wx]);
                atomicAdd(&cpn[g.colInd[srcStart + j]], clique_count[wx]);
            }
            __syncwarp(partMask);
        }
        __syncthreads();
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}

/* AtomicAdd to SharedMemory + Deep: Use atomicAdd to increase the local_clique counter in shared memory,
 *                                   whenever we find a clique. And increase the global counter later.
 *                                   This kernel is built on top of baseline_deep.
 * Extra shared memory: __shared__ uint64 local_clique_count[1024];
 *						__shared__ uint64 root_count;
 * Difference from baseline_deep:
 * 		Clear local_clique counter in shared memory:
 *			for (unsigned int idx = threadIdx.x; idx < srcLen; idx += blockDim.x)
 *			{
 *				local_clique_count[idx] = 0;
 *			}
 *			if (threadIdx.x == 0)
 *			{
 *				root_count = 0;
 *			}
 *			__syncthreads();
 * 		Whenever we find a clique:
 *			for (T k = lx; k < KCCOUNT - 2; k += CPARTSIZE)
 *			{
 *				atomicAdd(&local_clique_count[level_prev_index[wx][k] - 1], 1);
 *			}
 *			__syncwarp(partMask);
 *		After exploring a branch from the source node:
 * 			atomicAdd(&root_count, clique_count[wx]);
 *			atomicAdd(&local_clique_count[j], clique_count[wx]);
 *		After exploring a source node:
 * 			for (unsigned int idx = threadIdx.x; idx < srcLen; idx += blockDim.x)
 *			{
 *				atomicAdd(&cpn[g.colInd[srcStart + idx]], local_clique_count[idx]);
 *			}
 *			if (threadIdx.x == 0)
 *			{
 *				atomicAdd(&cpn[src], root_count);
 *			}
 *			__syncthreads();
 */

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_block_warp_binary_count_local_sharedmem_direct_deep(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;
    __shared__ unsigned short level_index[numPartitions][9];
    __shared__ unsigned short level_count[numPartitions][9];
    __shared__ unsigned short level_prev_index[numPartitions][9];

    __shared__ uint64 clique_count[numPartitions];
    __shared__ T l[numPartitions];
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen;

    __shared__ T num_divs_local, encode_offset, *encode;

    // shared memory counter for local_clique
    __shared__ uint64 local_clique_count[1024];
    __shared__ uint64 root_count;

    __syncthreads();

    if (threadIdx.x == 0)
    {
        sm_id = __mysmid();
        T temp = 0;
        while (atomicCAS(&(levelStats[(sm_id * CBPSM) + temp]), 0, 1) != 0)
        {
            temp++;
        }
        levelPtr = temp;
    }
    __syncthreads();

    for (unsigned long long i = blockIdx.x; i < (unsigned long long)current.count[0]; i += gridDim.x)
    {
        //block things
        if (threadIdx.x == 0)
        {
            src = current.queue[i];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;
        }
        __syncthreads();

        // Clear local_clique counter in shared memory
        for (unsigned int idx = threadIdx.x; idx < srcLen; idx += blockDim.x)
        {
            local_clique_count[idx] = 0;
        }
        if (threadIdx.x == 0)
        {
            root_count = 0;
        }
        __syncthreads();

        if (threadIdx.x == 0)
            num_divs_local = (srcLen + 32 - 1) / 32;
        else if (threadIdx.x == 1)
        {
            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];
        }
        __syncthreads();

        // Encode
        T partMask = (1 << CPARTSIZE) - 1;
        partMask = partMask << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
            __syncwarp(partMask);
            graph::warp_sorted_count_and_encode<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&g.colInd[srcStart], srcLen,
                &g.colInd[g.rowPtr[g.colInd[srcStart + j]]], g.rowPtr[g.colInd[srcStart + j] + 1] - g.rowPtr[g.colInd[srcStart + j]],
                &encode[j * num_divs_local]);
        }
        __syncthreads(); //Done encoding

        T* cl = &current_level[((sm_id * CBPSM + levelPtr) * numPartitions + wx) * NUMDIVS * 9];

        for (T j = wx; j < srcLen; j += numPartitions)
        {
            if (lx == 0)
            {
                level_count[wx][0] = 0;
                level_index[wx][0] = 0;
                level_prev_index[wx][0] = 0;
                l[wx] = 3;
                clique_count[wx] = 0;
            }

            uint64 warpCount = 0;
            for (unsigned long long k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask, warpCount);

            if (lx == 0)
            {
                if (KCCOUNT >= 3 && warpCount >= KCCOUNT - 2)
                {
                    level_count[wx][0] = warpCount;
                }
            }
            __syncwarp(partMask);

            while (level_count[wx][l[wx] - 3] > level_index[wx][l[wx] - 3])
            {
                //First Index
                T* from = l[wx] == 3 ? &(encode[num_divs_local * j]) : &(cl[num_divs_local * (l[wx] - 3)]);
                T maskBlock = level_prev_index[wx][l[wx] - 3] / 32;
                T maskIndex = ~((1 << (level_prev_index[wx][l[wx] - 3] & 0x1F)) -1);

                T newIndex = __ffs(from[maskBlock] & maskIndex);
                while(newIndex == 0)
                {
                    maskIndex = 0xFFFFFFFF;
                    maskBlock++;
                    newIndex = __ffs(from[maskBlock] & maskIndex);
                }
                newIndex =  32*maskBlock + newIndex - 1;

                if (lx == 0)
                {
                    level_prev_index[wx][l[wx] - 3] = newIndex + 1;
                    level_index[wx][l[wx] - 3]++;
                }
                __syncwarp(partMask);

                if (l[wx] < KCCOUNT)
                {
                    // Intersect
                    T* to = &(cl[num_divs_local * (l[wx] - 2)]);
                    warpCount = 0;
                    for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                    {
                        to[k] = from[k] & encode[newIndex * num_divs_local + k];
                        warpCount += __popc(to[k]);
                    }
                    reduce_part<T, CPARTSIZE>(partMask, warpCount);
                }

                // accumulate local counter
                if (l[wx] == KCCOUNT)
                {
                    for (T k = lx; k < KCCOUNT - 2; k += CPARTSIZE)
                    {
                        atomicAdd(&local_clique_count[level_prev_index[wx][k] - 1], 1);
                    }
                    __syncwarp(partMask);
                }

                if (lx == 0)
                {
                    if (l[wx] == KCCOUNT)
                        clique_count[wx] ++;
                    else if (l[wx] < KCCOUNT && warpCount >= KCCOUNT - l[wx])
                    {
                        (l[wx])++;
                        level_count[wx][l[wx] - 3] = warpCount;
                        level_index[wx][l[wx] - 3] = 0;
                        level_prev_index[wx][l[wx] - 3] = 0;
                    }
                
                    while (l[wx] > 3 && level_index[wx][l[wx] - 3] >= level_count[wx][l[wx] - 3])
                    {
                        (l[wx])--;
                    }
                }
                __syncwarp(partMask);
            }

            if (lx == 0)
            {
                atomicAdd(counter, clique_count[wx]);
                atomicAdd(&root_count, clique_count[wx]);
                atomicAdd(&local_clique_count[j], clique_count[wx]);
            }
            __syncwarp(partMask);
        }
        __syncthreads();

        for (unsigned int idx = threadIdx.x; idx < srcLen; idx += blockDim.x)
        {
            atomicAdd(&cpn[g.colInd[srcStart + idx]], local_clique_count[idx]);
        }
        if (threadIdx.x == 0)
        {
            atomicAdd(&cpn[src], root_count);
        }
        __syncthreads();
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}

/* AtomicAdd to GlobalMemory + Deep: Use atomicAdd to increase the local_clique counter in global memory,
 *                                   whenever we find a clique. This kernel is built on top of baseline_deep.
 * Extra shared memory: none
 * Difference from baseline_deep:
 * 		Whenever we find a clique:
 *			for (T k = lx; k < KCCOUNT - 2; k += CPARTSIZE)
 *			{
 *				atomicAdd(&cpn[g.colInd[srcStart + level_prev_index[wx][k] - 1]], 1);
 *			}
 *			__syncwarp(partMask);
 *		After exploring a branch from the source node:
 * 			atomicAdd(&cpn[current.queue[i]], clique_count[wx]);
 *			atomicAdd(&cpn[g.colInd[srcStart + j]], clique_count[wx]);
 */

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_block_warp_binary_count_local_globalmem_direct_deep(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;
    __shared__ unsigned short level_index[numPartitions][9];
    __shared__ unsigned short level_count[numPartitions][9];
    __shared__ unsigned short level_prev_index[numPartitions][9];

    __shared__ uint64 clique_count[numPartitions];
    __shared__ T l[numPartitions];
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen;

    __shared__ T num_divs_local, encode_offset, *encode;

    __syncthreads();

    if (threadIdx.x == 0)
    {
        sm_id = __mysmid();
        T temp = 0;
        while (atomicCAS(&(levelStats[(sm_id * CBPSM) + temp]), 0, 1) != 0)
        {
            temp++;
        }
        levelPtr = temp;
    }
    __syncthreads();

    for (unsigned long long i = blockIdx.x; i < (unsigned long long)current.count[0]; i += gridDim.x)
    {
        //block things
        if (threadIdx.x == 0)
        {
            src = current.queue[i];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;
        }
        __syncthreads();

        if (threadIdx.x == 0)
            num_divs_local = (srcLen + 32 - 1) / 32;
        else if (threadIdx.x == 1)
        {
            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];
        }
        __syncthreads();

        // Encode
        T partMask = (1 << CPARTSIZE) - 1;
        partMask = partMask << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
            __syncwarp(partMask);
            graph::warp_sorted_count_and_encode<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&g.colInd[srcStart], srcLen,
                &g.colInd[g.rowPtr[g.colInd[srcStart + j]]], g.rowPtr[g.colInd[srcStart + j] + 1] - g.rowPtr[g.colInd[srcStart + j]],
                &encode[j * num_divs_local]);
        }
        __syncthreads(); //Done encoding

        T* cl = &current_level[((sm_id * CBPSM + levelPtr) * numPartitions + wx) * NUMDIVS * 9];

        for (T j = wx; j < srcLen; j += numPartitions)
        {
            if (lx == 0)
            {
                level_count[wx][0] = 0;
                level_index[wx][0] = 0;
                level_prev_index[wx][0] = 0;
                l[wx] = 3;
                clique_count[wx] = 0;
            }

            uint64 warpCount = 0;
            for (unsigned long long k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask, warpCount);

            if (lx == 0)
            {
                if (KCCOUNT >= 3 && warpCount >= KCCOUNT - 2)
                {
                    level_count[wx][0] = warpCount;
                }
            }
            __syncwarp(partMask);

            while (level_count[wx][l[wx] - 3] > level_index[wx][l[wx] - 3])
            {
                //First Index
                T* from = l[wx] == 3 ? &(encode[num_divs_local * j]) : &(cl[num_divs_local * (l[wx] - 3)]);
                T maskBlock = level_prev_index[wx][l[wx] - 3] / 32;
                T maskIndex = ~((1 << (level_prev_index[wx][l[wx] - 3] & 0x1F)) -1);

                T newIndex = __ffs(from[maskBlock] & maskIndex);
                while(newIndex == 0)
                {
                    maskIndex = 0xFFFFFFFF;
                    maskBlock++;
                    newIndex = __ffs(from[maskBlock] & maskIndex);
                }
                newIndex =  32*maskBlock + newIndex - 1;

                if (lx == 0)
                {
                    level_prev_index[wx][l[wx] - 3] = newIndex + 1;
                    level_index[wx][l[wx] - 3]++;
                }
                __syncwarp(partMask);

                if (l[wx] < KCCOUNT)
                {
                    // Intersect
                    T* to = &(cl[num_divs_local * (l[wx] - 2)]);
                    warpCount = 0;
                    for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                    {
                        to[k] = from[k] & encode[newIndex * num_divs_local + k];
                        warpCount += __popc(to[k]);
                    }
                    reduce_part<T, CPARTSIZE>(partMask, warpCount);
                }

                // accumulate local counter
                if (l[wx] == KCCOUNT)
                {
                    for (T k = lx; k < KCCOUNT - 2; k += CPARTSIZE)
                    {
                        atomicAdd(&cpn[g.colInd[srcStart + level_prev_index[wx][k] - 1]], 1);
                    }
                    __syncwarp(partMask);
                }

                if (lx == 0)
                {
                    if (l[wx] == KCCOUNT)
                        clique_count[wx] ++;
                    else if (l[wx] < KCCOUNT && warpCount >= KCCOUNT - l[wx])
                    {
                        (l[wx])++;
                        level_count[wx][l[wx] - 3] = warpCount;
                        level_index[wx][l[wx] - 3] = 0;
                        level_prev_index[wx][l[wx] - 3] = 0;
                    }
                
                    while (l[wx] > 3 && level_index[wx][l[wx] - 3] >= level_count[wx][l[wx] - 3])
                    {
                        (l[wx])--;
                    }
                }
                __syncwarp(partMask);
            }

            if (lx == 0)
            {
                atomicAdd(counter, clique_count[wx]);
                atomicAdd(&cpn[current.queue[i]], clique_count[wx]);
                atomicAdd(&cpn[g.colInd[srcStart + j]], clique_count[wx]);
            }
            __syncwarp(partMask);
        }
        __syncthreads();
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}

/* Baseline + Deep
 * Intuition: In the original k-kclique counting kernel, we didn't search the k'th level,
 *            but stop at the (k - 1)'th level and accumulate the number of neighbors into the clique counter.
 *            This baseline_deep kernel reformulates the code to get the total clique count by getting into the k'th level,
 *            and increase the counted by 1 for each on-bit in the binary encoded list.
 * Expect:    The execution time of this kernel for k-clique should be similar as that
 *            of counting (k + 1)-clique in the original kernel. But should be a bit faster since we skip the last intersection 
 *            done in the original kernel. 
 */

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_block_warp_binary_count_local_baseline_deep(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;
    __shared__ unsigned short level_index[numPartitions][9];
    __shared__ unsigned short level_count[numPartitions][9];
    __shared__ unsigned short level_prev_index[numPartitions][9];

    __shared__ uint64 clique_count[numPartitions];
    __shared__ T l[numPartitions];
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen;

    __shared__ T num_divs_local, encode_offset, *encode;

    __syncthreads();

    if (threadIdx.x == 0)
    {
        sm_id = __mysmid();
        T temp = 0;
        while (atomicCAS(&(levelStats[(sm_id * CBPSM) + temp]), 0, 1) != 0)
        {
            temp++;
        }
        levelPtr = temp;
    }
    __syncthreads();

    for (unsigned long long i = blockIdx.x; i < (unsigned long long)current.count[0]; i += gridDim.x)
    {
        //block things
        if (threadIdx.x == 0)
        {
            src = current.queue[i];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;
        }
        __syncthreads();

        if (threadIdx.x == 0)
            num_divs_local = (srcLen + 32 - 1) / 32;
        else if (threadIdx.x == 1)
        {
            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];
        }
        __syncthreads();

        // Encode
        T partMask = (1 << CPARTSIZE) - 1;
        partMask = partMask << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
            __syncwarp(partMask);
            graph::warp_sorted_count_and_encode<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&g.colInd[srcStart], srcLen,
                &g.colInd[g.rowPtr[g.colInd[srcStart + j]]], g.rowPtr[g.colInd[srcStart + j] + 1] - g.rowPtr[g.colInd[srcStart + j]],
                &encode[j * num_divs_local]);
        }
        __syncthreads(); //Done encoding

        T* cl = &current_level[((sm_id * CBPSM + levelPtr) * numPartitions + wx) * NUMDIVS * 9];

        for (T j = wx; j < srcLen; j += numPartitions)
        {
            if (lx == 0)
            {
                level_count[wx][0] = 0;
                level_index[wx][0] = 0;
                level_prev_index[wx][0] = 0;
                l[wx] = 3;
                clique_count[wx] = 0;
            }

            uint64 warpCount = 0;
            for (unsigned long long k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask, warpCount);

            if (lx == 0)
            {
                if (KCCOUNT >= 3 && warpCount >= KCCOUNT - 2)
                {
                    level_count[wx][0] = warpCount;
                }
            }
            __syncwarp(partMask);

            while (level_count[wx][l[wx] - 3] > level_index[wx][l[wx] - 3])
            {
                //First Index
                T* from = l[wx] == 3 ? &(encode[num_divs_local * j]) : &(cl[num_divs_local * (l[wx] - 3)]);
                T maskBlock = level_prev_index[wx][l[wx] - 3] / 32;
                T maskIndex = ~((1 << (level_prev_index[wx][l[wx] - 3] & 0x1F)) -1);

                T newIndex = __ffs(from[maskBlock] & maskIndex);
                while(newIndex == 0)
                {
                    maskIndex = 0xFFFFFFFF;
                    maskBlock++;
                    newIndex = __ffs(from[maskBlock] & maskIndex);
                }
                newIndex =  32*maskBlock + newIndex - 1;

                if (lx == 0)
                {
                    level_prev_index[wx][l[wx] - 3] = newIndex + 1;
                    level_index[wx][l[wx] - 3]++;
                }
                __syncwarp(partMask);

                if (l[wx] < KCCOUNT)
                {
                    // Intersect
                    T* to = &(cl[num_divs_local * (l[wx] - 2)]);
                    warpCount = 0;
                    for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                    {
                        to[k] = from[k] & encode[newIndex * num_divs_local + k];
                        warpCount += __popc(to[k]);
                    }
                    reduce_part<T, CPARTSIZE>(partMask, warpCount);
                }

                if (lx == 0)
                {
                    if (l[wx] == KCCOUNT)
                        clique_count[wx] ++;
                    else if (l[wx] < KCCOUNT && warpCount >= KCCOUNT - l[wx])
                    {
                        (l[wx])++;
                        level_count[wx][l[wx] - 3] = warpCount;
                        level_index[wx][l[wx] - 3] = 0;
                        level_prev_index[wx][l[wx] - 3] = 0;
                    }
                
                    while (l[wx] > 3 && level_index[wx][l[wx] - 3] >= level_count[wx][l[wx] - 3])
                    {
                        (l[wx])--;
                    }
                }
                __syncwarp(partMask);
            }

            if (lx == 0)
            {
                atomicAdd(counter, clique_count[wx]);
            }
            __syncwarp(partMask);
        }
        __syncthreads();
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}

/* AtomicAdd to SharedMemory with accumulation through levels + Loop */
template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_block_warp_binary_count_local_sharedmem_lazy_loop(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;
    __shared__ unsigned short level_index[numPartitions][9];
    __shared__ unsigned short level_count[numPartitions][9];
    __shared__ unsigned short level_prev_index[numPartitions][9];

    __shared__ uint64 clique_count[numPartitions];
    __shared__ T l[numPartitions];
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen;

    __shared__ T num_divs_local, encode_offset, *encode;

    __shared__ uint64 local_clique_count[1024];
    __shared__ uint64 root_count;
    __shared__ uint64 level_local_clique_count[numPartitions][9];

    __syncthreads();

    if (threadIdx.x == 0)
    {
        sm_id = __mysmid();
        T temp = 0;
        while (atomicCAS(&(levelStats[(sm_id * CBPSM) + temp]), 0, 1) != 0)
        {
            temp++;
        }
        levelPtr = temp;
    }
    __syncthreads();

    for (unsigned long long i = blockIdx.x; i < (unsigned long long)current.count[0]; i += gridDim.x)
    {
        //block things
        if (threadIdx.x == 0)
        {
            src = current.queue[i];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;
        }
        __syncthreads();

        // Clear local_clique counter in shared memory
        for (unsigned int idx = threadIdx.x; idx < srcLen; idx += blockDim.x)
        {
            local_clique_count[idx] = 0;
        }
        if (threadIdx.x == 0)
        {
            root_count = 0;
        }
        __syncthreads();

        if (threadIdx.x == 0)
            num_divs_local = (srcLen + 32 - 1) / 32;
        else if (threadIdx.x == 1)
        {
            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];
        }
        __syncthreads();

        // Encode
        T partMask = (1 << CPARTSIZE) - 1;
        partMask = partMask << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
            __syncwarp(partMask);
            graph::warp_sorted_count_and_encode<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&g.colInd[srcStart], srcLen,
                &g.colInd[g.rowPtr[g.colInd[srcStart + j]]], g.rowPtr[g.colInd[srcStart + j] + 1] - g.rowPtr[g.colInd[srcStart + j]],
                &encode[j * num_divs_local]);
        }
        __syncthreads(); //Done encoding

        T* cl = &current_level[((sm_id * CBPSM + levelPtr) * numPartitions + wx) * NUMDIVS * 9];

        for (T j = wx; j < srcLen; j += numPartitions)
        {
            if (lx == 0)
            {
                level_count[wx][0] = 0;
                level_index[wx][0] = 0;
                level_prev_index[wx][0] = 0;
                l[wx] = 3;
                clique_count[wx] = 0;
                level_local_clique_count[wx][0] = 0;
            }

            uint64 warpCount = 0;
            for (unsigned long long k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask, warpCount);

            if (lx == 0)
            {
                if (l[wx] == KCCOUNT)
                {
                    clique_count[wx] += warpCount;
                }
                else if (KCCOUNT > 3 && warpCount >= KCCOUNT - 2)
                {
                    level_count[wx][0] = warpCount;
                }
            }
            __syncwarp(partMask);

            if (l[wx] == KCCOUNT)
            {
                for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                {
                    T cur = encode[j * num_divs_local + k], idx = 0;
                    while((idx = __ffs(cur)) != 0)
                    {
                        --idx;
                        cur ^= (1 << idx);
                        idx = 32 * k + idx;
                        atomicAdd(&local_clique_count[idx], 1);
                    }
                }
                __syncwarp(partMask);
            }

            while (level_count[wx][l[wx] - 3] > level_index[wx][l[wx] - 3])
            {
                //First Index
                T* from = l[wx] == 3 ? &(encode[num_divs_local * j]) : &(cl[num_divs_local * (l[wx] - 3)]);
                T* to = &(cl[num_divs_local * (l[wx] - 2)]);
                T maskBlock = level_prev_index[wx][l[wx] - 3] / 32;
                T maskIndex = ~((1 << (level_prev_index[wx][l[wx] - 3] & 0x1F)) -1);

                T newIndex = __ffs(from[maskBlock] & maskIndex);
                while(newIndex == 0)
                {
                    maskIndex = 0xFFFFFFFF;
                    maskBlock++;
                    newIndex = __ffs(from[maskBlock] & maskIndex);
                }
                newIndex =  32*maskBlock + newIndex - 1;

                if (lx == 0)
                {
                    level_prev_index[wx][l[wx] - 3] = newIndex + 1;
                    level_index[wx][l[wx] - 3]++;
                }

                warpCount = 0;
                for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                {
                    to[k] = from[k] & encode[newIndex * num_divs_local + k];
                    warpCount += __popc(to[k]);
                }
                reduce_part<T, CPARTSIZE>(partMask, warpCount);

                if (l[wx] + 1 == KCCOUNT)
                {
                    for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                    {
                        T cur = to[k], idx = 0;
                        while((idx = __ffs(cur)) != 0)
                        {
                            --idx;
                            cur ^= (1 << idx);
                            idx = 32 * k + idx;
                            atomicAdd(&local_clique_count[idx], 1);
                        }
                    }
                    __syncwarp(partMask);
                }

                if (lx == 0)
                {
                    if (l[wx] + 1 == KCCOUNT)
                    {
                        clique_count[wx] += warpCount;
                        atomicAdd(&local_clique_count[newIndex], warpCount);
                        level_local_clique_count[wx][l[wx] - 3] += warpCount;
                    }
                    else if (l[wx] + 1 < KCCOUNT && warpCount >= KCCOUNT - l[wx])
                    {
                        (l[wx])++;
                        level_count[wx][l[wx] - 3] = warpCount;
                        level_index[wx][l[wx] - 3] = 0;
                        level_prev_index[wx][l[wx] - 3] = 0;
                        level_local_clique_count[wx][l[wx] - 3] = 0;
                    }
                
                    while (l[wx] > 3 && level_index[wx][l[wx] - 3] >= level_count[wx][l[wx] - 3])
                    {
                        uint64 cur = level_local_clique_count[wx][l[wx] - 3];
                        level_local_clique_count[wx][l[wx] - 4] += cur;
                        atomicAdd(&local_clique_count[level_prev_index[wx][l[wx] - 4] - 1], cur);
                        (l[wx])--;
                    }
                }
                __syncwarp(partMask);
            }

            if (lx == 0)
            {
                atomicAdd(counter, clique_count[wx]);
                atomicAdd(&root_count, clique_count[wx]);
                atomicAdd(&local_clique_count[j], clique_count[wx]);
            }
            __syncwarp(partMask);
        }
        __syncthreads();

        for (unsigned int idx = threadIdx.x; idx < srcLen; idx += blockDim.x)
        {
            atomicAdd(&cpn[g.colInd[srcStart + idx]], local_clique_count[idx]);
        }
        if (threadIdx.x == 0)
        {
            atomicAdd(&cpn[src], root_count);
        }
        __syncthreads();
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}

/* AtomicAdd to GlobalMemory with accumulation through levels + Loop */
template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_block_warp_binary_count_local_globalmem_lazy_loop(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;
    __shared__ unsigned short level_index[numPartitions][9];
    __shared__ unsigned short level_count[numPartitions][9];
    __shared__ unsigned short level_prev_index[numPartitions][9];

    __shared__ uint64 clique_count[numPartitions];
    __shared__ T l[numPartitions];
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen;

    __shared__ T num_divs_local, encode_offset, *encode;

    __shared__ uint64 level_local_clique_count[numPartitions][9];

    __syncthreads();

    if (threadIdx.x == 0)
    {
        sm_id = __mysmid();
        T temp = 0;
        while (atomicCAS(&(levelStats[(sm_id * CBPSM) + temp]), 0, 1) != 0)
        {
            temp++;
        }
        levelPtr = temp;
    }
    __syncthreads();

    for (unsigned long long i = blockIdx.x; i < (unsigned long long)current.count[0]; i += gridDim.x)
    {
        //block things
        if (threadIdx.x == 0)
        {
            src = current.queue[i];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;
        }
        __syncthreads();

        if (threadIdx.x == 0)
            num_divs_local = (srcLen + 32 - 1) / 32;
        else if (threadIdx.x == 1)
        {
            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];
        }
        __syncthreads();

        // Encode
        T partMask = (1 << CPARTSIZE) - 1;
        partMask = partMask << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
            __syncwarp(partMask);
            graph::warp_sorted_count_and_encode<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&g.colInd[srcStart], srcLen,
                &g.colInd[g.rowPtr[g.colInd[srcStart + j]]], g.rowPtr[g.colInd[srcStart + j] + 1] - g.rowPtr[g.colInd[srcStart + j]],
                &encode[j * num_divs_local]);
        }
        __syncthreads(); //Done encoding

        T* cl = &current_level[((sm_id * CBPSM + levelPtr) * numPartitions + wx) * NUMDIVS * 9];

        for (T j = wx; j < srcLen; j += numPartitions)
        {
            if (lx == 0)
            {
                level_count[wx][0] = 0;
                level_index[wx][0] = 0;
                level_prev_index[wx][0] = 0;
                l[wx] = 3;
                clique_count[wx] = 0;
                level_local_clique_count[wx][0] = 0;
            }

            uint64 warpCount = 0;
            for (unsigned long long k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask, warpCount);

            if (lx == 0)
            {
                if (l[wx] == KCCOUNT)
                {
                    clique_count[wx] += warpCount;
                }
                else if (KCCOUNT > 3 && warpCount >= KCCOUNT - 2)
                {
                    level_count[wx][0] = warpCount;
                }
            }
            __syncwarp(partMask);

            if (l[wx] == KCCOUNT)
            {
                for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                {
                    T cur = encode[j * num_divs_local + k], idx = 0;
                    while((idx = __ffs(cur)) != 0)
                    {
                        --idx;
                        cur ^= (1 << idx);
                        idx = 32 * k + idx;
                        atomicAdd(&cpn[g.colInd[srcStart + idx]], 1);
                    }
                }
                __syncwarp(partMask);
            }

            while (level_count[wx][l[wx] - 3] > level_index[wx][l[wx] - 3])
            {
                //First Index
                T* from = l[wx] == 3 ? &(encode[num_divs_local * j]) : &(cl[num_divs_local * (l[wx] - 3)]);
                T* to = &(cl[num_divs_local * (l[wx] - 2)]);
                T maskBlock = level_prev_index[wx][l[wx] - 3] / 32;
                T maskIndex = ~((1 << (level_prev_index[wx][l[wx] - 3] & 0x1F)) -1);

                T newIndex = __ffs(from[maskBlock] & maskIndex);
                while(newIndex == 0)
                {
                    maskIndex = 0xFFFFFFFF;
                    maskBlock++;
                    newIndex = __ffs(from[maskBlock] & maskIndex);
                }
                newIndex =  32*maskBlock + newIndex - 1;

                if (lx == 0)
                {
                    level_prev_index[wx][l[wx] - 3] = newIndex + 1;
                    level_index[wx][l[wx] - 3]++;
                }

                warpCount = 0;
                for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                {
                    to[k] = from[k] & encode[newIndex * num_divs_local + k];
                    warpCount += __popc(to[k]);
                }
                reduce_part<T, CPARTSIZE>(partMask, warpCount);

                if (l[wx] + 1 == KCCOUNT)
                {
                    for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                    {
                        T cur = to[k], idx = 0;
                        while((idx = __ffs(cur)) != 0)
                        {
                            --idx;
                            cur ^= (1 << idx);
                            idx = 32 * k + idx;
                            atomicAdd(&cpn[g.colInd[srcStart + idx]], 1);
                        }
                    }
                    __syncwarp(partMask);
                }

                if (lx == 0)
                {
                    if (l[wx] + 1 == KCCOUNT)
                    {
                        clique_count[wx] += warpCount;
                        atomicAdd(&cpn[g.colInd[srcStart + newIndex]], warpCount);
                        level_local_clique_count[wx][l[wx] - 3] += warpCount;
                    }
                    else if (l[wx] + 1 < KCCOUNT && warpCount >= KCCOUNT - l[wx])
                    {
                        (l[wx])++;
                        level_count[wx][l[wx] - 3] = warpCount;
                        level_index[wx][l[wx] - 3] = 0;
                        level_prev_index[wx][l[wx] - 3] = 0;
                        level_local_clique_count[wx][l[wx] - 3] = 0;
                    }
                
                    while (l[wx] > 3 && level_index[wx][l[wx] - 3] >= level_count[wx][l[wx] - 3])
                    {
                        uint64 cur = level_local_clique_count[wx][l[wx] - 3];
                        level_local_clique_count[wx][l[wx] - 4] += cur;
                        atomicAdd(&cpn[g.colInd[srcStart + level_prev_index[wx][l[wx] - 4] - 1]], cur);
                        (l[wx])--;
                    }
                }
                __syncwarp(partMask);
            }

            if (lx == 0)
            {
                atomicAdd(counter, clique_count[wx]);
                atomicAdd(&cpn[current.queue[i]], clique_count[wx]);
                atomicAdd(&cpn[g.colInd[srcStart + j]], clique_count[wx]);
            }
            __syncwarp(partMask);
        }
        __syncthreads();
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}

/* AtomicAdd to SharedMemory Direct + Loop */
template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_block_warp_binary_count_local_sharedmem_direct_loop(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;
    __shared__ unsigned short level_index[numPartitions][9];
    __shared__ unsigned short level_count[numPartitions][9];
    __shared__ unsigned short level_prev_index[numPartitions][9];

    __shared__ uint64 clique_count[numPartitions];
    __shared__ T l[numPartitions];
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen;

    __shared__ T num_divs_local, encode_offset, *encode;
    __shared__ uint64 shared_warp_count[numPartitions];
    __shared__ uint64 local_clique_count[1024];
    __shared__ uint64 root_count;
    __syncthreads();

    if (threadIdx.x == 0)
    {
        sm_id = __mysmid();
        T temp = 0;
        while (atomicCAS(&(levelStats[(sm_id * CBPSM) + temp]), 0, 1) != 0)
        {
            temp++;
        }
        levelPtr = temp;
    }
    __syncthreads();

    for (unsigned long long i = blockIdx.x; i < (unsigned long long)current.count[0]; i += gridDim.x)
    {
        //block things
        if (threadIdx.x == 0)
        {
            src = current.queue[i];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;
        }
        __syncthreads();

        // Clear local_clique counter in shared memory
        for (unsigned int idx = threadIdx.x; idx < srcLen; idx += blockDim.x)
        {
            local_clique_count[idx] = 0;
        }
        if (threadIdx.x == 0)
        {
            root_count = 0;
        }
        __syncthreads();

        if (threadIdx.x == 0)
            num_divs_local = (srcLen + 32 - 1) / 32;
        else if (threadIdx.x == 1)
        {
            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];
        }
        __syncthreads();

        // Encode
        T partMask = (1 << CPARTSIZE) - 1;
        partMask = partMask << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
            __syncwarp(partMask);
            graph::warp_sorted_count_and_encode<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&g.colInd[srcStart], srcLen,
                &g.colInd[g.rowPtr[g.colInd[srcStart + j]]], g.rowPtr[g.colInd[srcStart + j] + 1] - g.rowPtr[g.colInd[srcStart + j]],
                &encode[j * num_divs_local]);
        }
        __syncthreads(); //Done encoding

        T* cl = &current_level[((sm_id * CBPSM + levelPtr) * numPartitions + wx) * NUMDIVS * 9];

        for (T j = wx; j < srcLen; j += numPartitions)
        {
            if (lx == 0)
            {
                level_count[wx][0] = 0;
                level_index[wx][0] = 0;
                level_prev_index[wx][0] = 0;
                l[wx] = 3;
                clique_count[wx] = 0;
            }

            uint64 warpCount = 0;
            for (unsigned long long k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask, warpCount);

            if (lx == 0)
            {
                if (l[wx] == KCCOUNT)
                {
                    clique_count[wx] += warpCount;
                }
                else if (KCCOUNT > 3 && warpCount >= KCCOUNT - 2)
                {
                    level_count[wx][0] = warpCount;
                }
            }
            __syncwarp(partMask);

            if (l[wx] == KCCOUNT)
            {
                for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                {
                    T cur = encode[j * num_divs_local + k], idx = 0;
                    while((idx = __ffs(cur)) != 0)
                    {
                        --idx;
                        cur ^= (1 << idx);
                        idx = 32 * k + idx;
                        atomicAdd(&local_clique_count[idx], 1);
                    }
                }
                __syncwarp(partMask);
            }

            while (level_count[wx][l[wx] - 3] > level_index[wx][l[wx] - 3])
            {
                //First Index
                T* from = l[wx] == 3 ? &(encode[num_divs_local * j]) : &(cl[num_divs_local * (l[wx] - 3)]);
                T* to = &(cl[num_divs_local * (l[wx] - 2)]);
                T maskBlock = level_prev_index[wx][l[wx] - 3] / 32;
                T maskIndex = ~((1 << (level_prev_index[wx][l[wx] - 3] & 0x1F)) -1);

                T newIndex = __ffs(from[maskBlock] & maskIndex);
                while(newIndex == 0)
                {
                    maskIndex = 0xFFFFFFFF;
                    maskBlock++;
                    newIndex = __ffs(from[maskBlock] & maskIndex);
                }
                newIndex =  32*maskBlock + newIndex - 1;

                if (lx == 0)
                {
                    level_prev_index[wx][l[wx] - 3] = newIndex + 1;
                    level_index[wx][l[wx] - 3]++;
                }

                warpCount = 0;
                for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                {
                    to[k] = from[k] & encode[newIndex * num_divs_local + k];
                    warpCount += __popc(to[k]);
                }
                reduce_part<T, CPARTSIZE>(partMask, warpCount);

                if (lx == 0)
                {
                    shared_warp_count[wx] = warpCount;
                }
                __syncwarp(partMask);

                if (l[wx] + 1 == KCCOUNT && shared_warp_count[wx] != 0)
                {
                    for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                    {
                        T cur = to[k], idx = 0;
                        while((idx = __ffs(cur)) != 0)
                        {
                            --idx;
                            cur ^= (1 << idx);
                            idx = 32 * k + idx;
                            atomicAdd(&local_clique_count[idx], 1);
                        }
                    }
                    for (T k = lx; k < KCCOUNT - 3; k += CPARTSIZE)
                    {
                        atomicAdd(&local_clique_count[level_prev_index[wx][k] - 1], shared_warp_count[wx]);
                    }
                    __syncwarp(partMask);
                }

                if (lx == 0)
                {
                    if (l[wx] + 1 == KCCOUNT)
                    {
                        clique_count[wx] += warpCount;
                    }
                    else if (l[wx] + 1 < KCCOUNT && warpCount >= KCCOUNT - l[wx])
                    {
                        (l[wx])++;
                        level_count[wx][l[wx] - 3] = warpCount;
                        level_index[wx][l[wx] - 3] = 0;
                        level_prev_index[wx][l[wx] - 3] = 0;
                    }
                
                    while (l[wx] > 3 && level_index[wx][l[wx] - 3] >= level_count[wx][l[wx] - 3])
                    {
                        (l[wx])--;
                    }
                }
                __syncwarp(partMask);
            }

            if (lx == 0)
            {
                atomicAdd(counter, clique_count[wx]);
                atomicAdd(&root_count, clique_count[wx]);
                atomicAdd(&local_clique_count[j], clique_count[wx]);
            }
            __syncwarp(partMask);
        }
        __syncthreads();

        for (unsigned int idx = threadIdx.x; idx < srcLen; idx += blockDim.x)
        {
            atomicAdd(&cpn[g.colInd[srcStart + idx]], local_clique_count[idx]);
        }
        if (threadIdx.x == 0)
        {
            atomicAdd(&cpn[src], root_count);
        }
        __syncthreads();
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}

/* AtomicAdd to GlobalMemory Direct + Loop */
template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_block_warp_binary_count_local_globalmem_direct_loop(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;
    __shared__ unsigned short level_index[numPartitions][9];
    __shared__ unsigned short level_count[numPartitions][9];
    __shared__ unsigned short level_prev_index[numPartitions][9];

    __shared__ uint64 clique_count[numPartitions];
    __shared__ T l[numPartitions];
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen;

    __shared__ T num_divs_local, encode_offset, *encode;
    __shared__ uint64 shared_warp_count[numPartitions];
    __syncthreads();

    if (threadIdx.x == 0)
    {
        sm_id = __mysmid();
        T temp = 0;
        while (atomicCAS(&(levelStats[(sm_id * CBPSM) + temp]), 0, 1) != 0)
        {
            temp++;
        }
        levelPtr = temp;
    }
    __syncthreads();

    for (unsigned long long i = blockIdx.x; i < (unsigned long long)current.count[0]; i += gridDim.x)
    {
        //block things
        if (threadIdx.x == 0)
        {
            src = current.queue[i];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;
        }
        __syncthreads();

        if (threadIdx.x == 0)
            num_divs_local = (srcLen + 32 - 1) / 32;
        else if (threadIdx.x == 1)
        {
            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];
        }
        __syncthreads();

        // Encode
        T partMask = (1 << CPARTSIZE) - 1;
        partMask = partMask << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
            __syncwarp(partMask);
            graph::warp_sorted_count_and_encode<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&g.colInd[srcStart], srcLen,
                &g.colInd[g.rowPtr[g.colInd[srcStart + j]]], g.rowPtr[g.colInd[srcStart + j] + 1] - g.rowPtr[g.colInd[srcStart + j]],
                &encode[j * num_divs_local]);
        }
        __syncthreads(); //Done encoding

        T* cl = &current_level[((sm_id * CBPSM + levelPtr) * numPartitions + wx) * NUMDIVS * 9];

        for (T j = wx; j < srcLen; j += numPartitions)
        {
            if (lx == 0)
            {
                level_count[wx][0] = 0;
                level_index[wx][0] = 0;
                level_prev_index[wx][0] = 0;
                l[wx] = 3;
                clique_count[wx] = 0;
            }

            uint64 warpCount = 0;
            for (unsigned long long k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask, warpCount);

            if (lx == 0)
            {
                if (l[wx] == KCCOUNT)
                {
                    clique_count[wx] += warpCount;
                }
                else if (KCCOUNT > 3 && warpCount >= KCCOUNT - 2)
                {
                    level_count[wx][0] = warpCount;
                }
            }
            __syncwarp(partMask);

            if (l[wx] == KCCOUNT)
            {
                for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                {
                    T cur = encode[j * num_divs_local + k], idx = 0;
                    while((idx = __ffs(cur)) != 0)
                    {
                        --idx;
                        cur ^= (1 << idx);
                        idx = 32 * k + idx;
                        atomicAdd(&cpn[g.colInd[srcStart + idx]], 1);
                    }
                }
                __syncwarp(partMask);
            }

            while (level_count[wx][l[wx] - 3] > level_index[wx][l[wx] - 3])
            {
                //First Index
                T* from = l[wx] == 3 ? &(encode[num_divs_local * j]) : &(cl[num_divs_local * (l[wx] - 3)]);
                T* to = &(cl[num_divs_local * (l[wx] - 2)]);
                T maskBlock = level_prev_index[wx][l[wx] - 3] / 32;
                T maskIndex = ~((1 << (level_prev_index[wx][l[wx] - 3] & 0x1F)) -1);

                T newIndex = __ffs(from[maskBlock] & maskIndex);
                while(newIndex == 0)
                {
                    maskIndex = 0xFFFFFFFF;
                    maskBlock++;
                    newIndex = __ffs(from[maskBlock] & maskIndex);
                }
                newIndex =  32*maskBlock + newIndex - 1;

                if (lx == 0)
                {
                    level_prev_index[wx][l[wx] - 3] = newIndex + 1;
                    level_index[wx][l[wx] - 3]++;
                }

                warpCount = 0;
                for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                {
                    to[k] = from[k] & encode[newIndex * num_divs_local + k];
                    warpCount += __popc(to[k]);
                }
                reduce_part<T, CPARTSIZE>(partMask, warpCount);

                if (lx == 0)
                {
                    shared_warp_count[wx] = warpCount;
                }
                __syncwarp(partMask);

                if (l[wx] + 1 == KCCOUNT && shared_warp_count[wx] != 0)
                {
                    for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                    {
                        T cur = to[k], idx = 0;
                        while((idx = __ffs(cur)) != 0)
                        {
                            --idx;
                            cur ^= (1 << idx);
                            idx = 32 * k + idx;
                            atomicAdd(&cpn[g.colInd[srcStart + idx]], 1);
                        }
                    }
                    for (T k = lx; k < KCCOUNT - 3; k += CPARTSIZE)
                    {
                        atomicAdd(&cpn[g.colInd[srcStart + level_prev_index[wx][k] - 1]], shared_warp_count[wx]);
                    }
                    __syncwarp(partMask);
                }

                if (lx == 0)
                {
                    if (l[wx] + 1 == KCCOUNT)
                    {
                        clique_count[wx] += warpCount;
                    }
                    else if (l[wx] + 1 < KCCOUNT && warpCount >= KCCOUNT - l[wx])
                    {
                        (l[wx])++;
                        level_count[wx][l[wx] - 3] = warpCount;
                        level_index[wx][l[wx] - 3] = 0;
                        level_prev_index[wx][l[wx] - 3] = 0;
                    }
                
                    while (l[wx] > 3 && level_index[wx][l[wx] - 3] >= level_count[wx][l[wx] - 3])
                    {
                        (l[wx])--;
                    }
                }
                __syncwarp(partMask);
            }

            if (lx == 0)
            {
                atomicAdd(counter, clique_count[wx]);
                atomicAdd(&cpn[current.queue[i]], clique_count[wx]);
                atomicAdd(&cpn[g.colInd[srcStart + j]], clique_count[wx]);
            }
            __syncwarp(partMask);
        }
        __syncthreads();
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}


/* Baseline + Loop */
template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_block_warp_binary_count_local_baseline_loop(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;
    __shared__ unsigned short level_index[numPartitions][9];
    __shared__ unsigned short level_count[numPartitions][9];
    __shared__ unsigned short level_prev_index[numPartitions][9];

    __shared__ uint64 clique_count[numPartitions];
    __shared__ T l[numPartitions];
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen;

    __shared__ T num_divs_local, encode_offset, *encode;

    __syncthreads();

    if (threadIdx.x == 0)
    {
        sm_id = __mysmid();
        T temp = 0;
        while (atomicCAS(&(levelStats[(sm_id * CBPSM) + temp]), 0, 1) != 0)
        {
            temp++;
        }
        levelPtr = temp;
    }
    __syncthreads();

    for (unsigned long long i = blockIdx.x; i < (unsigned long long)current.count[0]; i += gridDim.x)
    {
        //block things
        if (threadIdx.x == 0)
        {
            src = current.queue[i];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;
        }
        __syncthreads();

        if (threadIdx.x == 0)
            num_divs_local = (srcLen + 32 - 1) / 32;
        else if (threadIdx.x == 1)
        {
            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];
        }
        __syncthreads();

        // Encode
        T partMask = (1 << CPARTSIZE) - 1;
        partMask = partMask << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
            __syncwarp(partMask);
            graph::warp_sorted_count_and_encode<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&g.colInd[srcStart], srcLen,
                &g.colInd[g.rowPtr[g.colInd[srcStart + j]]], g.rowPtr[g.colInd[srcStart + j] + 1] - g.rowPtr[g.colInd[srcStart + j]],
                &encode[j * num_divs_local]);
        }
        __syncthreads(); //Done encoding

        T* cl = &current_level[((sm_id * CBPSM + levelPtr) * numPartitions + wx) * NUMDIVS * 9];

        for (T j = wx; j < srcLen; j += numPartitions)
        {
            if (lx == 0)
            {
                level_count[wx][0] = 0;
                level_index[wx][0] = 0;
                level_prev_index[wx][0] = 0;
                l[wx] = 3;
                clique_count[wx] = 0;
            }

            uint64 warpCount = 0;
            for (unsigned long long k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask, warpCount);

            if (lx == 0)
            {
                if (KCCOUNT > 3 && warpCount >= KCCOUNT - 2)
                {
                    level_count[wx][0] = warpCount;
                }
            }
            __syncwarp(partMask);

            if (l[wx] == KCCOUNT)
            {
                for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                {
                    T cur = encode[j * num_divs_local + k], idx = 0;
                    while((idx = __ffs(cur)) != 0)
                    {
                        --idx;
                        cur ^= (1 << idx);
                        idx = 32 * k + idx;
                        atomicAdd(&clique_count[wx], 1);
                    }
                }
                __syncwarp(partMask);
            }

            while (level_count[wx][l[wx] - 3] > level_index[wx][l[wx] - 3])
            {
                //First Index
                T* from = l[wx] == 3 ? &(encode[num_divs_local * j]) : &(cl[num_divs_local * (l[wx] - 3)]);
                T* to = &(cl[num_divs_local * (l[wx] - 2)]);
                T maskBlock = level_prev_index[wx][l[wx] - 3] / 32;
                T maskIndex = ~((1 << (level_prev_index[wx][l[wx] - 3] & 0x1F)) -1);

                T newIndex = __ffs(from[maskBlock] & maskIndex);
                while(newIndex == 0)
                {
                    maskIndex = 0xFFFFFFFF;
                    maskBlock++;
                    newIndex = __ffs(from[maskBlock] & maskIndex);
                }
                newIndex =  32*maskBlock + newIndex - 1;

                if (lx == 0)
                {
                    level_prev_index[wx][l[wx] - 3] = newIndex + 1;
                    level_index[wx][l[wx] - 3]++;
                }

                warpCount = 0;
                for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                {
                    to[k] = from[k] & encode[newIndex * num_divs_local + k];
                    warpCount += __popc(to[k]);
                }
                reduce_part<T, CPARTSIZE>(partMask, warpCount);

                if (l[wx] + 1 == KCCOUNT)
                {
                    for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                    {
                        T cur = to[k], idx = 0;
                        while((idx = __ffs(cur)) != 0)
                        {
                            --idx;
                            cur ^= (1 << idx);
                            idx = 32 * k + idx;
                            atomicAdd(&clique_count[wx], 1);
                        }
                    }
                    __syncwarp(partMask);
                }

                if (lx == 0)
                {
                    if (l[wx] + 1 == KCCOUNT)
                    {

                    }
                    else if (l[wx] + 1 < KCCOUNT && warpCount >= KCCOUNT - l[wx])
                    {
                        (l[wx])++;
                        level_count[wx][l[wx] - 3] = warpCount;
                        level_index[wx][l[wx] - 3] = 0;
                        level_prev_index[wx][l[wx] - 3] = 0;
                    }
                
                    while (l[wx] > 3 && level_index[wx][l[wx] - 3] >= level_count[wx][l[wx] - 3])
                    {
                        (l[wx])--;
                    }
                }
                __syncwarp(partMask);
            }

            if (lx == 0)
            {
                atomicAdd(counter, clique_count[wx]);
            }
            __syncwarp(partMask);
        }
        __syncthreads();
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}