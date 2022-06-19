#pragma once
#include "include/utils.cuh"
#include "config.cuh"

enum CounterName
{
    OTHER = 0,
    ALLOCATE,
    INITIALIZE,
    ENCODE,
    REDUCTION,
    INTERSECT,
    TRAVERSE,
    TOTAL,
    NUM_COUNTERS
};
char *Names[] =
    {
        "OTHER",
        "ALLOCATE",
        "INITIALIZE",
        "ENCODE",
        "REDUCTION",
        "INTERSECT",
        "TRAVERSE",
        "TOTAL",
        "NUM_COUNTERS"};

struct Counters
{
    unsigned long long tmp[NUM_COUNTERS];
    unsigned long long totalTime[NUM_COUNTERS];
};

template <typename T>
static __device__ void initializeCounters(Counters *counters, const size_t lx, const T mask)
{
    __syncwarp(mask);
    if (lx == 0)
    {
        for (unsigned int i = 0; i < NUM_COUNTERS; ++i)
        {
            counters->totalTime[i] = 0;
        }
    }
    __syncwarp(mask);
}

template <typename T>
static __device__ void startTime(CounterName counterName, Counters *counters, const size_t lx, const T mask)
{
    __syncwarp(mask);
    if (lx == 0)
    {
        counters->tmp[counterName] = clock64();
    }
    __syncwarp(mask);
}

template <typename T>
static __device__ void endTime(CounterName counterName, Counters *counters, const size_t lx, const T mask)
{
    __syncwarp(mask);
    if (lx == 0)
    {
        counters->totalTime[counterName] += clock64() - counters->tmp[counterName];
    }
    __syncwarp(mask);
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__global__ void sgm_kernel_compute_encoding(
    const graph::COOCSRGraph_d<T> g,
    const graph::GraphQueue_d<T, bool> current, const T head,
    T *adj_enc, int *offset, Counters *SM_times, uint64 *SM_edges)
{
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;

    __shared__ Counters times[numPartitions];
    T partMask = (1 << CPARTSIZE) - 1;
    partMask = partMask << ((wx % (32 / CPARTSIZE)) * CPARTSIZE);

    initializeCounters<T>(&times[wx], lx, partMask);
    startTime<T>(TOTAL, &times[wx], lx, partMask);

    __shared__ T src, srcStart, srcLen;
    __shared__ T num_divs_local, *encode, *orient_mask;
    const T i = blockIdx.x;
    if (threadIdx.x == 0)
    {
        src = current.queue[i + head];
        srcStart = g.rowPtr[src];
        srcLen = g.rowPtr[src + 1] - srcStart;

        num_divs_local = (srcLen + 32 - 1) / 32;

        offset[src] = (int)i;

        uint64 encode_offset = (uint64)i * NUMDIVS * MAXDEG;
        encode = &adj_enc[encode_offset];
    }
    __syncthreads();

    for (T j = wx; j < srcLen; j += numPartitions)
    {
        for (T k = lx; k < num_divs_local; k += CPARTSIZE)
        {
            encode[j * num_divs_local + k] = 0x00;
        }
        __syncwarp(partMask);

        graph::warp_sorted_count_and_encode_full<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&g.oriented_colInd[srcStart], srcLen,
                                                                                      &g.colInd[g.rowPtr[g.oriented_colInd[srcStart + j]]],
                                                                                      g.rowPtr[g.oriented_colInd[srcStart + j] + 1] - g.rowPtr[g.oriented_colInd[srcStart + j]],
                                                                                      j, num_divs_local,
                                                                                      encode);
    }

    endTime<T>(TOTAL, &times[wx], lx, partMask);
    if (lx == 0)
    {
        for (int i = 0; i < NUM_COUNTERS; i++)
            atomicAdd(&SM_times[__mysmid()].totalTime[i], times[wx].totalTime[i]);
    }
    if (threadIdx.x == 0)
    {
        atomicAdd(&SM_edges[__mysmid()], 1);
    }
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__global__ void sgm_kernel_pre_encoded_byEdge(
    uint64 *counter,
    const graph::COOCSRGraph_d<T> g,
    const mapping<T> *srcList, const T head,
    T *current_level,
    T *levelStats,
    T *adj_enc, T *node_count,
    int *offset, Counters *SM_times, uint64 *SM_edges)
{
    __shared__ uint32_t sm_id, levelPtr;

    if (threadIdx.x == 0)
    {
        sm_id = __mysmid();
        levelPtr = 0;
        while (atomicCAS(&(levelStats[(sm_id * CBPSM) + levelPtr]), 0, 1) != 0)
        {
            levelPtr++;
        }
    }
    __syncthreads();

    // will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;
    __shared__ Counters times[numPartitions];
    T partMask = (1 << CPARTSIZE) - 1;
    partMask = partMask << ((wx % (32 / CPARTSIZE)) * CPARTSIZE);

    initializeCounters<T>(&times[wx], lx, partMask);
    startTime<T>(TOTAL, &times[wx], lx, partMask);

    startTime<T>(ALLOCATE, &times[wx], lx, partMask);

    __shared__ T level_index[numPartitions][DEPTH];
    __shared__ T level_count[numPartitions][DEPTH];
    __shared__ T level_prev_index[numPartitions][DEPTH];
    __shared__ uint64 sg_count[numPartitions];
    __shared__ T l[numPartitions];
    __shared__ T src, srcStart, srcLen, srcSplit, dst, dstStart, dstLen, dstIdx;
    __shared__ T num_divs_local, *level_offset, *encode;
    __shared__ T to[BLOCK_DIM_X], newIndex[numPartitions], of[numPartitions];
    endTime<T>(ALLOCATE, &times[wx], lx, partMask);

    for (unsigned long long i = blockIdx.x; i < (unsigned long long)gridDim.x; i += gridDim.x)
    {
        startTime<T>(INITIALIZE, &times[wx], lx, partMask);
        // block things
        if (threadIdx.x == 0)
        {
            src = srcList[i + head].src;
            srcStart = g.rowPtr[src];
            srcSplit = g.splitPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;

            dstIdx = i + head - srcList[i + head].srcHead;
            dst = g.oriented_colInd[srcStart + dstIdx];
            dstStart = g.rowPtr[dst];
            dstLen = g.rowPtr[dst + 1] - dstStart;

            num_divs_local = (srcLen + 32 - 1) / 32;
            encode = &adj_enc[(uint64)offset[src] * NUMDIVS * MAXDEG];
            level_offset = &current_level[(uint64)((sm_id * CBPSM) + levelPtr) * (NUMDIVS * numPartitions * MAXLEVEL)];
        }
        if (lx == 0)
        {
            sg_count[wx] = 0;
        }
        __syncthreads();
        endTime<T>(INITIALIZE, &times[wx], lx, partMask);
        startTime<T>(TRAVERSE, &times[wx], lx, partMask);
        if (SYMNODE_PTR[2] == 1 && dstIdx < (srcSplit - srcStart))
        {
            continue;
        }

        endTime<T>(TRAVERSE, &times[wx], lx, partMask);

        // Compute triangles
        startTime<T>(INTERSECT, &times[wx], lx, partMask);
        uint64 warpCount = 0;
        for (T k = threadIdx.x; k < num_divs_local; k += BLOCK_DIM_X)
        {
            level_offset[k] = get_mask(srcLen, k) & unset_mask(dstIdx, k);

            if (QEDGE_PTR[3] - QEDGE_PTR[2] == 2)
                to[threadIdx.x] = encode[dstIdx * num_divs_local + k];
            else
                to[threadIdx.x] = level_offset[k];

            // Remove Redundancies
            for (T sym_idx = SYMNODE_PTR[2]; sym_idx < SYMNODE_PTR[3]; sym_idx++)
            {
                if (SYMNODE[sym_idx] > 0)
                    to[threadIdx.x] &= ~(level_offset[k] & get_mask(dstIdx, k));
            }
            level_offset[num_divs_local + k] = to[threadIdx.x];
            warpCount += __popc(to[threadIdx.x]);
        }
        endTime<T>(INTERSECT, &times[wx], lx, partMask);

        startTime<T>(REDUCTION, &times[wx], lx, partMask);
        typedef cub::BlockReduce<uint64, BLOCK_DIM_X> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        __syncthreads();
        warpCount = BlockReduce(temp_storage).Sum(warpCount); // whole block performs reduction here
        endTime<T>(REDUCTION, &times[wx], lx, partMask);

        startTime<T>(OTHER, &times[wx], lx, partMask);
        if (KCCOUNT == 3)
        {
            if (threadIdx.x == 0)
                atomicAdd(counter, warpCount);
            continue;
        }
        __syncthreads();
        endTime<T>(OTHER, &times[wx], lx, partMask);

        for (T j = wx; j < srcLen; j += numPartitions)
        {
            startTime<T>(TRAVERSE, &times[wx], lx, partMask);
            if ((level_offset[num_divs_local + j / 32] >> (j % 32)) % 2 == 0)
                continue;
            T *cl = level_offset + wx * (NUMDIVS * MAXLEVEL);
            endTime<T>(TRAVERSE, &times[wx], lx, partMask);

            startTime<T>(INITIALIZE, &times[wx], lx, partMask);
            for (T k = lx; k < DEPTH; k += CPARTSIZE)
            {
                level_count[wx][k] = 0;
                level_index[wx][k] = 0;
                level_prev_index[wx][k] = 0;
            }
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                cl[k] = get_mask(srcLen, k) & unset_mask(dstIdx, k) & unset_mask(j, k);
                cl[num_divs_local + k] = level_offset[num_divs_local + k];
            }
            __syncwarp(partMask);
            if (lx == 0)
            {
                l[wx] = 3;
                // sg_count[wx] = 0;
                level_prev_index[wx][1] = dstIdx + 1;
                level_prev_index[wx][2] = j + 1;
            }
            __syncwarp(partMask);
            endTime<T>(INITIALIZE, &times[wx], lx, partMask);

            startTime<T>(INTERSECT, &times[wx], lx, partMask);
            // get warp count ??
            warpCount = 0;
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                to[threadIdx.x] = cl[k];
                // Compute Intersection
                for (T q_idx = QEDGE_PTR[l[wx]] + 1; q_idx < QEDGE_PTR[l[wx] + 1]; q_idx++)
                {
                    to[threadIdx.x] &= encode[(level_prev_index[wx][QEDGE[q_idx]] - 1) * num_divs_local + k];
                }
                // Remove Redundancies
                for (T sym_idx = SYMNODE_PTR[l[wx]]; sym_idx < SYMNODE_PTR[l[wx] + 1]; sym_idx++)
                {
                    if (SYMNODE[sym_idx] > 0)
                        to[threadIdx.x] &= ~(get_mask(level_prev_index[wx][SYMNODE[sym_idx]] - 1, k));
                }
                warpCount += __popc(to[threadIdx.x]);
                cl[(l[wx] - 1) * num_divs_local + k] = to[threadIdx.x];
            }
            endTime<T>(INTERSECT, &times[wx], lx, partMask);
            startTime<T>(REDUCTION, &times[wx], lx, partMask);
            reduce_part<T, CPARTSIZE>(partMask, warpCount);
            endTime<T>(REDUCTION, &times[wx], lx, partMask);

            startTime<T>(OTHER, &times[wx], lx, partMask);
            if (lx == 0)
            {
                if (l[wx] + 1 == KCCOUNT)
                    sg_count[wx] += warpCount;
                else
                {
                    l[wx]++;
                    level_count[wx][l[wx] - 3] = warpCount;
                    level_index[wx][l[wx] - 3] = 0;
                    level_prev_index[wx][l[wx] - 1] = 0;
                }
            }
            __syncwarp(partMask);
            endTime<T>(OTHER, &times[wx], lx, partMask);

            while (level_count[wx][l[wx] - 3] > level_index[wx][l[wx] - 3])
            {
                startTime<T>(TRAVERSE, &times[wx], lx, partMask);
                // First Index
                if (lx == 0)
                {
                    T *from = &(cl[num_divs_local * (l[wx] - 2)]);
                    T maskBlock = level_prev_index[wx][l[wx] - 1] / 32;
                    T maskIndex = ~((1 << (level_prev_index[wx][l[wx] - 1] & 0x1F)) - 1);

                    newIndex[wx] = __ffs(from[maskBlock] & maskIndex);
                    while (newIndex[wx] == 0)
                    {
                        maskIndex = 0xFFFFFFFF;
                        maskBlock++;
                        newIndex[wx] = __ffs(from[maskBlock] & maskIndex);
                    }
                    newIndex[wx] = 32 * maskBlock + newIndex[wx] - 1;

                    level_prev_index[wx][l[wx] - 1] = newIndex[wx] + 1;
                    level_index[wx][l[wx] - 3]++;
                }
                __syncwarp(partMask);
                endTime<T>(TRAVERSE, &times[wx], lx, partMask);

                startTime<T>(INTERSECT, &times[wx], lx, partMask);
                compute_intersection<T, CPARTSIZE, true>(warpCount, of[wx], lx, partMask, num_divs_local,
                                                         newIndex[wx], l[wx], to, cl,
                                                         level_prev_index[wx], encode);
                endTime<T>(INTERSECT, &times[wx], lx, partMask);

                startTime<T>(TRAVERSE, &times[wx], lx, partMask);
                if (l[wx] + 1 < KCCOUNT)
                {
                    for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                        cl[k] &= unset_mask(level_prev_index[wx][l[wx] - 1] - 1, k);
                }
                endTime<T>(TRAVERSE, &times[wx], lx, partMask);

                startTime<T>(REDUCTION, &times[wx], lx, partMask);
                if (lx == 0)
                {
                    if (l[wx] + 1 == KCCOUNT)
                    {
                        sg_count[wx] += warpCount;
                    }
                }
                endTime<T>(REDUCTION, &times[wx], lx, partMask);
                startTime<T>(TRAVERSE, &times[wx], lx, partMask);
                if (lx == 0)
                {
                    if (l[wx] + 1 == KCCOUNT)
                    {
                        // nothing
                    }
                    else if (l[wx] + 1 < KCCOUNT) //&& warpCount >= KCCOUNT - l[wx])
                    {
                        (l[wx])++;
                        level_count[wx][l[wx] - 3] = warpCount;
                        level_index[wx][l[wx] - 3] = 0;
                        level_prev_index[wx][l[wx] - 1] = 0;

                        T idx = level_prev_index[wx][l[wx] - 2] - 1;
                        cl[idx / 32] &= ~(1 << (idx & 0x1F));
                    }
                    while (l[wx] > 4 && level_index[wx][l[wx] - 3] >= level_count[wx][l[wx] - 3])
                    {
                        (l[wx])--;
                        T idx = level_prev_index[wx][l[wx] - 1] - 1;
                        cl[idx / 32] |= 1 << (idx & 0x1F);
                    }
                }
                endTime<T>(TRAVERSE, &times[wx], lx, partMask);
                __syncwarp(partMask);
            }
            __syncwarp(partMask);
        }
        startTime<T>(REDUCTION, &times[wx], lx, partMask);
        if (lx == 0)
        {
            atomicAdd(counter, sg_count[wx]);
        }
        endTime<T>(REDUCTION, &times[wx], lx, partMask);
        __syncthreads();
    }

    endTime<T>(TOTAL, &times[wx], lx, partMask);
    if (lx == 0)
    {
        for (int i = 0; i < NUM_COUNTERS; i++)
            atomicAdd(&SM_times[__mysmid()].totalTime[i], times[wx].totalTime[i]);
    }
    if (threadIdx.x == 0)
    {
        atomicAdd(&SM_edges[__mysmid()], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}
