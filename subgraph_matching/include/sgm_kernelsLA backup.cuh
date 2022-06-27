#pragma once
#include "include/utils.cuh"
#include "config.cuh"

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__global__ void sgm_kernel_compute_encoding(
    const graph::COOCSRGraph_d<T> g,
    const graph::GraphQueue_d<T, bool> current, const T head,
    T *adj_enc, int *offset)
{
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;

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

    T partMask = (1 << CPARTSIZE) - 1;
    partMask = partMask << ((wx % (32 / CPARTSIZE)) * CPARTSIZE);
    for (T j = wx; j < srcLen; j += numPartitions)
    {
        for (T k = lx; k < num_divs_local; k += CPARTSIZE)
        {
            encode[j * num_divs_local + k] = 0x00;
        }
        __syncwarp(partMask);

        warp_sorted_count_and_encode_full<T, true, CPARTSIZE>(&g.oriented_colInd[srcStart], srcLen,
                                                              &g.colInd[g.rowPtr[g.oriented_colInd[srcStart + j]]],
                                                              g.rowPtr[g.oriented_colInd[srcStart + j] + 1] - g.rowPtr[g.oriented_colInd[srcStart + j]],
                                                              j, num_divs_local,
                                                              encode);
    }
}

template <typename T, uint BLOCK_DIM_X>
__global__ void sgm_kernel_pre_encoded_byEdge(
    uint64 *counter,
    const graph::COOCSRGraph_d<T> g,
    const mapping<T> *srcList, const T head,
    T *current_level, T *reuse_stats,
    T *levelStats, T *adj_enc,
    int *offset)
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

    const T tx = threadIdx.x;
    __shared__ T level_index[DEPTH];
    __shared__ T level_count[DEPTH];
    __shared__ T level_prev_index[DEPTH];
    __shared__ uint64 sg_count;
    __shared__ T lvl;
    __shared__ T src, srcStart, srcLen, srcSplit, dstIdx;

    __shared__ T num_divs_local, *level_offset, *reuse_offset, *encode;

    __shared__ T to[BLOCK_DIM_X], newIndex;

    // block things
    if (threadIdx.x == 0)
    {
        src = srcList[blockIdx.x + head].src;
        srcStart = g.rowPtr[src];
        srcSplit = g.splitPtr[src];
        srcLen = g.rowPtr[src + 1] - srcStart;

        dstIdx = blockIdx.x + head - srcList[blockIdx.x + head].srcHead;

        num_divs_local = (srcLen + 32 - 1) / 32;
        encode = &adj_enc[(uint64)offset[src] * NUMDIVS * MAXDEG];
        level_offset = &current_level[(uint64)((sm_id * CBPSM) + levelPtr) * (NUMDIVS * 1 * MAXLEVEL)]; // numpartition = 1
    }
    __syncthreads();

    uint64 warpCount = 0;

    if (SYMNODE_PTR[2] == 1 && dstIdx < (srcSplit - srcStart))
        goto end;
    // compute triangles
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

    typedef cub::BlockReduce<uint64, BLOCK_DIM_X> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    warpCount = BlockReduce(temp_storage).Sum(warpCount); // whole block performs reduction here
    __syncthreads();

    if (KCCOUNT == 3)
    {
        if (threadIdx.x == 0 && warpCount > 0)
            atomicAdd(counter, warpCount);
        goto end;
    }
    if (tx == 0)
    {

        sg_count = 0;
    }

    for (T j = 0; j < srcLen; j++) // no more partitions //can be equal to warpcount
    {
        if ((level_offset[num_divs_local + j / 32] >> (j % 32)) % 2 == 0)
            continue;
        T *cl = level_offset + 0 * (NUMDIVS * MAXLEVEL);

        for (T k = tx; k < DEPTH; k += BLOCK_DIM_X)
        {
            level_index[k] = 0;
            level_count[k] = 0;
            level_prev_index[k] = 0;
        }
        for (T k = tx; k < num_divs_local; k += BLOCK_DIM_X)
        {
            cl[k] = get_mask(srcLen, k) & unset_mask(dstIdx, k) & unset_mask(j, k);
            cl[num_divs_local + k] = level_offset[num_divs_local + k];
        }
        __syncthreads();
        if (tx == 0)
        {
            lvl = 3;
            level_prev_index[1] = dstIdx + 1;
            level_prev_index[2] = j + 1;
        }
        __syncthreads();

        // get warp count ??

        compute_intersection_block<T, BLOCK_DIM_X, true>(
            warpCount, tx, num_divs_local, UINT32_MAX, lvl, to, cl,
            level_prev_index, encode);

        if (tx == 0)
        {
            if (lvl + 1 == KCCOUNT)
                sg_count += warpCount;
            else
            {
                lvl++;
                level_count[lvl - 3] = warpCount;
                level_index[lvl - 3] = 0;
                level_prev_index[lvl - 1] = 0;
            }
        }
        __syncthreads();

        while (level_count[lvl - 3] > level_index[lvl - 3])
        {
            __syncthreads();
            // First Index
            if (tx == 0)
            {
                T *from = &(cl[num_divs_local * (lvl - 2)]);
                T maskBlock = level_prev_index[lvl - 1] / 32;
                T maskIndex = ~((1 << (level_prev_index[lvl - 1] & 0x1F)) - 1);

                newIndex = __ffs(from[maskBlock] & maskIndex);
                while (newIndex == 0)
                {
                    maskIndex = 0xFFFFFFFF;
                    maskBlock++;
                    newIndex = __ffs(from[maskBlock] & maskIndex);
                }
                newIndex = 32 * maskBlock + newIndex - 1;

                level_prev_index[lvl - 1] = newIndex + 1;
                level_index[lvl - 3]++;
            }
            __syncthreads();

            compute_intersection_block<T, BLOCK_DIM_X, true>(
                warpCount, tx, num_divs_local,
                newIndex, lvl, to, cl,
                level_prev_index, encode);

            if (tx == 0)
            {
                if (lvl + 1 == KCCOUNT)
                    sg_count += warpCount;
                else if (lvl + 1 < KCCOUNT) //&& warpCount >= KCCOUNT - l[wx])
                {
                    (lvl)++;
                    level_count[lvl - 3] = warpCount;
                    level_index[lvl - 3] = 0;
                    level_prev_index[lvl - 1] = 0;
                    T idx = level_prev_index[lvl - 2] - 1;
                    cl[idx / 32] &= ~(1 << (idx & 0x1F));
                }

                while (lvl > 4 && level_index[lvl - 3] >= level_count[lvl - 3])
                {
                    (lvl)--;
                    T idx = level_prev_index[lvl - 1] - 1;
                    cl[idx / 32] |= 1 << (idx & 0x1F);
                }
            }
            __syncthreads();
        }
        __syncthreads();
    }
    __syncthreads();
    if (tx == 0)
    {
        if (sg_count > 0)
        {
            atomicAdd(counter, sg_count);
            // atomicAdd(&node_count[src], sg_count[wx]);
        }
    }
    __syncthreads();
end:
    if (tx == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}