#pragma once
#include "kckernels.cuh"

__constant__ uint MAXUNDEG;

/* New Structure
 * | ... | s    | s    | ... | s    | s                  | s      | ... | s      | ... |
 * | ... | par1 | par2 | ... | parP | child1             | child2 | ... | childC | ... |
 * | ... |      |      |     |      | splitPtr[s] <here> |        | ... |        | ... |
 */

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
mckernel_node_block_warp_binary_encode_induced(
    graph::COOCSRGraph_d<T> gsplit, // split graph
    const T num_node, // |V|
    T* current_level,
    uint64* cpn, // clique per node
    T* levelStats,
    T* adj_enc,

    T* possible,
    T* x_level, // X for induced
    T* level_count_g,
    T* level_prev_g,
    T* buffer1_g, // X for undirected graph
    T* buffer2_g
)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;

    __shared__ T level_pivot[512];
    __shared__ T *buffer1, *buffer2, sz1[512], sz2;
    __shared__ bool path_more_explore, path_eliminated, vote;
    __shared__ T l;
    __shared__ T maxIntersection;
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen, curSZ, usrcStart, usrcLen, cnodeStart, cnodeLen, cnode;
    __shared__ bool partition_set[numPartitions], x_empty;

    __shared__ T num_divs_local, encode_offset, *encode;
    __shared__ T *pl, *cl, *xl;
    __shared__ T *level_count, *level_prev_index;

    __shared__ T lo, level_item_offset;

    __shared__ T maxCount[numPartitions], maxIndex[numPartitions], partMask[numPartitions];
    
    __shared__ 	T lastMask_i, lastMask_ii;

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

    for (T i = blockIdx.x; i < (T) num_node; i += gridDim.x)
    {
        __syncthreads();
        //block things

        if (threadIdx.x == 0)
        {
            src = i;
            srcStart = gsplit.splitPtr[src];
            srcLen = gsplit.rowPtr[src + 1] - srcStart;

            /* Locate source in the undirected graph*/
            usrcStart = gsplit.rowPtr[src];
            usrcLen = gsplit.splitPtr[src] - usrcStart;

            auto buf_offset = sm_id * CBPSM * (MAXUNDEG) + levelPtr * (MAXUNDEG);
            buffer1 = &buffer1_g[buf_offset];
            buffer2 = &buffer2_g[buf_offset];

            num_divs_local = (srcLen + 32 - 1) / 32;
            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];

            lo = sm_id * CBPSM * (NUMDIVS * (MAXDEG + 1)) + levelPtr * (NUMDIVS * (MAXDEG + 1)); // We will touch one more level in node centric
            cl = &current_level[lo]; 
            pl = &possible[lo]; 
            xl = &x_level[lo]; // X

            level_item_offset = sm_id * CBPSM * (MAXDEG + 1) + levelPtr * (MAXDEG + 1); // We will touch one more level in node centric
            level_count = &level_count_g[level_item_offset];
            level_prev_index = &level_prev_g[level_item_offset];

            level_count[0] = 0;
            level_prev_index[0] = 0;
            l = 2;

            level_pivot[0] = 0xFFFFFFFF;

            path_more_explore = false;
            maxIntersection = 0;

            lastMask_i = srcLen / 32;
            lastMask_ii = (1 << (srcLen & 0x1F)) - 1;
            sz1[1] = usrcLen;
            sz2 = 0;
        }
        __syncthreads();

        for(T j = threadIdx.x; j < usrcLen;j += BLOCK_DIM_X)
        {
            buffer1[j] = j;
        }
        __syncthreads();

        //Encode Clear
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
        }
        __syncthreads();

        // Full Encode
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            graph::warp_sorted_count_and_encode_full<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&gsplit.colInd[srcStart], srcLen,
                &gsplit.colInd[gsplit.splitPtr[gsplit.colInd[srcStart + j]]], gsplit.rowPtr[gsplit.colInd[srcStart + j] + 1] - gsplit.splitPtr[gsplit.colInd[srcStart + j]],
                j, num_divs_local, encode);
        }
        __syncthreads(); // Done encoding

        // Find the first pivot
        if(lx == 0)
        {
            maxCount[wx] = srcLen + 1;
            maxIndex[wx] = 0;
            partition_set[wx] = false;
            partMask[wx] = CPARTSIZE == 32? 0xFFFFFFFF : (1 << CPARTSIZE) - 1;
            partMask[wx] = partMask[wx] << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        }
        __syncthreads();

        for (T j = wx; j < srcLen; j += numPartitions)
        {
            uint64 warpCount = 0;
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);

            if(lx == 0 && maxCount[wx] == srcLen + 1)
            {
                path_more_explore = true; //shared, unsafe, but okay
                partition_set[wx] = true;
                maxCount[wx] = warpCount;
                maxIndex[wx] = j;
            }
            else if(lx == 0 && maxCount[wx] < warpCount)
            {
                maxCount[wx] = warpCount;
                maxIndex[wx] = j;
            }
        }
        __syncthreads();

        if(path_more_explore)
        {
            if(lx == 0 && partition_set[wx])
            {
                atomicMax(&(maxIntersection), maxCount[wx]);
            }
            __syncthreads();
            
            if(lx == 0)
            {
                if(maxIntersection == maxCount[wx])
                {
                    atomicMin(&(level_pivot[0]), maxIndex[wx]);
                }
            }
            __syncthreads();

            //Prepare the Possible and Intersection Encode Lists
            uint64 warpCount = 0;
            for (T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
            {
                T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
                pl[j] = ~(encode[(level_pivot[0])*num_divs_local + j]) & m;
                cl[j] = m;
                xl[j] = 0;
                warpCount += __popc(pl[j]);
            }
            reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
            if(lx == 0 && threadIdx.x < num_divs_local)
            {
                atomicAdd(&(level_count[0]), (T)warpCount);
            }
            __syncthreads();
        }

        // Explore the tree
        while(level_count[l - 2] > 0)
        {
            T maskBlock = level_prev_index[l - 2] / 32;
            T maskIndex = ~((1 << (level_prev_index[l - 2] & 0x1F)) -1);
            T newIndex = __ffs(pl[num_divs_local*(l-2) + maskBlock] & maskIndex);
            while(newIndex == 0)
            {
                maskIndex = 0xFFFFFFFF;
                maskBlock++;
                newIndex = __ffs(pl[num_divs_local*(l - 2) + maskBlock] & maskIndex);
            }
            newIndex =  32 * maskBlock + newIndex - 1;
            T sameBlockMask = (~((1 << (newIndex & 0x1F)) - 1)) | ~pl[num_divs_local*(l-2) + maskBlock];
            __syncthreads();
            
            if (threadIdx.x == 0)
            {
                level_prev_index[l - 2] = newIndex + 1;
                level_count[l - 2]--;
                level_pivot[l - 1] = 0xFFFFFFFF;
                path_more_explore = false;
                maxIntersection = 0;

                /* c stands for current */
                cnode = gsplit.colInd[srcStart + newIndex];
                cnodeStart = gsplit.rowPtr[cnode];
                cnodeLen = gsplit.splitPtr[cnode] - cnodeStart;
                sz1[l] = sz2 = 0;
            }
            __syncthreads();

            /* Only check vertices in X that were deleted no earlier than this level .
             * Note that storing level into first_removed makes us happy when backtracking.
             */
            for(T j = threadIdx.x; j < sz1[l - 1];j += BLOCK_DIM_X)
            {
                bool found = false;
                T cur = buffer1[j];
                graph::binary_search(gsplit.colInd + cnodeStart, 0u, cnodeLen, gsplit.colInd[usrcStart + cur], found);
                if(found) {
                    buffer2[atomicAdd(&sz1[l], 1)] = cur;
                }
                else {
                    buffer2[sz1[l - 1] - atomicAdd(&sz2, 1) - 1] = cur;
                }
            }
            __syncthreads();
            for(T j = threadIdx.x; j < sz1[l - 1];j += BLOCK_DIM_X)
            {
                buffer1[j] = buffer2[j];
            }
               __syncthreads();

            // Now prepare intersection list
            T* from_cl = &(cl[num_divs_local * (l - 2)]);
            T* to_cl =  &(cl[num_divs_local * (l - 1)]);
            T* from_xl = &(xl[num_divs_local * (l - 2)]);
            T* to_xl = &(xl[num_divs_local * (l - 1)]);

            for (T k = threadIdx.x; k < num_divs_local; k += BLOCK_DIM_X)
            {
                // binary encoding operations of X in induced subgraph
                // similar as the inverse of P(cl)
                to_xl[k] = from_xl[k] | ( (maskBlock < k) ? pl[num_divs_local * (l - 2) + k] : ( (maskBlock > k) ? 0 : ~sameBlockMask) );
                to_xl[k] &= encode[newIndex * num_divs_local + k];
                to_cl[k] = from_cl[k] & encode[newIndex * num_divs_local + k];
                to_cl[k] = to_cl[k] & ( (maskBlock < k) ? ~pl[num_divs_local * (l - 2) + k] : ( (maskBlock > k) ? 0xFFFFFFFF : sameBlockMask) );
            }
            if(lx == 0)
            {	
                partition_set[wx] = false;
                maxCount[wx] = srcLen + 1; //make it shared !!
                maxIndex[wx] = 0;
            }
            __syncthreads();
            
            /* Note that choosing a pivot from X can remove all vertices from P */
            uint64 warpCount = 0;
            curSZ = 0;
            path_eliminated = false;

            for (T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
            {
                warpCount += __popc(to_cl[j]);
            }
            reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);

            __syncthreads();
            if(lx == 0 && threadIdx.x < num_divs_local)
            {
                atomicAdd(&curSZ, warpCount);
            }
            __syncthreads();

            for (T j = wx; j < srcLen; j += numPartitions)
            {
                warpCount = 0;
                T bi = j / 32;
                T ii = j & 0x1F;
                if( (to_cl[bi] & (1 << ii)) != 0 || (to_xl[bi] & (1 << ii)) != 0 )
                {
                    for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                    {
                        warpCount += __popc(to_cl[k] & encode[j * num_divs_local + k]);
                    }
                    reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);

                    /* A pivot from X removes all vertices from P */
                    if(lx == 0 && curSZ == warpCount)
                    {
                        path_eliminated = true;
                    }

                    if(lx == 0 && maxCount[wx] == srcLen + 1)
                    {
                        partition_set[wx] = true;
                        path_more_explore = true; //shared, unsafe, but okay
                        maxCount[wx] = warpCount;
                        maxIndex[wx] = j;
                    }
                    else if(lx == 0 && maxCount[wx] < warpCount)
                    {
                        maxCount[wx] = warpCount;
                        maxIndex[wx] = j;
                    }	
                }
            }

            __syncthreads();
            if(!path_more_explore || path_eliminated)
            {
                __syncthreads();
                
                if (threadIdx.x == 0)
                {
                    x_empty = true;
                }
                __syncthreads();

                /* Check if X in induced subgraph is empty */
                for (T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
                {
                    if(to_xl[j])
                    {
                        x_empty = false;
                    }
                }
                __syncthreads();

                if(threadIdx.x == 0)
                {	
                    if(x_empty && sz1[l] == 0)
                    {
                        ++cpn[src];
                    }

                    /* No need to maintain X when backtracking. */
                    while (l > 2 && level_count[l - 2] == 0)
                    {
                        (l)--;
                    }
                }
                __syncthreads();
            }
            else
            {
                if(lx == 0 && partition_set[wx])
                {
                    atomicMax(&(maxIntersection), maxCount[wx]);
                }
                __syncthreads();

                if(lx == 0 && maxIntersection == maxCount[wx])
                {	
                    atomicMin(&(level_pivot[l-1]), maxIndex[wx]);
                }
                __syncthreads();

                warpCount = 0;
                for (T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
                {
                    T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
                    pl[(l - 1)*num_divs_local + j] = ~(encode[level_pivot[l - 1] * num_divs_local + j]) & to_cl[j] & m;
                    warpCount += __popc(pl[(l - 1)*num_divs_local + j]);
                }
                reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
                __syncthreads(); // Need this for degeneracy > 1024

                if(threadIdx.x == 0)
                {
                    l++;
                    level_count[l-2] = 0;
                    level_prev_index[l-2] = 0;
                }

                __syncthreads();
                if(lx == 0 && threadIdx.x < num_divs_local)
                {
                    atomicAdd(&(level_count[l - 2]), warpCount);
                }
            }
            __syncthreads();
        }
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
mckernel_edge_block_warp_binary_encode_induced(
    graph::COOCSRGraph_d<T> gsplit, // original split graph
    const T num_edge, // |E|
    T* current_level,
    uint64* cpn, // clique per node
    T* levelStats,
    T* adj_enc,

    T* possible,
    T* x_level, // X for induced
    T* level_count_g,
    T* level_prev_g,
    T* buffer1_g, // X for undirected graph
    T* buffer2_g,
    T* adj_tri
)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;

    __shared__ T level_pivot[512];
    __shared__ bool path_more_explore, path_eliminated, vote;
    __shared__ T l;
    __shared__ T maxIntersection;
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen, src2, src2Start, src2Len, curSZ, usrcStart, usrcLen, cnodeStart, cnodeLen, cnode;
    __shared__ bool partition_set[numPartitions], x_empty, rev_edge;

    __shared__ T num_divs_local, encode_offset, *encode;
    __shared__ T *pl, *cl, *xl, *tri, scounter;
    __shared__ T *level_count, *level_prev_index;

    __shared__ T lo, level_item_offset;

    __shared__ T maxCount[numPartitions], maxIndex[numPartitions], partMask[numPartitions];
    
    __shared__ T lastMask_i, lastMask_ii;
    __shared__ T *buffer1, *buffer2, sz1[512], sz2;

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

    for (T i = blockIdx.x; i < (T) num_edge; i += gridDim.x)
    {
        __syncthreads();
        //block things

        if (threadIdx.x == 0) {
            rev_edge = i < gsplit.splitPtr[gsplit.rowInd[i]];
        }
        __syncthreads();
        if (rev_edge)
        {
            continue;
        }

        if (threadIdx.x == 0)
        {
            src = gsplit.rowInd[i];
            srcStart = gsplit.splitPtr[src];
            srcLen = gsplit.rowPtr[src + 1] - srcStart;
            src2 = gsplit.colInd[i];
            src2Start = gsplit.splitPtr[src2];
            src2Len = gsplit.rowPtr[src2 + 1] - src2Start;
            usrcStart = gsplit.rowPtr[src];
            usrcLen = gsplit.splitPtr[src] - usrcStart;
            cnode = src2;
            cnodeStart = gsplit.rowPtr[cnode];
            cnodeLen = gsplit.splitPtr[cnode] - cnodeStart;
            auto tri_offset = sm_id * CBPSM * (MAXDEG) + levelPtr * (MAXDEG);
            auto buf_offset = sm_id * CBPSM * (MAXUNDEG) + levelPtr * (MAXUNDEG);
            buffer1 = &buffer1_g[buf_offset];
            buffer2 = &buffer2_g[buf_offset];
            tri = &adj_tri[tri_offset];
            scounter = 0;

            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];

            lo = sm_id * CBPSM * (NUMDIVS * MAXDEG) + levelPtr * (NUMDIVS * MAXDEG);
            cl = &current_level[lo]; 
            pl = &possible[lo]; 
            xl = &x_level[lo]; // X

            level_item_offset = sm_id * CBPSM * (MAXDEG) + levelPtr * (MAXDEG);
            level_count = &level_count_g[level_item_offset];
            level_prev_index = &level_prev_g[level_item_offset];

            level_count[0] = 0;
            level_prev_index[0] = 0;
            l = 3;

            level_pivot[0] = 0xFFFFFFFF;

            path_more_explore = false;
            maxIntersection = 0;
            sz1[2] = 0;
        }
        __syncthreads();

        for(T j = threadIdx.x; j < usrcLen + srcLen;j += BLOCK_DIM_X)
        {
            bool found = false;
            T cur = gsplit.colInd[usrcStart + j];
            graph::binary_search(gsplit.colInd + cnodeStart, 0u, cnodeLen, cur, found);
            if(found) {
                buffer1[atomicAdd(&sz1[2], 1)] = j;
            }
        }
        __syncthreads();

        graph::block_count_and_set_tri<BLOCK_DIM_X, T>(&gsplit.colInd[srcStart], srcLen, &gsplit.colInd[src2Start], src2Len,
            tri, &scounter);
        
        __syncthreads();

        if (threadIdx.x == 0)
        {
            num_divs_local = (scounter + 32 - 1) / 32;
            lastMask_i = scounter / 32;
            lastMask_ii = (1 << (scounter & 0x1F)) - 1;
        }
        __syncthreads();

        //Encode Clear
        for (T j = wx; j < scounter; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
        }
        __syncthreads();

        // Full Encode
        for (T j = wx; j < scounter; j += numPartitions)
        {
            graph::warp_sorted_count_and_encode_full<WARPS_PER_BLOCK, T, true, CPARTSIZE>(tri, scounter,
                &gsplit.colInd[gsplit.splitPtr[tri[j]]], gsplit.rowPtr[tri[j] + 1] - gsplit.splitPtr[tri[j]],
                j, num_divs_local, encode);
        }
        __syncthreads(); // Done encoding

        // Find the first pivot
        if(lx == 0)
        {
            maxCount[wx] = scounter + 1;
            maxIndex[wx] = 0;
            partition_set[wx] = false;
            partMask[wx] = CPARTSIZE == 32? 0xFFFFFFFF : (1 << CPARTSIZE) - 1;
            partMask[wx] = partMask[wx] << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        }
        __syncthreads();

        for (T j = wx; j < scounter; j += numPartitions)
        {
            uint64 warpCount = 0;
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);

            if(lx == 0 && maxCount[wx] == scounter + 1)
            {
                path_more_explore = true; //shared, unsafe, but okay
                partition_set[wx] = true;
                maxCount[wx] = warpCount;
                maxIndex[wx] = j;
            }
            else if(lx == 0 && maxCount[wx] < warpCount)
            {
                maxCount[wx] = warpCount;
                maxIndex[wx] = j;
            }
        }
        __syncthreads();

        if(path_more_explore)
        {
            if(lx == 0 && partition_set[wx])
            {
                atomicMax(&(maxIntersection), maxCount[wx]);
            }
            __syncthreads();
            
            if(lx == 0)
            {
                if(maxIntersection == maxCount[wx])
                {
                    atomicMin(&(level_pivot[0]), maxIndex[wx]);
                }
            }
            __syncthreads();

            //Prepare the Possible and Intersection Encode Lists
            uint64 warpCount = 0;
            for (T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
            {
                T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
                pl[j] = ~(encode[(level_pivot[0])*num_divs_local + j]) & m;
                cl[j] = m;
                xl[j] = 0;
                warpCount += __popc(pl[j]);
            }
            reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
            if(lx == 0 && threadIdx.x < num_divs_local)
            {
                atomicAdd(&(level_count[0]), (T)warpCount);
            }
            __syncthreads();
        }
        else
        {
            if(threadIdx.x == 0 && sz1[2] == 0) 
            {
                atomicAdd(cpn + src, 1);
            }
            __syncthreads();
        }

        // Explore the tree
        while(level_count[l - 3] > 0)
        {
            T maskBlock = level_prev_index[l - 3] / 32;
            T maskIndex = ~((1 << (level_prev_index[l - 3] & 0x1F)) -1);
            T newIndex = __ffs(pl[num_divs_local*(l-3) + maskBlock] & maskIndex);
            while(newIndex == 0)
            {
                maskIndex = 0xFFFFFFFF;
                maskBlock++;
                newIndex = __ffs(pl[num_divs_local*(l - 3) + maskBlock] & maskIndex);
            }
            newIndex =  32 * maskBlock + newIndex - 1;
            T sameBlockMask = (~((1 << (newIndex & 0x1F)) - 1)) | ~pl[num_divs_local*(l-3) + maskBlock];
            __syncthreads();
            
            if (threadIdx.x == 0)
            {
                level_prev_index[l - 3] = newIndex + 1;
                level_count[l - 3]--;
                level_pivot[l - 2] = 0xFFFFFFFF;
                path_more_explore = false;
                maxIntersection = 0;
                /* c stands for current */
                cnode = tri[newIndex];
                cnodeStart = gsplit.rowPtr[cnode];
                cnodeLen = gsplit.splitPtr[cnode] - cnodeStart;
                sz1[l] = sz2 = 0;
            }
            __syncthreads();

            for(T j = threadIdx.x; j < sz1[l - 1];j += BLOCK_DIM_X)
            {
                bool found = false;
                T cur = buffer1[j];
                graph::binary_search(gsplit.colInd + cnodeStart, 0u, cnodeLen, gsplit.colInd[usrcStart + cur], found);
                if(found) {
                    buffer2[atomicAdd(&sz1[l], 1)] = cur;
                }
                else {
                    buffer2[sz1[l - 1] - atomicAdd(&sz2, 1) - 1] = cur;
                }
            }
            __syncthreads();
            for(T j = threadIdx.x; j < sz1[l - 1];j += BLOCK_DIM_X)
            {
                buffer1[j] = buffer2[j];
            }
               __syncthreads();

            // Now prepare intersection list
            T* from_cl = &(cl[num_divs_local * (l - 3)]);
            T* to_cl =  &(cl[num_divs_local * (l - 2)]);
            T* from_xl = &(xl[num_divs_local * (l - 3)]);
            T* to_xl = &(xl[num_divs_local * (l - 2)]);

            for (T k = threadIdx.x; k < num_divs_local; k += BLOCK_DIM_X)
            {
                // binary encoding operations of X in induced subgraph
                // similar as the inverse of P(cl)
                to_xl[k] = from_xl[k] | ( (maskBlock < k) ? pl[num_divs_local * (l - 3) + k] : ( (maskBlock > k) ? 0 : ~sameBlockMask) );
                to_xl[k] &= encode[newIndex * num_divs_local + k];
                to_cl[k] = from_cl[k] & encode[newIndex * num_divs_local + k];
                to_cl[k] = to_cl[k] & ( (maskBlock < k) ? ~pl[num_divs_local * (l - 3) + k] : ( (maskBlock > k) ? 0xFFFFFFFF : sameBlockMask) );
            }
            if(lx == 0)
            {	
                partition_set[wx] = false;
                maxCount[wx] = scounter + 1; //make it shared !!
                maxIndex[wx] = 0;
            }
            __syncthreads();
            
            /* Note that choosing a pivot from X can remove all vertices from P */
            uint64 warpCount = 0;
            curSZ = 0;
            path_eliminated = false;

            for (T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
            {
                warpCount += __popc(to_cl[j]);
            }
            reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);

            __syncthreads();
            if(lx == 0 && threadIdx.x < num_divs_local)
            {
                atomicAdd(&curSZ, warpCount);
            }
            __syncthreads();

            for (T j = wx; j < scounter; j += numPartitions)
            {
                warpCount = 0;
                T bi = j / 32;
                T ii = j & 0x1F;
                if( (to_cl[bi] & (1 << ii)) != 0 || (to_xl[bi] & (1 << ii)) != 0 )
                {
                    for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                    {
                        warpCount += __popc(to_cl[k] & encode[j * num_divs_local + k]);
                    }
                    reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);

                    /* A pivot from X removes all vertices from P */
                    if(lx == 0 && curSZ == warpCount)
                    {
                        path_eliminated = true;
                    }

                    if(lx == 0 && maxCount[wx] == scounter + 1)
                    {
                        partition_set[wx] = true;
                        path_more_explore = true; //shared, unsafe, but okay
                        maxCount[wx] = warpCount;
                        maxIndex[wx] = j;
                    }
                    else if(lx == 0 && maxCount[wx] < warpCount)
                    {
                        maxCount[wx] = warpCount;
                        maxIndex[wx] = j;
                    }	
                }
            }

            __syncthreads();
            if(!path_more_explore || path_eliminated)
            {
                __syncthreads();
                
                if (threadIdx.x == 0)
                {
                    x_empty = true;
                }
                __syncthreads();

                /* Check if X in induced subgraph is empty */
                for (T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
                {
                    if(to_xl[j])
                    {
                        x_empty = false;
                    }
                }
                __syncthreads();

                if(threadIdx.x == 0)
                {	
                    if(x_empty && sz1[l] == 0)
                    {
                        atomicAdd(cpn + src, 1);
                    }

                    /* No need to maintain X when backtracking. */
                    while (l > 3 && level_count[l - 3] == 0)
                    {
                        (l)--;
                    }
                }
                __syncthreads();
            }
            else
            {
                if(lx == 0 && partition_set[wx])
                {
                    atomicMax(&(maxIntersection), maxCount[wx]);
                }
                __syncthreads();

                if(lx == 0 && maxIntersection == maxCount[wx])
                {	
                    atomicMin(&(level_pivot[l-2]), maxIndex[wx]);
                }
                __syncthreads();

                warpCount = 0;
                for (T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
                {
                    T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
                    pl[(l - 2)*num_divs_local + j] = ~(encode[level_pivot[l - 2] * num_divs_local + j]) & to_cl[j] & m;
                    warpCount += __popc(pl[(l - 2)*num_divs_local + j]);
                }
                reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
                __syncthreads(); // Need this for degeneracy > 1024

                if(threadIdx.x == 0)
                {
                    l++;
                    level_count[l-3] = 0;
                    level_prev_index[l-3] = 0;
                }

                __syncthreads();
                if(lx == 0 && threadIdx.x < num_divs_local)
                {
                    atomicAdd(&(level_count[l - 3]), warpCount);
                }
            }
            __syncthreads();
        }
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}


// This use (P + X) by P bit structure, and pivot on all P + X
template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
mckernel_node_block_warp_binary_encode_half(
    graph::COOCSRGraph_d<T> gsplit, // split graph
    const T num_node, // |V|
    T* current_level,
    uint64* cpn, // clique per node
    T* levelStats,
    T* adj_enc,

    T* possible,
    T* x_level, // X for induced
    T* level_count_g,
    T* level_prev_g,
    T* buffer1_g, // X for undirected graph
    T* buffer2_g
)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;

    __shared__ T level_pivot[512];
    __shared__ T *buffer1, *buffer2, sz1[512], sz2;
    __shared__ bool path_more_explore, path_eliminated, vote;
    __shared__ T l;
    __shared__ T maxIntersection;
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen, curSZ, usrcStart, usrcLen, cnodeStart, cnodeLen, cnode;
    __shared__ bool partition_set[numPartitions], x_empty;

    __shared__ T num_divs_local, encode_offset, *encode;
    __shared__ T *pl, *cl, *xl;
    __shared__ T *level_count, *level_prev_index;

    __shared__ T lo, level_item_offset;

    __shared__ T maxCount[numPartitions], maxIndex[numPartitions], partMask[numPartitions];
    
    __shared__ 	T lastMask_i, lastMask_ii;

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

    for (T i = blockIdx.x; i < (T) num_node; i += gridDim.x)
    {
        __syncthreads();
        //block things

        if (threadIdx.x == 0)
        {
            src = i;
            srcStart = gsplit.splitPtr[src];
            srcLen = gsplit.rowPtr[src + 1] - srcStart;

            /* Locate source in the undirected graph*/
            usrcStart = gsplit.rowPtr[src];
            usrcLen = gsplit.splitPtr[src] - usrcStart;

            auto buf_offset = sm_id * CBPSM * (MAXUNDEG) + levelPtr * (MAXUNDEG);
            buffer1 = &buffer1_g[buf_offset];
            buffer2 = &buffer2_g[buf_offset];

            num_divs_local = (srcLen + 32 - 1) / 32;
            encode_offset = sm_id * CBPSM * (MAXUNDEG * NUMDIVS) + levelPtr * (MAXUNDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];

            lo = sm_id * CBPSM * (NUMDIVS * (MAXDEG + 1)) + levelPtr * (NUMDIVS * (MAXDEG + 1)); // We will touch one more level in node centric
            cl = &current_level[lo]; 
            pl = &possible[lo]; 
            xl = &x_level[lo]; // X

            level_item_offset = sm_id * CBPSM * (MAXDEG + 1) + levelPtr * (MAXDEG + 1); // We will touch one more level in node centric
            level_count = &level_count_g[level_item_offset];
            level_prev_index = &level_prev_g[level_item_offset];

            level_count[0] = 0;
            level_prev_index[0] = 0;
            l = 2;

            level_pivot[0] = 0xFFFFFFFF;

            path_more_explore = false;
            path_eliminated = false;
            maxIntersection = 0;

            lastMask_i = srcLen / 32;
            lastMask_ii = (1 << (srcLen & 0x1F)) - 1;
            sz1[1] = usrcLen;
            sz2 = 0;
        }
        __syncthreads();

        for(T j = threadIdx.x; j < usrcLen;j += BLOCK_DIM_X)
        {
            buffer1[j] = j + srcLen;
        }
        __syncthreads();

        //Encode Clear
        for (T j = wx; j < usrcLen + srcLen; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
        }
        __syncthreads();

        // Full Encode
        for (T j = wx; j < usrcLen + srcLen; j += numPartitions)
        {
            graph::warp_sorted_count_and_encode_full_mclique<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&gsplit.colInd[srcStart], srcLen,
                &gsplit.colInd[gsplit.splitPtr[gsplit.colInd[usrcStart + j]]], gsplit.rowPtr[gsplit.colInd[usrcStart + j] + 1] - gsplit.splitPtr[gsplit.colInd[usrcStart + j]],
                j, num_divs_local, encode, usrcLen);
            /*   P
             * P| |
             * X| |
             */
        }
        __syncthreads(); // Done encoding

        // Find the first pivot
        if(lx == 0)
        {
            maxCount[wx] = srcLen + 1;
            maxIndex[wx] = 0;
            partition_set[wx] = false;
            partMask[wx] = CPARTSIZE == 32? 0xFFFFFFFF : (1 << CPARTSIZE) - 1;
            partMask[wx] = partMask[wx] << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        }
        __syncthreads();

        for (T j = wx; j < srcLen; j += numPartitions)
        {
            uint64 warpCount = 0;
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);

            if(lx == 0 && maxCount[wx] == srcLen + 1)
            {
                path_more_explore = true; //shared, unsafe, but okay
                partition_set[wx] = true;
                maxCount[wx] = warpCount;
                maxIndex[wx] = j;
            }
            else if(lx == 0 && maxCount[wx] < warpCount)
            {
                maxCount[wx] = warpCount;
                maxIndex[wx] = j;
            }
        }
        __syncthreads();

        for (T j = wx; j < sz1[1]; j += numPartitions)
        {
            uint64 warpCount = 0;
            T cur = buffer1[j];
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[cur * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);

            /* A pivot from X removes all vertices from P */
            if(lx == 0 && srcLen == warpCount)
            {
                path_eliminated = true;
            }

            if(lx == 0 && maxCount[wx] == srcLen + 1)
            {
                path_more_explore = true; //shared, unsafe, but okay
                partition_set[wx] = true;
                maxCount[wx] = warpCount;
                maxIndex[wx] = cur;
            }
            else if(lx == 0 && maxCount[wx] < warpCount)
            {
                maxCount[wx] = warpCount;
                maxIndex[wx] = cur;
            }
        }
        __syncthreads();

        if(path_more_explore && !path_eliminated)
        {
            if(lx == 0 && partition_set[wx])
            {
                atomicMax(&(maxIntersection), maxCount[wx]);
            }
            __syncthreads();
            
            if(lx == 0)
            {
                if(maxIntersection == maxCount[wx])
                {
                    atomicMin(&(level_pivot[0]), maxIndex[wx]);
                }
            }
            __syncthreads();

            //Prepare the Possible and Intersection Encode Lists
            uint64 warpCount = 0;
            for (T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
            {
                T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
                pl[j] = ~(encode[(level_pivot[0])*num_divs_local + j]) & m;
                cl[j] = m;
                xl[j] = 0;
                warpCount += __popc(pl[j]);
            }
            reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
            if(lx == 0 && threadIdx.x < num_divs_local)
            {
                atomicAdd(&(level_count[0]), (T)warpCount);
            }
            __syncthreads();
        }

        // Explore the tree
        while(level_count[l - 2] > 0)
        {
            T maskBlock = level_prev_index[l - 2] / 32;
            T maskIndex = ~((1 << (level_prev_index[l - 2] & 0x1F)) -1);
            T newIndex = __ffs(pl[num_divs_local*(l-2) + maskBlock] & maskIndex);
            while(newIndex == 0)
            {
                maskIndex = 0xFFFFFFFF;
                maskBlock++;
                newIndex = __ffs(pl[num_divs_local*(l - 2) + maskBlock] & maskIndex);
            }
            newIndex =  32 * maskBlock + newIndex - 1;
            T sameBlockMask = (~((1 << (newIndex & 0x1F)) - 1)) | ~pl[num_divs_local*(l-2) + maskBlock];
            __syncthreads();
            
            if (threadIdx.x == 0)
            {
                level_prev_index[l - 2] = newIndex + 1;
                level_count[l - 2]--;
                level_pivot[l - 1] = 0xFFFFFFFF;
                path_more_explore = false;
                maxIntersection = 0;

                sz1[l] = sz2 = 0;
            }
            __syncthreads();

            /* Only check vertices in X that were deleted no earlier than this level .
             * Note that storing level into first_removed makes us happy when backtracking.
             */
            for(T j = threadIdx.x; j < sz1[l - 1];j += BLOCK_DIM_X)
            {
                T cur = buffer1[j];
                if((encode[cur * num_divs_local + (newIndex >> 5)] >> (newIndex & 0x1F)) & 1) {
                    buffer2[atomicAdd(&sz1[l], 1)] = cur;
                } else {
                    buffer2[sz1[l - 1] - atomicAdd(&sz2, 1) - 1] = cur;
                }
            }
            __syncthreads();
            for(T j = threadIdx.x; j < sz1[l - 1];j += BLOCK_DIM_X)
            {
                buffer1[j] = buffer2[j];
            }
               __syncthreads();

            // Now prepare intersection list
            T* from_cl = &(cl[num_divs_local * (l - 2)]);
            T* to_cl =  &(cl[num_divs_local * (l - 1)]);
            T* from_xl = &(xl[num_divs_local * (l - 2)]);
            T* to_xl = &(xl[num_divs_local * (l - 1)]);

            for (T k = threadIdx.x; k < num_divs_local; k += BLOCK_DIM_X)
            {
                // binary encoding operations of X in induced subgraph
                // similar as the inverse of P(cl)
                to_xl[k] = from_xl[k] | ( (maskBlock < k) ? pl[num_divs_local * (l - 2) + k] : ( (maskBlock > k) ? 0 : ~sameBlockMask) );
                to_xl[k] &= encode[newIndex * num_divs_local + k];

                to_cl[k] = from_cl[k] & encode[newIndex * num_divs_local + k];
                to_cl[k] = to_cl[k] & ( (maskBlock < k) ? ~pl[num_divs_local * (l - 2) + k] : ( (maskBlock > k) ? 0xFFFFFFFF : sameBlockMask) );
            }
            if(lx == 0)
            {	
                partition_set[wx] = false;
                maxCount[wx] = srcLen + 1; //make it shared !!
                maxIndex[wx] = 0;
            }
            __syncthreads();
            
            /* Note that choosing a pivot from X can remove all vertices from P */
            uint64 warpCount = 0;
            curSZ = 0;
            path_eliminated = false;

            for (T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
            {
                warpCount += __popc(to_cl[j]);
            }
            reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);

            __syncthreads();
            if(lx == 0 && threadIdx.x < num_divs_local)
            {
                atomicAdd(&curSZ, warpCount);
            }
            __syncthreads();

            for (T j = wx;j < sz1[l];j += numPartitions)
            {
                warpCount = 0;
                T cur = buffer1[j];
                for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                {
                    warpCount += __popc(to_cl[k] & encode[cur * num_divs_local + k]);
                }
                reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);

                /* A pivot from X removes all vertices from P */
                if(lx == 0 && curSZ == warpCount)
                {
                    path_eliminated = true;
                }

                if(lx == 0 && maxCount[wx] == srcLen + 1)
                {
                    partition_set[wx] = true;
                    path_more_explore = true; //shared, unsafe, but okay
                    maxCount[wx] = warpCount;
                    maxIndex[wx] = cur;
                }
                else if(lx == 0 && maxCount[wx] < warpCount)
                {
                    maxCount[wx] = warpCount;
                    maxIndex[wx] = cur;
                }	
            }
            __syncthreads();

            for (T j = wx; j < srcLen; j += numPartitions)
            {
                warpCount = 0;
                T bi = j / 32;
                T ii = j & 0x1F;
                if( (to_cl[bi] & (1 << ii)) != 0 || (to_xl[bi] & (1 << ii)) != 0 )
                {
                    for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                    {
                        warpCount += __popc(to_cl[k] & encode[j * num_divs_local + k]);
                    }
                    reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);

                    /* A pivot from X removes all vertices from P */
                    if(lx == 0 && curSZ == warpCount)
                    {
                        path_eliminated = true;
                    }

                    if(lx == 0 && maxCount[wx] == srcLen + 1)
                    {
                        partition_set[wx] = true;
                        path_more_explore = true; //shared, unsafe, but okay
                        maxCount[wx] = warpCount;
                        maxIndex[wx] = j;
                    }
                    else if(lx == 0 && maxCount[wx] < warpCount)
                    {
                        maxCount[wx] = warpCount;
                        maxIndex[wx] = j;
                    }	
                }
            }

            __syncthreads();
            if(!path_more_explore || path_eliminated)
            {
                __syncthreads();
                
                if (threadIdx.x == 0)
                {
                    x_empty = true;
                }
                __syncthreads();

                /* Check if X in induced subgraph is empty */
                for (T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
                {
                    if(to_xl[j])
                    {
                        x_empty = false;
                    }
                }
                __syncthreads();

                if(threadIdx.x == 0)
                {	
                    if(x_empty && sz1[l] == 0)
                    {
                        ++cpn[src];
                    }

                    /* No need to maintain X when backtracking. */
                    while (l > 2 && level_count[l - 2] == 0)
                    {
                        (l)--;
                    }
                }
                __syncthreads();
            }
            else
            {
                if(lx == 0 && partition_set[wx])
                {
                    atomicMax(&(maxIntersection), maxCount[wx]);
                }
                __syncthreads();

                if(lx == 0 && maxIntersection == maxCount[wx])
                {	
                    atomicMin(&(level_pivot[l-1]), maxIndex[wx]);
                }
                __syncthreads();

                warpCount = 0;
                for (T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
                {
                    T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
                    pl[(l - 1)*num_divs_local + j] = ~(encode[level_pivot[l - 1] * num_divs_local + j]) & to_cl[j] & m;
                    warpCount += __popc(pl[(l - 1)*num_divs_local + j]);
                }
                reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
                __syncthreads(); // Need this for degeneracy > 1024

                if(threadIdx.x == 0)
                {
                    l++;
                    level_count[l-2] = 0;
                    level_prev_index[l-2] = 0;
                }

                __syncthreads();
                if(lx == 0 && threadIdx.x < num_divs_local)
                {
                    atomicAdd(&(level_count[l - 2]), warpCount);
                }
            }
            __syncthreads();
        }
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}


// This use (P + X) by P bit structure, and pivot on P + X
template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
mckernel_edge_block_warp_binary_encode_half(
    graph::COOCSRGraph_d<T> gsplit, // original split graph
    const T num_edge, // |E|
    T* current_level,
    uint64* cpn, // clique per node
    T* levelStats,
    T* adj_enc,

    T* possible,
    T* x_level, // X for induced
    T* level_count_g,
    T* level_prev_g,
    T* buffer1_g, // X for undirected graph
    T* buffer2_g,
    T* adj_tri
)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;

    __shared__ T level_pivot[512];
    __shared__ bool path_more_explore, path_eliminated, vote;
    __shared__ T l;
    __shared__ T maxIntersection;
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen, src2, src2Start, src2Len, curSZ, usrcStart, usrcLen, cnodeStart, cnodeLen, cnode;
    __shared__ bool partition_set[numPartitions], x_empty, rev_edge;

    __shared__ T num_divs_local, encode_offset, *encode;
    __shared__ T *pl, *cl, *xl, *tri, scounter;
    __shared__ T *level_count, *level_prev_index;

    __shared__ T lo, level_item_offset;

    __shared__ T maxCount[numPartitions], maxIndex[numPartitions], partMask[numPartitions];
    
    __shared__ T lastMask_i, lastMask_ii;
    __shared__ T *buffer1, *buffer2, sz1[512], sz2;

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

    for (T i = blockIdx.x; i < (T) num_edge; i += gridDim.x)
    {
        __syncthreads();
        //block things

        if (threadIdx.x == 0) {
            rev_edge = i < gsplit.splitPtr[gsplit.rowInd[i]];
        }
        __syncthreads();
        if (rev_edge)
        {
            continue;
        }

        if (threadIdx.x == 0)
        {
            src = gsplit.rowInd[i];
            srcStart = gsplit.splitPtr[src];
            srcLen = gsplit.rowPtr[src + 1] - srcStart;
            src2 = gsplit.colInd[i];
            src2Start = gsplit.splitPtr[src2];
            src2Len = gsplit.rowPtr[src2 + 1] - src2Start;
            usrcStart = gsplit.rowPtr[src];
            usrcLen = gsplit.splitPtr[src] - usrcStart;
            cnode = src2;
            cnodeStart = gsplit.rowPtr[cnode];
            cnodeLen = gsplit.splitPtr[cnode] - cnodeStart;
            auto tri_offset = sm_id * CBPSM * (MAXDEG) + levelPtr * (MAXDEG);
            auto buf_offset = sm_id * CBPSM * (MAXUNDEG) + levelPtr * (MAXUNDEG);
            buffer1 = &buffer1_g[buf_offset];
            buffer2 = &buffer2_g[buf_offset];
            tri = &adj_tri[tri_offset];
            scounter = 0;

            encode_offset = sm_id * CBPSM * (MAXUNDEG * NUMDIVS) + levelPtr * (MAXUNDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];

            lo = sm_id * CBPSM * (NUMDIVS * MAXDEG) + levelPtr * (NUMDIVS * MAXDEG);
            cl = &current_level[lo]; 
            pl = &possible[lo]; 
            xl = &x_level[lo]; // X

            level_item_offset = sm_id * CBPSM * (MAXDEG) + levelPtr * (MAXDEG);
            level_count = &level_count_g[level_item_offset];
            level_prev_index = &level_prev_g[level_item_offset];

            level_count[0] = 0;
            level_prev_index[0] = 0;
            l = 3;

            level_pivot[0] = 0xFFFFFFFF;

            path_more_explore = false;
            path_eliminated = false;
            maxIntersection = 0;
            sz1[2] = 0;
        }
        __syncthreads();

        for(T j = threadIdx.x; j < usrcLen + srcLen;j += BLOCK_DIM_X)
        {
            bool found = false;
            T cur = gsplit.colInd[usrcStart + j];
            graph::binary_search(gsplit.colInd + cnodeStart, 0u, cnodeLen, cur, found);
            if(found) {
                buffer1[atomicAdd(&sz1[2], 1)] = usrcStart + j;
            }
        }
        __syncthreads();

        graph::block_count_and_set_tri<BLOCK_DIM_X, T>(&gsplit.colInd[srcStart], srcLen, &gsplit.colInd[src2Start], src2Len,
            tri, &scounter);
        
        __syncthreads();

        if (threadIdx.x == 0)
        {
            num_divs_local = (scounter + 32 - 1) / 32;
            lastMask_i = scounter / 32;
            lastMask_ii = (1 << (scounter & 0x1F)) - 1;
        }
        __syncthreads();

        //Encode Clear
        for (T j = wx; j < scounter + sz1[2]; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
        }
        __syncthreads();

        // Full Encode
        for (T j = wx; j < scounter; j += numPartitions)
        {
            graph::warp_sorted_count_and_encode_full_mclique<WARPS_PER_BLOCK, T, true, CPARTSIZE>(tri, scounter,
                &gsplit.colInd[gsplit.splitPtr[tri[j]]], gsplit.rowPtr[tri[j] + 1] - gsplit.splitPtr[tri[j]],
                j + sz1[2], num_divs_local, encode, sz1[2]);
        }
        for (T j = wx; j < sz1[2]; j += numPartitions)
        {
            graph::warp_sorted_count_and_encode_full_mclique<WARPS_PER_BLOCK, T, true, CPARTSIZE>(tri, scounter,
                &gsplit.colInd[gsplit.splitPtr[gsplit.colInd[buffer1[j]]]], gsplit.rowPtr[gsplit.colInd[buffer1[j]] + 1] - gsplit.splitPtr[gsplit.colInd[buffer1[j]]],
                j, num_divs_local, encode, sz1[2]);
        }
        __syncthreads(); // Done encoding

        for (T j = threadIdx.x; j < sz1[2]; j += BLOCK_DIM_X)
        {
            buffer1[j] = j + scounter;
        }
        __syncthreads(); // Done encoding

        // Find the first pivot
        if(lx == 0)
        {
            maxCount[wx] = scounter + 1;
            maxIndex[wx] = 0;
            partition_set[wx] = false;
            partMask[wx] = CPARTSIZE == 32? 0xFFFFFFFF : (1 << CPARTSIZE) - 1;
            partMask[wx] = partMask[wx] << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        }
        __syncthreads();

        for (T j = wx; j < scounter; j += numPartitions)
        {
            uint64 warpCount = 0;
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);

            if(lx == 0 && maxCount[wx] == scounter + 1)
            {
                path_more_explore = true; //shared, unsafe, but okay
                partition_set[wx] = true;
                maxCount[wx] = warpCount;
                maxIndex[wx] = j;
            }
            else if(lx == 0 && maxCount[wx] < warpCount)
            {
                maxCount[wx] = warpCount;
                maxIndex[wx] = j;
            }
        }
        __syncthreads();

        for (T j = wx; j < sz1[2]; j += numPartitions)
        {
            uint64 warpCount = 0;
            T cur = buffer1[j];
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                warpCount += __popc(encode[cur * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);

            /* A pivot from X removes all vertices from P */
            if(lx == 0 && scounter == warpCount)
            {
                path_eliminated = true;
            }

            if(lx == 0 && maxCount[wx] == scounter + 1)
            {
                path_more_explore = true; //shared, unsafe, but okay
                partition_set[wx] = true;
                maxCount[wx] = warpCount;
                maxIndex[wx] = cur;
            }
            else if(lx == 0 && maxCount[wx] < warpCount)
            {
                maxCount[wx] = warpCount;
                maxIndex[wx] = cur;
            }
        }
        __syncthreads();

        if(path_more_explore && !path_eliminated)
        {
            if(lx == 0 && partition_set[wx])
            {
                atomicMax(&(maxIntersection), maxCount[wx]);
            }
            __syncthreads();
            
            if(lx == 0)
            {
                if(maxIntersection == maxCount[wx])
                {
                    atomicMin(&(level_pivot[0]), maxIndex[wx]);
                }
            }
            __syncthreads();

            //Prepare the Possible and Intersection Encode Lists
            uint64 warpCount = 0;
            for (T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
            {
                T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
                pl[j] = ~(encode[(level_pivot[0])*num_divs_local + j]) & m;
                cl[j] = m;
                xl[j] = 0;
                warpCount += __popc(pl[j]);
            }
            reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
            if(lx == 0 && threadIdx.x < num_divs_local)
            {
                atomicAdd(&(level_count[0]), (T)warpCount);
            }
            __syncthreads();
        }
        else
        {
            if(threadIdx.x == 0 && sz1[2] == 0) 
            {
                atomicAdd(cpn + src, 1);
            }
            __syncthreads();
        }

        // Explore the tree
        while(level_count[l - 3] > 0)
        {
            T maskBlock = level_prev_index[l - 3] / 32;
            T maskIndex = ~((1 << (level_prev_index[l - 3] & 0x1F)) -1);
            T newIndex = __ffs(pl[num_divs_local*(l-3) + maskBlock] & maskIndex);
            while(newIndex == 0)
            {
                maskIndex = 0xFFFFFFFF;
                maskBlock++;
                newIndex = __ffs(pl[num_divs_local*(l - 3) + maskBlock] & maskIndex);
            }
            newIndex =  32 * maskBlock + newIndex - 1;
            T sameBlockMask = (~((1 << (newIndex & 0x1F)) - 1)) | ~pl[num_divs_local*(l-3) + maskBlock];
            __syncthreads();
            
            if (threadIdx.x == 0)
            {
                level_prev_index[l - 3] = newIndex + 1;
                level_count[l - 3]--;
                level_pivot[l - 2] = 0xFFFFFFFF;
                path_more_explore = false;
                maxIntersection = 0;
                sz1[l] = sz2 = 0;
            }
            __syncthreads();

            for(T j = threadIdx.x; j < sz1[l - 1];j += BLOCK_DIM_X)
            {
                T cur = buffer1[j];
                if((encode[cur * num_divs_local + (newIndex >> 5)] >> (newIndex & 0x1F)) & 1) {
                    buffer2[atomicAdd(&sz1[l], 1)] = cur;
                } else {
                    buffer2[sz1[l - 1] - atomicAdd(&sz2, 1) - 1] = cur;
                }
            }
            __syncthreads();
            for(T j = threadIdx.x; j < sz1[l - 1];j += BLOCK_DIM_X)
            {
                buffer1[j] = buffer2[j];
            }
            __syncthreads();

            // Now prepare intersection list
            T* from_cl = &(cl[num_divs_local * (l - 3)]);
            T* to_cl =  &(cl[num_divs_local * (l - 2)]);
            T* from_xl = &(xl[num_divs_local * (l - 3)]);
            T* to_xl = &(xl[num_divs_local * (l - 2)]);

            for (T k = threadIdx.x; k < num_divs_local; k += BLOCK_DIM_X)
            {
                // binary encoding operations of X in induced subgraph
                // similar as the inverse of P(cl)
                to_xl[k] = from_xl[k] | ( (maskBlock < k) ? pl[num_divs_local * (l - 3) + k] : ( (maskBlock > k) ? 0 : ~sameBlockMask) );
                to_xl[k] &= encode[newIndex * num_divs_local + k];
                to_cl[k] = from_cl[k] & encode[newIndex * num_divs_local + k];
                to_cl[k] = to_cl[k] & ( (maskBlock < k) ? ~pl[num_divs_local * (l - 3) + k] : ( (maskBlock > k) ? 0xFFFFFFFF : sameBlockMask) );
            }
            if(lx == 0)
            {	
                partition_set[wx] = false;
                maxCount[wx] = scounter + 1; //make it shared !!
                maxIndex[wx] = 0;
            }
            __syncthreads();
            
            /* Note that choosing a pivot from X can remove all vertices from P */
            uint64 warpCount = 0;
            curSZ = 0;
            path_eliminated = false;

            for (T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
            {
                warpCount += __popc(to_cl[j]);
            }
            reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);

            __syncthreads();
            if(lx == 0 && threadIdx.x < num_divs_local)
            {
                atomicAdd(&curSZ, warpCount);
            }
            __syncthreads();

            for (T j = wx;j < sz1[l];j += numPartitions)
            {
                warpCount = 0;
                T cur = buffer1[j];
                for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                {
                    warpCount += __popc(to_cl[k] & encode[cur * num_divs_local + k]);
                }
                reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);

                /* A pivot from X removes all vertices from P */
                if(lx == 0 && curSZ == warpCount)
                {
                    path_eliminated = true;
                }

                if(lx == 0 && maxCount[wx] == scounter + 1)
                {
                    partition_set[wx] = true;
                    path_more_explore = true; //shared, unsafe, but okay
                    maxCount[wx] = warpCount;
                    maxIndex[wx] = cur;
                }
                else if(lx == 0 && maxCount[wx] < warpCount)
                {
                    maxCount[wx] = warpCount;
                    maxIndex[wx] = cur;
                }	
            }
            __syncthreads();

            for (T j = wx; j < scounter; j += numPartitions)
            {
                warpCount = 0;
                T bi = j / 32;
                T ii = j & 0x1F;
                if( (to_cl[bi] & (1 << ii)) != 0 || (to_xl[bi] & (1 << ii)) != 0 )
                {
                    for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                    {
                        warpCount += __popc(to_cl[k] & encode[j * num_divs_local + k]);
                    }
                    reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);

                    /* A pivot from X removes all vertices from P */
                    if(lx == 0 && curSZ == warpCount)
                    {
                        path_eliminated = true;
                    }

                    if(lx == 0 && maxCount[wx] == scounter + 1)
                    {
                        partition_set[wx] = true;
                        path_more_explore = true; //shared, unsafe, but okay
                        maxCount[wx] = warpCount;
                        maxIndex[wx] = j;
                    }
                    else if(lx == 0 && maxCount[wx] < warpCount)
                    {
                        maxCount[wx] = warpCount;
                        maxIndex[wx] = j;
                    }	
                }
            }

            __syncthreads();
            if(!path_more_explore || path_eliminated)
            {
                __syncthreads();
                
                if (threadIdx.x == 0)
                {
                    x_empty = true;
                }
                __syncthreads();

                /* Check if X in induced subgraph is empty */
                for (T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
                {
                    if(to_xl[j])
                    {
                        x_empty = false;
                    }
                }
                __syncthreads();

                if(threadIdx.x == 0)
                {	
                    if(x_empty && sz1[l] == 0)
                    {
                        atomicAdd(cpn + src, 1);
                    }

                    /* No need to maintain X when backtracking. */
                    while (l > 3 && level_count[l - 3] == 0)
                    {
                        (l)--;
                    }
                }
                __syncthreads();
            }
            else
            {
                if(lx == 0 && partition_set[wx])
                {
                    atomicMax(&(maxIntersection), maxCount[wx]);
                }
                __syncthreads();

                if(lx == 0 && maxIntersection == maxCount[wx])
                {	
                    atomicMin(&(level_pivot[l-2]), maxIndex[wx]);
                }
                __syncthreads();

                warpCount = 0;
                for (T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
                {
                    T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
                    pl[(l - 2)*num_divs_local + j] = ~(encode[level_pivot[l - 2] * num_divs_local + j]) & to_cl[j] & m;
                    warpCount += __popc(pl[(l - 2)*num_divs_local + j]);
                }
                reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
                __syncthreads(); // Need this for degeneracy > 1024

                if(threadIdx.x == 0)
                {
                    l++;
                    level_count[l-3] = 0;
                    level_prev_index[l-3] = 0;
                }

                __syncthreads();
                if(lx == 0 && threadIdx.x < num_divs_local)
                {
                    atomicAdd(&(level_count[l - 3]), warpCount);
                }
            }
            __syncthreads();
        }
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}