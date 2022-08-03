#pragma once
#include "host_utils.cuh"
#include "../../include/Timer.h"
#include "common_utils.cuh"

#define fundef_LD template <typename T, uint BLOCK_DIM_X, uint NP> \
__device__ __forceinline__

template <typename T>
struct GLOBAL_HANDLE
{
    uint64 *global_counter;
    uint64 *counter;
    graph::COOCSRGraph_d<T> g;
    mapping<T> *srcList;
    graph::GraphQueue_d<T, bool> current;

    T *current_level;
    MessageBlock *Message;
    int *offset;
    T *adj_enc;
    uint64 *work_list_head1;
    uint64 *work_list_head2;
    uint64 work_list_tail;
    int devId;
    T stride;
    T boundary;
    T cutoff;

    // for worker queue
    cuda::atomic<uint32_t, cuda::thread_scope_device> *work_ready;
};

template <typename T, uint BLOCK_DIM_X, uint NP>
__device__ struct SHARED_HANDLE
{
    T level_index[NP][DEPTH];
    T level_count[NP][DEPTH];
    T level_prev_index[NP][DEPTH];

    uint64 sg_count[NP];
    T src, srcStart, srcLen, srcSplit, dstIdx;

    T l[NP];
    T num_divs_local, *level_offset, *encode;
    T to[BLOCK_DIM_X], newIndex[NP];
    T tc, wtc[NP];

    // For Worker Queue
    cuda::atomic<uint32_t, cuda::thread_scope_device> *work_ready;
    T state, root_sm_block_id, sm_block_id, worker_pos[NP], shared_other_sm_block_id[NP];
    bool fork[NP];
};

__device__ struct LOCAL_HANDLE
{
    uint64 warpCount = 0;
};

fundef_LD void
init_sm(SHARED_HANDLE<T, BLOCK_DIM_X, NP> &sh, GLOBAL_HANDLE<T> &gh)
{
    __syncthreads();
    if (threadIdx.x == 0)
    {
        uint64 pos = atomicAdd(gh.work_list_head1, 1);
        if (pos < gh.boundary)
        {
            sh.src = gh.current.queue[pos];
            sh.srcStart = gh.g.rowPtr[sh.src];
            sh.srcSplit = gh.g.splitPtr[sh.src];
            sh.srcLen = gh.g.rowPtr[sh.src + 1] - sh.srcStart;
            sh.num_divs_local = (sh.srcLen + 32 - 1) / 32;
            sh.tc = 0;
        }
        else
        {
            uint64 index = atomicAdd(gh.work_list_head2, (uint64)gh.stride);
            if (index < gh.work_list_tail)
            {
                sh.src = gh.current.queue[index];
                sh.srcStart = gh.g.rowPtr[sh.src];
                sh.srcSplit = gh.g.splitPtr[sh.src];
                sh.srcLen = gh.g.rowPtr[sh.src + 1] - sh.srcStart;
                sh.num_divs_local = (sh.srcLen + 32 - 1) / 32;
                sh.tc = 0;
            }
            else
            {
                // sh.state = 100;
                // 1: Block ready for other work, 100: terminate block
                sh.state = 1;
            }
        }
    }
    __syncthreads();
}

fundef_LD void
encode(SHARED_HANDLE<T, BLOCK_DIM_X, NP> &sh,
       GLOBAL_HANDLE<T> &gh)
{
    __syncthreads();
    constexpr T CPARTSIZE = BLOCK_DIM_X / NP;
    const T wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const T lx = threadIdx.x % CPARTSIZE;

    for (T j = wx; j < sh.srcLen; j += NP)
    {
        for (T k = lx; k < sh.num_divs_local; k += CPARTSIZE)
        {
            sh.encode[j * sh.num_divs_local + k] = 0x00;
        }
    }
    __syncthreads();

    for (T j = wx; j < sh.srcLen; j += NP)
    {
        warp_sorted_count_and_encode_full_undirected<WARPS_PER_BLOCK, T, true, CPARTSIZE>(
            &gh.g.oriented_colInd[sh.srcStart], sh.srcLen,
            &gh.g.colInd[gh.g.rowPtr[gh.g.oriented_colInd[sh.srcStart + j]]],
            gh.g.rowPtr[gh.g.oriented_colInd[sh.srcStart + j] + 1] - gh.g.rowPtr[gh.g.oriented_colInd[sh.srcStart + j]],
            j, sh.num_divs_local,
            sh.encode);
    }
    __syncthreads();
}

fundef_LD void
init_stack(SHARED_HANDLE<T, BLOCK_DIM_X, NP> &sh, GLOBAL_HANDLE<T> &gh, const T partMask, T j)
{
    constexpr T CPARTSIZE = BLOCK_DIM_X / NP;
    const T wx = threadIdx.x / CPARTSIZE;
    const T lx = threadIdx.x % CPARTSIZE;
    for (T k = lx; k < DEPTH; k += CPARTSIZE)
    {
        sh.level_count[wx][k] = 0;
        sh.level_index[wx][k] = 0;
        sh.level_prev_index[wx][k] = 0;
    }
    __syncwarp(partMask);
    if (lx == 1)
    {
        sh.l[wx] = 2;
        sh.level_prev_index[wx][0] = sh.srcSplit - sh.srcStart + 1; // for symmetry breaking purposes
        sh.level_prev_index[wx][1] = j + 1;
    }
    __syncwarp(partMask);
}

fundef_LD void
init_stack_block(SHARED_HANDLE<T, BLOCK_DIM_X, NP> &sh, GLOBAL_HANDLE<T> &gh, T *cl, T j, const T partMask)
{
    constexpr T CPARTSIZE = BLOCK_DIM_X / NP;
    const T wx = threadIdx.x / CPARTSIZE;
    const T lx = threadIdx.x % CPARTSIZE;
    if (lx == 0)
    {
        sh.l[wx] = gh.Message[blockIdx.x].level_ + 1;
        sh.level_prev_index[wx][sh.l[wx] - 1] = j + 1;
    }
    __syncwarp(partMask);
    for (T k = lx + sh.l[wx]; k < DEPTH; k += CPARTSIZE)
    {
        sh.level_count[wx][k] = 0;
        sh.level_index[wx][k] = 0;
        sh.level_prev_index[wx][k] = 0;
    }
    for (T k = lx; k < sh.num_divs_local; k += CPARTSIZE)
    {
        cl[k] = get_mask(sh.srcLen, k);
        for (T l = 1; l < sh.l[wx]; l++)
        {
            cl[k] &= unset_mask(sh.level_prev_index[wx][l] - 1, k);
        }
    }
}

fundef_LD void
count_tri(LOCAL_HANDLE &lh, SHARED_HANDLE<T, BLOCK_DIM_X, NP> &sh,
          GLOBAL_HANDLE<T> &gh, const T partMask, T *cl, const T j)
{
    constexpr T CPARTSIZE = BLOCK_DIM_X / NP;
    const T wx = threadIdx.x / CPARTSIZE;
    const T lx = threadIdx.x % CPARTSIZE;
    lh.warpCount = 0;
    for (T k = lx; k < sh.num_divs_local; k += CPARTSIZE)
    {
        cl[k] = get_mask(sh.srcLen, k) & unset_mask(j, k);      // unset mask returns all ones except at bit j
                                                                // if (j == 32 * (srcLen / 32) + 1)
        if (QEDGE_PTR[sh.l[wx] + 1] - QEDGE_PTR[sh.l[wx]] == 2) // i.e. if connected to both nodes at level 0 and 1
            sh.to[threadIdx.x] = sh.encode[j * sh.num_divs_local + k];
        else
            sh.to[threadIdx.x] = cl[k]; // if only connected to central node, everything is a candidate.

        // Remove Redundancies
        for (T sym_idx = SYMNODE_PTR[sh.l[wx]]; sym_idx < SYMNODE_PTR[sh.l[wx] + 1]; sym_idx++)
        {
            if (SYMNODE[sym_idx] > 0)
            {
                sh.to[threadIdx.x] &= ~(cl[k] & get_mask(j, k)); // if symmetric to level 1 (lexicographic symmetry breaking)
            }
        }
        // to[threadIdx.x] &= ~EP_mask[1 * num_divs_local + k];
        cl[(sh.l[wx] - 1) * sh.num_divs_local + k] = sh.to[threadIdx.x]; // candidates for level 2
        lh.warpCount += __popc(sh.to[threadIdx.x]);
    }
    reduce_part<T, CPARTSIZE>(partMask, lh.warpCount);
}

fundef_LD void
get_wc_block(LOCAL_HANDLE &lh, SHARED_HANDLE<T, BLOCK_DIM_X, NP> &sh,
             GLOBAL_HANDLE<T> &gh)
{
    lh.warpCount = 0;
    constexpr T CPARTSIZE = BLOCK_DIM_X / NP;
    const T wx = threadIdx.x / CPARTSIZE;
    // count triangles
    T dstIdx = sh.level_prev_index[wx][sh.l[wx] - 1] - 1;
    for (T k = threadIdx.x; k < sh.num_divs_local; k += BLOCK_DIM_X)
    {
        sh.level_offset[k] &= get_mask(sh.srcLen, k) & unset_mask(dstIdx, k);

        if (QEDGE_PTR[sh.l[wx] + 1] - QEDGE_PTR[sh.l[wx]] == 2)
            sh.to[threadIdx.x] = sh.encode[dstIdx * sh.num_divs_local + k];
        else
            sh.to[threadIdx.x] = sh.level_offset[k];

        // Remove Redundancies
        for (T sym_idx = SYMNODE_PTR[sh.l[wx]]; sym_idx < SYMNODE_PTR[sh.l[wx] + 1]; sym_idx++)
        {
            if (SYMNODE[sym_idx] > 0)
                sh.to[threadIdx.x] &= ~(sh.level_offset[k] & get_mask(dstIdx, k));
        }
        sh.level_offset[sh.num_divs_local * (sh.l[wx] - 1) + k] = sh.to[threadIdx.x];
        lh.warpCount += __popc(sh.to[threadIdx.x]);
    }
    typedef cub::BlockReduce<uint64, BLOCK_DIM_X> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    lh.warpCount = BlockReduce(temp_storage).Sum(lh.warpCount); // whole block performs reduction here
    __syncthreads();
}

fundef_LD void
check_terminate(LOCAL_HANDLE &lh, SHARED_HANDLE<T, BLOCK_DIM_X, NP> &sh, const T partMask)
{
    constexpr T CPARTSIZE = BLOCK_DIM_X / NP;
    const T wx = threadIdx.x / CPARTSIZE;
    const T lx = threadIdx.x % CPARTSIZE;
    if (lx == 0)
    {
        sh.l[wx]++;
        if (sh.l[wx] == KCCOUNT - LUNMAT) // If reached last level
        {
            sh.sg_count[wx] += lh.warpCount; // code reaches here only if counting triangles
        }
        else
        {
            sh.level_count[wx][sh.l[wx]] = lh.warpCount; // since 2 levels already finalized and l[wx] indexed from 1
            sh.level_index[wx][sh.l[wx]] = 0;            // This index to iterated from 0 to warpCount
        }
    }
    __syncwarp(partMask); // all triangles found till here!
}

fundef_LD void
get_newIndex(LOCAL_HANDLE &lh, SHARED_HANDLE<T, BLOCK_DIM_X, NP> &sh, const T partMask, const T *cl)
{
    constexpr T CPARTSIZE = BLOCK_DIM_X / NP;
    const T wx = threadIdx.x / CPARTSIZE;
    const T lx = threadIdx.x % CPARTSIZE;
    __syncwarp(partMask);
    if (lx == 0)
    {
        const T *from = &(cl[sh.num_divs_local * (sh.l[wx] - 2)]);                  // all the current candidates
        T maskBlock = sh.level_prev_index[wx][sh.l[wx] - 1] / 32;                   // to identify which 32 bits to pick from num_divs_local
        T maskIndex = ~((1 << (sh.level_prev_index[wx][sh.l[wx] - 1] & 0x1F)) - 1); // to unset previously visited index

        sh.newIndex[wx] = __ffs(from[maskBlock] & maskIndex); //__ffs is find first set bit returns 0 if nothing set
        while (sh.newIndex[wx] == 0)                          // if not found, look into next block
        {
            maskIndex = 0xFFFFFFFF;
            maskBlock++;
            sh.newIndex[wx] = __ffs(from[maskBlock] & maskIndex);
        }
        sh.newIndex[wx] = 32 * maskBlock + sh.newIndex[wx] - 1; // actual new index

        sh.level_prev_index[wx][sh.l[wx] - 1] = sh.newIndex[wx] + 1; // level prev index is numbered from 1
        sh.level_index[wx][sh.l[wx]]++;
    }
    __syncwarp(partMask);
}

fundef_LD void
backtrack(LOCAL_HANDLE &lh, SHARED_HANDLE<T, BLOCK_DIM_X, NP> &sh, const T partMask, T *cl)
{
    constexpr T CPARTSIZE = BLOCK_DIM_X / NP;
    const T wx = threadIdx.x / CPARTSIZE;
    const T lx = threadIdx.x % CPARTSIZE;

    if (lx == 0)
    {
        if (sh.l[wx] + 1 == KCCOUNT - LUNMAT)
        {
            sh.sg_count[wx] += lh.warpCount;
        }
        else if (sh.l[wx] + 1 < KCCOUNT - LUNMAT) // Not at last level yet
        {
            (sh.l[wx])++;                                // go further
            sh.level_count[wx][sh.l[wx]] = lh.warpCount; // save level_count (warpcount updated during compute intersection)
            sh.level_index[wx][sh.l[wx]] = 0;            // initialize level index
            sh.level_prev_index[wx][sh.l[wx] - 1] = 0;   // initialize level previous index

            T idx = sh.level_prev_index[wx][sh.l[wx] - 2] - 1; //-1 since 1 is always added to level previous index
            cl[idx / 32] &= ~(1 << (idx & 0x1F));              // idx & 0x1F gives remainder after dividing by 32
                                                               // this puts the newindex at correct place in current level
        }
        while (sh.l[wx] > 3 && sh.level_index[wx][sh.l[wx]] >= sh.level_count[wx][sh.l[wx]]) // reset memory since will be going out of while loop (i.e. backtracking)
        {
            (sh.l[wx])--;
            T idx = sh.level_prev_index[wx][sh.l[wx] - 1] - 1;
            cl[idx / 32] |= 1 << (idx & 0x1F);
        }
    }
    __syncwarp(partMask);
}

fundef_LD void
try_dequeue(SHARED_HANDLE<T, BLOCK_DIM_X, NP> &sh, GLOBAL_HANDLE<T> &gh,
            queue_callee(queue, tickets, head, tail))
{
    constexpr T CPARTSIZE = BLOCK_DIM_X / NP;
    const T wx = threadIdx.x / CPARTSIZE;
    if (gh.work_list_head2[0] >= gh.work_list_tail)
        queue_dequeue(queue, tickets, head, tail, CB, sh.fork[wx], sh.worker_pos[wx], N_RECEPIENTS);
}

fundef_LD void
do_fork(SHARED_HANDLE<T, BLOCK_DIM_X, NP> &sh, GLOBAL_HANDLE<T> &gh, const T j,
        queue_callee(queue, tickets, head, tail))
{
    constexpr T CPARTSIZE = BLOCK_DIM_X / NP;
    const T wx = threadIdx.x / CPARTSIZE;
    const T lx = threadIdx.x % CPARTSIZE;
    if (lx == 0)
    {
        for (T iter = 0; iter < N_RECEPIENTS; iter++)
        {
            queue_wait_ticket(queue, tickets, head, tail, CB, sh.worker_pos[wx], sh.shared_other_sm_block_id[wx]);

            T other_sm_block_id = sh.shared_other_sm_block_id[wx];
            gh.Message[other_sm_block_id].src_ = sh.src;
            gh.Message[other_sm_block_id].dstIdx_ = j;
            gh.Message[other_sm_block_id].encode_ = sh.encode;
            gh.Message[other_sm_block_id].root_sm_block_id_ = sh.sm_block_id;
            gh.Message[other_sm_block_id].level_ = sh.l[wx];

            for (T l = 0; l < sh.l[wx]; l++)
            {
                gh.Message[other_sm_block_id].level_prev_index_[l] = sh.level_prev_index[wx][l];
            }

            gh.work_ready[other_sm_block_id].store(1, cuda::memory_order_release);
            sh.worker_pos[wx]++;
        }
        if (sh.l[wx] < 3)
            sh.wtc[wx] = atomicAdd(&(sh.tc), 1);
    }
}

fundef_LD void
setup_stack(SHARED_HANDLE<T, BLOCK_DIM_X, NP> &sh, GLOBAL_HANDLE<T> &gh)
{
    __syncthreads();
    constexpr T CPARTSIZE = BLOCK_DIM_X / NP;
    const T wx = threadIdx.x / CPARTSIZE;
    const T lx = threadIdx.x % CPARTSIZE;
    if (threadIdx.x == 0)
    {
        sh.src = gh.Message[blockIdx.x].src_;
        sh.encode = gh.Message[blockIdx.x].encode_;
        sh.root_sm_block_id = gh.Message[blockIdx.x].root_sm_block_id_;
        sh.dstIdx = gh.Message[blockIdx.x].dstIdx_;

        // setup SM
        sh.srcStart = gh.g.rowPtr[sh.src];
        sh.srcSplit = gh.g.splitPtr[sh.src];
        sh.srcLen = gh.g.rowPtr[sh.src + 1] - sh.srcStart;
        sh.sm_block_id = blockIdx.x;
        sh.num_divs_local = (sh.srcLen + 32 - 1) / 32;
        sh.tc = 0;
    }
    if (lx == 0)
    {
        sh.l[wx] = gh.Message[blockIdx.x].level_;
        for (T l = 0; l < sh.l[wx]; l++)
        {
            sh.level_prev_index[wx][l] = gh.Message[blockIdx.x].level_prev_index_[l];
        }
    }
    __syncthreads();
}

fundef_LD void
clear_messages(SHARED_HANDLE<T, BLOCK_DIM_X, NP> &sh, GLOBAL_HANDLE<T> &gh)
{
    constexpr T CPARTSIZE = BLOCK_DIM_X / NP;
    const T wx = threadIdx.x / CPARTSIZE;
    const T lx = threadIdx.x % CPARTSIZE;
    if (threadIdx.x == 0)
    {
        gh.Message[blockIdx.x].src_ = 0;
        gh.Message[blockIdx.x].encode_ = 0;
        gh.Message[blockIdx.x].root_sm_block_id_ = 0;
        gh.Message[blockIdx.x].dstIdx_ = 0;
        gh.Message[blockIdx.x].level_ = 0;
    }
    if (lx == 0)
    {
        for (T l = 0; l < KCCOUNT; l++)
            gh.Message[blockIdx.x].level_prev_index_[l] = 0;
    }
    __syncthreads();
}

template <typename T, uint CPARTSIZE, bool MAT>
__device__ __forceinline__ void compute_intersection(
    uint64 &wc,
    const size_t lx, const T partMask,
    const T num_divs_local, const T maskIdx, const T lvl,
    T *to, T *cl, const T *level_prev_index, const T *encode)
{
    wc = 0;
    for (T k = lx; k < num_divs_local; k += CPARTSIZE)
    {
        to[threadIdx.x] = cl[k] & unset_mask(maskIdx, k);
        // Compute Intersection
        for (T q_idx = QEDGE_PTR[lvl] + 1; q_idx < QEDGE_PTR[lvl + 1]; q_idx++)
        {
            to[threadIdx.x] &= encode[(level_prev_index[QEDGE[q_idx]] - 1) * num_divs_local + k];
        }

        // Remove Redundancies
        for (T sym_idx = SYMNODE_PTR[lvl]; sym_idx < SYMNODE_PTR[lvl + 1]; sym_idx++)
        {
            if (!MAT && SYMNODE[sym_idx] == lvl - 1)
                continue;
            if (SYMNODE[sym_idx] > 0)
                to[threadIdx.x] &= ~(get_mask(level_prev_index[SYMNODE[sym_idx]] - 1, k));
        }

        wc += __popc(to[threadIdx.x]);                        // counts number of set bits
        cl[(lvl - 1) * num_divs_local + k] = to[threadIdx.x]; // saves candidates list in cl
    }
    reduce_part<T, CPARTSIZE>(partMask, wc);
}

template <typename T, uint BLOCK_DIM_X, bool MAT>
__device__ __forceinline__ void compute_intersection_block(
    uint64 &wc, const T num_divs_local, const T maskIdx,
    const T lvl, T *to, T *cl, const T *level_prev_index, const T *encode)
{
    wc = 0;
    for (T k = threadIdx.x; k < num_divs_local; k += BLOCK_DIM_X)
    {
        to[threadIdx.x] = cl[k] & unset_mask(maskIdx, k);

        // Compute Intersection
        if (QEDGE_PTR[lvl] + 1 == QEDGE_PTR[lvl + 1])
        {
            to[threadIdx.x] = cl[k];
        }
        else
        {
            for (T q_idx = QEDGE_PTR[lvl] + 1; q_idx < QEDGE_PTR[lvl + 1]; q_idx++)
            {
                to[threadIdx.x] &= encode[(level_prev_index[QEDGE[q_idx]] - 1) * num_divs_local + k];
            }
        }
        // Remove Redundancies
        for (T sym_idx = SYMNODE_PTR[lvl]; sym_idx < SYMNODE_PTR[lvl + 1]; sym_idx++)
        {
            if (SYMNODE[sym_idx] > 0)
                to[threadIdx.x] &= ~(get_mask(level_prev_index[SYMNODE[sym_idx]] - 1, k));
        }
        cl[(lvl - 1) * num_divs_local + k] = to[threadIdx.x];
        wc += __popc(to[threadIdx.x]);
    }
    __syncthreads();

    typedef cub::BlockReduce<uint64, BLOCK_DIM_X> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    wc = BlockReduce(temp_storage).Sum(wc); // whole block performs reduction here
    __syncthreads();
}