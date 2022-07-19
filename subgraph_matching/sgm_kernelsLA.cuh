#pragma once
#include "include/utils.cuh"
#include "config.cuh"
#include "../include/queue.cuh"

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X)
    __global__ void sgm_kernel_compute_encoding(
        const graph::COOCSRGraph_d<T> g,
        const graph::GraphQueue_d<T, bool> current,
        const T head,
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
    GLOBAL_HANDLE<T> gh, const T eq_head,
    queue_callee(queue, tickets, head, tail))
{
    __shared__ SHARED_HANDLE<T, BLOCK_DIM_X> sh;

    if (threadIdx.x == 0)
    {
        sh.root_sm_block_id = sh.sm_block_id = blockIdx.x;
        sh.state = 0;
    }
    __syncthreads();
    while (sh.state != 100)
    {
        LOCAL_HANDLE<T> lh;
        lh.warpCount = 0;
        lh.go_ahead = true;

        if (sh.state == 0)
        {
            init_sm(sh, gh, eq_head);
            if (sh.state == 1)
            {
                if (threadIdx.x == 0)
                    queue_enqueue(queue, tickets, head, tail, CB, sh.sm_block_id);
                __syncthreads();
                continue;
            }

            // compute triangles/count till L3
            compute_triangles(sh, gh, lh);
            __syncthreads();
            if (!lh.go_ahead)
                continue;
        }
        else if (sh.state == 1)
        {
            __syncthreads();
            if (threadIdx.x == 0)
            {
                wait_for_donor(gh.work_ready[sh.sm_block_id], sh.state,
                               queue_caller(queue, tickets, head, tail));
            }
            __syncthreads();
            continue;
        }
        else if (sh.state == 2)
        {
            __syncthreads();

            // if (threadIdx.x == 0)
            // {
            //     sh.state = 1;
            //     queue_enqueue(queue, tickets, head, tail, CB, sh.sm_block_id);
            // }
            // __syncthreads();
            // continue;

            setup_stack_recepient(gh, sh);
            __syncthreads();
        }
        __syncthreads();

        // DFST work
        for (T j = 0; j < sh.srcLen; j++)
        {
            init_stack(sh, gh, lh, j);
            if (!lh.go_ahead)
                continue;

            // try dequeue here
            if (threadIdx.x == 0 && sh.state == 0)
            {
                sh.fork = false;
                try_dequeue(gh, sh, j, queue_caller(queue, tickets, head, tail));
            }
            __syncthreads();
            if (sh.fork)
                do_fork(gh, sh, j, queue_caller(queue, tickets, head, tail));

            __syncthreads();
            compute_intersection_block<T, BLOCK_DIM_X, true>(
                lh.warpCount, threadIdx.x, sh.num_divs_local, UINT32_MAX, sh.lvl, sh.to, sh.cl,
                sh.level_prev_index, sh.encode);

            check_terminate(sh, lh);

            while (sh.level_count[sh.lvl - 3] > sh.level_index[sh.lvl - 3])
            {
                // First Index
                __syncthreads();
                get_newIndex(sh, lh);
                __syncthreads();

                compute_intersection_block<T, BLOCK_DIM_X, true>(
                    lh.warpCount, threadIdx.x, sh.num_divs_local,
                    sh.newIndex, sh.lvl, sh.to, sh.cl,
                    sh.level_prev_index, sh.encode);

                backtrack(sh, lh);
                __syncthreads();
            }
            __syncthreads();
        }
        if (threadIdx.x == 0 && sh.state == 2)
        {
            sh.state = 1;
            queue_enqueue(queue, tickets, head, tail, CB, sh.sm_block_id);
        }
        __syncthreads();
        // finalize_count(sh, gh, lh);
        // __syncthreads();
        if (threadIdx.x == 0)
        {
            if (sh.sg_count > 0)
                atomicAdd(gh.counter, sh.sg_count);
        }
        __syncthreads();
    }
}