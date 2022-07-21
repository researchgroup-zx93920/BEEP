#pragma once
#include "include/utils.cuh"
#include "include/utils_LD.cuh"
#include "include/common_utils.cuh"
#include "config.cuh"

enum CounterName
{
    OTHER = 0,
    ACTIVE,
    INACTIVE,
    TOTAL,
    NUM_COUNTERS
};
char *Names[] = {
    "OTHER",
    "ACTIVE",
    "INACTIVE",
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
__launch_bounds__(BLOCK_DIM_X)
    __global__ void sgm_kernel_central_node_function(
        Counters *SM_times, double *butil,
        GLOBAL_HANDLE<T> gh, queue_callee(queue, tickets, head, tail))
{
    constexpr T NP = BLOCK_DIM_X / CPARTSIZE;
    const T wx = threadIdx.x / CPARTSIZE;
    const T lx = threadIdx.x % CPARTSIZE;
    const T partMask = ((1 << CPARTSIZE) - 1) << ((wx % (32 / CPARTSIZE)) * CPARTSIZE);

    __shared__ Counters btimes;
    __shared__ Counters wtimes[NP];
    __shared__ SHARED_HANDLE_LD<T, BLOCK_DIM_X, NP> sh;
    LOCAL_HANDLE_LD lh;

    initializeCounters<T>(&btimes, threadIdx.x, 0XFFFFFFFF);
    startTime<T>(TOTAL, &btimes, threadIdx.x, 0XFFFFFFFF);
    if (threadIdx.x == 0)
    {
        sh.root_sm_block_id = sh.sm_block_id = blockIdx.x;
        sh.state = 0;
    }
    __syncthreads();
    while (sh.state != 100)
    {
        lh.warpCount = 0;

        if (sh.state == 0)
        {
            startTime<T>(ACTIVE, &btimes, threadIdx.x, 0XFFFFFFFF);
            init_sm(sh, gh);

            if (sh.state == 1)
            {
                if (threadIdx.x == 0)
                    queue_enqueue(queue, tickets, head, tail, CB, sh.sm_block_id);
                __syncthreads();
                continue;
            }
            // if (sh.state == 100)
            // 	break;
            if (lx == 0)
            {
                sh.sg_count[wx] = 0;
            }

            encode(sh, gh);
            if (lx == 0)
                sh.wtc[wx] = atomicAdd(&(sh.tc), 1);
            __syncwarp(partMask);

            while (sh.wtc[wx] < sh.srcLen)
            {
                T j = sh.wtc[wx];
                if (!(SYMNODE_PTR[2] == 1 && j < (sh.srcSplit - sh.srcStart)))
                {
                    T *cl = sh.level_offset + wx * (NUMDIVS * MAXLEVEL);
                    init_stack(sh, gh, partMask, j);

                    // try dequeue here
                    if (lx == 0 && sh.state == 0)
                    {
                        sh.fork[wx] = false;
                        LD_try_dequeue(sh, gh, j, queue_caller(queue, tickets, head, tail));
                    }
                    __syncwarp(partMask);
                    if (sh.fork[wx])
                    {
                        if (lx == 0)
                        {
                            LD_do_fork(sh, gh, j, queue_caller(queue, tickets, head, tail));
                            sh.wtc[wx] = atomicAdd(&(sh.tc), 1);
                        }
                        __syncwarp(partMask);
                        continue;
                    }
                    __syncwarp(partMask);

                    // get wc
                    count_tri(lh, sh, gh, partMask, cl, j);

                    check_terminate(lh, sh, partMask);
                    while (sh.level_index[wx][sh.l[wx] - 3] < sh.level_count[wx][sh.l[wx] - 3])
                    {
                        get_newIndex(lh, sh, partMask, cl);
                        compute_intersection<T, CPARTSIZE, true>(
                            lh.warpCount, lx, partMask,
                            sh.num_divs_local, sh.newIndex[wx], sh.l[wx],
                            sh.to, cl, sh.level_prev_index[wx], sh.encode);

                        backtrack(lh, sh, partMask, cl);
                    }
                }
                if (lx == 0)
                    sh.wtc[wx] = atomicAdd(&(sh.tc), 1);
                __syncwarp(partMask);
            }
            if (lx == 0 && sh.sg_count[wx] > 0)
            {
                atomicAdd(gh.counter, sh.sg_count[wx]);
            }
            __syncthreads();
            endTime(ACTIVE, &btimes, threadIdx.x, 0XFFFFFFFF);
        }
        else if (sh.state == 1)
        {
            startTime(INACTIVE, &btimes, threadIdx.x, 0XFFFFFFFF);
            __syncthreads();
            if (threadIdx.x == 0)
            {
                wait_for_donor(gh.work_ready[sh.sm_block_id], sh.state,
                               queue_caller(queue, tickets, head, tail));
            }
            __syncthreads();
            endTime(INACTIVE, &btimes, threadIdx.x, 0XFFFFFFFF);
        }
        else if (sh.state == 2)
        {
            startTime<T>(ACTIVE, &btimes, threadIdx.x, 0XFFFFFFFF);
            LD_setup_stack_recepient(sh, gh);

            count_tri_block(lh, sh, gh);

            if (KCCOUNT == 3)
            {
                if (threadIdx.x == 0 && lh.warpCount > 0)
                    atomicAdd(gh.counter, lh.warpCount);
            }
            else
            {
                if (lx == 0)
                {
                    sh.sg_count[wx] = 0;
                    sh.wtc[wx] = atomicAdd(&(sh.tc), 1);
                }
                __syncwarp(partMask);
                while (sh.wtc[wx] < sh.srcLen)
                {
                    T j = sh.wtc[wx];
                    if (!((sh.level_offset[sh.num_divs_local + j / 32] >> (j % 32)) % 2 == 0))
                    {
                        T *cl = sh.level_offset + wx * (NUMDIVS * MAXLEVEL);

                        // init stack --block
                        init_stack_block(sh, gh, cl, j);
                        __syncwarp(partMask);

                        compute_intersection<T, CPARTSIZE, true>(
                            lh.warpCount, lx, partMask,
                            sh.num_divs_local, UINT32_MAX, sh.l[wx], sh.to, cl,
                            sh.level_prev_index[wx], sh.encode);

                        if (lx == 0)
                        {
                            if (sh.l[wx] + 1 == KCCOUNT)
                                sh.sg_count[wx] += lh.warpCount;
                            else
                            {
                                sh.l[wx]++;
                                sh.level_count[wx][sh.l[wx] - 3] = lh.warpCount;
                                sh.level_index[wx][sh.l[wx] - 3] = 0;
                                sh.level_prev_index[wx][sh.l[wx] - 1] = 0;
                            }
                        }
                        __syncwarp(partMask);
                        while (sh.level_count[wx][sh.l[wx] - 3] > sh.level_index[wx][sh.l[wx] - 3])
                        {
                            get_newIndex(lh, sh, partMask, cl);
                            compute_intersection<T, CPARTSIZE, true>(
                                lh.warpCount, lx, partMask, sh.num_divs_local,
                                sh.newIndex[wx], sh.l[wx], sh.to, cl,
                                sh.level_prev_index[wx], sh.encode);

                            if (lx == 0)
                            {
                                if (sh.l[wx] + 1 == KCCOUNT)
                                    sh.sg_count[wx] += lh.warpCount;
                                else if (sh.l[wx] + 1 < KCCOUNT) //&& warpCount >= KCCOUNT - l[wx])
                                {
                                    (sh.l[wx])++;
                                    sh.level_count[wx][sh.l[wx] - 3] = lh.warpCount;
                                    sh.level_index[wx][sh.l[wx] - 3] = 0;
                                    sh.level_prev_index[wx][sh.l[wx] - 1] = 0;
                                    T idx = sh.level_prev_index[wx][sh.l[wx] - 2] - 1;
                                    cl[idx / 32] &= ~(1 << (idx & 0x1F));
                                }

                                while (sh.l[wx] > 4 && sh.level_index[wx][sh.l[wx] - 3] >= sh.level_count[wx][sh.l[wx] - 3])
                                {
                                    (sh.l[wx])--;
                                    T idx = sh.level_prev_index[wx][sh.l[wx] - 1] - 1;
                                    cl[idx / 32] |= 1 << (idx & 0x1F);
                                }
                            }
                            __syncwarp(partMask);
                        }
                        __syncwarp(partMask);
                    }
                    if (lx == 0)
                        sh.wtc[wx] = atomicAdd(&(sh.tc), 1);
                    __syncwarp(partMask);
                }
                if (lx == 0 && sh.sg_count[wx] > 0)
                {
                    atomicAdd(gh.counter, sh.sg_count[wx]);
                }
            }

            __syncthreads();
            if (threadIdx.x == 0)
            {
                sh.state = 1; // done with donated task, back to inactive state
                queue_enqueue(queue, tickets, head, tail, CB, sh.sm_block_id);
            }
            __syncthreads();
            endTime(ACTIVE, &btimes, threadIdx.x, 0XFFFFFFFF);
        }
    }
    endTime<T>(TOTAL, &btimes, threadIdx.x, 0XFFFFFFFF);
    __syncthreads();
    if (threadIdx.x == 0)
    {
        butil[blockIdx.x] = (btimes.totalTime[ACTIVE] * 1.0) / (btimes.totalTime[INACTIVE] + btimes.totalTime[ACTIVE]);

        for (int i = 0; i < NUM_COUNTERS; i++)
            atomicAdd(&SM_times[__mysmid()].totalTime[i], btimes.totalTime[i]);
    }
}