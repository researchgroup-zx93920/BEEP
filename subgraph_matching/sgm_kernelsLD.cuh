#pragma once
#include "include/host_utils.cuh"
#include "include/device_utils.cuh"
#include "include/common_utils.cuh"
#include "config.cuh"

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X)
		__global__ void sgm_kernel_central_node_function(
				GLOBAL_HANDLE<T> gh, // T *cpn,
				queue_callee(queue, tickets, head, tail))
{
	constexpr T NP = BLOCK_DIM_X / CPARTSIZE;
	const T wx = threadIdx.x / CPARTSIZE;
	const T lx = threadIdx.x % CPARTSIZE;
	const T partMask = ((1 << CPARTSIZE) - 1) << ((wx % (32 / CPARTSIZE)) * CPARTSIZE);
	__shared__ SHARED_HANDLE<T, BLOCK_DIM_X, NP> sh;
	LOCAL_HANDLE lh;

	if (threadIdx.x == 0)
	{
		sh.root_sm_block_id = sh.sm_block_id = blockIdx.x;
		sh.state = 0;
		sh.encode = &gh.adj_enc[(uint64)blockIdx.x * NUMDIVS * MAXDEG];
		sh.level_offset = &gh.current_level[(uint64)(blockIdx.x * NUMDIVS * NP * MAXLEVEL)];
	}
	__syncthreads();
	while (sh.state != 100)
	{
		lh.warpCount = 0;
		if (sh.state == 0)
		{
			init_sm(sh, gh);

			// if (sh.state == 100)
			// {
			// 	continue;
			// }
			if (sh.state == 1)
			{
				if (threadIdx.x == 0)
					queue_enqueue(queue, tickets, head, tail, CB, sh.sm_block_id);
				__syncthreads();
				continue;
			}

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
				if ((!(SYMNODE_PTR[2] == 1 && j < (sh.srcSplit - sh.srcStart))) && ((sh.srcLen > gh.cutoff && (j % NDEV) == gh.devId) || sh.srcLen <= gh.cutoff))
				{
					T *cl = sh.level_offset + wx * (NUMDIVS * MAXLEVEL);
					init_stack(sh, gh, partMask, j);

					// try dequeue here
					if (lx == 0)
					{
						sh.fork[wx] = false;
						try_dequeue(sh, gh, queue_caller(queue, tickets, head, tail));
					}
					__syncwarp(partMask);
					if (sh.fork[wx])
					{
						do_fork(sh, gh, j, queue_caller(queue, tickets, head, tail));
						__syncwarp(partMask);
						continue;
					}

					__syncwarp(partMask);

					// get wc
					count_tri(lh, sh, gh, partMask, cl, j);
					__syncwarp(partMask);

					check_terminate(lh, sh, partMask);
					while (sh.level_index[wx][sh.l[wx]] < sh.level_count[wx][sh.l[wx]])
					{
						get_newIndex(lh, sh, partMask, cl);
						if (sh.l[wx] <= (KCCOUNT + 1) / 2 /*&& sh.level_index[wx][sh.l[wx]] - sh.level_count[wx][sh.l[wx]] > NP*/)
						{
							// try L3 dequeue here
							if (lx == 0)
							{
								sh.fork[wx] = false;
								try_dequeue(sh, gh, queue_caller(queue, tickets, head, tail));
							}
							__syncwarp(partMask);
							if (sh.fork[wx])
							{
								do_fork(sh, gh, j, queue_caller(queue, tickets, head, tail));
								lh.warpCount = 0;
								__syncwarp(partMask);
								backtrack(lh, sh, partMask, cl);
								continue;
							}
						}

						compute_intersection<T, CPARTSIZE, true>(
								lh.warpCount, lx, partMask,
								sh.num_divs_local, sh.newIndex[wx], sh.l[wx],
								sh.to, cl, sh.level_prev_index[wx], sh.encode);

						__syncwarp(partMask);

						backtrack(lh, sh, partMask, cl);

						__syncwarp(partMask);
					}
				}
				__syncwarp(partMask);
				if (lx == 0)
				{
					sh.wtc[wx] = atomicAdd(&(sh.tc), 1);
				}
				__syncwarp(partMask);
			}
			if (lx == 0 && sh.sg_count[wx] > 0)
			{
				atomicAdd(gh.counter, sh.sg_count[wx]);
				// atomicAdd(&cpn[sh.src], sh.sg_count[wx]);
			}

			__syncthreads();
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
		}
		else if (sh.state == 2)
		{
			T *cl = sh.level_offset + wx * (NUMDIVS * MAXLEVEL);
			constexpr T CPARTSIZE = BLOCK_DIM_X / NP;
			const T wx = threadIdx.x / CPARTSIZE;
			const T lx = threadIdx.x % CPARTSIZE;

			setup_stack(sh, gh);
			__syncthreads();

			for (T p = threadIdx.x; p < sh.num_divs_local; p += BLOCK_DIM_X)
			{
				sh.level_offset[p] = get_mask(sh.srcLen, p) & unset_mask(sh.level_prev_index[wx][1] - 1, p);
				for (T l = 2; l < sh.l[wx] - 1; l++)
				{
					sh.level_offset[p] &= unset_mask(sh.level_prev_index[wx][l] - 1, p);
				}
			}
			__syncthreads();

			compute_intersection_block<T, BLOCK_DIM_X, true>(
					lh.warpCount, sh.num_divs_local,
					sh.level_prev_index[wx][sh.l[wx] - 1] - 1, sh.l[wx], sh.to,
					sh.level_offset, sh.level_prev_index[wx], sh.encode);

			// copy to all individual warp stacks;
			for (T p = lx; p < sh.num_divs_local; p += CPARTSIZE)
				cl[sh.num_divs_local * (sh.l[wx] - 1) + p] = sh.level_offset[sh.num_divs_local * (sh.l[wx] - 1) + p];

			__syncthreads();

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
					if (!((sh.level_offset[sh.num_divs_local * (gh.Message[blockIdx.x].level_ - 1) + j / 32] >> (j % 32)) % 2 == 0))
					{

						// init stack --block
						init_stack_block(sh, gh, cl, j, partMask);
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
								sh.level_count[wx][sh.l[wx]] = lh.warpCount;
								sh.level_index[wx][sh.l[wx]] = 0;
								sh.level_prev_index[wx][sh.l[wx] - 1] = 0;
							}
						}
						__syncwarp(partMask);
						while (sh.level_count[wx][sh.l[wx]] > sh.level_index[wx][sh.l[wx]])
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
									sh.level_count[wx][sh.l[wx]] = lh.warpCount;
									sh.level_index[wx][sh.l[wx]] = 0;
									sh.level_prev_index[wx][sh.l[wx] - 1] = 0;
									T idx = sh.level_prev_index[wx][sh.l[wx] - 2] - 1;
									cl[idx / 32] &= ~(1 << (idx & 0x1F));
								}

								while (sh.l[wx] > gh.Message[blockIdx.x].level_ + 2 &&
											 sh.level_index[wx][sh.l[wx]] >= sh.level_count[wx][sh.l[wx]])
								{
									(sh.l[wx])--;
									T idx = sh.level_prev_index[wx][sh.l[wx] - 1] - 1;
									cl[idx / 32] |= 1 << (idx & 0x1F);
								}
							}
							__syncwarp(partMask);
						}
					}
					__syncwarp(partMask);
					if (lx == 0)
						sh.wtc[wx] = atomicAdd(&(sh.tc), 1);
					__syncwarp(partMask);
				}
				__syncwarp(partMask);
				if (lx == 0 && sh.sg_count[wx] > 0)
				{
					atomicAdd(gh.counter, sh.sg_count[wx]);
					// atomicAdd(&cpn[sh.src], sh.sg_count[wx]);
				}
			}

			__syncthreads();
			clear_messages(sh, gh);
			if (threadIdx.x == 0)
			{
				sh.state = 1; // done with donated task, back to inactive state
				queue_enqueue(queue, tickets, head, tail, CB, sh.sm_block_id);
			}
			__syncthreads();
		}
	}
}

// baseline
#pragma region
template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__global__ void sgm_kernel_central_node_function_byNode(
		uint64 *counter, uint64 *work_list_head,
		const graph::COOCSRGraph_d<T> g,
		const graph::GraphQueue_d<T, bool> current,
		T *current_level,
		T *adj_enc)
{
	// will be removed later
	constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
	const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
	const size_t lx = threadIdx.x % CPARTSIZE;
	__shared__ T level_index[numPartitions][DEPTH];
	__shared__ T level_count[numPartitions][DEPTH];
	__shared__ T level_prev_index[numPartitions][DEPTH];

	__shared__ uint64 sg_count[numPartitions];
	__shared__ T src, srcStart, srcLen, srcSplit;

	__shared__ T l[numPartitions];

	__shared__ T num_divs_local, *level_offset, *encode;
	__shared__ T to[BLOCK_DIM_X], newIndex[numPartitions];
	__shared__ T tc, wtc[numPartitions];

	__shared__ T state;

	//  block things
	if (threadIdx.x == 0)
	{
		state = 0;
	}
	__syncthreads();
	while (state != 100)
	{
		if (threadIdx.x == 0)
		{
			uint64 index = atomicAdd(work_list_head, 1);
			if (index < current.count[0])
			{

				src = current.queue[index];
				srcStart = g.rowPtr[src];
				srcSplit = g.splitPtr[src];
				srcLen = g.rowPtr[src + 1] - srcStart;

				num_divs_local = (srcLen + 32 - 1) / 32;

				encode = &adj_enc[(uint64)blockIdx.x * NUMDIVS * MAXDEG]; /*srcStart[wx]*/
				level_offset = &current_level[blockIdx.x * NUMDIVS * (numPartitions * MAXLEVEL)];
				tc = 0;
			}
			else
				state = 100;
		}
		__syncthreads();
		if (state == 100)
			break;
		if (lx == 0)
		{
			sg_count[wx] = 0;
		}
		__syncthreads();

		// Encode
		T partMask = (1 << CPARTSIZE) - 1;
		partMask = partMask << ((wx % (32 / CPARTSIZE)) * CPARTSIZE); // set of 8 set bits for getting candidates
		for (T j = wx; j < srcLen; j += numPartitions)
		{
			for (T k = lx; k < num_divs_local; k += CPARTSIZE)
			{
				encode[j * num_divs_local + k] = 0x00;
			}
		}
		__syncthreads(); // cleared encode
		for (T j = wx; j < srcLen; j += numPartitions)
		{
			warp_sorted_count_and_encode_full_undirected<WARPS_PER_BLOCK, T, true, CPARTSIZE>(
					&g.oriented_colInd[srcStart], srcLen,
					&g.colInd[g.rowPtr[g.oriented_colInd[srcStart + j]]],
					g.rowPtr[g.oriented_colInd[srcStart + j] + 1] - g.rowPtr[g.oriented_colInd[srcStart + j]],
					j, num_divs_local,
					encode);
		}
		__syncthreads(); // Done full encoding

		if (lx == 0)
			wtc[wx] = atomicAdd(&(tc), 1);
		__syncwarp(partMask);

		while (wtc[wx] < srcLen)
		{
			T j = wtc[wx];

			if (!(SYMNODE_PTR[2] == 1 && j < (srcSplit - srcStart)))
			{

				T *cl = level_offset + wx * (NUMDIVS * MAXLEVEL);
				for (T k = lx; k < DEPTH; k += CPARTSIZE)
				{
					level_count[wx][k] = 0;
					level_index[wx][k] = 0;
					level_prev_index[wx][k] = 0;
				}
				__syncwarp(partMask);
				if (lx == 1)
				{
					l[wx] = 2;
					level_prev_index[wx][0] = srcSplit - srcStart + 1; // possible position of src (L0)
					level_prev_index[wx][1] = j + 1;									 // L1 node
				}
				__syncwarp(partMask);

				// get warp count ??
				uint64 warpCount = 0; // level 2 reached (base level being 0)
				for (T k = lx; k < num_divs_local; k += CPARTSIZE)
				{
					cl[k] = get_mask(srcLen, k) & unset_mask(j, k);		// unset mask returns all ones except at bit j
																														// if (j == 32 * (srcLen / 32) + 1)
					if (QEDGE_PTR[l[wx] + 1] - QEDGE_PTR[l[wx]] == 2) // i.e. if connected to both nodes at level 0 and 1
						to[threadIdx.x] = encode[j * num_divs_local + k];
					else
						to[threadIdx.x] = cl[k]; // if only connected to central node, everything is a candidate.

					// Remove Redundancies
					for (T sym_idx = SYMNODE_PTR[l[wx]]; sym_idx < SYMNODE_PTR[l[wx] + 1]; sym_idx++)
					{
						if (SYMNODE[sym_idx] > 0)
						{
							to[threadIdx.x] &= ~(cl[k] & get_mask(j, k)); // if symmetric to level 1 (lexicographic symmetry breaking)
						}
					}
					// to[threadIdx.x] &= ~EP_mask[1 * num_divs_local + k];
					cl[(l[wx] - 1) * num_divs_local + k] = to[threadIdx.x]; // candidates for level 2
					warpCount += __popc(to[threadIdx.x]);
				}
				reduce_part<T, CPARTSIZE>(partMask, warpCount);

				if (lx == 0)
				{
					(l[wx])++;
					if (l[wx] == KCCOUNT - LUNMAT) // If reached last level
					{
						sg_count[wx] += warpCount; // code reaches here only if counting triangles
					}
					else
					{
						level_count[wx][l[wx]] = warpCount; // since 2 levels already finalized and l[wx] indexed from 1
						level_index[wx][l[wx]] = 0;					// This index to iterated from 0 to warpCount
					}
				}
				__syncwarp(partMask); // all triangles found till here!

				while (level_index[wx][l[wx]] < level_count[wx][l[wx]]) // limits work per warp.. [0 to 32*32]
				{
					__syncwarp(partMask);
					if (lx == 0)
					{
						T *from = &(cl[num_divs_local * (l[wx] - 2)]);												// all the current candidates
						T maskBlock = level_prev_index[wx][l[wx] - 1] / 32;										// to identify which 32 bits to pick from num_divs_local
						T maskIndex = ~((1 << (level_prev_index[wx][l[wx] - 1] & 0x1F)) - 1); // to unset previously visited index

						newIndex[wx] = __ffs(from[maskBlock] & maskIndex); //__ffs is find first set bit returns 0 if nothing set
						while (newIndex[wx] == 0)													 // if not found, look into next block
						{
							maskIndex = 0xFFFFFFFF;
							maskBlock++;
							newIndex[wx] = __ffs(from[maskBlock] & maskIndex);
						}
						newIndex[wx] = 32 * maskBlock + newIndex[wx] - 1; // actual new index

						level_prev_index[wx][l[wx] - 1] = newIndex[wx] + 1; // level prev index is numbered from 1
						level_index[wx][l[wx]]++;
					}
					__syncwarp(partMask);

					compute_intersection<T, CPARTSIZE, true>(
							warpCount, lx, partMask,
							num_divs_local, newIndex[wx], l[wx],
							to, cl, level_prev_index[wx], encode);

					if (lx == 0)
					{
						if (l[wx] + 1 == KCCOUNT - LUNMAT) // reached last level
						{
							sg_count[wx] += warpCount;
						}
						else if (l[wx] + 1 < KCCOUNT - LUNMAT) // Not at last level yet
						{
							(l[wx])++;													 // go further
							level_count[wx][l[wx]] = warpCount;	 // save level_count (warpcount updated during compute intersection)
							level_index[wx][l[wx]] = 0;					 // initialize level index
							level_prev_index[wx][l[wx] - 1] = 0; // initialize level previous index

							T idx = level_prev_index[wx][l[wx] - 2] - 1; //-1 since 1 is always added to level previous index
							cl[idx / 32] &= ~(1 << (idx & 0x1F));				 // idx & 0x1F gives remainder after dividing by 32
																													 // this puts the newindex at correct place in current level
						}
						// reset memory since will be going out of while loop (i.e. backtracking)
						while (l[wx] > 3 && level_index[wx][l[wx]] >= level_count[wx][l[wx]])
						{
							(l[wx])--;
							T idx = level_prev_index[wx][l[wx] - 1] - 1;
							cl[idx / 32] |= 1 << (idx & 0x1F); // reset cl for next subtree
						}
					}
					__syncwarp(partMask);
				}
			}
			if (lx == 0)
				wtc[wx] = atomicAdd(&(tc), 1);
			__syncwarp(partMask);
		}
		if (lx == 0 && sg_count[wx] > 0)
		{
			atomicAdd(counter, sg_count[wx]);
		}
		__syncthreads();
	}
}

#pragma endregion