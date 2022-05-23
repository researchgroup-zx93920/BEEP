#pragma once
#include "utils.cuh"

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__device__ __forceinline__ void sgm_kernel_central_node_function_byNode(
	T blockOffset,
	uint64 *counter, T *cpn, uint64 *intersection_count,
	const graph::COOCSRGraph_d<T> &g,
	const graph::GraphQueue_d<T, bool> current,
	T *current_level, T *reuse_stats,
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
	__shared__ T offset[numPartitions];

#ifdef IC_COUNT
	__shared__ uint64 icount[numPartitions];
#endif

	__shared__ T num_divs_local, *level_offset, *encode, *reuse_offset;
	__shared__ T to[BLOCK_DIM_X], newIndex[numPartitions];

	for (unsigned long long i = blockIdx.x; i < (unsigned long long)current.count[0]; i += gridDim.x)
	{
		// block things
		if (threadIdx.x == 0)
		{
			src = current.queue[i];
			srcStart = g.rowPtr[src];
			srcSplit = g.splitPtr[src];
			srcLen = g.rowPtr[src + 1] - srcStart;

			num_divs_local = (srcLen + 32 - 1) / 32;

			T orient_offset = blockOffset * NUMDIVS;

			uint64 encode_offset = (uint64)orient_offset * MAXDEG;
			encode = &adj_enc[encode_offset]; /*srcStart[wx]*/

			level_offset = &current_level[orient_offset * (numPartitions * MAXLEVEL)];
			reuse_offset = &reuse_stats[orient_offset * (numPartitions * MAXLEVEL)];
		}
#ifdef IC_COUNT
		if (lx == 0)
		{
			icount[wx] = 0;
			offset[wx] = 0;
		}
#endif
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
			__syncwarp(partMask);

			graph::warp_sorted_count_and_encode_full<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&g.oriented_colInd[srcStart], srcLen,
																						  &g.colInd[g.rowPtr[g.oriented_colInd[srcStart + j]]],
																						  g.rowPtr[g.oriented_colInd[srcStart + j] + 1] - g.rowPtr[g.oriented_colInd[srcStart + j]],
																						  j, num_divs_local,
																						  encode);
		}
		__syncthreads(); // Done full encoding

		// start DFS
		for (T j = wx; j < srcLen; j += numPartitions) // each warp processes an edge coming from central node
		{
			if (SYMNODE_PTR[2] == 1 && j < (srcSplit - srcStart))
				continue;

			T *cl = level_offset + wx * (NUMDIVS * MAXLEVEL);
			T *reuse = reuse_offset + wx * (NUMDIVS * MAXLEVEL);

			for (T k = lx; k < DEPTH; k += CPARTSIZE)
			{
				level_count[wx][k] = 0;
				level_index[wx][k] = 0;
				level_prev_index[wx][k] = 0;
			}
			if (lx == 1)
			{
				l[wx] = 3; // i.e. at level 2	l[wx] is +1 at all places
				sg_count[wx] = 0;
				level_prev_index[wx][0] = srcSplit - srcStart + 1;
				level_prev_index[wx][1] = j + 1;
			}
			__syncwarp(partMask);

			// get warp count ??
			uint64 warpCount = 0; // level 2 reached (base level being 0)
			for (T k = lx; k < num_divs_local; k += CPARTSIZE)
			{
				cl[k] = get_mask(srcLen, k) & unset_mask(j, k); // unset mask returns all ones except at bit j
																// if (j == 32 * (srcLen / 32) + 1)
				// atomicAdd(&icount[wx], 1);
				if (QEDGE_PTR[3] - QEDGE_PTR[2] == 2) // i.e. if connected to both nodes at level 0 and 1
					to[threadIdx.x] = encode[j * num_divs_local + k];
				else
					to[threadIdx.x] = cl[k]; // if only connected to central node, everything is a candidate.

				// Remove Redundancies
				for (T sym_idx = SYMNODE_PTR[l[wx] - 1]; sym_idx < SYMNODE_PTR[l[wx]]; sym_idx++)
				{
					if (SYMNODE[sym_idx] > 0)
					{
						to[threadIdx.x] &= ~(cl[k] & get_mask(j, k)); // if symmetric to level 1 (lexicographic symmetry breaking)
#ifdef IC_COUNT
						atomicAdd(&icount[wx], 1);
#endif
					}
					// else
					// 	to[threadIdx.x] &= ~get_mask(srcSplit - srcStart, k);
				}
				cl[1 * num_divs_local + k] = to[threadIdx.x]; // candidates for level 2
				warpCount += __popc(to[threadIdx.x]);
			}
			reduce_part<T, CPARTSIZE>(partMask, warpCount);

			if (l[wx] == KCCOUNT - LUNMAT && LUNMAT == 1)
			{
				// uint64 tmpCount;
				// compute_intersection<T, CPARTSIZE, false>(
				// 	tmpCount, lx, partMask, num_divs_local, j, l[wx], to, cl, level_prev_index[wx], encode);
				// warpCount *= tmpCount;

				// tmpCount = 0;
				// for (T k = lx; k < num_divs_local; k += CPARTSIZE)
				// {
				// 	tmpCount += __popc(cl[num_divs_local + k] & cl[2 * num_divs_local + k]);
				// }
				// reduce_part<T, CPARTSIZE>(partMask, tmpCount);

				// warpCount -= tmpCount;

				// if (SYMNODE_PTR[l[wx] + 1] > SYMNODE_PTR[l[wx]] &&
				// 	SYMNODE[SYMNODE_PTR[l[wx] + 1] - 1] == l[wx] - 1)
				// 	warpCount /= 2;
			}

			if (lx == 0)
			{
				if (l[wx] == KCCOUNT - LUNMAT) // If reached last level
				{
					sg_count[wx] += warpCount; // code reaches here only if counting triangles
				}
				else
				{
					level_count[wx][l[wx] - 3] = warpCount; // since 2 levels already finalized and l[wx] indexed from 1
					level_index[wx][l[wx] - 3] = 0;			// This index to iterated from 0 to warpCount
				}
			}
			__syncwarp(partMask); // all triangles found till here!

			while (level_index[wx][l[wx] - 3] < level_count[wx][l[wx] - 3]) // limits work per warp.. [0 to 32*32]
			{
				// __syncwarp(partMask);
				if (lx == 0)
				{
					T *from = &(cl[num_divs_local * (l[wx] - 2)]);						  // all the current candidates
					T maskBlock = level_prev_index[wx][l[wx] - 1] / 32;					  // to identify which 32 bits to pick from num_divs_local
					T maskIndex = ~((1 << (level_prev_index[wx][l[wx] - 1] & 0x1F)) - 1); // to unset previously visited index

					newIndex[wx] = __ffs(from[maskBlock] & maskIndex); //__ffs is find first set bit returns 0 if nothing set
					while (newIndex[wx] == 0)						   // if not found, look into next block
					{
						maskIndex = 0xFFFFFFFF;
						maskBlock++;
						newIndex[wx] = __ffs(from[maskBlock] & maskIndex);
					}
					newIndex[wx] = 32 * maskBlock + newIndex[wx] - 1; // actual new index

					level_prev_index[wx][l[wx] - 1] = newIndex[wx] + 1; // level prev index is numbered from 1
					level_index[wx][l[wx] - 3]++;
				}
				__syncwarp(partMask);

#ifdef IC_COUNT
#ifdef REUSE
				compute_intersection_ic_reuse<T, CPARTSIZE, true>(
					warpCount, icount[wx], offset[wx], srcSplit - srcStart,
					lx, partMask, num_divs_local, newIndex[wx], l[wx],
					to, cl, reuse, level_prev_index[wx], encode);
#else
				// compute_intersection_ic<T, CPARTSIZE, true>(
				// 	warpCount, icount[wx], offset[wx], srcSplit - srcStart,
				// lx, partMask,num_divs_local, newIndex[wx], l[wx],
				// 	to, cl, level_prev_index[wx], encode);
#endif
#else
#ifdef REUSE
				compute_intersection_reuse<T, CPARTSIZE, true>(
					warpCount, offset[wx], lx, partMask,
					num_divs_local, newIndex[wx], l[wx],
					to, cl, reuse, level_prev_index[wx], encode);
#else
				compute_intersection<T, CPARTSIZE, true>(
					warpCount, offset[wx], lx, partMask,
					num_divs_local, newIndex[wx], l[wx],
					to, cl, level_prev_index[wx], encode);

#endif
#endif
				if (l[wx] + 1 == KCCOUNT - LUNMAT && LUNMAT == 1)
				{
					// 	uint64 tmpCount;
					// 	compute_intersection<T, CPARTSIZE, false>(
					// 		tmpCount, srcSplit - srcStart, lx, partMask, num_divs_local, newIndex[wx], l[wx] + 1, to, cl, level_prev_index[wx], encode);
					// 	warpCount *= tmpCount;

					// 	tmpCount = 0;
					// 	for (T k = lx; k < num_divs_local; k += CPARTSIZE)
					// 	{
					// 		tmpCount += __popc(cl[(l[wx] - 1) * num_divs_local + k] & cl[l[wx] * num_divs_local + k]);
					// 	}
					// 	reduce_part<T, CPARTSIZE>(partMask, tmpCount);

					// 	warpCount -= tmpCount;

					// 	if (SYMNODE_PTR[l[wx] + 2] > SYMNODE_PTR[l[wx] + 1] &&
					// 		SYMNODE[SYMNODE_PTR[l[wx] + 2] - 1] == l[wx])
					// 		warpCount /= 2;
				}

				// unsetting to avoid repeats not needed

				if (lx == 0)
				{
					if (l[wx] + 1 == KCCOUNT - LUNMAT) // reached last level
					{
						sg_count[wx] += warpCount;
					}
					else if (l[wx] + 1 < KCCOUNT - LUNMAT) // Not at last level yet
					{
						(l[wx])++;								// go further
						level_count[wx][l[wx] - 3] = warpCount; // save level_count (warpcount updated during compute intersection)
						level_index[wx][l[wx] - 3] = 0;			// initialize level index
						level_prev_index[wx][l[wx] - 1] = 0;	// initialize level previous index

						T idx = level_prev_index[wx][l[wx] - 2] - 1; //-1 since 1 is always added to level previous index
						cl[idx / 32] &= ~(1 << (idx & 0x1F));		 // idx & 0x1F gives remainder after dividing by 32
																	 // this puts the newindex at correct place in current level
					}
					while (l[wx] > 3 && level_index[wx][l[wx] - 3] >= level_count[wx][l[wx] - 3]) // reset memory since will be going out of while loop (i.e. backtracking)
					{
						(l[wx])--;
						T idx = level_prev_index[wx][l[wx] - 1] - 1;
						cl[idx / 32] |= 1 << (idx & 0x1F);
					}
				}
				__syncwarp(partMask);
			}
			if (lx == 0)
			{
				atomicAdd(counter, sg_count[wx]);
#ifdef IC_COUNT
				atomicAdd(intersection_count, icount[wx]);
#endif
			}
			__syncwarp(partMask);
		}
		__syncthreads();
	}
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__device__ __forceinline__ void sgm_kernel_central_node_function_byEdge(
	T blockOffset,
	uint64 *counter,
	const graph::COOCSRGraph_d<T> g,
	const graph::GraphQueue_d<T, bool> current,
	T *current_level, T *reuse,
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
	__shared__ T l[numPartitions];
	__shared__ T src, srcStart, srcLen, dst, dstStart, dstLen, dstIdx;

	__shared__ T num_divs_local, *level_offset, *encode;

	__shared__ T to[BLOCK_DIM_X], newIndex[numPartitions];

	for (unsigned long long i = blockIdx.x; i < (unsigned long long)current.count[0]; i += gridDim.x)
	{
		// block things
		if (threadIdx.x == 0)
		{
			src = g.rowInd[current.queue[i]];
			srcStart = g.rowPtr[src];
			srcLen = g.rowPtr[src + 1] - srcStart;

			dst = g.colInd[current.queue[i]];
			dstStart = g.rowPtr[dst];
			dstLen = g.rowPtr[dst + 1] - dstStart;
			dstIdx = current.queue[i] - srcStart;

			num_divs_local = (srcLen + 32 - 1) / 32;

			T orient_offset = blockOffset * NUMDIVS;

			uint64 encode_offset = (uint64)orient_offset * MAXDEG;
			encode = &adj_enc[encode_offset /*srcStart[wx]*/];

			level_offset = &current_level[orient_offset * (numPartitions * MAXLEVEL)];
		}
		__syncthreads();

		if (SYMNODE_PTR[2] == 1)
		{
			bool keep = (dstLen > srcLen || ((dstLen == srcLen) && src < dst));
			if (!keep)
				continue;
		}

		// Encode
		T partMask = (1 << CPARTSIZE) - 1;
		partMask = partMask << ((wx % (32 / CPARTSIZE)) * CPARTSIZE);
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
		__syncthreads(); // Done encoding

		// Compute triangles
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
				// if (SYMNODE[sym_idx] > 0)
				to[threadIdx.x] &= ~(level_offset[k] & get_mask(dstIdx, k));
			}
			level_offset[num_divs_local + k] = to[threadIdx.x];
			warpCount += __popc(to[threadIdx.x]);
		}

		typedef cub::BlockReduce<uint64, BLOCK_DIM_X> BlockReduce;
		__shared__ typename BlockReduce::TempStorage temp_storage;
		__syncthreads();
		warpCount = BlockReduce(temp_storage).Sum(warpCount);

		if (KCCOUNT == 3)
		{
			if (threadIdx.x == 0)
				atomicAdd(counter, warpCount);
			// if (threadIdx.x == 0) printf("Src: %d, Dst: %d, count: %d\n", src, dst, warpCount);
			continue;
		}
		__syncthreads();

		for (T j = wx; j < srcLen; j += numPartitions)
		{
			if ((level_offset[num_divs_local + j / 32] >> (j % 32)) % 2 == 0)
				continue;
			T *cl = level_offset + wx * (NUMDIVS * MAXLEVEL);

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
				sg_count[wx] = 0;
				level_prev_index[wx][1] = dstIdx + 1;
				level_prev_index[wx][2] = j + 1;
			}
			__syncwarp(partMask);

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
						to[threadIdx.x] &= ~(cl[(SYMNODE[sym_idx] - 1) * num_divs_local + k] & get_mask(level_prev_index[wx][SYMNODE[sym_idx]] - 1, k));
				}
				warpCount += __popc(to[threadIdx.x]);
				cl[(l[wx] - 1) * num_divs_local + k] = to[threadIdx.x];
			}
			reduce_part<T, CPARTSIZE>(partMask, warpCount);
			// warpCount += __shfl_down_sync(partMask, warpCount, 16);
			// warpCount += __shfl_down_sync(partMask, warpCount, 8);
			// warpCount += __shfl_down_sync(partMask, warpCount, 4);
			// warpCount += __shfl_down_sync(partMask, warpCount, 2);
			// warpCount += __shfl_down_sync(partMask, warpCount, 1);

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

			while (level_count[wx][l[wx] - 3] > level_index[wx][l[wx] - 3])
			{
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

				// Intersect
				warpCount = 0;
				for (T k = lx; k < num_divs_local; k += CPARTSIZE)
				{
					to[threadIdx.x] = cl[k] & unset_mask(newIndex[wx], k);
					// Compute Intersection
					for (T q_idx = QEDGE_PTR[l[wx]] + 1; q_idx < QEDGE_PTR[l[wx] + 1]; q_idx++)
					{
						to[threadIdx.x] &= encode[(level_prev_index[wx][QEDGE[q_idx]] - 1) * num_divs_local + k];
					}
					// Remove Redundancies
					for (T sym_idx = SYMNODE_PTR[l[wx]]; sym_idx < SYMNODE_PTR[l[wx] + 1]; sym_idx++)
					{
						if (SYMNODE[sym_idx] > 0)
							to[threadIdx.x] &= ~(cl[(SYMNODE[sym_idx] - 1) * num_divs_local + k] & get_mask(level_prev_index[wx][SYMNODE[sym_idx]] - 1, k));
					}
					warpCount += __popc(to[threadIdx.x]);
					cl[(l[wx] - 1) * num_divs_local + k] = to[threadIdx.x];
				}
				reduce_part<T, CPARTSIZE>(partMask, warpCount);
				// warpCount += __shfl_down_sync(partMask, warpCount, 16);
				// warpCount += __shfl_down_sync(partMask, warpCount, 8);
				// warpCount += __shfl_down_sync(partMask, warpCount, 4);
				// warpCount += __shfl_down_sync(partMask, warpCount, 2);
				// warpCount += __shfl_down_sync(partMask, warpCount, 1);

				if (l[wx] + 1 < KCCOUNT)
				{
					for (T k = lx; k < num_divs_local; k += CPARTSIZE)
						cl[k] &= unset_mask(level_prev_index[wx][l[wx] - 1] - 1, k);
				}
				if (lx == 0)
				{
					if (l[wx] + 1 == KCCOUNT)
					{
						sg_count[wx] += warpCount;
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
				__syncwarp(partMask);
			}
			if (lx == 0)
			{
				atomicAdd(counter, sg_count[wx]);
				// cpn[current.queue[i]] = sg_count[wx];
			}
			__syncwarp(partMask);
		}
		__syncthreads();
	}
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X)
	__global__ void sgm_kernel_central_node_base_binary(
		uint64 *counter, T *cpn, uint64 *intersection_count,
		const graph::COOCSRGraph_d<T> g,
		const graph::GraphQueue_d<T, bool> current,
		T *current_level, T *reuse,
		T *adj_enc,
		const bool byNode)
{
	if (byNode)
		sgm_kernel_central_node_function_byNode<T, BLOCK_DIM_X, CPARTSIZE>(blockIdx.x,
																		   counter, cpn, intersection_count, g, current, current_level, reuse, adj_enc);
	else
		sgm_kernel_central_node_function_byEdge<T, BLOCK_DIM_X, CPARTSIZE>(blockIdx.x,
																		   counter, g, current, current_level, reuse, adj_enc);
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X)
	__global__ void sgm_kernel_central_node_base_binary_persistant(
		uint64 *counter, T *cpn, uint64 *intersection_count,
		const graph::COOCSRGraph_d<T> g,
		const graph::GraphQueue_d<T, bool> current,
		T *current_level, T *reuse,
		T *levelStats,
		T *adj_enc,
		const bool byNode)
{

	// only needed for persistant launches
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

	if (byNode)
		sgm_kernel_central_node_function_byNode<T, BLOCK_DIM_X, CPARTSIZE>((sm_id * CBPSM) + levelPtr,
																		   counter, cpn, intersection_count, g, current, current_level, reuse, adj_enc);
	else
		sgm_kernel_central_node_function_byEdge<T, BLOCK_DIM_X, CPARTSIZE>((sm_id * CBPSM) + levelPtr,
																		   counter, g, current, current_level, reuse, adj_enc);

	__syncthreads();

	if (threadIdx.x == 0)
		atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
}
