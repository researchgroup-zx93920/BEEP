#pragma once
#include "utils.cuh"
#include "config.cuh"

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__device__ __forceinline__ void sgm_kernel_central_node_function_byNode(
	T blockOffset,
	uint64 *counter, const graph::COOCSRGraph_d<T> &g,
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

	//  block things
	if (threadIdx.x == 0)
	{
		src = current.queue[blockIdx.x];
		srcStart = g.rowPtr[src];
		srcSplit = g.splitPtr[src];
		srcLen = g.rowPtr[src + 1] - srcStart;

		num_divs_local = (srcLen + 32 - 1) / 32;

		encode = &adj_enc[(uint64)blockOffset * NUMDIVS * MAXDEG]; /*srcStart[wx]*/
		level_offset = &current_level[blockOffset * NUMDIVS * (numPartitions * MAXLEVEL)];
#ifdef REUSE
		reuse_offset = &reuse_stats[blockOffset * NUMDIVS * (numPartitions * MAXLEVEL)];
#endif
	}

	if (lx == 0)
	{
#ifdef IC_COUNT
		icount[wx] = 0;
#endif
#ifdef SYMOPT
		offset[wx] = 0;
#endif
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
#ifdef REUSE
		T *reuse = reuse_offset + wx * (NUMDIVS * MAXLEVEL);
#endif
		for (T k = lx; k < DEPTH; k += CPARTSIZE)
		{
			level_count[wx][k] = 0;
			level_index[wx][k] = 0;
			level_prev_index[wx][k] = 0;
		}
		if (lx == 1)
		{
			l[wx] = 3; // i.e. at level 2	l[wx] is +1 at all places
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
			if (QEDGE_PTR[3] - QEDGE_PTR[2] == 2)			// i.e. if connected to both nodes at level 0 and 1
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
			compute_intersection_ic<T, CPARTSIZE, true>(
				warpCount, icount[wx], offset[wx], srcSplit - srcStart,
				lx, partMask, num_divs_local, newIndex[wx], l[wx],
				to, cl, level_prev_index[wx], encode);
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
		__syncwarp(partMask);
	}
	if (lx == 0 && sg_count[wx] > 0)
	{
		atomicAdd(counter, sg_count[wx]);
#ifdef IC_COUNT
		atomicAdd(intersection_count, icount[wx]);
#endif
	}
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X)
	__global__ void sgm_kernel_central_node_base_binary(
		uint64 *counter,
		const graph::COOCSRGraph_d<T> g,
		const graph::GraphQueue_d<T, bool> current,
		T *current_level, T *reuse,
		T *adj_enc)
{
	sgm_kernel_central_node_function_byNode<T, BLOCK_DIM_X, CPARTSIZE>(blockIdx.x,
																	   counter, g, current, current_level, reuse, adj_enc);
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X)
	__global__ void sgm_kernel_central_node_base_binary_persistant(
		uint64 *counter,
		const graph::COOCSRGraph_d<T> g,
		const graph::GraphQueue_d<T, bool> current,
		T *current_level, T *reuse,
		T *levelStats,
		T *adj_enc)
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

	sgm_kernel_central_node_function_byNode<T, BLOCK_DIM_X, CPARTSIZE>((sm_id * CBPSM) + levelPtr,
																	   counter, g, current, current_level, reuse, adj_enc);

	if (threadIdx.x == 0)
		atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
}
