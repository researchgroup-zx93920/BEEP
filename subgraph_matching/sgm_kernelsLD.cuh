#pragma once
#include "include/utils.cuh"
#include "include/common_utils.cuh"
#include "config.cuh"

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
						}
					}
					// to[threadIdx.x] &= ~EP_mask[1 * num_divs_local + k];
					cl[1 * num_divs_local + k] = to[threadIdx.x]; // candidates for level 2
					warpCount += __popc(to[threadIdx.x]);
				}
				reduce_part<T, CPARTSIZE>(partMask, warpCount);

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
					__syncwarp(partMask);
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
