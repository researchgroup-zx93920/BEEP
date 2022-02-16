#pragma once
const uint DEPTH = 10;

__constant__ uint MINLEVEL;
__constant__ uint LUNMAT;

__constant__ uint QEDGE[DEPTH * (DEPTH - 1) / 2];
__constant__ uint QEDGE_PTR[DEPTH + 1];
__constant__ uint QDEG[DEPTH];

__constant__ uint SYMNODE[DEPTH * (DEPTH - 1) / 2];
__constant__ uint SYMNODE_PTR[DEPTH + 1];

template <typename T>
__device__ __forceinline__ T get_mask(T idx, T partition)
{
	if (idx / 32 > partition)
		return 0xFFFFFFFF;
	if (idx / 32 < partition)
		return 0;
	return (0xFFFFFFFF >> (32 - (idx - partition * 32)));
}

template <typename T>
__device__ __forceinline__ T unset_mask(T idx, T partition)
{
	// Use with BITWISE AND. All bits 1 except at idx.
	if (idx / 32 == partition)
		return (~(1 << (idx - partition * 32)));
	else
		return 0xFFFFFFFF;
}

template <typename T>
__device__ __forceinline__ T set_mask(T idx, T partition)
{
	// Use with BITWISE OR. All bits 0 except at idx
	if (idx / 32 == partition)
		return (1 << (idx - partition * 32));
	else
		return 0;
}

template <typename T, uint CPARTSIZE>
__device__ __forceinline__ void reduce_part(T partMask, uint64 &warpCount)
{
	for (int i = CPARTSIZE / 2; i >= 1; i /= 2)
		warpCount += __shfl_down_sync(partMask, warpCount, i);
}

template <typename T>
__global__ void
remove_edges_connected_to_node(
	const graph::COOCSRGraph_d<T> g,
	const graph::GraphQueue_d<T, bool> node_queue,
	bool *keep)
{
	const int partition = 1;
	auto lx = threadIdx.x % partition;
	auto wx = threadIdx.x / partition;
	auto numPart = blockDim.x / partition;
	for (auto i = wx + blockIdx.x * numPart; i < node_queue.count[0]; i += numPart * gridDim.x)
	{
		T src = node_queue.queue[i];
		T srcStart = g.rowPtr[src];
		T srcEnd = g.rowPtr[src + 1];
		for (T j = srcStart + lx; j < srcEnd; j += partition)
		{
			keep[j] = false;
			T dst = g.colInd[j];
			for (T k = g.rowPtr[dst]; k < g.rowPtr[dst + 1]; k++)
			{
				if (g.colInd[k] == src)
				{
					keep[k] = false;
					break;
				}
			}
		}
	}
}

template <typename T, uint CPARTSIZE, bool MAT>
__device__ __forceinline__ void compute_intersection(
	uint64 &wc,
	const size_t lx, const T partMask,
	const T num_divs_local, const T maskIdx, const T lvl,
	T *to, T *cl, T *level_prev_index, T *encode, T *orient_mask)
{
	wc = 0;
	to[threadIdx.x] = 0x00000000;
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
				to[threadIdx.x] &= ~(cl[(SYMNODE[sym_idx] - 1) * num_divs_local + k] & get_mask(level_prev_index[SYMNODE[sym_idx]] - 1, k));
			else
				to[threadIdx.x] &= orient_mask[k];
		}
		wc += __popc(to[threadIdx.x]);						  //counts number of set bits
		cl[(lvl - 1) * num_divs_local + k] = to[threadIdx.x]; //saves candidates list in cl
	}
	reduce_part<T, CPARTSIZE>(partMask, wc); //also does __syncwarp(partMask)
	__syncwarp(partMask);
}

template <typename T, uint CPARTSIZE>
__device__ __forceinline__ void intersect_adj(
	const T *A, const T aSz,
	const T *B, const T bSz,
	T *out, T *outSz, const T partMask,
	const T cutoff, const size_t lx)
{
	// Assume A is lower degree than B
	if (lx == 0)
		outSz[0] = 0;
	__syncwarp(partMask);

	for (T i = lx; i < aSz; i += CPARTSIZE)
	{
		const T searchVal = A[i];
		if (searchVal < cutoff) //rest taken care by oriented datagraph
		{
			bool found = false;
			graph::binary_search<T>(B, 0, bSz, searchVal, found);
			if (found)
			{
				T sz = atomicAdd(outSz, 1);
				out[sz] = searchVal;
			}
			// printf("\033[0;36m lx: %lu, searchvalue %u, result %s\n\033[0;37m", lx, searchVal, found ? "true" : "false");
		}
	}
}

template <typename T, uint CPARTSIZE, bool MAT>
__device__ __forceinline__ void ptr_intersection(
	const uint64 &count, uint64 &asym_count,
	const size_t lx, const size_t wx, const T partMask,
	const T src, const T lvl,
	T *scratchpad, T *fairpad, T *scratch, T *level_prev_index,
	const graph::COOCSRGraph_d<T> &g, const graph::COOCSRGraph_d<T> &o_g)
{
	// if (lx == 0 && blockIdx.x == 5 && wx == 0)
	// {
	// 	printf("\033[0;31m Reached here\t src: %u,  warp: %d\n\033[0;37m", src, wx);
	// }

	asym_count = 0;
	// __syncwarp(partMask);
	const T start = g.rowPtr[src];
	T end = g.rowPtr[src + 1];
	const T srcLen = end - start; //unditected source degree
	for (T k = lx; k < srcLen; k += CPARTSIZE)
	{
		const T neighbor = g.colInd[k + start];
		if (neighbor < src)
			fairpad[k] = g.colInd[k + start];
	}
	__syncwarp(partMask);
	// if (lx < srcLen)
	// 	printf("\033[0;31m lx: %lu, Source adj %u\n\033[0;37m", lx, fairpad[lx]);
	bool first = true;
	for (T q_idx = QEDGE_PTR[lvl] + 1; q_idx < QEDGE_PTR[lvl + 1]; q_idx++)
	{
		T *swap = fairpad;
		fairpad = scratchpad;
		scratchpad = swap; //output always written in fairpad
		// __syncwarp(partMask);
		// if (q_idx == QEDGE_PTR[lvl] + 1 && lx < srcLen)
		// 	printf("\033[0;33m Now in scratchpad lx: %lu, Source adj %u\n\033[0;37m", lx, scratchpad[lx]);
		T src2start = g.rowPtr[o_g.colInd[o_g.rowPtr[src] + level_prev_index[QEDGE[q_idx]] - 1]];
		T src2end = g.rowPtr[o_g.colInd[o_g.rowPtr[src] + level_prev_index[QEDGE[q_idx]] - 1] + 1];
		T src2Len = src2end - src2start;
		if (first)
		{
			if (srcLen < src2Len)
			{
				intersect_adj<T, CPARTSIZE>(scratchpad, srcLen, &g.colInd[src2start], src2Len, fairpad, scratch, partMask, src, lx);
			}
			else
			{
				intersect_adj<T, CPARTSIZE>(&g.colInd[src2start], src2Len, scratchpad, srcLen, fairpad, scratch, partMask, src, lx);
			}
			first = false;
		}
		else
		{
			intersect_adj<T, CPARTSIZE>(scratchpad, srcLen, &g.colInd[src2start], src2Len, fairpad, scratch, partMask, src, lx);
		}
		__syncwarp(partMask);
	}
	for (T k = lx; k < srcLen; k += CPARTSIZE)
	{
		fairpad[k] = 0XFFFFFFFF;
		scratchpad[k] = 0XFFFFFFFF;
	}

	// if (blockIdx.x == 5 && lx == 0 && wx == 0)
	// {
	// 	printf("\033[0;31m outsz %u\033[0;37m \n", scratch[0]);
	// }
	if (lx == 0)
	{
		asym_count = (((long)scratch[0]) * ((long)scratch[0] - 1) / 2) + count * (long)scratch[0]; //since last 2 levels have to be symmetric

		// printf("asymmetric count %u, symmetric count %lu\n", scratch[0], count);
	}
	__syncwarp(partMask);
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__device__ __forceinline__ void sgm_kernel_central_node_function_byNode(
	T blockOffset,
	uint64 *counter, T *cpn,
	const graph::COOCSRGraph_d<T> &g, const graph::COOCSRGraph_d<T> &o_g,
	const graph::GraphQueue_d<T, bool> current,
	T *current_level, T *asymmetric_levelstats,
	T *adj_enc,
	T *orient)
{
	//will be removed later
	constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
	const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
	const size_t lx = threadIdx.x % CPARTSIZE;
	__shared__ T level_index[numPartitions][DEPTH];
	__shared__ T level_count[numPartitions][DEPTH];
	__shared__ T level_prev_index[numPartitions][DEPTH];

	__shared__ uint64 sg_count[numPartitions];
	__shared__ T l[numPartitions];
	__shared__ T src, srcStart, srcLen;
	__shared__ T scratch_space[numPartitions];	 //added later
	__shared__ uint64 asym_count[numPartitions]; //added later
	__shared__ T num_divs_local, *level_offset, *encode, *orient_mask;

	__shared__ T to[BLOCK_DIM_X], newIndex[numPartitions];

	for (unsigned long long i = blockIdx.x; i < (unsigned long long)current.count[0]; i += gridDim.x)
	{
		//block things
		if (threadIdx.x == 0)
		{
			src = current.queue[i];
			srcStart = o_g.rowPtr[src];
			srcLen = o_g.rowPtr[src + 1] - srcStart;

			num_divs_local = (srcLen + 32 - 1) / 32;

			T orient_offset = blockOffset * NUMDIVS;
			orient_mask = &orient[orient_offset];

			uint64 encode_offset = (uint64)orient_offset * MAXDEG;
			encode = &adj_enc[encode_offset /*srcStart[wx]*/];

			level_offset = &current_level[orient_offset * (numPartitions * MAXLEVEL)];
			// asym_level = &asymmetric_levelstats[blockOffset * UNDIRECTED_MAXDEG * numPartitions];

			// if (src == 1)
			// {
			// 	printf("src: %lu, start pointer: %u, length: %u\n", i, srcStart, srcLen);
			// }
		}
		__syncthreads();

		//Encode
		T partMask = (1 << CPARTSIZE) - 1;
		partMask = partMask << ((wx % (32 / CPARTSIZE)) * CPARTSIZE); //set of 8 set bits for getting candidates
		for (T j = wx; j < srcLen; j += numPartitions)
		{
			for (T k = lx; k < num_divs_local; k += CPARTSIZE)
			{
				encode[j * num_divs_local + k] = 0x00;
			}
		}
		// __syncwarp(partMask);
		__syncthreads();
		for (T j = wx; j < srcLen; j += numPartitions)
		{
			// if (src == 1 && (j >= 0 && j <= 4))
			// {
			graph::warp_sorted_count_and_encode_full<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&o_g.colInd[srcStart], srcLen,
																						  &o_g.colInd[o_g.rowPtr[o_g.colInd[srcStart + j]]],
																						  o_g.rowPtr[o_g.colInd[srcStart + j] + 1] - o_g.rowPtr[o_g.colInd[srcStart + j]],
																						  j, num_divs_local,
																						  encode);
			// if (lx == 0)
			// 	printf("Testing encode: %u, for adjacency of %u\n", encode[j], o_g.colInd[srcStart + j]);
			// }
		}
		__syncthreads(); //Done encoding, has all upper and lower triangular elements..

		// Compute orientation mask (For any level symmetric to node 0 i.e. the central node)
		for (T tid = threadIdx.x; tid < srcLen; tid += blockDim.x)
		{
			T dst = o_g.colInd[srcStart + tid];
			// T dstLen = g.rowPtr[dst + 1] - g.rowPtr[dst];
			T mask = get_mask(srcLen, tid / 32); //returns all 1
			// bool keep = (dstLen > srcLen || ((dstLen == srcLen) && src < dst));
			bool keep = dst > src;
			orient_mask[tid / 32] = __ballot_sync(mask, keep);
		}
		__syncthreads();

		for (T j = wx; j < srcLen; j += numPartitions) //each warp processes an edge coming from central node
		{
			if (SYMNODE_PTR[2] == 1 && (orient_mask[j / 32] >> (j % 32)) % 2 == 0) //if eliminated by orienting due to being symmetric to central node
				continue;														   // last bit zero hence %2 is zero

			T *cl = level_offset + wx * (NUMDIVS * MAXLEVEL);
			T *asym_level = &asymmetric_levelstats[UNDIRECTED_MAXDEG * 2 * (blockOffset * numPartitions + wx)];
			for (T k = lx; k < DEPTH; k += CPARTSIZE)
			{
				level_count[wx][k] = 0;
				level_index[wx][k] = 0;
				level_prev_index[wx][k] = 0;
			}
			if (lx == 1)
			{

				l[wx] = 3; //i.e. at level 2	l[wx] is +1 at all places
				sg_count[wx] = 0;
				level_prev_index[wx][1] = j + 1;
			}
			// if (lx == 1)
			// {
			// 	level_prev_index[wx][1] = j + 1;
			// }
			__syncwarp(partMask);

			// get warp count ??
			uint64 warpCount = 0; //level 2 reached (base level being 0)
			for (T k = lx; k < num_divs_local; k += CPARTSIZE)
			{
				cl[k] = get_mask(srcLen, k) & unset_mask(j, k); //unset mask returns all ones except k
				if (QEDGE_PTR[3] - QEDGE_PTR[2] == 2)			//i.e. if connected to both nodes at level 0 and 1
					to[threadIdx.x] = encode[j * num_divs_local + k];
				else
					to[threadIdx.x] = cl[k]; //if only connected to central node, everything is a candidate..

				// Remove Redundancies
				for (T sym_idx = SYMNODE_PTR[l[wx] - 1]; sym_idx < SYMNODE_PTR[l[wx]]; sym_idx++)
				{
					if (SYMNODE[sym_idx] > 0)
						to[threadIdx.x] &= ~(cl[k] & get_mask(j, k)); //if symmetric to level 1 (lexicographic symmetry breaking)
					else
						to[threadIdx.x] &= orient_mask[k]; //if symmetric to central node
				}
				cl[1 * num_divs_local + k] = to[threadIdx.x]; //candidates for level 2
				warpCount += __popc(to[threadIdx.x]);
			}
			reduce_part<T, CPARTSIZE>(partMask, warpCount);
			// warpCount += __shfl_down_sync(partMask, warpCount, 16);
			// warpCount += __shfl_down_sync(partMask, warpCount, 8);
			// warpCount += __shfl_down_sync(partMask, warpCount, 4);
			// warpCount += __shfl_down_sync(partMask, warpCount, 2);
			// warpCount += __shfl_down_sync(partMask, warpCount, 1);

			if (l[wx] == KCCOUNT - LUNMAT && LUNMAT == 1)
			{
				uint64 tmpCount;
				compute_intersection<T, CPARTSIZE, false>(
					tmpCount, lx, partMask, num_divs_local, j, l[wx], to, cl, level_prev_index[wx], encode, orient_mask);
				warpCount *= tmpCount;

				tmpCount = 0;
				for (T k = lx; k < num_divs_local; k += CPARTSIZE)
				{
					tmpCount += __popc(cl[num_divs_local + k] & cl[2 * num_divs_local + k]);
				}
				reduce_part<T, CPARTSIZE>(partMask, tmpCount);

				warpCount -= tmpCount;

				if (SYMNODE_PTR[l[wx] + 1] > SYMNODE_PTR[l[wx]] &&
					SYMNODE[SYMNODE_PTR[l[wx] + 1] - 1] == l[wx] - 1)
					warpCount /= 2;
			}
			if (lx == 0)
			{
				if (l[wx] == KCCOUNT - LUNMAT) //If reached last level
				{
					sg_count[wx] += warpCount; //code reaches here only if counting triangles
				}
				else
				{
					level_count[wx][l[wx] - 3] = warpCount; //since 2 levels already finalized and l[wx] indexed from 1
					level_index[wx][l[wx] - 3] = 0;			//This index to iterated from 0 to warpCount
				}
			}
			__syncwarp(partMask);
			// if (src == 1 && wx == 0 && lx == 0 && j == 0)
			// {
			// 	printf("j: %u, level count: %u\n", j, level_count[wx][l[wx] - 3]);
			// }
			while (level_index[wx][l[wx] - 3] < level_count[wx][l[wx] - 3]) //limits work per warp.. [0 to 32*32]
			{
				__syncwarp(partMask);
				if (lx == 0)
				{
					scratch_space[wx] = 0;

					T *from = &(cl[num_divs_local * (l[wx] - 2)]);						  //all the current candidates
					T maskBlock = level_prev_index[wx][l[wx] - 1] / 32;					  //to identify which 32 bits to pick from num_divs_local
					T maskIndex = ~((1 << (level_prev_index[wx][l[wx] - 1] & 0x1F)) - 1); //to unset previously visited index
					newIndex[wx] = __ffs(from[maskBlock] & maskIndex);					  //__ffs is find first set bit returns 0 if nothing set

					while (newIndex[wx] == 0) //if not found, look into next block
					{
						maskIndex = 0xFFFFFFFF;
						maskBlock++;
						newIndex[wx] = __ffs(from[maskBlock] & maskIndex);
					}
					newIndex[wx] = 32 * maskBlock + newIndex[wx] - 1; //actual new index
					// if (src == 1 && wx == 0 && j == 0 && l[wx] == 3)
					// {
					// 	printf("src: %u, wx: %u, index: %u\n", src, wx, o_g.colInd[srcStart + newIndex[wx]]);
					// }

					level_prev_index[wx][l[wx] - 1] = newIndex[wx] + 1; //level prev index is numbered from 1
					level_index[wx][l[wx] - 3]++;
				}
				__syncwarp(partMask);

				// compute_intersection<T, CPARTSIZE, true>(
				// 	warpCount, lx, partMask, num_divs_local, newIndex[wx], l[wx], to, cl, level_prev_index[wx], encode, orient_mask);
				// if (src == 1 && wx == 0 && j == 0 && l[wx] == 3 && o_g.colInd[srcStart + newIndex[wx]] == 6)
				// {
				warpCount = 0;
				to[threadIdx.x] = 0;
				for (T k = lx; k < num_divs_local; k += CPARTSIZE)
				{
					to[threadIdx.x] = cl[k] & unset_mask(newIndex[wx], k);
					//compute intersection
					for (T q_idx = QEDGE_PTR[l[wx]] + 1; q_idx < QEDGE_PTR[l[wx] + 1]; q_idx++)
					{
						// printf("query index: %u, Testing Encoding: %u\n", QEDGE[q_idx], encode[(level_prev_index[wx][QEDGE[q_idx]] - 1) * num_divs_local + k]);
						to[threadIdx.x] &= encode[(level_prev_index[wx][QEDGE[q_idx]] - 1) * num_divs_local + k];
					}
					//remove Redundancies
					for (T sym_idx = SYMNODE_PTR[l[wx]]; sym_idx < SYMNODE_PTR[l[wx] + 1]; sym_idx++)
					{
						// if (!MAT && SYMNODE[sym_idx] == lvl - 1)
						// 	continue;
						if (SYMNODE[sym_idx] > 0)
							to[threadIdx.x] &= ~(cl[(SYMNODE[sym_idx] - 1) * num_divs_local + k] & get_mask(level_prev_index[wx][SYMNODE[sym_idx]] - 1, k));
						else
							to[threadIdx.x] &= orient_mask[k];
					}
					warpCount += __popc(to[threadIdx.x]);
					cl[(l[wx] - 1) * num_divs_local + k] = to[threadIdx.x];
				}
				reduce_part<T, CPARTSIZE>(partMask, warpCount);
				// __syncwarp(partMask);
				// if (lx == 0)
				// {
				// 	printf("src: %u, wx: %u, index: %u, count %lu\n", src, wx, o_g.colInd[srcStart + newIndex[wx]], warpCount);
				// }
				// }

				if (l[wx] + 1 == KCCOUNT - 1) //takes care of asymmetric nodes for both levels
				{
					//use binary search on original graph till src and use compute_intersection for rest
					if (lx == 0)
						asym_count[wx] = 0;
					// if (src == 11 && wx == 1)
					// {
					ptr_intersection<T, CPARTSIZE, true>(
						warpCount, asym_count[wx], lx, wx, partMask, src, l[wx], asym_level, &asym_level[UNDIRECTED_MAXDEG],
						&scratch_space[wx], level_prev_index[wx], g, o_g);

					if (lx == 0)
					{
						// 	printf("\033[0;33m warp: %d, stacktrace: %u, %u, %u, asym_count: %u, warp_count: %lu, formula count: %lu\n\033[0;37m", wx, src, o_g.colInd[srcStart + level_prev_index[wx][1] - 1],
						// 		   o_g.colInd[srcStart + level_prev_index[wx][2] - 1], scratch_space[wx], warpCount, asym_count[wx]);
						sg_count[wx] += asym_count[wx];
					}
					// }
				}

				if (l[wx] + 1 == KCCOUNT - LUNMAT && LUNMAT == 1)
				{
					uint64 tmpCount;
					compute_intersection<T, CPARTSIZE, false>(
						tmpCount, lx, partMask, num_divs_local, newIndex[wx], l[wx] + 1, to, cl, level_prev_index[wx], encode, orient_mask);
					warpCount *= tmpCount;

					tmpCount = 0;
					for (T k = lx; k < num_divs_local; k += CPARTSIZE)
					{
						tmpCount += __popc(cl[(l[wx] - 1) * num_divs_local + k] & cl[l[wx] * num_divs_local + k]);
					}
					reduce_part<T, CPARTSIZE>(partMask, tmpCount);

					warpCount -= tmpCount;

					if (SYMNODE_PTR[l[wx] + 2] > SYMNODE_PTR[l[wx] + 1] &&
						SYMNODE[SYMNODE_PTR[l[wx] + 2] - 1] == l[wx])
						warpCount /= 2;
				}

				if (l[wx] + 1 < KCCOUNT - LUNMAT) //not at last level yet, then unset previously visited indexes
				{
					for (T k = lx; k < num_divs_local; k += CPARTSIZE)
						cl[k] &= unset_mask(level_prev_index[wx][l[wx] - 1] - 1, k); //make sure visited elements don't repeat
				}
				if (lx == 0)
				{
					if (l[wx] + 1 == KCCOUNT - LUNMAT) //reached last level
					{
						// printf("Enumeration: %u, %u, %u, %u, count %lu\n", src, g.colInd[srcStart + level_prev_index[wx][1] - 1], g.colInd[srcStart + level_prev_index[wx][2] - 1],
						// 	   g.colInd[srcStart + level_prev_index[wx][3] - 1], warpCount /*, g.colInd[srcStart + level_prev_index[wx][4] - 1]*/);
						sg_count[wx] += warpCount;
					}
					else if (l[wx] + 1 < KCCOUNT - LUNMAT) //Not at last level yet
					{
						(l[wx])++;								//go further
						level_count[wx][l[wx] - 3] = warpCount; //save level_count (warpcount updated during compute intersection)
						level_index[wx][l[wx] - 3] = 0;			//initialize level index
						level_prev_index[wx][l[wx] - 1] = 0;	//initialize level previous index

						T idx = level_prev_index[wx][l[wx] - 2] - 1; //-1 since +1 is always added to level previous index
						cl[idx / 32] &= ~(1 << (idx & 0x1F));		 //idx&0x1F gives remainder after dividing by 32
																	 //this puts the newindex at correct place in current level
					}

					while (l[wx] > 3 && level_index[wx][l[wx] - 3] >= level_count[wx][l[wx] - 3]) //reset memory since will be going out of while loop (i.e. backtracking)
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
				// atomicAdd(&cpn[src], sg_count[wx]);
			}
			__syncwarp(partMask);
		}
		// __syncthreads();
		// if (threadIdx.x == 0)
		// {
		// 	printf("count for node: %u is: %u\n", src, cpn[src]);
		// }
		__syncthreads();
	}
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__device__ __forceinline__ void sgm_kernel_central_node_function_byEdge(
	T blockOffset,
	uint64 *counter,
	const graph::COOCSRGraph_d<T> g,
	const graph::GraphQueue_d<T, bool> current,
	T *current_level, T *asymmetric_levelstats,
	T *adj_enc,
	T *orient)
{
	//will be removed later
	constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
	const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
	const size_t lx = threadIdx.x % CPARTSIZE;
	__shared__ T level_index[numPartitions][DEPTH];
	__shared__ T level_count[numPartitions][DEPTH];
	__shared__ T level_prev_index[numPartitions][DEPTH];

	__shared__ uint64 sg_count[numPartitions];
	__shared__ T l[numPartitions];
	__shared__ T src, srcStart, srcLen, dst, dstStart, dstLen, dstIdx;

	__shared__ T num_divs_local, *level_offset, *encode, *orient_mask;

	__shared__ T to[BLOCK_DIM_X], newIndex[numPartitions];

	for (unsigned long long i = blockIdx.x; i < (unsigned long long)current.count[0]; i += gridDim.x)
	{
		//block things
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
			orient_mask = &orient[orient_offset];

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

		//Encode
		T partMask = (1 << CPARTSIZE) - 1;
		partMask = partMask << ((wx % (32 / CPARTSIZE)) * CPARTSIZE);
		for (T j = wx; j < srcLen; j += numPartitions)
		{
			for (T k = lx; k < num_divs_local; k += CPARTSIZE)
			{
				encode[j * num_divs_local + k] = 0x00;
			}
			__syncwarp(partMask);
			graph::warp_sorted_count_and_encode_full<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&g.colInd[srcStart], srcLen,
																						  &g.colInd[g.rowPtr[g.colInd[srcStart + j]]], g.rowPtr[g.colInd[srcStart + j] + 1] - g.rowPtr[g.colInd[srcStart + j]],
																						  j, num_divs_local,
																						  encode);
		}
		__syncthreads(); //Done encoding

		// Compute orientation mask (If any level is symmetric to node 0)
		for (T tid = threadIdx.x; tid < srcLen; tid += blockDim.x)
		{
			T tmp = g.colInd[srcStart + tid];
			T tmpLen = g.rowPtr[tmp + 1] - g.rowPtr[tmp];
			T mask = get_mask(srcLen, tid / 32);
			bool keep = (tmpLen > srcLen || ((tmpLen == srcLen) && src < tmp));
			orient_mask[tid / 32] = __ballot_sync(mask, keep);
		}
		__syncthreads();

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
				if (SYMNODE[sym_idx] > 0)
					to[threadIdx.x] &= ~(level_offset[k] & get_mask(dstIdx, k));
				else
					to[threadIdx.x] &= orient_mask[k];
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
			//if (threadIdx.x == 0) printf("Src: %d, Dst: %d, count: %d\n", src, dst, warpCount);
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

			//get warp count ??
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
					else
						to[threadIdx.x] &= orient_mask[k];
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
				//First Index
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

				//Intersect
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
						else
							to[threadIdx.x] &= orient_mask[k];
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
				//cpn[current.queue[i]] = sg_count[wx];
			}
			__syncwarp(partMask);
		}
		__syncthreads();
	}
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X)
	__global__ void sgm_kernel_central_node_base_binary(
		uint64 *counter, T *cpn,
		const graph::COOCSRGraph_d<T> g, const graph::COOCSRGraph_d<T> o_g,
		const graph::GraphQueue_d<T, bool> current,
		T *current_level, T *asymmetric_levelstats,
		T *adj_enc,
		T *orient,
		const bool byNode)
{
	if (byNode)
		sgm_kernel_central_node_function_byNode<T, BLOCK_DIM_X, CPARTSIZE>(blockIdx.x,
																		   counter, cpn, g, o_g, current, current_level, asymmetric_levelstats, adj_enc, orient);
	else
		sgm_kernel_central_node_function_byEdge<T, BLOCK_DIM_X, CPARTSIZE>(blockIdx.x,
																		   counter, g, current, current_level, asymmetric_levelstats, adj_enc, orient);
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X)
	__global__ void sgm_kernel_central_node_base_binary_persistant(
		uint64 *counter, T *cpn,
		const graph::COOCSRGraph_d<T> g, const graph::COOCSRGraph_d<T> o_g,
		const graph::GraphQueue_d<T, bool> current,
		T *current_level, T *asymmetric_levelstats,
		T *levelStats,
		T *adj_enc,
		T *orient,
		const bool byNode)
{

	//only needed for persistant launches
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
																		   counter, cpn, g, o_g, current, current_level, asymmetric_levelstats, adj_enc, orient);
	else
		sgm_kernel_central_node_function_byEdge<T, BLOCK_DIM_X, CPARTSIZE>((sm_id * CBPSM) + levelPtr,
																		   counter, g, current, current_level, asymmetric_levelstats, adj_enc, orient);

	__syncthreads();

	if (threadIdx.x == 0)
	{
		atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
	}
}