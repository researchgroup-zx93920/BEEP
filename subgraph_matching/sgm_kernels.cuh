#pragma once

const uint DEPTH = 6;

__constant__ uint QEDGE[DEPTH * (DEPTH-1) / 2];
__constant__ uint QEDGE_PTR[DEPTH + 1];

__constant__ uint SYMNODE[DEPTH * (DEPTH - 1) / 2];
__constant__ uint SYMNODE_PTR[DEPTH + 1];

template<typename T>
__device__ __forceinline__ T get_mask(T idx, T partition) {
	return 0xFFFFFFFF >> (32 - (idx - partition * 32));
}

template<typename T>
__device__ __forceinline__ T unset_mask(T idx, T partition) {
	return ~(1 << (idx - partition * 32));
}

template<typename T>
__device__ __forceinline__ T set_mask(T idx, T partition) {
	return 1 << (idx - partition * 32);
}


template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__global__ void
sgm_kernel_central_node_base_binary(
	uint64* counter,
	const graph::COOCSRGraph_d<T> g,
	const graph::GraphQueue_d<T, bool>  current,
	T* current_level,
	T* adj_enc
)
{
	//will be removed later
	constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
	const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
	const size_t lx = threadIdx.x % CPARTSIZE;
	__shared__ T level_index[numPartitions][DEPTH];
	__shared__ T level_count[numPartitions][DEPTH];
	__shared__ T level_prev_index[numPartitions][DEPTH];

	__shared__ T  level_offset[numPartitions];
	__shared__ uint64 sg_count[numPartitions];
	__shared__ T l[numPartitions];
	__shared__ T src, srcStart, srcLen;

	__shared__ T num_divs_local, encode_offset, *encode;

	//__shared__ T scl[896];
	__syncthreads();

	for (unsigned long long i = blockIdx.x; i < (unsigned long long) current.count[0]; i += gridDim.x)
	{
		//block things
		if (threadIdx.x == 0)
		{
			T src = current.queue[i];
			srcStart = g.rowPtr[src];
			srcLen = g.rowPtr[src + 1] - srcStart;

			num_divs_local = (srcLen + 32 - 1) / 32;

			encode_offset = blockIdx.x * (MAXDEG * NUMDIVS);
			encode = &adj_enc[encode_offset  /*srcStart[wx]*/];
		}
		__syncthreads();

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

		for (T j = wx; j < srcLen; j += numPartitions)
		{
			level_offset[wx] = blockIdx.x * (numPartitions * NUMDIVS * DEPTH);
			T* cl = &current_level[level_offset[wx] + wx * (NUMDIVS * DEPTH)];


			if (lx < DEPTH)
			{
				level_count[wx][lx] = 0;
				level_index[wx][lx] = 0;
				level_prev_index[wx][lx] = 0;
			}
			if (lx == 0)
			{
				l[wx] = 3;
				sg_count[wx] = 0;
			}

			//get warp count ??
			uint64 warpCount = 0;
			for (T k = lx; k < num_divs_local; k += CPARTSIZE)
			{
				cl[k] = get_mask(srcLen, k) & unset_mask(j, k);
				cl[num_divs_local + k] = (QEDGE_PTR[3] - QEDGE_PTR[2] == 2) ? 
										encode[j * num_divs_local + k] :
										cl[k];
				// Remove Redundancies
				for (T sym_idx = SYMNODE_PTR[l[wx] - 1]; sym_idx < SYMNODE_PTR[l[wx]]; sym_idx++) {
					if (SYMNODE[sym_idx] > 0) cl[num_divs_local + k] &= ~(cl[k] & get_mask(j, k));
				}
				warpCount += __popc(cl[num_divs_local + k]); 
			}
			reduce_part<T>(partMask, warpCount);
			// warpCount += __shfl_down_sync(partMask, warpCount, 16);
			// warpCount += __shfl_down_sync(partMask, warpCount, 8);
			// warpCount += __shfl_down_sync(partMask, warpCount, 4);
			// warpCount += __shfl_down_sync(partMask, warpCount, 2);
			// warpCount += __shfl_down_sync(partMask, warpCount, 1);

			if (lx == 0 && l[wx] == KCCOUNT)
				sg_count[wx] += warpCount;
			else if (lx == 0 && KCCOUNT > 3 ) // && warpCount >= KCCOUNT - 2)
			{
				level_count[wx][l[wx] - 3] = warpCount;
				level_index[wx][l[wx] - 3] = 0;
				level_prev_index[wx][1] = j + 1;
				level_prev_index[wx][2] = 0;
			}


			__syncwarp(partMask);
			while (level_count[wx][l[wx] - 3] > level_index[wx][l[wx] - 3])
			{
				//First Index
				T* from = &(cl[num_divs_local * (l[wx] - 2)]);
				T* to = &(cl[num_divs_local * (l[wx] - 1)]);
				T maskBlock = level_prev_index[wx][l[wx] - 1] / 32;
				T maskIndex = ~((1 << (level_prev_index[wx][l[wx] - 1] & 0x1F)) - 1);

				T newIndex = __ffs(from[maskBlock] & maskIndex);
				while (newIndex == 0)
				{
					maskIndex = 0xFFFFFFFF;
					maskBlock++;
					newIndex = __ffs(from[maskBlock] & maskIndex);
				}
				newIndex = 32 * maskBlock + newIndex - 1;

				if (lx == 0)
				{
					level_prev_index[wx][l[wx] - 1] = newIndex + 1;
					level_index[wx][l[wx] - 3]++;
				}

				//Intersect
				uint64 warpCount = 0;
				for (T k = lx; k < num_divs_local; k += CPARTSIZE)
				{
					to[k] = cl[k] & unset_mask(newIndex, k);
					// Compute Intersection
					for (T q_idx = QEDGE_PTR[l[wx]] + 1; q_idx < QEDGE_PTR[l[wx]+1]; q_idx++) {
						to[k] = to[k] & encode[(level_prev_index[wx][QEDGE[q_idx]] - 1) * num_divs_local + k]; 
					}
					// Remove Redundancies
					for (T sym_idx = SYMNODE_PTR[l[wx]]; sym_idx < SYMNODE_PTR[l[wx]+1]; sym_idx++) {
						if (SYMNODE[sym_idx] > 0) to[k] &= ~(cl[(SYMNODE[sym_idx] - 1) * num_divs_local + k] & get_mask(level_prev_index[wx][SYMNODE[sym_idx]] - 1, k));
					}
					warpCount += __popc(to[k]);
				}
				reduce_part<T>(partMask, warpCount);
				// warpCount += __shfl_down_sync(partMask, warpCount, 16);
				// warpCount += __shfl_down_sync(partMask, warpCount, 8);
				// warpCount += __shfl_down_sync(partMask, warpCount, 4);
				// warpCount += __shfl_down_sync(partMask, warpCount, 2);
				// warpCount += __shfl_down_sync(partMask, warpCount, 1);

				if (lx == 0)
				{
					if (l[wx] + 1 == KCCOUNT) {
						sg_count[wx] += warpCount;
					}
					else if (l[wx] + 1 < KCCOUNT) //&& warpCount >= KCCOUNT - l[wx])
					{
						(l[wx])++;
						level_count[wx][l[wx] - 3] = warpCount;
						level_index[wx][l[wx] - 3] = 0;
						level_prev_index[wx][l[wx] - 1] = 0;

					}

					while (l[wx] > 3 && level_index[wx][l[wx] - 3] >= level_count[wx][l[wx] - 3])
					{
						(l[wx])--;
						for (T k = lx; k < num_divs_local; k += CPARTSIZE) 
							cl[k] &= set_mask(level_prev_index[wx][l[wx]-1]-1, k);
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
	}
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__global__ void
sgm_kernel_central_node_base_binary_persistant(
	uint64* counter,
	const graph::COOCSRGraph_d<T> g,
	const graph::GraphQueue_d<T, bool>  current,
	T* current_level,
	T* levelStats,
	T* adj_enc
)
{
	//will be removed later
	constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
	const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
	const size_t lx = threadIdx.x % CPARTSIZE;
	__shared__ T level_index[numPartitions][DEPTH];
	__shared__ T level_count[numPartitions][DEPTH];
	__shared__ T level_prev_index[numPartitions][DEPTH];

	__shared__ T  level_offset[numPartitions];
	__shared__ uint64 sg_count[numPartitions];
	__shared__ T l[numPartitions];
	__shared__ uint32_t  sm_id, levelPtr;
	__shared__ T src, srcStart, srcLen;

	__shared__ T num_divs_local, encode_offset, *encode;

	//__shared__ T scl[896];
	__syncthreads();

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

	for (unsigned long long i = blockIdx.x; i < (unsigned long long) current.count[0]; i += gridDim.x)
	{
		//block things
		if (threadIdx.x == 0)
		{
			T src = current.queue[i];
			srcStart = g.rowPtr[src];
			srcLen = g.rowPtr[src + 1] - srcStart;

			num_divs_local = (srcLen + 32 - 1) / 32;

			encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
			encode = &adj_enc[encode_offset  /*srcStart[wx]*/];
		}
		__syncthreads();

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

		for (T j = wx; j < srcLen; j += numPartitions)
		{
			level_offset[wx] = sm_id * CBPSM * (numPartitions * NUMDIVS * DEPTH) + levelPtr * (numPartitions * NUMDIVS * DEPTH);
			T* cl = &current_level[level_offset[wx] + wx * (NUMDIVS * DEPTH)];


			if (lx < DEPTH)
			{
				level_count[wx][lx] = 0;
				level_index[wx][lx] = 0;
				level_prev_index[wx][lx] = 0;
			}
			if (lx == 0)
			{
				l[wx] = 3;
				sg_count[wx] = 0;
			}

			//get warp count ??
			uint64 warpCount = 0;
			for (T k = lx; k < num_divs_local; k += CPARTSIZE)
			{
				cl[k] = get_mask(srcLen, k) & unset_mask(j, k);
				cl[num_divs_local + k] = (QEDGE_PTR[3] - QEDGE_PTR[2] == 2) ? 
										encode[j * num_divs_local + k] :
										cl[k];
				// Remove Redundancies
				for (T sym_idx = SYMNODE_PTR[l[wx] - 1]; sym_idx < SYMNODE_PTR[l[wx]]; sym_idx++) {
					if (SYMNODE[sym_idx] > 0) cl[num_divs_local + k] &= ~(cl[k] & get_mask(j, k));
				}
				warpCount += __popc(cl[num_divs_local + k]); 
			}
			reduce_part<T>(partMask, warpCount);
			// warpCount += __shfl_down_sync(partMask, warpCount, 16);
			// warpCount += __shfl_down_sync(partMask, warpCount, 8);
			// warpCount += __shfl_down_sync(partMask, warpCount, 4);
			// warpCount += __shfl_down_sync(partMask, warpCount, 2);
			// warpCount += __shfl_down_sync(partMask, warpCount, 1);

			if (lx == 0 && l[wx] == KCCOUNT)
				sg_count[wx] += warpCount;
			else if (lx == 0 && KCCOUNT > 3 ) // && warpCount >= KCCOUNT - 2)
			{
				level_count[wx][l[wx] - 3] = warpCount;
				level_index[wx][l[wx] - 3] = 0;
				level_prev_index[wx][1] = j + 1;
				level_prev_index[wx][2] = 0;
			}


			__syncwarp(partMask);
			while (level_count[wx][l[wx] - 3] > level_index[wx][l[wx] - 3])
			{
				//First Index
				T* from = &(cl[num_divs_local * (l[wx] - 2)]);
				T* to = &(cl[num_divs_local * (l[wx] - 1)]);
				T maskBlock = level_prev_index[wx][l[wx] - 1] / 32;
				T maskIndex = ~((1 << (level_prev_index[wx][l[wx] - 1] & 0x1F)) - 1);

				T newIndex = __ffs(from[maskBlock] & maskIndex);
				while (newIndex == 0)
				{
					maskIndex = 0xFFFFFFFF;
					maskBlock++;
					newIndex = __ffs(from[maskBlock] & maskIndex);
				}
				newIndex = 32 * maskBlock + newIndex - 1;

				if (lx == 0)
				{
					level_prev_index[wx][l[wx] - 1] = newIndex + 1;
					level_index[wx][l[wx] - 3]++;
				}

				//Intersect
				uint64 warpCount = 0;
				for (T k = lx; k < num_divs_local; k += CPARTSIZE)
				{
					to[k] = cl[k] & unset_mask(newIndex, k);
					// Compute Intersection
					for (T q_idx = QEDGE_PTR[l[wx]] + 1; q_idx < QEDGE_PTR[l[wx]+1]; q_idx++) {
						to[k]  &= encode[(level_prev_index[wx][QEDGE[q_idx]] - 1) * num_divs_local + k]; 
					}
					// Remove Redundancies
					for (T sym_idx = SYMNODE_PTR[l[wx]]; sym_idx < SYMNODE_PTR[l[wx]+1]; sym_idx++) {
						if (SYMNODE[sym_idx] > 0) to[k] &= ~(cl[(SYMNODE[sym_idx] - 1) * num_divs_local + k] & get_mask(level_prev_index[wx][SYMNODE[sym_idx]] - 1, k));
					}
					warpCount += __popc(to[k]);
				}
				reduce_part<T>(partMask, warpCount);
				// warpCount += __shfl_down_sync(partMask, warpCount, 16);
				// warpCount += __shfl_down_sync(partMask, warpCount, 8);
				// warpCount += __shfl_down_sync(partMask, warpCount, 4);
				// warpCount += __shfl_down_sync(partMask, warpCount, 2);
				// warpCount += __shfl_down_sync(partMask, warpCount, 1);

				if (lx == 0)
				{
					if (l[wx] + 1 == KCCOUNT) {
						sg_count[wx] += warpCount;
					}
					else if (l[wx] + 1 < KCCOUNT) //&& warpCount >= KCCOUNT - l[wx])
					{
						(l[wx])++;
						level_count[wx][l[wx] - 3] = warpCount;
						level_index[wx][l[wx] - 3] = 0;
						level_prev_index[wx][l[wx] - 1] = 0;

						for (T k = lx; k < num_divs_local; k += CPARTSIZE) 
							cl[k] &= unset_mask(level_prev_index[wx][l[wx]-2]-1, k);
					}

					while (l[wx] > 3 && level_index[wx][l[wx] - 3] >= level_count[wx][l[wx] - 3])
					{
						(l[wx])--;
						for (T k = lx; k < num_divs_local; k += CPARTSIZE)
							cl[k] &= set_mask(level_prev_index[wx][l[wx]-1]-1, k);
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
	}

	__syncthreads();
	if (threadIdx.x == 0)
	{
		atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
	}
}