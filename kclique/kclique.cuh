


#pragma once
#define QUEUE_SIZE 1024

#include <driver_types.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include "../include/utils.cuh"
#include "../include/Logger.cuh"
#include "../include/CGArray.cuh"


#include "../triangle_counting/TcBase.cuh"
#include "../triangle_counting/TcSerial.cuh"
#include "../triangle_counting/TcBinary.cuh"
#include "../triangle_counting/TcVariablehash.cuh"
#include "../triangle_counting/testHashing.cuh"
#include "../triangle_counting/TcBmp.cuh"

#include "../include/GraphQueue.cuh"

#include "kckernels.cuh"






template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_block_warp_binary_pivot_count(
	uint64* counter,
	graph::COOCSRGraph_d<T> g,
	const  graph::GraphQueue_d<T, bool>  current,
	T* current_level,
	uint64* cpn,
	T* levelStats,
	T* adj_enc,

	T* possible,
	T* level_index_g,
	T* level_count_g,
	T* level_prev_g,
	T* level_r,
	T* level_d,
	T* level_tmp,
	unsigned long long* nCR
)
{
	//will be removed later
	constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
	const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
	const size_t lx = threadIdx.x % CPARTSIZE;

	//__shared__ T  level_offset[numPartitions], level_item_offset[numPartitions]; //for l and p
	__shared__ T level_pivot[512];
	__shared__ uint64 clique_count[numPartitions];
	__shared__ uint64 path_more_explore;
	__shared__ T l;
	__shared__ T maxIntersection;
	__shared__ uint32_t  sm_id, levelPtr;
	__shared__ T src, srcStart, srcLen;
	__shared__ bool  partition_set[numPartitions];

	__shared__ T num_divs_local, encode_offset, * encode;
	__shared__ T* pl, * cl;
	__shared__ T* level_count, * level_index, * level_prev_index, * rsize, * drop;

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

	for (unsigned long long i = blockIdx.x; i < (unsigned long long)current.count[0]; i += gridDim.x)
	{
		__syncthreads();
		//block things
		if (threadIdx.x == 0)
		{
			src = current.queue[i];
			srcStart = g.rowPtr[src];
			srcLen = g.rowPtr[src + 1] - srcStart;

			//printf("src = %u, srcLen = %u\n", src, srcLen);

			num_divs_local = (srcLen + 32 - 1) / 32;
			encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
			encode = &adj_enc[encode_offset  /*srcStart[wx]*/];

			lo = sm_id * CBPSM * (/*numPartitions **/ NUMDIVS * MAXDEG) + levelPtr * (/*numPartitions **/ NUMDIVS * MAXDEG);
			cl = &current_level[lo/*level_offset[wx]/* + wx * (NUMDIVS * MAXDEG)*/];
			pl = &possible[lo/*level_offset[wx] /*+ wx * (NUMDIVS * MAXDEG)*/];

			level_item_offset = sm_id * CBPSM * (/*numPartitions **/ MAXDEG)+levelPtr * (/*numPartitions **/ MAXDEG);
			level_count = &level_count_g[level_item_offset /*+ wx*MAXDEG*/];
			level_index = &level_index_g[level_item_offset /*+ wx*MAXDEG*/];
			level_prev_index = &level_prev_g[level_item_offset /*+ wx*MAXDEG*/];
			rsize = &level_r[level_item_offset /*+ wx*MAXDEG*/]; // will be removed
			drop = &level_d[level_item_offset /*+ wx*MAXDEG*/];  //will be removed

			level_count[0] = 0;
			level_prev_index[0] = 0;
			level_index[0] = 0;
			l = 2;
			rsize[0] = 1;
			drop[0] = 0;

			level_pivot[0] = 0xFFFFFFFF;

			maxIntersection = 0;

			lastMask_i = srcLen / 32;
			lastMask_ii = (1 << (srcLen & 0x1F)) - 1;
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
		//Full Encode
		for (T j = wx; j < srcLen; j += numPartitions)
		{
			// if(current.queue[i] == 40 && lx == 0)
			// 	printf("%llu -> %u, ", j, g.colInd[srcStart + j]);
			graph::warp_sorted_count_and_encode_full<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&g.colInd[srcStart], srcLen,
				&g.colInd[g.rowPtr[g.colInd[srcStart + j]]], g.rowPtr[g.colInd[srcStart + j] + 1] - g.rowPtr[g.colInd[srcStart + j]],
				j, num_divs_local, encode);
		}
		__syncthreads(); //Done encoding

		//Find the first pivot
		if (lx == 0)
		{
			maxCount[wx] = 0;
			maxIndex[wx] = 0xFFFFFFFF;
			partMask[wx] = CPARTSIZE == 32 ? 0xFFFFFFFF : (1 << CPARTSIZE) - 1;
			partMask[wx] = partMask[wx] << ((wx % (32 / CPARTSIZE)) * CPARTSIZE);
		}
		__syncthreads();

		for (T j = wx; j < srcLen; j += numPartitions)
		{
			uint64 warpCount = 0;
			for (T k = lx; k < num_divs_local; k += CPARTSIZE)
			{
				warpCount += __popc(encode[j * num_divs_local + k]);
			}
			reduce_part<T>(partMask[wx], warpCount);

			if (lx == 0 && maxCount[wx] < warpCount)
			{
				maxCount[wx] = warpCount;
				maxIndex[wx] = j;
			}
		}
		if (lx == 0)
		{
			atomicMax(&(maxIntersection), maxCount[wx]);
		}
		__syncthreads();
		if (lx == 0)
		{
			if (maxIntersection == maxCount[wx]) // unsafe, but okay I need any one with this max count
			{
				atomicMin(&(level_pivot[0]), maxIndex[wx]);
			}
		}
		__syncthreads();

		//Prepare the Possible and Intersection Encode Lists
		uint64 warpCount = 0;

		for (T j = threadIdx.x; j < num_divs_local && maxIntersection > 0; j += BLOCK_DIM_X)
		{
			T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
			pl[j] = ~(encode[(level_pivot[0]) * num_divs_local + j]) & m;
			cl[j] = 0xFFFFFFFF;
			warpCount += __popc(pl[j]);
		}
		reduce_part<T>(partMask[wx], warpCount);
		if (lx == 0 && threadIdx.x < num_divs_local)
		{
			atomicAdd(&(level_count[0]), (T)warpCount);
		}
		__syncthreads();

		//Explore the tree
		while ((level_count[l - 2] > level_index[l - 2]))
		{
			T maskBlock = level_prev_index[l - 2] / 32;
			T maskIndex = ~((1 << (level_prev_index[l - 2] & 0x1F)) - 1);
			T newIndex = __ffs(pl[num_divs_local * (l - 2) + maskBlock] & maskIndex);
			while (newIndex == 0)
			{
				maskIndex = 0xFFFFFFFF;
				maskBlock++;
				newIndex = __ffs(pl[num_divs_local * (l - 2) + maskBlock] & maskIndex);
			}
			newIndex = 32 * maskBlock + newIndex - 1;
			T sameBlockMask = (~((1 << (newIndex & 0x1F)) - 1)) | ~pl[num_divs_local * (l - 2) + maskBlock];
			__syncthreads();
			if (threadIdx.x == 0)
			{
				level_prev_index[l - 2] = newIndex + 1;
				level_index[l - 2]++;
				level_pivot[l - 1] = 0xFFFFFFFF;
				path_more_explore = false;
				maxIntersection = 0;
				rsize[l - 1] = rsize[l - 2] + 1;
				drop[l - 1] = drop[l - 2];
				if (newIndex == level_pivot[l - 2])
					drop[l - 1] = drop[l - 2] + 1;
			}
			__syncthreads();
			//assert(level_prev_index[l - 2] == newIndex + 1);

			if (rsize[l - 1] - drop[l - 1] > KCCOUNT)
			{
				__syncthreads();
				//printf("Stop Here, %u %u\n", rsize[l-1], drop[l-1]);
				if (threadIdx.x == 0)
				{
					T c = rsize[l - 1] - KCCOUNT;
					unsigned long long ncr = nCR[drop[l - 1] * 401 + c];
					atomicAdd(counter, ncr/*rsize[l-1]*/);

					//printf, go back
					while (l > 2 && level_index[l - 2] >= level_count[l - 2])
					{
						(l)--;
					}
				}
				__syncthreads();
			}
			else
			{
				// Now prepare intersection list
				T* from = &(cl[num_divs_local * (l - 2)]);
				T* to = &(cl[num_divs_local * (l - 1)]);
				for (T k = threadIdx.x; k < num_divs_local; k += BLOCK_DIM_X)
				{
					to[k] = from[k] & encode[newIndex * num_divs_local + k];
					//remove previous pivots from here
					to[k] = to[k] & ((maskBlock < k) ? ~pl[num_divs_local * (l - 2) + k] : ((maskBlock > k) ? 0xFFFFFFFF : sameBlockMask));
				}
				if (lx == 0)
				{
					partition_set[wx] = false;
					maxCount[wx] = srcLen + 1; //make it shared !!
					maxIndex[wx] = 0;
				}
				__syncthreads();
				//////////////////////////////////////////////////////////////////////
				//Now new pivot generation, then check to extend to new level or not

				//T limit = (srcLen + numPartitions -1)/numPartitions;
				for (T j = wx; j < /*numPartitions*limit*/srcLen; j += numPartitions)
				{
					uint64 warpCount = 0;
					T bi = j / 32;
					T ii = j & 0x1F;
					if ((to[bi] & (1 << ii)) != 0)
					{
						for (T k = lx; k < num_divs_local; k += CPARTSIZE)
						{
							warpCount += __popc(to[k] & encode[j * num_divs_local + k]);
						}
						reduce_part<T>(partMask[wx], warpCount);
						if (lx == 0 && maxCount[wx] == srcLen + 1)
						{
							partition_set[wx] = true;
							path_more_explore = true; //shared, unsafe, but okay
							maxCount[wx] = warpCount;
							maxIndex[wx] = j;
						}
						else if (lx == 0 && maxCount[wx] < warpCount)
						{
							maxCount[wx] = warpCount;
							maxIndex[wx] = j;
						}
					}
				}

				__syncthreads();
				if (!path_more_explore)
				{
					__syncthreads();
					if (threadIdx.x == 0)
					{
						if (rsize[l - 1] >= KCCOUNT)
						{
							T c = rsize[l - 1] - KCCOUNT;
							unsigned long long ncr = nCR[drop[l - 1] * 401 + c];
							atomicAdd(counter, ncr/*rsize[l-1]*/);
						}
						//printf, go back
						while (l > 2 && level_index[l - 2] >= level_count[l - 2])
						{
							(l)--;
						}
					}
					__syncthreads();
				}
				else
				{
					if (lx == 0 && partition_set[wx])
					{
						atomicMax(&(maxIntersection), maxCount[wx]);
					}
					__syncthreads();

					if (lx == 0 && maxIntersection == maxCount[wx])
					{
						atomicMin(&(level_pivot[l - 1]), maxIndex[wx]);
					}
					__syncthreads();

					uint64 warpCount = 0;
					for (T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
					{
						T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
						pl[(l - 1) * num_divs_local + j] = ~(encode[level_pivot[l - 1] * num_divs_local + j]) & to[j] & m;
						warpCount += __popc(pl[(l - 1) * num_divs_local + j]);
					}
					reduce_part<T>(partMask[wx], warpCount);

					if (threadIdx.x == 0)
					{
						l++;
						level_count[l - 2] = 0;
						level_prev_index[l - 2] = 0;
						level_index[l - 2] = 0;
					}

					__syncthreads();
					if (lx == 0 && threadIdx.x < num_divs_local)
					{
						atomicAdd(&(level_count[l - 2]), warpCount);
					}
				}

			}
			__syncthreads();
			/////////////////////////////////////////////////////////////////////////
		}
	}

	__syncthreads();
	if (threadIdx.x == 0)
	{
		atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
	}
}


template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
//__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_edge_block_warp_binary_pivot_count(
	uint64* counter,
	graph::COOCSRGraph_d<T> g,
	const  graph::GraphQueue_d<T, bool>  current,
	T* current_level,
	uint64* cpn,
	T* levelStats,
	T* adj_enc,
	T* adj_tri,

	T* possible,
	T* level_index_g,
	T* level_count_g,
	T* level_prev_g,
	T* level_r,
	T* level_d,
	T* level_tmp,
	unsigned long long* nCR
)
{
	//will be removed later
	constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
	const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
	const size_t lx = threadIdx.x % CPARTSIZE;

	__shared__ T level_pivot[512];
	__shared__ uint64 clique_count[numPartitions];
	__shared__ uint64 path_more_explore;
	__shared__ T l;
	__shared__ T maxIntersection;
	__shared__ uint32_t  sm_id, levelPtr;
	__shared__ T src, srcStart, srcLen, src2, src2Start, src2Len, scounter;
	__shared__ bool  partition_set[numPartitions];
	__shared__ T num_divs_local, encode_offset, * encode, tri_offset, * tri;
	__shared__ T* pl, * cl;
	__shared__ T* level_count, * level_index, * level_prev_index, * rsize, * drop;
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

	for (unsigned long long i = blockIdx.x; i < (unsigned long long)current.count[0]; i += gridDim.x)
	{
		//block things
		if (threadIdx.x == 0)
		{
			src = g.rowInd[current.queue[i]];
			srcStart = g.rowPtr[src];
			srcLen = g.rowPtr[src + 1] - srcStart;
			src2 = g.colInd[current.queue[i]];
			src2Start = g.rowPtr[src2];
			src2Len = g.rowPtr[src2 + 1] - src2Start;
			tri_offset = sm_id * CBPSM * (MAXDEG)+levelPtr * (MAXDEG);
			tri = &adj_tri[tri_offset  /*srcStart[wx]*/];
			scounter = 0;

			encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
			encode = &adj_enc[encode_offset  /*srcStart[wx]*/];

			lo = sm_id * CBPSM * (/*numPartitions **/ NUMDIVS * MAXDEG) + levelPtr * (/*numPartitions **/ NUMDIVS * MAXDEG);
			cl = &current_level[lo/*level_offset[wx]/* + wx * (NUMDIVS * MAXDEG)*/];
			pl = &possible[lo/*level_offset[wx] /*+ wx * (NUMDIVS * MAXDEG)*/];

			level_item_offset = sm_id * CBPSM * (/*numPartitions **/ MAXDEG)+levelPtr * (/*numPartitions **/ MAXDEG);
			level_count = &level_count_g[level_item_offset /*+ wx*MAXDEG*/];
			level_index = &level_index_g[level_item_offset /*+ wx*MAXDEG*/];
			level_prev_index = &level_prev_g[level_item_offset /*+ wx*MAXDEG*/];
			rsize = &level_r[level_item_offset /*+ wx*MAXDEG*/]; // will be removed
			drop = &level_d[level_item_offset /*+ wx*MAXDEG*/];  //will be removed

			level_count[0] = 0;
			level_prev_index[0] = 0;
			level_index[0] = 0;
			l = 3;
			rsize[0] = 1;
			drop[0] = 0;

			level_pivot[0] = 0xFFFFFFFF;

			maxIntersection = 0;


		}

		// //get tri list: by block :!!
		__syncthreads();
		graph::block_sorted_count_and_set_tri<BLOCK_DIM_X, T>(&g.colInd[srcStart], srcLen, &g.colInd[src2Start], src2Len,
			tri, &scounter);

		__syncthreads();

		if (threadIdx.x == 0)
		{
			num_divs_local = (scounter + 32 - 1) / 32;
			lastMask_i = scounter / 32;
			lastMask_ii = (1 << (scounter & 0x1F)) - 1;
		}

		if (KCCOUNT == 3 && threadIdx.x == 0)
			atomicAdd(counter, scounter);


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

		//Full Encode
		for (T j = wx; j < scounter; j += numPartitions)
		{
			graph::warp_sorted_count_and_encode_full<WARPS_PER_BLOCK, T, true, CPARTSIZE>(tri, scounter,
				&g.colInd[g.rowPtr[tri[j]]], g.rowPtr[tri[j] + 1] - g.rowPtr[tri[j]],
				j, num_divs_local, encode);
		}
		__syncthreads(); //Done encoding

		if (lx == 0)
		{
			maxCount[wx] = 0;
			maxIndex[wx] = 0xFFFFFFFF;
			partMask[wx] = CPARTSIZE == 32 ? 0xFFFFFFFF : (1 << CPARTSIZE) - 1;
			partMask[wx] = partMask[wx] << ((wx % (32 / CPARTSIZE)) * CPARTSIZE);
		}
		__syncthreads();

		//Find the first pivot
		for (T j = wx; j < scounter; j += numPartitions)
		{
			uint64 warpCount = 0;
			for (T k = lx; k < num_divs_local; k += CPARTSIZE)
			{
				warpCount += __popc(encode[j * num_divs_local + k]);
			}
			reduce_part<T>(partMask[wx], warpCount);

			if (lx == 0 && maxCount[wx] < warpCount)
			{
				maxCount[wx] = warpCount;
				maxIndex[wx] = j;
			}
		}
		if (lx == 0)
		{
			atomicMax(&(maxIntersection), maxCount[wx]);
		}
		__syncthreads();
		if (lx == 0)
		{
			if (maxIntersection == maxCount[wx]) // unsafe, but okay I need any one with this max count
			{
				atomicMin(&(level_pivot[0]), maxIndex[wx]);
			}
		}
		__syncthreads();

		//Prepare the Possible and Intersection Encode Lists
		uint64 warpCount = 0;
		for (T j = threadIdx.x; j < num_divs_local && maxIntersection > 0; j += BLOCK_DIM_X)
		{
			T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
			pl[j] = ~(encode[(level_pivot[0]) * num_divs_local + j]) & m;
			cl[j] = 0xFFFFFFFF;
			warpCount += __popc(pl[j]);
		}
		reduce_part<T>(partMask[wx], warpCount);
		if (lx == 0 && threadIdx.x < num_divs_local)
		{
			atomicAdd(&(level_count[0]), (T)warpCount);
		}
		__syncthreads();
		while ((level_count[l - 3] > level_index[l - 3]))
		{
			T maskBlock = level_prev_index[l - 3] / 32;
			T maskIndex = ~((1 << (level_prev_index[l - 3] & 0x1F)) - 1);
			T newIndex = __ffs(pl[num_divs_local * (l - 3) + maskBlock] & maskIndex);
			while (newIndex == 0)
			{
				maskIndex = 0xFFFFFFFF;
				maskBlock++;
				newIndex = __ffs(pl[num_divs_local * (l - 3) + maskBlock] & maskIndex);
			}
			newIndex = 32 * maskBlock + newIndex - 1;
			T sameBlockMask = (~((1 << (newIndex & 0x1F)) - 1)) | ~pl[num_divs_local * (l - 3) + maskBlock];
			__syncthreads();
			if (threadIdx.x == 0)
			{
				level_prev_index[l - 3] = newIndex + 1;
				level_index[l - 3]++;
				level_pivot[l - 2] = 0xFFFFFFFF;
				path_more_explore = false;
				maxIntersection = 0;
				rsize[l - 2] = rsize[l - 3] + 1;
				drop[l - 2] = drop[l - 3];
				if (newIndex == level_pivot[l - 3])
					drop[l - 2] = drop[l - 3] + 1;
			}
			__syncthreads();
			//assert(level_prev_index[l - 2] == newIndex + 1);

			if (rsize[l - 2] - drop[l - 2] > KCCOUNT)
			{
				__syncthreads();
				//printf("Stop Here, %u %u\n", rsize[l-1], drop[l-1]);
				if (threadIdx.x == 0)
				{
					T c = rsize[l - 2] - KCCOUNT;
					unsigned long long ncr = nCR[drop[l - 2] * 401 + c];
					atomicAdd(counter, ncr/*rsize[l-1]*/);

					//printf, go back
					while (l > 3 && level_index[l - 3] >= level_count[l - 3])
					{
						(l)--;
					}
				}
				__syncthreads();
			}
			else
			{
				// Now prepare intersection list
				T* from = &(cl[num_divs_local * (l - 3)]);
				T* to = &(cl[num_divs_local * (l - 2)]);
				for (T k = threadIdx.x; k < num_divs_local; k += BLOCK_DIM_X)
				{
					to[k] = from[k] & encode[newIndex * num_divs_local + k];
					//remove previous pivots from here
					to[k] = to[k] & ((maskBlock < k) ? ~pl[num_divs_local * (l - 3) + k] : ((maskBlock > k) ? 0xFFFFFFFF : sameBlockMask));
				}
				if (lx == 0)
				{
					partition_set[wx] = false;
					maxCount[wx] = scounter + 1; //make it shared !!
					maxIndex[wx] = 0;
				}
				__syncthreads();
				//////////////////////////////////////////////////////////////////////
				//Now new pivot generation, then check to extend to new level or not

				//T limit = (srcLen + numPartitions -1)/numPartitions;
				for (T j = wx; j < /*numPartitions*limit*/scounter; j += numPartitions)
				{
					uint64 warpCount = 0;
					T bi = j / 32;
					T ii = j & 0x1F;
					if ((to[bi] & (1 << ii)) != 0)
					{
						for (T k = lx; k < num_divs_local; k += CPARTSIZE)
						{
							warpCount += __popc(to[k] & encode[j * num_divs_local + k]);
						}
						reduce_part<T>(partMask[wx], warpCount);
						if (lx == 0 && maxCount[wx] == scounter + 1)
						{
							partition_set[wx] = true;
							path_more_explore = true; //shared, unsafe, but okay
							maxCount[wx] = warpCount;
							maxIndex[wx] = j;
						}
						else if (lx == 0 && maxCount[wx] < warpCount)
						{
							maxCount[wx] = warpCount;
							maxIndex[wx] = j;
						}
					}
				}

				__syncthreads();
				if (!path_more_explore)
				{
					__syncthreads();
					if (threadIdx.x == 0)
					{
						if (rsize[l - 2] >= KCCOUNT)
						{
							T c = rsize[l - 2] - KCCOUNT;
							unsigned long long ncr = nCR[drop[l - 2] * 401 + c];
							atomicAdd(counter, ncr/*rsize[l-1]*/);
						}
						//printf, go back
						while (l > 3 && level_index[l - 3] >= level_count[l - 3])
						{
							(l)--;
						}
					}
					__syncthreads();
				}
				else
				{
					if (lx == 0 && partition_set[wx])
					{
						atomicMax(&(maxIntersection), maxCount[wx]);
					}
					__syncthreads();

					if (lx == 0 && maxIntersection == maxCount[wx])
					{
						atomicMin(&(level_pivot[l - 2]), maxIndex[wx]);
					}
					__syncthreads();

					uint64 warpCount = 0;
					for (T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
					{
						T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
						pl[(l - 2) * num_divs_local + j] = ~(encode[level_pivot[l - 2] * num_divs_local + j]) & to[j] & m;
						warpCount += __popc(pl[(l - 2) * num_divs_local + j]);
					}
					reduce_part<T>(partMask[wx], warpCount);

					if (threadIdx.x == 0)
					{
						l++;
						level_count[l - 3] = 0;
						level_prev_index[l - 3] = 0;
						level_index[l - 3] = 0;
					}

					__syncthreads();
					if (lx == 0 && threadIdx.x < num_divs_local)
					{
						atomicAdd(&(level_count[l - 3]), warpCount);
					}
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

namespace graph
{
	template<typename T>
	class SingleGPU_Kclique
	{
	private:
		int dev_;
		cudaStream_t stream_;

		//Outputs:
		//Max k of a complete ktruss kernel
		int k;


		//Percentage of deleted edges for a specific k
		float percentage_deleted_k;

		//Same Function for any comutation
		void bucket_scan(
			GPUArray<T> nodeDegree, T node_num, T level, T span,
			GraphQueue<T, bool>& current,
			GPUArray<T> asc,
			GraphQueue<T, bool>& bucket,
			T& bucket_level_end_)
		{
			static bool is_first = true;
			static int multi = 1;
			if (is_first)
			{
				current.mark.setAll(false, true);
				bucket.mark.setAll(false, true);
				is_first = false;
			}

			if (level == bucket_level_end_)
			{
				// Clear the bucket_removed_indicator


				long grid_size = (node_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
				execKernel((filter_window<T, T>), grid_size, BLOCK_SIZE, dev_, false,
					nodeDegree.gdata(), node_num, bucket.mark.gdata(), level, bucket_level_end_ + KCL_NODE_LEVEL_SKIP_SIZE);

				multi++;

				bucket.count.gdata()[0] = CUBSelect(asc.gdata(), bucket.queue.gdata(), bucket.mark.gdata(), node_num, dev_);
				bucket_level_end_ += KCL_NODE_LEVEL_SKIP_SIZE;
			}
			// SCAN the window.
			if (bucket.count.gdata()[0] != 0)
			{
				current.count.gdata()[0] = 0;
				long grid_size = (bucket.count.gdata()[0] + BLOCK_SIZE - 1) / BLOCK_SIZE;
				execKernel((filter_with_random_append<T, T>), grid_size, BLOCK_SIZE, dev_, false,
					bucket.queue.gdata(), bucket.count.gdata()[0], nodeDegree.gdata(), current.mark.gdata(), current.queue.gdata(), &(current.count.gdata()[0]), level, span);
			}
			else
			{
				current.count.gdata()[0] = 0;
			}
			//Log(LogPriorityEnum::info, "Level: %d, curr: %d/%d", level, current.count.gdata()[0], bucket.count.gdata()[0]);
		}


		//Same Function for any comutation
		void bucket_edge_scan(
			GPUArray<T> nodeDegree, T node_num, T level, T span,
			GraphQueue<T, bool>& current,
			GPUArray<T> asc,
			GraphQueue<T, bool>& bucket,
			T& bucket_level_end_)
		{
			static bool is_first = true;
			static int multi = 1;
			if (is_first)
			{
				current.mark.setAll(false, true);
				bucket.mark.setAll(false, true);
				is_first = false;
			}

			if (level == bucket_level_end_)
			{
				// Clear the bucket_removed_indicator


				long grid_size = (node_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
				execKernel(filter_window, grid_size, BLOCK_SIZE, dev_, false,
					nodeDegree.gdata(), node_num, bucket.mark.gdata(), level, bucket_level_end_ + KCL_EDGE_LEVEL_SKIP_SIZE);

				multi++;

				bucket.count.gdata()[0] = CUBSelect(asc.gdata(), bucket.queue.gdata(), bucket.mark.gdata(), node_num, dev_);
				bucket_level_end_ += KCL_EDGE_LEVEL_SKIP_SIZE;
			}
			// SCAN the window.
			if (bucket.count.gdata()[0] != 0)
			{
				current.count.gdata()[0] = 0;
				long grid_size = (bucket.count.gdata()[0] + BLOCK_SIZE - 1) / BLOCK_SIZE;
				execKernel((filter_with_random_append<T>), grid_size, BLOCK_SIZE, dev_, false,
					bucket.queue.gdata(), bucket.count.gdata()[0], nodeDegree.gdata(), current.mark.gdata(), current.queue.gdata(), &(current.count.gdata()[0]), level, span);
			}
			else
			{
				current.count.gdata()[0] = 0;
			}
			//Log(LogPriorityEnum::info, "Level: %d, curr: %d/%d", level, current.count.gdata()[0], bucket.count.gdata()[0]);
		}

		void AscendingGpu(T n, GPUArray<T>& identity_arr_asc)
		{
			long grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
			identity_arr_asc.initialize("Identity Array Asc", AllocationTypeEnum::gpu, n, dev_);
			execKernel(init_asc, grid_size, BLOCK_SIZE, dev_, false, identity_arr_asc.gdata(), n);
		}

	public:
		GPUArray<T> nodeDegree;
		GPUArray<T> edgePtr;
		graph::GraphQueue<T, bool> bucket_q;
		graph::GraphQueue<T, bool> current_q;
		GPUArray<T> identity_arr_asc;

		SingleGPU_Kclique(int dev, COOCSRGraph_d<T>& g) : dev_(dev) {
			CUDA_RUNTIME(cudaSetDevice(dev_));
			CUDA_RUNTIME(cudaStreamCreate(&stream_));
			CUDA_RUNTIME(cudaStreamSynchronize(stream_));

			bucket_q.Create(unified, g.numEdges, dev_);
			current_q.Create(unified, g.numEdges, dev_);
			AscendingGpu(g.numEdges, identity_arr_asc);

			edgePtr.initialize("Edge Support", unified, g.numEdges, dev_);
		}

		SingleGPU_Kclique() : SingleGPU_Kclique(0) {}


		void getNodeDegree(COOCSRGraph_d<T>& g, T* maxDegree,
			const size_t nodeOffset = 0, const size_t edgeOffset = 0)
		{
			const int dimBlock = 128;
			nodeDegree.initialize("Edge Support", unified, g.numNodes, dev_);
			uint dimGridNodes = (g.numNodes + dimBlock - 1) / dimBlock;
			execKernel((getNodeDegree_kernel<T, dimBlock>), dimGridNodes, dimBlock, dev_, false, nodeDegree.gdata(), g, maxDegree);
		}

		void findKclqueIncremental_node_async(int kcount, COOCSRGraph_d<T>& g,
			ProcessingElementEnum pe, const size_t nodeOffset = 0, const size_t edgeOffset = 0)
		{
			CUDA_RUNTIME(cudaSetDevice(dev_));
			const auto block_size = 128;
			CUDAContext context;
			T num_SMs = context.num_SMs;
			T level = 0;
			T span = 1024;
			T bucket_level_end_ = level;
			T todo = g.numNodes;
			GPUArray <uint64> counter("Temp level Counter", unified, 1, dev_);
			GPUArray <uint64> cpn("Temp level Counter", unified, g.numNodes, dev_);
			GPUArray <T> maxDegree("Temp Degree", unified, 1, dev_);

			T conc_blocks_per_SM = context.GetConCBlocks(block_size);
			GPUArray<T> d_bitmap_states("bmp bitmap stats", AllocationTypeEnum::gpu, num_SMs * conc_blocks_per_SM, dev_);
			T factor = (pe == Block) ? 1 : (block_size / 32);


			// cpn.setAll(0, true);
			// GPUArray<T>
			// 	filter_level("Temp filter Counter", unified, g.numEdges, dev_),
			// 	filter_scan("Temp scan Counter", unified, g.numEdges, dev_);
			counter.setSingle(0, 0, true);
			maxDegree.setSingle(0, 0, true);
			d_bitmap_states.setAll(0, true);
			getNodeDegree(g, maxDegree.gdata());
			bucket_scan(nodeDegree, g.numNodes, 0, kcount - 1, current_q, identity_arr_asc, bucket_q, bucket_level_end_);
			todo -= current_q.count.gdata()[0];
			current_q.count.gdata()[0] = 0;
			level = kcount - 1;
			bucket_level_end_ = level;

			/*level = 32;
			bucket_level_end_ = level;*/
			while (todo > 0)
			{
				CUDA_RUNTIME(cudaGetLastError());
				cudaDeviceSynchronize();

				//1 bucket fill
				bucket_scan(nodeDegree, g.numNodes, level, span, current_q, identity_arr_asc, bucket_q, bucket_level_end_);
				CUDA_RUNTIME(cudaGetLastError());
				cudaDeviceSynchronize();

				todo -= current_q.count.gdata()[0];
				if (current_q.count.gdata()[0] > 0)
				{

					// std::sort(current_q.queue.gdata(), current_q.queue.gdata() + current_q.count.gdata()[0]);
					// current_q.count.gdata()[0] = current_q.count.gdata()[0]< 128? current_q.count.gdata()[0]: 128;
					//current_q.count.gdata()[0] = 1; 
					if (pe == Warp)
					{
						GPUArray<char> current_level("Temp level Counter", unified, num_SMs * conc_blocks_per_SM * factor * maxDegree.gdata()[0], dev_);
						current_level.setAll(2, true);

						auto grid_block_size = (32 * current_q.count.gdata()[0] + block_size - 1) / block_size;
						execKernel((kckernel_node_warp_sync_count<T, block_size>), grid_block_size, block_size, dev_, false,
							counter.gdata(),
							g,
							kcount,
							maxDegree.gdata()[0],
							current_q.device_queue->gdata()[0],
							current_level.gdata(), cpn.gdata()
							, conc_blocks_per_SM, d_bitmap_states.gdata());

						current_level.freeGPU();
					}
					else if (pe == Block)
					{
						GPUArray<char> current_level("Temp level Counter", unified, num_SMs * conc_blocks_per_SM * factor * maxDegree.gdata()[0], dev_);
						current_level.setAll(2, true);

						auto grid_block_size = current_q.count.gdata()[0];
						execKernel((kckernel_node_block_count<T, block_size>), grid_block_size, block_size, dev_, false,
							counter.gdata(),
							g,
							kcount,
							maxDegree.gdata()[0],
							current_q.device_queue->gdata()[0],
							current_level.gdata(), cpn.gdata(),
							conc_blocks_per_SM, d_bitmap_states.gdata());

						current_level.freeGPU();
					}

					else if (pe == BlockWarp)
					{

						const T partitionSize = PART_SIZE;
						factor = (block_size / partitionSize);

						const uint dv = 32;
						const uint max_level = 7;
						uint num_divs = (maxDegree.gdata()[0] + dv - 1) / dv;
						const uint64 level_size = num_SMs * conc_blocks_per_SM * factor * max_level * num_divs;
						const uint64 encode_size = num_SMs * conc_blocks_per_SM * maxDegree.gdata()[0] * num_divs;
						printf("Level Size = %llu, Encode Size = %llu\n", level_size, encode_size);
						GPUArray<T> current_level2("Temp level Counter", unified, level_size, dev_);
						GPUArray<T> node_be("Temp level Counter", unified, encode_size, dev_);
						current_level2.setAll(0, true);
						node_be.setAll(0, true);



						const T numPartitions = block_size / partitionSize;
						cudaMemcpyToSymbol(KCCOUNT, &kcount, sizeof(KCCOUNT));
						cudaMemcpyToSymbol(PARTSIZE, &partitionSize, sizeof(PARTSIZE));
						cudaMemcpyToSymbol(NUMPART, &numPartitions, sizeof(NUMPART));
						cudaMemcpyToSymbol(MAXLEVEL, &max_level, sizeof(MAXLEVEL));
						cudaMemcpyToSymbol(NUMDIVS, &num_divs, sizeof(NUMDIVS));
						cudaMemcpyToSymbol(MAXDEG, &(maxDegree.gdata()[0]), sizeof(MAXDEG));
						cudaMemcpyToSymbol(CBPSM, &(conc_blocks_per_SM), sizeof(CBPSM));

						auto grid_block_size = current_q.count.gdata()[0];
						execKernel((kckernel_node_block_warp_binary_count<T, block_size, partitionSize>), grid_block_size, block_size, dev_, false,
							counter.gdata(),
							g,
							current_q.device_queue->gdata()[0],
							current_level2.gdata(), cpn.gdata(),
							d_bitmap_states.gdata(), node_be.gdata());


						current_level2.freeGPU();
					}
					std::cout.imbue(std::locale(""));
					std::cout << "------------- Level = " << level << " Nodes = " << current_q.count.gdata()[0] << " Counter = " << counter.gdata()[0] << "\n";
				}
				level += span;
			}

			counter.freeGPU();
			cpn.freeGPU();

			d_bitmap_states.freeGPU();
			k = level;
			printf("Max Degree (+span) = %d\n", k - 1);
		}
		void findKclqueIncremental_edge_async(int kcount, COOCSRGraph_d<T>& g,
			ProcessingElementEnum pe, const size_t nodeOffset = 0, const size_t edgeOffset = 0)
		{
			CUDA_RUNTIME(cudaSetDevice(dev_));

			const auto block_size = 128;
			CUDAContext context;
			T num_SMs = context.num_SMs;

			T level = 0;
			T span = 1024;
			T bucket_level_end_ = level;
			T todo = g.numEdges;

			GPUArray <uint64> counter("Temp level Counter", unified, 1, dev_);
			GPUArray <T> maxDegree("Temp Degree", unified, 1, dev_);

			T conc_blocks_per_SM = context.GetConCBlocks(block_size);
			GPUArray<T> d_bitmap_states("bmp bitmap stats", AllocationTypeEnum::gpu, num_SMs * conc_blocks_per_SM, dev_);
			T factor = (pe == Block) ? 1 : (block_size / 32);



			counter.setSingle(0, 0, true);
			maxDegree.setSingle(0, 0, true);
			execKernel((get_max_degree<T, 128>), (g.numEdges + 128 - 1) / 128, 128, dev_, false, g, edgePtr.gdata(), maxDegree.gdata());
			GPUArray<char> current_level("Temp level Counter", unified, num_SMs * conc_blocks_per_SM * factor * maxDegree.gdata()[0], dev_);


			printf("Max Dgree = %u vs %u\n", maxDegree.gdata()[0], g.numEdges);
			bucket_edge_scan(edgePtr, g.numEdges, 0, kcount - 2, current_q, identity_arr_asc, bucket_q, bucket_level_end_);
			todo -= current_q.count.gdata()[0];
			current_q.count.gdata()[0] = 0;
			level = kcount - 2;
			bucket_level_end_ = level;
			CUDA_RUNTIME(cudaGetLastError());
			cudaDeviceSynchronize();

			/*	GPUArray <uint64> cpn("Temp Degree", unified, g.numEdges, dev_);
				cpn.setAll(0, true);*/

			while (todo > 0)
			{
				//1 bucket fill
				bucket_edge_scan(edgePtr, g.numEdges, level, span, current_q, identity_arr_asc, bucket_q, bucket_level_end_);
				todo -= current_q.count.gdata()[0];
				if (current_q.count.gdata()[0] > 0)
				{
					//std::sort(current_q.queue.gdata(), current_q.queue.gdata() + current_q.count.gdata()[0]);
					//current_q.count.gdata()[0] = current_q.count.gdata()[0]< 5000? current_q.count.gdata()[0]: 5000;
					//current_q.count.gdata()[0] = 1;

					if (pe == Warp)
					{
						// auto grid_block_size = (32 * current_q.count.gdata()[0] + block_size - 1) / block_size;
						// execKernel((kckernel_edge_warp_count2_shared<T, block_size>), grid_block_size, block_size, dev_, false,
						// 	counter.gdata(),
						// 	g,
						// 	kcount,
						// 	maxDegree.gdata()[0],
						// 	current_q.device_queue->gdata()[0],
						// 	current_level.gdata(),
						// 	NULL,
						// 	conc_blocks_per_SM, d_bitmap_states.gdata());



						const uint dv = 32;
						const uint max_level = 7;
						uint num_divs = (maxDegree.gdata()[0] + dv - 1) / dv;
						const uint64 level_size = num_SMs * conc_blocks_per_SM * factor * max_level * num_divs;
						const uint64 encode_size = num_SMs * conc_blocks_per_SM * factor * maxDegree.gdata()[0] * num_divs;
						const uint64 tri_size = num_SMs * conc_blocks_per_SM * factor * maxDegree.gdata()[0];
						printf("Level Size = %llu, Encode Size = %llu\n", level_size, encode_size);
						GPUArray<T> current_level2("Temp level Counter", unified, level_size, dev_);
						GPUArray<T> node_be("Temp level Counter", unified, encode_size, dev_);
						GPUArray<T> tri_list("Temp level Counter", unified, tri_size, dev_);
						current_level2.setAll(0, true);
						node_be.setAll(0, true);
						tri_list.setAll(0, true);

						auto grid_block_size = (32 * current_q.count.gdata()[0] + block_size - 1) / block_size;
						execKernel((kckernel_edge_warp_binary_count<T, block_size>), grid_block_size, block_size, dev_, false,
							counter.gdata(),
							g,
							kcount,
							maxDegree.gdata()[0],
							current_q.device_queue->gdata()[0],
							current_level2.gdata(), NULL,
							conc_blocks_per_SM, d_bitmap_states.gdata(), node_be.gdata(), tri_list.gdata());


						current_level2.freeGPU();


					}
					else if (pe == Block)
					{
						auto grid_block_size = current_q.count.gdata()[0];
						execKernel((kckernel_edge_block_count<T, block_size>), grid_block_size, block_size, dev_, false,
							counter.gdata(),
							g,
							kcount,
							maxDegree.gdata()[0],
							current_q.device_queue->gdata()[0],
							current_level.gdata(),
							NULL,
							conc_blocks_per_SM, d_bitmap_states.gdata());
					}
					else if (pe == BlockWarp)
					{
						const T partitionSize = PART_SIZE;
						factor = (block_size / partitionSize);

						const uint dv = 32;
						const uint max_level = 7;
						uint num_divs = (maxDegree.gdata()[0] + dv - 1) / dv;
						const uint64 level_size = num_SMs * conc_blocks_per_SM * factor * max_level * num_divs;
						const uint64 encode_size = num_SMs * conc_blocks_per_SM * maxDegree.gdata()[0] * num_divs;
						const uint64 tri_size = num_SMs * conc_blocks_per_SM * maxDegree.gdata()[0];
						printf("Level Size = %llu, Encode Size = %llu\n", level_size, encode_size);
						GPUArray<T> current_level2("Temp level Counter", unified, level_size, dev_);
						GPUArray<T> node_be("Temp level Counter", unified, encode_size, dev_);
						GPUArray<T> tri_list("Temp level Counter", unified, tri_size, dev_);

						// simt::atomic<KCTask<T>, simt::thread_scope_device> *queue_data;
						// CUDA_RUNTIME(cudaMalloc((void **)&queue_data, (num_SMs * conc_blocks_per_SM * QUEUE_SIZE) * sizeof(simt::atomic<KCTask<T>, simt::thread_scope_device>)));

						GPUArray<KCTask<T>> queue_data("test", unified, num_SMs * conc_blocks_per_SM * QUEUE_SIZE, dev_);
						GPUArray<T> queue_encode("test", unified, num_SMs * conc_blocks_per_SM * QUEUE_SIZE * num_divs, dev_);





						current_level2.setAll(0, true);
						node_be.setAll(0, true);
						tri_list.setAll(0, true);

						const T numPartitions = block_size / partitionSize;
						cudaMemcpyToSymbol(KCCOUNT, &kcount, sizeof(KCCOUNT));
						cudaMemcpyToSymbol(PARTSIZE, &partitionSize, sizeof(PARTSIZE));
						cudaMemcpyToSymbol(NUMPART, &numPartitions, sizeof(NUMPART));
						cudaMemcpyToSymbol(MAXLEVEL, &max_level, sizeof(MAXLEVEL));
						cudaMemcpyToSymbol(NUMDIVS, &num_divs, sizeof(NUMDIVS));
						cudaMemcpyToSymbol(MAXDEG, &(maxDegree.gdata()[0]), sizeof(MAXDEG));
						cudaMemcpyToSymbol(CBPSM, &(conc_blocks_per_SM), sizeof(CBPSM));

						auto grid_block_size = current_q.count.gdata()[0];
						// execKernel((kckernel_edge_block_warp_binary_count<T, block_size, partitionSize>), grid_block_size, block_size, dev_, false,
						// 	counter.gdata(),
						// 	g,
						// 	current_q.device_queue->gdata()[0],
						// 	current_level2.gdata(), NULL,
						// 	d_bitmap_states.gdata(), node_be.gdata(), tri_list.gdata(),
						// 	queue_data.gdata(),
						// 	queue_encode.gdata()

						// );


						execKernel((kckernel_edge_block_warp_binary_count_o<T, block_size, partitionSize>), grid_block_size, block_size, dev_, false,
							counter.gdata(),
							g,
							current_q.device_queue->gdata()[0],
							current_level2.gdata(), NULL,
							d_bitmap_states.gdata(), node_be.gdata(), tri_list.gdata()

						);


						current_level2.freeGPU();
						// node_be.freeGPU();
						// tri_list.freeGPU();
					}

				}
				level += span;

				std::cout.imbue(std::locale(""));
				std::cout << "Level = " << level << " Nodes = " << current_q.count.gdata()[0] << " Counter = " << counter.gdata()[0] << "\n";
			}

			counter.freeGPU();
			//cpn.freeGPU();
			current_level.freeGPU();

			d_bitmap_states.freeGPU();
			maxDegree.freeGPU();
			k = level;

			printf("Max Edge Min Degree = %d\n", k - 1);

		}






		void findKclqueIncremental_node_pivot_async(int kcount, COOCSRGraph_d<T>& g,
			ProcessingElementEnum pe, const size_t nodeOffset = 0, const size_t edgeOffset = 0)
		{

			FILE* infile;
			GPUArray<unsigned long long> nCr("bmp bitmap stats", AllocationTypeEnum::gpu, 1001 * 401, dev_);
			infile = fopen("/home/almasri3/mewcp-gpu/kclique/nCr.txt", "r"); //change it
			double d = 0;
			if (infile == NULL)
			{
				printf("file could not be opened\n");
				exit(1);
			}

			for (int row = 0; row < 1001; ++row)
			{
				for (int col = 0; col < 401; ++col)
				{
					if (!fscanf(infile, "%lf,", &d))
						fprintf(stderr, "Error\n");
					// fprintf(stderr, "%lf\n", d);
					nCr.cdata()[row * 401 + col] = (unsigned long long)d;
				}
			}
			fclose(infile);
			printf("Test, 5c4 = %llu\n", nCr.cdata()[7 * 401 + 4]);

			nCr.switch_to_gpu();


			CUDA_RUNTIME(cudaSetDevice(dev_));
			const auto block_size = 128;
			CUDAContext context;
			T num_SMs = context.num_SMs;
			T level = 0;
			T span = 1024;
			T bucket_level_end_ = level;
			T todo = g.numNodes;
			GPUArray <uint64> counter("Temp level Counter", unified, 1, dev_);
			GPUArray <uint64> cpn("Temp level Counter", unified, g.numNodes, dev_);
			GPUArray <T> maxDegree("Temp Degree", unified, 1, dev_);

			T conc_blocks_per_SM = context.GetConCBlocks(block_size);
			GPUArray<T> d_bitmap_states("bmp bitmap stats", AllocationTypeEnum::gpu, num_SMs * conc_blocks_per_SM, dev_);
			T factor = (pe == Block) ? 1 : (block_size / 32);

			// cpn.setAll(0, true);
			// GPUArray<T>
			// 	filter_level("Temp filter Counter", unified, g.numEdges, dev_),
			// 	filter_scan("Temp scan Counter", unified, g.numEdges, dev_);
			counter.setSingle(0, 0, true);
			maxDegree.setSingle(0, 0, true);
			d_bitmap_states.setAll(0, true);
			getNodeDegree(g, maxDegree.gdata());
			bucket_scan(nodeDegree, g.numNodes, 0, kcount - 1, current_q, identity_arr_asc, bucket_q, bucket_level_end_);
			todo -= current_q.count.gdata()[0];
			current_q.count.gdata()[0] = 0;
			level = kcount - 1;
			bucket_level_end_ = level;


			const T partitionSize = PART_SIZE; //Defined
			factor = (block_size / partitionSize);

			const uint dv = 32;
			const uint max_level = maxDegree.gdata()[0];
			uint num_divs = (maxDegree.gdata()[0] + dv - 1) / dv;

			const uint64 encode_size = num_SMs * conc_blocks_per_SM * maxDegree.gdata()[0] * num_divs; //per block
			GPUArray<T> node_be("Temp level Counter", unified, encode_size, dev_);

			const uint64 level_size = num_SMs * conc_blocks_per_SM * factor * max_level * num_divs; //per partition
			const uint64 level_item_size = num_SMs * conc_blocks_per_SM * factor * max_level; //per partition
			const uint64 level_partition_size = num_SMs * conc_blocks_per_SM * factor * num_divs; //per partition

			GPUArray<T> current_level2("Temp level Counter", unified, level_size, dev_);
			GPUArray<T> possible("Temp level Counter", unified, level_size, dev_);

			GPUArray<T> level_index("Temp level Counter", unified, level_item_size, dev_);
			GPUArray<T> level_count("Temp level Counter", unified, level_item_size, dev_);
			GPUArray<T> level_prev("Temp level Counter", unified, level_item_size, dev_);
			GPUArray<T> level_r("Temp level Counter", unified, level_item_size, dev_);
			GPUArray<T> level_d("Temp level Counter", unified, level_item_size, dev_);
			GPUArray<T> level_temp("Temp level Counter", unified, level_partition_size, dev_);

			printf("Level Size = %llu, Encode Size = %llu\n", 2 * level_size + 5 * level_item_size + 1 * level_partition_size, encode_size);

			// current_level2.setAll(0, true);
			// node_be.setAll(0, true);
			const T numPartitions = block_size / partitionSize;
			cudaMemcpyToSymbol(KCCOUNT, &kcount, sizeof(KCCOUNT));
			cudaMemcpyToSymbol(PARTSIZE, &partitionSize, sizeof(PARTSIZE));
			cudaMemcpyToSymbol(NUMPART, &numPartitions, sizeof(NUMPART));
			cudaMemcpyToSymbol(MAXLEVEL, &max_level, sizeof(MAXLEVEL));
			cudaMemcpyToSymbol(NUMDIVS, &num_divs, sizeof(NUMDIVS));
			cudaMemcpyToSymbol(MAXDEG, &(maxDegree.gdata()[0]), sizeof(MAXDEG));
			cudaMemcpyToSymbol(CBPSM, &(conc_blocks_per_SM), sizeof(CBPSM));

			//while (todo > 0)
			{
				CUDA_RUNTIME(cudaGetLastError());
				cudaDeviceSynchronize();

				//1 bucket fill
				bucket_scan(nodeDegree, g.numNodes, level, span, current_q, identity_arr_asc, bucket_q, bucket_level_end_);
				CUDA_RUNTIME(cudaGetLastError());
				cudaDeviceSynchronize();

				todo -= current_q.count.gdata()[0];
				if (current_q.count.gdata()[0] > 0)
				{
					auto grid_block_size = current_q.count.gdata()[0];
					execKernel((kckernel_node_block_warp_binary_pivot_count<T, block_size, partitionSize>), grid_block_size, block_size, dev_, false,
						counter.gdata(),
						g,
						current_q.device_queue->gdata()[0],
						current_level2.gdata(), cpn.gdata(),
						d_bitmap_states.gdata(), node_be.gdata(),

						possible.gdata(),
						level_index.gdata(),
						level_count.gdata(),
						level_prev.gdata(),
						level_r.gdata(),
						level_d.gdata(),
						level_temp.gdata(),
						nCr.gdata()
					);



					std::cout.imbue(std::locale(""));
					std::cout << "------------- Level = " << level << " Nodes = " << current_q.count.gdata()[0] << " Counter = " << counter.gdata()[0] << "\n";
				}
				level += span;
			}

			counter.freeGPU();
			cpn.freeGPU();
			current_level2.freeGPU();
			d_bitmap_states.freeGPU();
			node_be.freeGPU();
			possible.freeGPU();
			level_index.freeGPU();
			level_count.freeGPU();
			level_prev.freeGPU();
			level_r.freeGPU();
			level_d.freeGPU();
			nCr.freeGPU();


			k = level;
			printf("Max Degree (+span) = %d\n", k - 1);
		}



		void findKclqueIncremental_edge_pivot_async(int kcount, COOCSRGraph_d<T>& g,
			ProcessingElementEnum pe, const size_t nodeOffset = 0, const size_t edgeOffset = 0)
		{


			//Please be carful
			kcount = kcount - 1;

			FILE* infile;
			GPUArray<unsigned long long> nCr("bmp bitmap stats", AllocationTypeEnum::gpu, 1001 * 401, dev_);
			infile = fopen("/home/almasri3/mewcp-gpu/kclique/nCr.txt", "r"); //change it
			double d = 0;
			if (infile == NULL)
			{
				printf("file could not be opened\n");
				exit(1);
			}
			for (int row = 0; row < 1001; ++row)
			{
				for (int col = 0; col < 401; ++col)
				{
					if (!fscanf(infile, "%lf,", &d))
						fprintf(stderr, "Error\n");
					// fprintf(stderr, "%lf\n", d);
					nCr.cdata()[row * 401 + col] = (unsigned long long)d;
				}
			}
			fclose(infile);
			printf("Test, 5c4 = %llu\n", nCr.cdata()[7 * 401 + 4]);
			nCr.switch_to_gpu();

			CUDA_RUNTIME(cudaSetDevice(dev_));

			const auto block_size = 64;
			CUDAContext context;
			T num_SMs = context.num_SMs;

			T level = 0;
			T span = 1024;
			T bucket_level_end_ = level;
			T todo = g.numEdges;

			GPUArray <uint64> counter("Temp level Counter", unified, 1, dev_);
			GPUArray <T> maxDegree("Temp Degree", unified, 1, dev_);

			T conc_blocks_per_SM = context.GetConCBlocks(block_size);
			GPUArray<T> d_bitmap_states("bmp bitmap stats", AllocationTypeEnum::gpu, num_SMs * conc_blocks_per_SM, dev_);
			T factor = (pe == Block) ? 1 : (block_size / 32);
			counter.setSingle(0, 0, true);
			maxDegree.setSingle(0, 0, true);
			execKernel((get_max_degree<T, 128>), (g.numEdges + 128 - 1) / 128, 128, dev_, false, g, edgePtr.gdata(), maxDegree.gdata());
			GPUArray<char> current_level("Temp level Counter", unified, num_SMs * conc_blocks_per_SM * factor * maxDegree.gdata()[0], dev_);


			printf("Max Dgree = %u vs %u\n", maxDegree.gdata()[0], g.numEdges);
			bucket_edge_scan(edgePtr, g.numEdges, 0, kcount - 2, current_q, identity_arr_asc, bucket_q, bucket_level_end_);
			todo -= current_q.count.gdata()[0];
			current_q.count.gdata()[0] = 0;
			level = kcount - 2;
			bucket_level_end_ = level;

			const T partitionSize = PART_SIZE;
			factor = (block_size / partitionSize);

			const uint dv = 32;
			const uint max_level = maxDegree.gdata()[0];
			uint num_divs = (maxDegree.gdata()[0] + dv - 1) / dv;

			const uint64 encode_size = num_SMs * conc_blocks_per_SM * maxDegree.gdata()[0] * num_divs; //per block
			const uint64 tri_size = num_SMs * conc_blocks_per_SM * maxDegree.gdata()[0]; //per block
			GPUArray<T> node_be("Temp level Counter", unified, encode_size, dev_);
			GPUArray<T> tri_list("Temp level Counter", unified, tri_size, dev_);

			const uint64 level_size = num_SMs * conc_blocks_per_SM * /*factor **/ max_level * num_divs; //per partition
			const uint64 level_item_size = num_SMs * conc_blocks_per_SM * /*factor **/ max_level; //per partition
			const uint64 level_partition_size = num_SMs * conc_blocks_per_SM * /*factor **/ num_divs; //per partition

			GPUArray<T> current_level2("Temp level Counter", unified, level_size, dev_);
			GPUArray<T> possible("Temp level Counter", unified, level_size, dev_);

			GPUArray<T> level_index("Temp level Counter", unified, level_item_size, dev_);
			GPUArray<T> level_count("Temp level Counter", unified, level_item_size, dev_);
			GPUArray<T> level_prev("Temp level Counter", unified, level_item_size, dev_);
			GPUArray<T> level_r("Temp level Counter", unified, level_item_size, dev_);
			GPUArray<T> level_d("Temp level Counter", unified, level_item_size, dev_);
			GPUArray<T> level_temp("Temp level Counter", unified, level_partition_size, dev_);

			printf("Level Size = %llu, Encode Size = %llu, Tri size = %llu\n", 2 * level_size + 5 * level_item_size + 1 * level_partition_size, encode_size, tri_size);

			// current_level2.setAll(0, true);
			// node_be.setAll(0, true);
			// tri_list.setAll(0, true);

			const T numPartitions = block_size / partitionSize;
			cudaMemcpyToSymbol(KCCOUNT, &kcount, sizeof(KCCOUNT));
			cudaMemcpyToSymbol(PARTSIZE, &partitionSize, sizeof(PARTSIZE));
			cudaMemcpyToSymbol(NUMPART, &numPartitions, sizeof(NUMPART));
			cudaMemcpyToSymbol(MAXLEVEL, &max_level, sizeof(MAXLEVEL));
			cudaMemcpyToSymbol(NUMDIVS, &num_divs, sizeof(NUMDIVS));
			cudaMemcpyToSymbol(MAXDEG, &(maxDegree.gdata()[0]), sizeof(MAXDEG));
			cudaMemcpyToSymbol(CBPSM, &(conc_blocks_per_SM), sizeof(CBPSM));

			/*	GPUArray <uint64> cpn("Temp Degree", unified, g.numEdges, dev_);
				cpn.setAll(0, true);*/

			while (todo > 0)
			{
				//1 bucket fill
				bucket_edge_scan(edgePtr, g.numEdges, level, span, current_q, identity_arr_asc, bucket_q, bucket_level_end_);
				todo -= current_q.count.gdata()[0];
				if (current_q.count.gdata()[0] > 0)
				{
					//std::sort(current_q.queue.gdata(), current_q.queue.gdata() + current_q.count.gdata()[0]);
					//current_q.count.gdata()[0] = current_q.count.gdata()[0]< 5000? current_q.count.gdata()[0]: 5000;
					//current_q.count.gdata()[0] = 1;

					auto grid_block_size = current_q.count.gdata()[0];

					execKernel((kckernel_edge_block_warp_binary_pivot_count<T, block_size, partitionSize>), grid_block_size, block_size, dev_, false,
						counter.gdata(),
						g,
						current_q.device_queue->gdata()[0],
						current_level2.gdata(), NULL,
						d_bitmap_states.gdata(), node_be.gdata(), tri_list.gdata(),

						possible.gdata(),
						level_index.gdata(),
						level_count.gdata(),
						level_prev.gdata(),
						level_r.gdata(),
						level_d.gdata(),
						level_temp.gdata(),
						nCr.gdata()
					);


					current_level2.freeGPU();
				}
				level += span;

				std::cout.imbue(std::locale(""));
				std::cout << "Level = " << level << " Nodes = " << current_q.count.gdata()[0] << " Counter = " << counter.gdata()[0] << "\n";
			}

			counter.freeGPU();
			//cpn.freeGPU();
			current_level.freeGPU();

			d_bitmap_states.freeGPU();
			maxDegree.freeGPU();
			k = level;

			printf("Max Edge Min Degree = %d\n", k - 1);

		}



		uint findKtrussIncremental_sync(int kmin, int kmax, TcBase<T>* tcCounter, EidGraph_d<T>& g, int* reverseIndex, EncodeDataType* bitMap, const size_t nodeOffset = 0, const size_t edgeOffset = 0)
		{
			findKtrussIncremental_async(kmin, kmax, tcCounter, g, reverseIndex, bitMap, nodeOffset, edgeOffset);
			sync();
			return count();
		}


		void free()
		{
			current_q.free();
			bucket_q.free();
			identity_arr_asc.freeGPU();

			nodeDegree.freeGPU();
			edgePtr.freeGPU();

		}
		void sync() { CUDA_RUNTIME(cudaStreamSynchronize(stream_)); }

		uint count() const { return k - 1; }
		int device() const { return dev_; }
		cudaStream_t stream() const { return stream_; }
	};

} // namespace pangolin
