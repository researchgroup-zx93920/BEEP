//Defined in KCLIQUE CODE
// __constant__ uint KCCOUNT;
// __constant__ uint MAXDEG;
// __constant__ uint PARTSIZE;
// __constant__ uint NUMPART;
// __constant__ uint MAXLEVEL;
// __constant__ uint NUMDIVS;
// __constant__ uint CBPSM;

#include "../cub/cub.cuh"
#include <../cub/block/block_load.cuh>
#include <../cub/block/block_store.cuh>
#include <../cub/block/block_radix_sort.cuh>


template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
maximal_node_block_warp_binary_pivot_count(
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
	__shared__ T level_pivot[1024];
	__shared__ uint64 clique_count;
	__shared__ bool path_more_explore;
	__shared__ T l;
	__shared__ uint64 maxIntersection;
	__shared__ uint32_t  sm_id, levelPtr;
	__shared__ T src, srcStart, srcLen;
	__shared__ bool  partition_set[numPartitions];

	__shared__ T num_divs_local, encode_offset, *encode;
	__shared__ T *pl, *cl;
	__shared__ T *level_count, *level_index, *level_prev_index, *rsize, *drop;

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

			level_item_offset = sm_id * CBPSM * (/*numPartitions **/ MAXDEG) + levelPtr * (/*numPartitions **/ MAXDEG);
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

			clique_count = 0;
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
		if(lx == 0)
		{
			maxCount[wx] = 0;
			maxIndex[wx] = 0xFFFFFFFF;
			partMask[wx] = CPARTSIZE ==32? 0xFFFFFFFF : (1 << CPARTSIZE) - 1;
			partMask[wx] = partMask[wx] << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
		}
		__syncthreads();
	
		for (T j = wx; j < srcLen; j += numPartitions)
		{
			uint64 warpCount = 0;
			for (T k = lx; k < num_divs_local; k += CPARTSIZE)
			{
				warpCount += __popc(encode[j * num_divs_local + k]);
			}
			reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);

			if(lx == 0 && maxCount[wx] < warpCount)
			{
				maxCount[wx] = warpCount;
				maxIndex[wx] = j;
			}	
		}
		if(lx == 0)
		{
			atomicMax(&(maxIntersection), maxCount[wx]);
		}
		__syncthreads();
		if(lx == 0)
		{
			if(maxIntersection == maxCount[wx]) // unsafe, but okay I need any one with this max count
			{
				atomicMin(&(level_pivot[0]),maxIndex[wx]);
			}
		}
		__syncthreads();

		//Prepare the Possible and Intersection Encode Lists
		uint64 warpCount = 0;
	
		for (T j = threadIdx.x; j < num_divs_local && maxIntersection > 0; j += BLOCK_DIM_X)
		{
			T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
			pl[j] = ~(encode[(level_pivot[0])*num_divs_local + j]) & m;
			cl[j] = 0xFFFFFFFF;
			warpCount += __popc(pl[j]);
		}
		reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
		if(lx == 0 && threadIdx.x < num_divs_local)
		{
			atomicAdd(&(level_count[0]), (T)warpCount);
		}
		__syncthreads();

		//Explore the tree
		while((level_count[l - 2] > level_index[l - 2]))
		{
			T maskBlock = level_prev_index[l- 2] / 32;
			T maskIndex = ~((1 << (level_prev_index[l - 2] & 0x1F)) -1);
			T newIndex = __ffs(pl[num_divs_local*(l-2) + maskBlock] & maskIndex);
			while(newIndex == 0)
			{
				maskIndex = 0xFFFFFFFF;
				maskBlock++;
				newIndex = __ffs(pl[num_divs_local*(l-2) + maskBlock] & maskIndex);
			}
			newIndex =  32*maskBlock + newIndex - 1;
			T sameBlockMask = (~((1 << (newIndex & 0x1F)) - 1))   | ~pl[num_divs_local*(l-2) + maskBlock];
			__syncthreads();
			if (threadIdx.x == 0)
			{
				level_prev_index[l - 2] = newIndex + 1;
				level_index[l - 2]++;
				level_pivot[l - 1] = 0xFFFFFFFF;
				path_more_explore = false;
				maxIntersection = 0;
				rsize[l-1] = rsize[l-2] + 1;
				drop[l-1] = drop[l-2];
				if(newIndex == level_pivot[l-2])
					drop[l-1] = drop[l-2] + 1;
			}
			__syncthreads();
			//assert(level_prev_index[l - 2] == newIndex + 1);

			// if(rsize[l-1] - drop[l-1] > KCCOUNT)
			// {	
			// 	__syncthreads();
			// 	//printf("Stop Here, %u %u\n", rsize[l-1], drop[l-1]);
			// 	if(threadIdx.x == 0)
			// 	{
			// 		T c = rsize[l-1] - KCCOUNT;
			// 		unsigned long long ncr = nCR[ drop[l-1] * 401 + c  ];
			// 		atomicAdd(counter, ncr/*rsize[l-1]*/);
					
			// 		//printf, go back
			// 		while (l > 2 && level_index[l - 2] >= level_count[l - 2])
			// 		{
			// 			(l)--;
			// 		}
			// 	}
			// 	__syncthreads();
			// }
			// else
			{
				// Now prepare intersection list
				T* from = &(cl[num_divs_local * (l - 2)]);
				T* to =  &(cl[num_divs_local * (l - 1)]);
				for (T k = threadIdx.x; k < num_divs_local; k += BLOCK_DIM_X)
				{
					to[k] = from[k] & encode[newIndex* num_divs_local + k];
					//remove previous pivots from here
					to[k] = to[k] & ( (maskBlock < k) ? ~pl[num_divs_local*(l-2) + k] : ( (maskBlock > k) ? 0xFFFFFFFF:  sameBlockMask) );
				}
				if(lx == 0)
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
					if( (to[bi] & (1<<ii)) != 0)
					{
						for (T k = lx; k < num_divs_local; k += CPARTSIZE)
						{
							warpCount += __popc(to[k] & encode[j * num_divs_local + k]);
						}
						reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
						if(lx == 0 && maxCount[wx] == srcLen + 1)
						{
							partition_set[wx] = true;
							path_more_explore = true; //shared, unsafe, but okay
							maxCount[wx] = warpCount;
							maxIndex[wx] = j;
						}
						else if(lx == 0 && maxCount[wx] < warpCount)
						{
							maxCount[wx] = warpCount;
							maxIndex[wx] = j;
						}	
					}
				}
		
				__syncthreads();
				if(!path_more_explore)
				{
					__syncthreads();
					if(threadIdx.x == 0)
					{	
						if(rsize[l-1] >= KCCOUNT)
						{
							T c = rsize[l-1] - KCCOUNT;
							unsigned long long ncr = nCR[ drop[l-1] * 401 + c  ];
							atomicAdd(&clique_count, 1/*rsize[l-1]*/);
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
					if(lx == 0 && partition_set[wx])
					{
						atomicMax(&(maxIntersection), maxCount[wx]);
					}
					__syncthreads();

					if(lx == 0 && maxIntersection == maxCount[wx])
					{	
							atomicMin(&(level_pivot[l-1]), maxIndex[wx]);
					}
					__syncthreads();

					uint64 warpCount = 0;
					for (T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
					{
						T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
						pl[(l-1)*num_divs_local + j] = ~(encode[level_pivot[l - 1] * num_divs_local + j]) & to[j] & m;
						warpCount += __popc(pl[(l-1)*num_divs_local + j]);
					}
					reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);

					if(threadIdx.x == 0)
					{
						l++;
						level_count[l-2] = 0;
						level_prev_index[l-2] = 0;
						level_index[l-2] = 0;
					}

					__syncthreads();
					if(lx == 0 && threadIdx.x < num_divs_local)
					{
						atomicAdd(&(level_count[l-2]), warpCount);
					}
				}
				
			}
			__syncthreads();
			/////////////////////////////////////////////////////////////////////////
		}

		__syncthreads();
		if(threadIdx.x == 0)
		{
			atomicAdd(counter, clique_count);
		}
	}

	__syncthreads();
	if (threadIdx.x == 0)
	{
		atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
	}
}


template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
maximal_edge_block_warp_binary_pivot_count(
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

	__shared__ T level_pivot[1024];
	__shared__ uint64 clique_count;
	__shared__ bool path_more_explore;
	__shared__ T l;
	__shared__ T maxIntersection;
	__shared__ uint32_t  sm_id, levelPtr;
	__shared__ T src, srcStart, srcLen, src2, src2Start, src2Len, scounter;
	__shared__ bool  partition_set[numPartitions];
	__shared__ T num_divs_local, encode_offset, *encode, tri_offset, *tri;
	__shared__ T *pl, *cl;
	__shared__ T *level_count, *level_index, *level_prev_index, *rsize, *drop;
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
			src = g.rowInd[current.queue[i]];
			srcStart = g.rowPtr[src];
			srcLen = g.rowPtr[src + 1] - srcStart;
			src2 = g.colInd[current.queue[i]];
			src2Start = g.rowPtr[src2];
			src2Len = g.rowPtr[src2 + 1] - src2Start;
			tri_offset = sm_id * CBPSM * (MAXDEG) + levelPtr * (MAXDEG);
			tri = &adj_tri[tri_offset  /*srcStart[wx]*/];
			scounter = 0;

			encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
			encode = &adj_enc[encode_offset  /*srcStart[wx]*/];

			lo = sm_id * CBPSM * (/*numPartitions **/ NUMDIVS * MAXDEG) + levelPtr * (/*numPartitions **/ NUMDIVS * MAXDEG);
			cl = &current_level[lo/*level_offset[wx]/* + wx * (NUMDIVS * MAXDEG)*/];
			pl = &possible[lo/*level_offset[wx] /*+ wx * (NUMDIVS * MAXDEG)*/];

			level_item_offset = sm_id * CBPSM * (/*numPartitions **/ MAXDEG) + levelPtr * (/*numPartitions **/ MAXDEG);
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
			clique_count = 0;

		
		}

		// //get tri list: by block :!!
		__syncthreads();

		//if(src == 7053 && src2 == 1301)
		{

			graph::block_count_and_set_tri<BLOCK_DIM_X, T>(&g.colInd[srcStart], srcLen, &g.colInd[src2Start], src2Len,
				tri, &scounter);
			
			__syncthreads();

			if (threadIdx.x == 0)
			{
				num_divs_local = (scounter + 32 - 1) / 32;
				lastMask_i = scounter / 32;
				lastMask_ii = (1 << (scounter & 0x1F)) - 1;
			}
			
			// if(KCCOUNT == 3 && threadIdx.x == 0)
			// 	atomicAdd(counter, scounter);

				
		//Just for testing
		T vals[2];
		if(threadIdx.x < scounter)
			vals[0] = tri[threadIdx.x];
		else
			vals[0] = 0xFFFFFFFF;
		
		if(BLOCK_DIM_X + threadIdx.x < scounter)
			vals[1] = tri[BLOCK_DIM_X + threadIdx.x];
		else
			vals[1] = 0xFFFFFFFF;
		
		__syncthreads();

		typedef cub::BlockRadixSort<T, BLOCK_DIM_X, 2> BlockRadixSort;
		__shared__ typename BlockRadixSort::TempStorage temp_storage;

		BlockRadixSort(temp_storage).Sort(vals);

		__syncthreads();

		if(threadIdx.x * 2 < scounter)
			tri[2*threadIdx.x] = vals[0];
		if(threadIdx.x * 2 + 1 < scounter)
			tri[2*threadIdx.x + 1] = vals[1];

		/////////////////

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
					j, num_divs_local,  encode);
			}
			__syncthreads(); //Done encoding



			

			if(lx == 0)
			{
				maxCount[wx] = 0;
				maxIndex[wx] = 0xFFFFFFFF;
				partMask[wx] = CPARTSIZE ==32? 0xFFFFFFFF : (1 << CPARTSIZE) - 1;
				partMask[wx] = partMask[wx] << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
			}
			__syncthreads();

			//Find the first pivot
			for (T j = wx; j < scounter; j += numPartitions)
			{
				T warpCount = 0;
				for (T k = lx; k < num_divs_local; k += CPARTSIZE)
				{
					warpCount += __popc(encode[j * num_divs_local + k]);
				}
				reduce_partT<T, CPARTSIZE>(partMask[wx], warpCount);

				if(lx == 0 && maxCount[wx] < warpCount)
				{
					maxCount[wx] = warpCount;
					maxIndex[wx] = j;
				}	
			}
			__syncthreads();
			if(lx == 0)
			{
				atomicMax(&(maxIntersection), maxCount[wx]);
			}
			__syncthreads();
			if(lx == 0)
			{
				if(maxIntersection == maxCount[wx]) // unsafe, but okay I need any one with this max count
				{
					atomicMin(&(level_pivot[0]),maxIndex[wx]);
				}
			}
			__syncthreads();


			// if(threadIdx.x == 0)
			// {
			// 	printf("\n FIRST PIVOT L = %u, P = %u, Max Intersect = %u \n", l, level_pivot[0], maxIntersection);
			// }

			__syncthreads();

			//Prepare the Possible and Intersection Encode Lists
			uint64 warpCount = 0;
			for (T j = threadIdx.x; j < num_divs_local && maxIntersection > 0; j += BLOCK_DIM_X)
			{
				T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
				pl[j] = ~(encode[(level_pivot[0])*num_divs_local + j]) & m;
				cl[j] = 0xFFFFFFFF;
				warpCount += __popc(pl[j]);
			}
			reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
			if(lx == 0 && threadIdx.x < num_divs_local)
			{
				atomicAdd(&(level_count[0]), (T)warpCount);
			}
			__syncthreads();
			while((level_count[l - 3] > level_index[l - 3]))
			{
				T maskBlock = level_prev_index[l- 3] / 32;
				T maskIndex = ~((1 << (level_prev_index[l - 3] & 0x1F)) -1);
				T newIndex = __ffs(pl[num_divs_local*(l-3) + maskBlock] & maskIndex);
				while(newIndex == 0)
				{
					maskIndex = 0xFFFFFFFF;
					maskBlock++;
					newIndex = __ffs(pl[num_divs_local*(l-3) + maskBlock] & maskIndex);
				}
				newIndex =  32*maskBlock + newIndex - 1;
				T sameBlockMask = (~((1 << (newIndex & 0x1F)) - 1))   | ~pl[num_divs_local*(l-3) + maskBlock];
				__syncthreads();
				if (threadIdx.x == 0)
				{
					level_prev_index[l - 3] = newIndex + 1;
					level_index[l - 3]++;
					level_pivot[l - 2] = 0xFFFFFFFF;
					path_more_explore = false;
					maxIntersection = 0;
					rsize[l-2] = rsize[l-3] + 1;
					drop[l-2] = drop[l-3];
					if(newIndex == level_pivot[l-3])
						drop[l-2] = drop[l-3] + 1;
				}
				__syncthreads();
				//assert(level_prev_index[l - 2] == newIndex + 1);

				// if(rsize[l-2] - drop[l-2] > KCCOUNT)
				// {	
				// 	__syncthreads();
				// 	//printf("Stop Here, %u %u\n", rsize[l-1], drop[l-1]);
				// 	if(threadIdx.x == 0)
				// 	{
				// 		T c = rsize[l-2] - KCCOUNT;
				// 		unsigned long long ncr = nCR[ drop[l-2] * 401 + c  ];
				// 		atomicAdd(counter, ncr/*rsize[l-1]*/);
						
				// 		//printf, go back
				// 		while (l > 3 && level_index[l - 3] >= level_count[l - 3])
				// 		{
				// 			(l)--;
				// 		}
				// 	}
				// 	__syncthreads();
				// }
				// else
				{
					// Now prepare intersection list
					T* from = &(cl[num_divs_local * (l - 3)]);
					T* to =  &(cl[num_divs_local * (l - 2)]);
					for (T k = threadIdx.x; k < num_divs_local; k += BLOCK_DIM_X)
					{
						to[k] = from[k] & encode[newIndex* num_divs_local + k];
						//remove previous pivots from here
						to[k] = to[k] & ( (maskBlock < k) ? ~pl[num_divs_local*(l-3) + k] : ( (maskBlock > k) ? 0xFFFFFFFF:  sameBlockMask) );
					}
					if(lx == 0)
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
						T warpCount = 0;
						T bi = j / 32;
						T ii = j & 0x1F;
						if( (to[bi] & (1<<ii)) != 0)
						{
							for (T k = lx; k < num_divs_local; k += CPARTSIZE)
							{
								warpCount += __popc(to[k] & encode[j * num_divs_local + k]);
							}
							reduce_partT<T, CPARTSIZE>(partMask[wx], warpCount);

							if(lx == 0 && maxCount[wx] == scounter + 1)
							{
								partition_set[wx] = true;
								path_more_explore = true; //shared, unsafe, but okay
								maxCount[wx] = warpCount;
								maxIndex[wx] = j;
							}
							else if(lx == 0 && maxCount[wx] < warpCount && partition_set[wx])
							{
								maxCount[wx] = warpCount;
								maxIndex[wx] = j;
							}	
						}
						__syncwarp(partMask[wx]);
					}
			
					__syncthreads();
					if(!path_more_explore)
					{
						__syncthreads();
						if(threadIdx.x == 0)
						{	
							if(rsize[l-2] >= KCCOUNT)
							{
								T c = rsize[l-2] - KCCOUNT;
								unsigned long long ncr = nCR[ drop[l-2] * 401 + c  ];
								atomicAdd(&clique_count, 1/*rsize[l-1]*/);
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
						if(lx == 0 && partition_set[wx])
						{
							atomicMax(&(maxIntersection), maxCount[wx]);
						}
						__syncthreads();

						if(lx == 0 && maxIntersection == maxCount[wx])
						{	
							atomicMin(&(level_pivot[l-2]), maxIndex[wx]);
						}
						__syncthreads();

						// if(threadIdx.x == 0)
						// {
						// 	printf("L = %u, P = %u, Inter = %u\n", l, level_pivot[l-2], maxIntersection);
						// 	// for (int i = 0; i < BLOCK_DIM_X/CPARTSIZE; i++)
						// 	// 	printf("%d, Max Count = %u, Index = %u\n", i, maxCount[i], maxIndex[i]);
							
						// }	


						__syncthreads();
						uint64 warpCount = 0;
						for (T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
						{
							T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
							pl[(l-2)*num_divs_local + j] = ~(encode[level_pivot[l - 2] * num_divs_local + j]) & to[j] & m;
							warpCount += __popc(pl[(l-2)*num_divs_local + j]);
						}
						reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);

						if(threadIdx.x == 0)
						{
							l++;
							level_count[l-3] = 0;
							level_prev_index[l-3] = 0;
							level_index[l-3] = 0;
						}

						__syncthreads();
						if(lx == 0 && threadIdx.x < num_divs_local)
						{
							atomicAdd(&(level_count[l-3]), warpCount);
						}
					}
					
				}
				__syncthreads();
			}
		}
		__syncthreads();
		if(threadIdx.x == 0)
		{
			// if(src == 7053 && src2 == 1301)
			// 	printf("%u, %u, %llu\n", src, src2, clique_count);
			atomicAdd(counter, clique_count);
		}

	}

	

	__syncthreads();
	if (threadIdx.x == 0)
	{
		atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
	}
}



