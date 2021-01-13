
__constant__ uint KCCOUNT;
__constant__ uint MAXDEG;
__constant__ uint PARTSIZE;
__constant__ uint NUMPART;
__constant__ uint MAXLEVEL;
__constant__ uint NUMDIVS;
__constant__ uint CBPSM;

#define PART_SIZE 16
template<typename T>
__inline__ __device__ void reduce_part(T mask, uint64 &count)
{
	// count += __shfl_down_sync(mask, count, 16);
	count += __shfl_down_sync(mask, count, 8);
	count += __shfl_down_sync(mask, count, 4);
	count += __shfl_down_sync(mask, count, 2);
	count += __shfl_down_sync(mask, count, 1);
}



template <typename T, int BLOCK_DIM_X>
__global__ void try_block_scan(T count, T* output)
{
	typedef cub::BlockScan<T, BLOCK_DIM_X> BlockScan;
	__shared__ typename BlockScan::TempStorage temp_storage;
	auto tid = threadIdx.x;
	T srcLenBlocks = (count + BLOCK_DIM_X - 1) / BLOCK_DIM_X;
	T threadData = 0;
	T aggreagtedData = 0;
	T total = 0;

	for (int k = 0; k < srcLenBlocks; k++)
	{
		T index = k * BLOCK_DIM_X + tid;
		threadData = 0;
		aggreagtedData = 0;
		__syncthreads();

		if (index < count)
		{
			threadData = 1;
		}

		__syncthreads();
		BlockScan(temp_storage).ExclusiveSum(threadData, threadData, aggreagtedData);

		__syncthreads();



		if (index < count)
			output[threadData + total] = threadData;


		total += aggreagtedData;
	}
}


__device__ __inline__ uint32_t __mysmid() {
	unsigned int r;
	asm("mov.u32 %0, %%smid;" : "=r"(r));
	return r;
}


template <typename T, int BLOCK_DIM_X>
__global__ void get_max_degree(graph::COOCSRGraph_d<T> g, T* edgePtr, T* maxDegree)
{
	const T gtid = (BLOCK_DIM_X * blockIdx.x + threadIdx.x);
	typedef cub::BlockReduce<T, BLOCK_DIM_X> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	T degree = 0;

	if (gtid < g.numEdges)
	{
		T src = g.rowInd[gtid];
		T dst = g.colInd[gtid];

		T srcDeg = g.rowPtr[src + 1] - g.rowPtr[src];
		T dstDeg = g.rowPtr[dst + 1] - g.rowPtr[dst];

		degree = srcDeg < dstDeg ? srcDeg : dstDeg;
		edgePtr[gtid] = degree;
	}

	__syncthreads();
	T aggregate = BlockReduce(temp_storage).Reduce(degree, cub::Max());


	if (threadIdx.x == 0)
		atomicMax(maxDegree, aggregate);


}


template<typename T, int BLOCK_DIM_X>
__global__ void getNodeDegree_kernel(T* nodeDegree, graph::COOCSRGraph_d<T> g, T* maxDegree)
{
	T gtid = threadIdx.x + blockIdx.x * blockDim.x;
	typedef cub::BlockReduce<T, BLOCK_DIM_X> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	T degree = 0;
	if (gtid < g.numNodes)
	{
		degree = g.rowPtr[gtid + 1] - g.rowPtr[gtid];
		nodeDegree[gtid] = degree;
	}

	T aggregate = BlockReduce(temp_storage).Reduce(degree, cub::Max());
	if (threadIdx.x == 0)
		atomicMax(maxDegree, aggregate);
}


/////////////////////////////////////////// Latest Kernels ///////////////////////////////////////////////////
template <typename T, uint BLOCK_DIM_X>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_warp_count(
	uint64* counter,
	graph::COOCSRGraph_d<T> g,
	T kclique,
	T maxDeg,
	graph::GraphQueue_d<T, bool> current,
	char* current_level,
	uint64* cpn,
	T conc_blocks_per_SM,
	T* levelStats
)
{
	constexpr T warpsPerBlock = BLOCK_DIM_X / 32;
	const int wx = threadIdx.x / 32; // which warp in thread block
	const size_t lx = threadIdx.x % 32;
	const T gwx = (BLOCK_DIM_X * blockIdx.x + threadIdx.x) / 32;

	__shared__ T level_index[warpsPerBlock][5];
	__shared__ T level_count[warpsPerBlock][5];
	__shared__ T level_prev_index[warpsPerBlock][5];
	__shared__ T current_node_index[warpsPerBlock];
	__shared__ uint64 clique_count[warpsPerBlock];
	__shared__ char l[warpsPerBlock];
	__shared__ char new_level[warpsPerBlock];
	__shared__ uint32_t nodeId[warpsPerBlock], sm_id, levelPtr;
	__shared__ T srcStart[warpsPerBlock], srcLen[warpsPerBlock], srcLenBlocks[warpsPerBlock];
	__syncthreads();

	if (threadIdx.x == 0)
	{
		sm_id = __mysmid();
		T temp = 0;
		while (atomicCAS(&(levelStats[(sm_id * conc_blocks_per_SM) + temp]), 0, 1) != 0)
		{
			temp++;
		}
		levelPtr = temp;
	}
	__syncthreads();

	for (unsigned long long i = gwx; i < (unsigned long long)current.count[0]; i += warpsPerBlock * gridDim.x) //warp per node
	{
		if (lx < 5)
		{
			level_index[wx][lx] = 0;
			level_prev_index[wx][lx] = 0;
		}
		else if (lx == 6)
		{
			l[wx] = 2;
			nodeId[wx] = current.queue[i];
			srcStart[wx] = g.rowPtr[nodeId[wx]];
			srcLen[wx] = g.rowPtr[nodeId[wx] + 1] - srcStart[wx];
			srcLenBlocks[wx] = (srcLen[wx] + 32 - 1) / 32;
			level_count[wx][l[wx] - 2] = srcLen[wx];
		}
		else if (lx == 7)
		{

			current_node_index[wx] = UINT_MAX;
			clique_count[wx] = 0;
		}

		__syncwarp();
		T blockOffset = sm_id * conc_blocks_per_SM * (warpsPerBlock * maxDeg)
			+ levelPtr * (warpsPerBlock * maxDeg);
		char* cl = &current_level[blockOffset + wx * maxDeg /*srcStart[wx]*/];
		while (level_count[wx][l[wx] - 2] > level_index[wx][l[wx] - 2])
		{
			for (T k = 0; k < srcLenBlocks[wx]; k++)
			{
				T index = level_prev_index[wx][l[wx] - 2] + k * 32 + lx;
				if (index < srcLen[wx] && cl[index] == l[wx])
				{
					atomicMin(&(current_node_index[wx]), index);
				}
				__syncwarp();
				if (current_node_index[wx] != UINT_MAX)
					break;

				__syncwarp();
			}


			//Warp shuffle
			// for (T k = 0; k < srcLenBlocks[wx]; k++)
			// {
			// 	T index = level_prev_index[wx][l[wx] - 2] + k * 32 + lx;
			// 	int condition = index < srcLen[wx] && cl[index] == l[wx];
			// 	unsigned int newmask = __ballot_sync(0xFFFFFFFF, condition);
			// 	if(newmask > 0)
			// 	{
			// 		int elected_lane_deq = __ffs(newmask) - 1;
			// 		if(lx == elected_lane_deq)
			// 		{
			// 			current_node_index[wx] = index;
			// 		}
			// 		break;
			// 	}
			// }


			if (lx == 0)
			{
				//current_node_index[0] = finalIndex;
				level_prev_index[wx][l[wx] - 2] = current_node_index[wx] + 1;
				level_index[wx][l[wx] - 2]++;
				new_level[wx] = l[wx];
			}

			__syncwarp();

			uint64 warpCount = 0;
			const T dst = g.colInd[current_node_index[wx] + srcStart[wx]];
			const T dstStart = g.rowPtr[dst];
			const T dstStop = g.rowPtr[dst + 1];
			const T dstLen = dstStop - dstStart;

			if (dstLen >= kclique - l[wx])
			{
				if (dstLen > srcLen[wx])
				{
					warpCount = graph::warp_sorted_count_and_set_binary<WARPS_PER_BLOCK, T, true>(0, &g.colInd[srcStart[wx]], srcLen[wx],
						&g.colInd[dstStart], dstLen, true, srcStart[wx], cl, l[wx] + 1, kclique);
				}
				else {
					warpCount = graph::warp_sorted_count_and_set_binary<WARPS_PER_BLOCK, T, true>(0, &g.colInd[dstStart], dstLen,
						&g.colInd[srcStart[wx]], srcLen[wx], false, srcStart[wx], cl, l[wx] + 1, kclique);
				}

				__syncwarp();
				if (lx == 0 && warpCount > 0)
				{
					if (l[wx] + 1 == kclique)
						clique_count[wx] += warpCount;
					else if (l[wx] + 1 < kclique)
					{
						(l[wx])++;
						(new_level[wx])++;
						level_count[wx][l[wx] - 2] = warpCount;
						level_index[wx][l[wx] - 2] = 0;
						level_prev_index[wx][l[wx] - 2] = 0;
					}

				}
			}

			__syncwarp();
			if (lx == 0)
			{
				while (new_level[wx] > 2 && level_index[wx][new_level[wx] - 2] >= level_count[wx][new_level[wx] - 2])
				{
					(new_level[wx])--;
				}
			}

			__syncwarp();
			if (new_level[wx] < l[wx])
			{
				for (auto k = 0; k < srcLenBlocks[wx]; k++)
				{
					T index = k * 32 + lx;
					if (index < srcLen[wx] && cl[index] > new_level[wx])
						cl[index] = new_level[wx];
				}
			}

			__syncwarp();

			if (lx == 0)
			{
				l[wx] = new_level[wx];
				current_node_index[wx] = UINT_MAX;
			}
			__syncwarp();



		}

		if (lx == 0)
		{
			atomicAdd(counter, clique_count[wx]);
			//cpn[nodeId] = clique_count;
		}
	}

	__syncthreads();
	if (threadIdx.x == 0)
	{
		atomicCAS(&levelStats[sm_id * conc_blocks_per_SM + levelPtr], 1, 0);
	}
}


template <typename T, uint BLOCK_DIM_X>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_warp_sync_count(
	uint64* counter,
	graph::COOCSRGraph_d<T> g,
	T kclique,
	T maxDeg,
	graph::GraphQueue_d<T, bool> current,
	char* current_level,
	uint64* cpn,
	T conc_blocks_per_SM,
	T* levelStats
)
{
	constexpr T warpsPerBlock = BLOCK_DIM_X / 32;
	const int wx = threadIdx.x / 32; // which warp in thread block
	const size_t lx = threadIdx.x % 32;
	const T gwx = (BLOCK_DIM_X * blockIdx.x + threadIdx.x) / 32;

	T level_index[5];
	T level_count[5];
	T level_prev_index[5];
	//char* current_level_s = &current_level_s_all[wx][0];
	T current_node_index;
	uint64 clique_count;
	char l;
	char new_level;

	__shared__ uint32_t nodeId[warpsPerBlock], sm_id, levelPtr;
	__shared__ T srcStart[warpsPerBlock], srcLen[warpsPerBlock], srcLenBlocks[warpsPerBlock];

	for (unsigned long long i = gwx; i < (unsigned long long)current.count[0]; i += warpsPerBlock * gridDim.x) //warp per node
	{

		if (threadIdx.x == BLOCK_DIM_X - 1)
		{
			sm_id = __mysmid();
			T temp = 0;
			while (atomicCAS(&(levelStats[(sm_id * conc_blocks_per_SM) + temp]), 0, 1) != 0)
			{
				temp++;
			}
			levelPtr = temp;
		}
		__syncthreads();
		T blockOffset = sm_id * conc_blocks_per_SM * (warpsPerBlock * maxDeg)
			+ levelPtr * (warpsPerBlock * maxDeg);

		if (lx == 0)
		{
			nodeId[wx] = current.queue[i];
			srcStart[wx] = g.rowPtr[nodeId[wx]];
			srcLen[wx] = g.rowPtr[nodeId[wx] + 1] - srcStart[wx];
			srcLenBlocks[wx] = (srcLen[wx] + 32 - 1) / 32;
		}
		__syncwarp();
		for (T kk = 0; kk < 5; kk++)
		{
			level_index[kk] = 0;
			level_prev_index[kk] = 0;
		}

		char* cl = &current_level[blockOffset + wx * maxDeg];
		l = 2;
		clique_count = 0;
		level_count[l - 2] = srcLen[wx];

		while (level_count[l - 2] > level_index[l - 2])
		{
			//Warp shuffle
			//current_node_index = srcLen;
			for (T k = 0; k < srcLenBlocks[wx]; k++)
			{
				T index = level_prev_index[l - 2] + k * 32 + lx;
				int condition = index < srcLen[wx] && cl[index] == l;
				unsigned int newmask = __ballot_sync(0xFFFFFFFF, condition);
				if (newmask != 0)
				{
					int elected_lane_deq = __ffs(newmask) - 1;
					current_node_index = __shfl_sync(0xFFFFFFFF, index, elected_lane_deq, 32);
					break;
				}
			}

			__syncwarp();

			level_prev_index[l - 2] = current_node_index + 1;
			level_index[l - 2]++;
			new_level = l;

			uint64 warpCount = 0;
			const T dst = g.colInd[current_node_index + srcStart[wx]];
			const T dstStart = g.rowPtr[dst];
			const T dstLen = g.rowPtr[dst + 1] - dstStart;
			if (dstLen >= kclique - l)
			{
				if (dstLen > srcLen[wx])
				{
					warpCount = graph::warp_sorted_count_and_set_binary<WARPS_PER_BLOCK, T, true>(0, &g.colInd[srcStart[wx]], srcLen[wx],
						&g.colInd[dstStart], dstLen, true, srcStart[wx], cl, l + 1, kclique);
				}
				else {
					warpCount = graph::warp_sorted_count_and_set_binary<WARPS_PER_BLOCK, T, true>(0, &g.colInd[dstStart], dstLen,
						&g.colInd[srcStart[wx]], srcLen[wx], false, srcStart[wx], cl, l + 1, kclique);
				}

				warpCount = __shfl_sync(0xFFFFFFFF, warpCount, 0, 32);

				if (warpCount > 0)
				{
					if (l + 1 == kclique)
						clique_count += warpCount;
					else if (l + 1 < kclique)
					{
						l++;
						new_level++;
						level_count[l - 2] = warpCount;
						level_index[l - 2] = 0;
						level_prev_index[l - 2] = 0;
					}
				}
			}

			while (new_level > 2 && level_index[new_level - 2] >= level_count[new_level - 2])
			{
				new_level--;
			}

			if (new_level < l)
			{
				for (auto k = 0; k < srcLenBlocks[wx]; k++)
				{
					T index = k * 32 + lx;
					if (index < srcLen[wx] && cl[index] > new_level)
						cl[index] = new_level;
				}
			}

			l = new_level;
		}

		if (lx == 0)
		{
			atomicAdd(counter, clique_count);
			//cpn[nodeId] = clique_count;
		}

		__syncthreads();
		if (threadIdx.x == 0)
		{
			atomicCAS(&levelStats[sm_id * conc_blocks_per_SM + levelPtr], 1, 0);
		}
	}
}

template <typename T, uint BLOCK_DIM_X>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_warp_sync_s_count(
	uint64* counter,
	graph::COOCSRGraph_d<T> g,
	T kclique,
	T maxDeg,
	graph::GraphQueue_d<T, bool> current,
	char* current_level,
	uint64* cpn,
	T conc_blocks_per_SM,
	T* levelStats
)
{
	constexpr T warpsPerBlock = BLOCK_DIM_X / 32;
	const int wx = threadIdx.x / 32; // which warp in thread block
	const size_t lx = threadIdx.x % 32;
	const T gwx = (BLOCK_DIM_X * blockIdx.x + threadIdx.x) / 32;

	T level_index[5];
	T level_count[5];
	T level_prev_index[5];
	//char* current_level_s = &current_level_s_all[wx][0];
	T current_node_index;
	uint64 clique_count;
	char l;
	char new_level;

	__shared__ uint32_t nodeId[warpsPerBlock], sm_id, levelPtr;
	__shared__ T srcStart[warpsPerBlock], srcLen[warpsPerBlock], srcLenBlocks[warpsPerBlock];

	//////////////////////////////////
	const int num32perWarp = 1;
	const T pwMaxSize = num32perWarp * 32;
	const T startIndex = wx * pwMaxSize;
	const T s = (pwMaxSize * warpsPerBlock);
	//__shared__ T first[warpsPerBlock * pwMaxSize];
	__shared__ T srcShared[s];
	//////////////////////////////////

	for (unsigned long long i = gwx; i < (unsigned long long)current.count[0]; i += warpsPerBlock * gridDim.x) //warp per node
	{

		if (threadIdx.x == BLOCK_DIM_X - 1)
		{
			sm_id = __mysmid();
			T temp = 0;
			while (atomicCAS(&(levelStats[(sm_id * conc_blocks_per_SM) + temp]), 0, 1) != 0)
			{
				temp++;
			}
			levelPtr = temp;
		}
		__syncthreads();
		T blockOffset = sm_id * conc_blocks_per_SM * (warpsPerBlock * maxDeg)
			+ levelPtr * (warpsPerBlock * maxDeg);

		if (lx == 0)
		{
			nodeId[wx] = current.queue[i];
			srcStart[wx] = g.rowPtr[nodeId[wx]];
			srcLen[wx] = g.rowPtr[nodeId[wx] + 1] - srcStart[wx];
			srcLenBlocks[wx] = (srcLen[wx] + 32 - 1) / 32;
		}
		__syncwarp();


		T par = (srcLen[wx] + pwMaxSize - 1) / (pwMaxSize);
		T numElements = srcLen[wx] < pwMaxSize ? srcLen[wx] : (srcLen[wx] + par - 1) / par;

		for (T f = 0; f < num32perWarp; f++)
		{
			T sharedIndex = startIndex + 32 * f + lx;
			T realIndex = srcStart[wx] + (lx + f * 32) * par;
			srcShared[sharedIndex] = (lx + 32 * f) <= numElements ? g.colInd[realIndex] : 0;
		}

		for (T kk = 0; kk < 5; kk++)
		{
			level_index[kk] = 0;
			level_prev_index[kk] = 0;
		}

		char* cl = &current_level[blockOffset + wx * maxDeg];
		l = 2;
		clique_count = 0;
		level_count[l - 2] = srcLen[wx];

		while (level_count[l - 2] > level_index[l - 2])
		{
			//Warp shuffle
			//current_node_index = srcLen;
			for (T k = 0; k < srcLenBlocks[wx]; k++)
			{
				T index = level_prev_index[l - 2] + k * 32 + lx;
				int condition = index < srcLen[wx] && cl[index] == l;
				unsigned int newmask = __ballot_sync(0xFFFFFFFF, condition);
				if (newmask != 0)
				{
					int elected_lane_deq = __ffs(newmask) - 1;
					current_node_index = __shfl_sync(0xFFFFFFFF, index, elected_lane_deq, 32);
					break;
				}
			}

			__syncwarp();

			level_prev_index[l - 2] = current_node_index + 1;
			level_index[l - 2]++;
			new_level = l;

			uint64 warpCount = 0;
			const T dst = g.colInd[current_node_index + srcStart[wx]];
			const T dstStart = g.rowPtr[dst];
			const T dstLen = g.rowPtr[dst + 1] - dstStart;
			if (dstLen >= kclique - l)
			{

				if (dstLen >= srcLen[wx])
				{

					//T par = (dstLen + pwMaxSize - 1) / (pwMaxSize);
					//T numElements = dstLen < pwMaxSize ? dstLen : (dstLen + par - 1) / par;

					//for (T f = 0; f < num32perWarp; f++)
					//{
					//	T sharedIndex = startIndex + 32 * f + lx;
					//	T realIndex = dstStart + (lx + f * 32) * par;
					//	first[sharedIndex] = (lx + 32 * f) <= numElements ? g.colInd[realIndex] : 0;
					//}


					//warpCount = graph::warp_sorted_count_set_binary_s<WARPS_PER_BLOCK, T, true>(&g.colInd[srcStart[wx]], srcLen[wx],
					//	&g.colInd[dstStart], dstLen, 
					//	&(first[startIndex]), par, numElements, pwMaxSize,
					//	true, cl, l + 1, kclique);


					warpCount = graph::warp_sorted_count_and_set_binary<WARPS_PER_BLOCK, T, true>(0, &g.colInd[srcStart[wx]], srcLen[wx],
						&g.colInd[dstStart], dstLen, true, srcStart[wx], cl, l + 1, kclique);
				}
				else {

					T par = (srcLen[wx] + pwMaxSize - 1) / (pwMaxSize);
					T numElements = srcLen[wx] < pwMaxSize ? srcLen[wx] : (srcLen[wx] + par - 1) / par;
					warpCount = graph::warp_sorted_count_set_binary_s<WARPS_PER_BLOCK, T, true>(&g.colInd[dstStart], dstLen,
						&g.colInd[srcStart[wx]], srcLen[wx],
						&(srcShared[startIndex]), par, numElements, pwMaxSize,
						false, cl, l + 1, kclique);
				}

				warpCount = __shfl_sync(0xFFFFFFFF, warpCount, 0, 32);

				if (warpCount > 0)
				{
					if (l + 1 == kclique)
						clique_count += warpCount;
					else if (l + 1 < kclique)
					{
						l++;
						new_level++;
						level_count[l - 2] = warpCount;
						level_index[l - 2] = 0;
						level_prev_index[l - 2] = 0;
					}
				}
			}

			while (new_level > 2 && level_index[new_level - 2] >= level_count[new_level - 2])
			{
				new_level--;
			}

			if (new_level < l)
			{
				for (auto k = 0; k < srcLenBlocks[wx]; k++)
				{
					T index = k * 32 + lx;
					if (index < srcLen[wx] && cl[index] > new_level)
						cl[index] = new_level;
				}
			}

			l = new_level;
		}

		if (lx == 0)
		{
			atomicAdd(counter, clique_count);
			//cpn[nodeId] = clique_count;
		}

		__syncthreads();
		if (threadIdx.x == 0)
		{
			atomicCAS(&levelStats[sm_id * conc_blocks_per_SM + levelPtr], 1, 0);
		}
	}
}



template <typename T, uint BLOCK_DIM_X>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_block_count(
	uint64* counter,
	graph::COOCSRGraph_d<T> g,
	T kclique,
	T maxDeg,
	const  graph::GraphQueue_d<T, bool>  current,
	char* current_level,
	uint64* cpn,
	T conc_blocks_per_SM,
	T* levelStats
)
{
	//will be removed later
	__shared__ T level_index[5];
	__shared__ T level_count[5];
	__shared__ T level_prev_index[5];


	__shared__ T current_node_index;
	__shared__ uint64 clique_count;

	__shared__ char l;
	__shared__ char new_level;
	__shared__ uint32_t nodeId, sm_id, levelPtr;
	__shared__ T srcStart, srcLen, srcLenBlocks;

	__syncthreads();

	for (unsigned long long i = blockIdx.x; i < (unsigned long long)current.count[0]; i += gridDim.x)
	{
		if (threadIdx.x < 5)
		{
			level_index[threadIdx.x] = 0;
			level_prev_index[threadIdx.x] = 0;
		}
		else if (threadIdx.x == 6)
		{
			l = 2;
			current_node_index = UINT_MAX;
			clique_count = 0;
			nodeId = current.queue[i];
			srcStart = g.rowPtr[nodeId];
			srcLen = g.rowPtr[nodeId + 1] - srcStart;
			srcLenBlocks = (srcLen + BLOCK_DIM_X - 1) / BLOCK_DIM_X;
			level_count[l - 2] = srcLen;

		}
		else if (threadIdx.x == 32)
		{
			sm_id = __mysmid();
			T temp = 0;
			while (atomicCAS(&levelStats[sm_id * conc_blocks_per_SM + temp], 0, 1) != 0)
			{
				temp++;
			}
			levelPtr = temp;
		}

		__syncthreads();

		char* cl = &current_level[(sm_id * conc_blocks_per_SM + levelPtr) * maxDeg]; //300 is just an arbitrary number
		while (level_count[l - 2] > level_index[l - 2])
		{
			for (T k = 0; k < srcLenBlocks; k++)
			{
				T index = level_prev_index[l - 2] + k * BLOCK_DIM_X + threadIdx.x;
				if (index < srcLen && cl[index] == l)
				{
					atomicMin(&current_node_index, index);
				}

				__syncthreads();
				if (current_node_index != UINT_MAX)
					break;

				__syncthreads();
			}

			if (threadIdx.x == 0)
			{
				level_prev_index[l - 2] = current_node_index + 1;
				level_index[l - 2]++;
				new_level = l;
			}

			__syncthreads();

			uint64 blockCount = 0;
			const T dst = g.colInd[current_node_index + srcStart];
			const T dstStart = g.rowPtr[dst];
			const T dstLen = g.rowPtr[dst + 1] - dstStart;
			if (dstLen >= kclique - l)
			{
				if (dstLen > srcLen)
				{
					blockCount = graph::block_sorted_count_and_set_binary2<BLOCK_DIM_X, T, true>(&g.colInd[srcStart], srcLen,
						&g.colInd[dstStart], dstLen, true, srcStart, cl, l + 1, kclique);
				}
				else {
					blockCount = graph::block_sorted_count_and_set_binary2<BLOCK_DIM_X, T, true>(&g.colInd[dstStart], dstLen,
						&g.colInd[srcStart], srcLen, false, srcStart, cl, l + 1, kclique);
				}

				__syncthreads();
				if (threadIdx.x == 0 && blockCount > 0)
				{
					if (l + 1 == kclique)
						clique_count += blockCount;
					else if (l + 1 < kclique)
					{
						l++;
						new_level++;
						level_count[l - 2] = blockCount;
						level_index[l - 2] = 0;
						level_prev_index[l - 2] = 0;
					}

				}
			}

			__syncthreads();
			if (threadIdx.x == 0)
			{
				while (new_level > 2 && level_index[new_level - 2] >= level_count[new_level - 2])
				{
					new_level--;
				}
			}

			__syncthreads();
			if (new_level < l)
			{
				for (auto k = 0; k < srcLenBlocks; k++)
				{
					T index = k * BLOCK_DIM_X + threadIdx.x;
					if (index < srcLen && cl[index] > new_level)
						cl[index] = new_level;
				}
				__syncthreads();
			}

			if (threadIdx.x == 0)
			{
				l = new_level;
				current_node_index = UINT_MAX;
			}
			__syncthreads();
		}

		//release and count
		if (threadIdx.x == 0)
		{
			atomicCAS(&levelStats[sm_id * conc_blocks_per_SM + levelPtr], 1, 0);
			atomicAdd(counter, clique_count);
			//cpn[nodeId] = clique_count;
		}
	}
}


template <typename T, int BLOCK_DIM_X>
__launch_bounds__(BLOCK_DIM_X)
__global__ void
kckernel_edge_block_count(
	uint64* counter,
	graph::COOCSRGraph_d<T> g,
	T kclique,
	T maxDeg,
	graph::GraphQueue_d<T, bool> current,
	char* current_level,
	uint64* cpn,
	T conc_blocks_per_SM,
	T* levelStats
)
{
	__shared__ T level_index[5];
	__shared__ T level_count[5];
	__shared__ T level_prev_index[5];
	__shared__ T current_node_index;
	__shared__ uint64 clique_count;
	__shared__ char l;
	__shared__ char new_level;
	__shared__ uint32_t edgeId, src, src2, sm_id, levelPtr;
	__shared__ T srcStart, srcLen, srcLenBlocks, src2Start, src2Len, refIndex, refLen;
	__syncthreads();

	for (unsigned long long i = blockIdx.x; i < (unsigned long long)current.count[0]; i += gridDim.x)
	{
		if (threadIdx.x < 5)
		{
			level_count[threadIdx.x] = 0;
			level_index[threadIdx.x] = 0;
			level_prev_index[threadIdx.x] = 0;
		}
		else if (threadIdx.x == 6)
		{
			l = 2;
			new_level = 2;
			current_node_index = UINT_MAX;
			clique_count = 0;
			edgeId = current.queue[i];

			src = g.rowInd[edgeId];
			src2 = g.colInd[edgeId];

			srcStart = g.rowPtr[src];
			srcLen = g.rowPtr[src + 1] - srcStart;

			src2Start = g.rowPtr[src2];
			src2Len = g.rowPtr[src2 + 1] - src2Start;

			refIndex = srcLen < src2Len ? srcStart : src2Start;
			refLen = srcLen < src2Len ? srcLen : src2Len;;
			srcLenBlocks = (refLen + BLOCK_DIM_X - 1) / BLOCK_DIM_X;

		}
		else if (threadIdx.x == 32)
		{
			sm_id = __mysmid();
			T temp = 0;
			while (atomicCAS(&levelStats[sm_id * conc_blocks_per_SM + temp], 0, 1) != 0)
			{
				temp++;
			}
			levelPtr = temp;
		}

		__syncthreads();


		char* cl = &current_level[(sm_id * conc_blocks_per_SM + levelPtr) * maxDeg];
		for (unsigned long long k = threadIdx.x; k < refLen; k += BLOCK_DIM_X)
		{
			cl[k] = 2;
		}

		__syncthreads();
		uint64 blockCount = 0;
		if (src2Len >= kclique - l)
		{
			if (srcLen < src2Len)
			{
				blockCount += graph::block_sorted_count_and_set_binary2<BLOCK_DIM_X, T, true>(&g.colInd[srcStart], srcLen,
					&g.colInd[src2Start], src2Len, true, srcStart, cl, l + 1, kclique);
			}
			else {
				blockCount += graph::block_sorted_count_and_set_binary2<BLOCK_DIM_X, T, true>(&g.colInd[src2Start], src2Len,
					&g.colInd[srcStart], srcLen, true, srcStart, cl, l + 1, kclique);
			}

			__syncthreads();
			if (blockCount > 0)
			{
				if (l + 1 == kclique && threadIdx.x == 0)
					clique_count += blockCount;
				else if (l + 1 < kclique)
				{
					if (threadIdx.x == 0)
					{
						l++;
						new_level++;
						level_count[l - 2] = blockCount;
						level_index[l - 2] = 0;
						level_prev_index[l - 2] = 0;
					}
				}

			}
		}


		__syncthreads();

		while (level_count[l - 2] > level_index[l - 2])
		{
			for (auto k = 0; k < srcLenBlocks; k++)
			{
				T index = level_prev_index[l - 2] + k * BLOCK_DIM_X + threadIdx.x;
				if (index < refLen && cl[index] == l)
				{
					atomicMin(&current_node_index, index);
				}
				__syncthreads();
				if (current_node_index != UINT_MAX)
					break;

				__syncthreads();
			}

			if (threadIdx.x == 0)
			{
				level_prev_index[l - 2] = current_node_index + 1;
				level_index[l - 2]++;
				new_level = l;
			}

			__syncthreads();

			uint64 blockCountIn = 0;
			const T dst = g.colInd[current_node_index + refIndex];

			const T dstStart = g.rowPtr[dst];
			const T dstStop = g.rowPtr[dst + 1];
			const T dstLen = dstStop - dstStart;
			if (dstLen >= kclique - l)
			{
				if (dstLen > refLen)
				{
					blockCountIn += graph::block_sorted_count_and_set_binary2<BLOCK_DIM_X, T, true>(&g.colInd[refIndex], refLen,
						&g.colInd[dstStart], dstLen, true, srcStart, cl, l + 1, kclique);
				}
				else {
					blockCountIn += graph::block_sorted_count_and_set_binary2<BLOCK_DIM_X, T, true>(&g.colInd[dstStart], dstLen,
						&g.colInd[refIndex], refLen, false, srcStart, cl, l + 1, kclique);
				}

				__syncthreads();
				if (blockCountIn > 0)
				{
					if (l + 1 == kclique && threadIdx.x == 0)
						clique_count += blockCountIn;
					else if (l + 1 < kclique)
					{
						if (threadIdx.x == 0)
						{
							l++;
							new_level++;
							level_count[l - 2] = blockCountIn;
							level_index[l - 2] = 0;
							level_prev_index[l - 2] = 0;
						}
					}

				}
			}

			__syncthreads();
			if (threadIdx.x == 0)
			{

				while (new_level > 3 && level_index[new_level - 2] >= level_count[new_level - 2])
				{
					new_level--;
				}
			}

			__syncthreads();
			if (new_level < l)
			{
				for (T k = 0; k < srcLenBlocks; k++)
				{
					T index = k * BLOCK_DIM_X + threadIdx.x;
					if (index < refLen && cl[index] > new_level)
						cl[index] = new_level;
				}

				__syncthreads();
			}



			if (threadIdx.x == 0)
			{
				l = new_level;
				current_node_index = UINT_MAX;
			}
			__syncthreads();
		}

		if (threadIdx.x == 0)
		{
			atomicCAS(&levelStats[sm_id * conc_blocks_per_SM + levelPtr], 1, 0);
			atomicAdd(counter, clique_count);
		}
	}
}


template <typename T, int BLOCK_DIM_X>
__launch_bounds__(BLOCK_DIM_X)
__global__ void
kckernel_edge_warp_count(
	uint64* counter,
	graph::COOCSRGraph_d<T> g,
	T kclique,
	T maxDeg,
	graph::GraphQueue_d<T, bool> current,
	char* current_level,
	uint64* cpn,
	T conc_blocks_per_SM,
	T* levelStats
)
{
	constexpr T warpsPerBlock = BLOCK_DIM_X / 32;
	const int wx = threadIdx.x / 32; // which warp in thread block
	const size_t lx = threadIdx.x % 32;
	const T gwx = (BLOCK_DIM_X * blockIdx.x + threadIdx.x) / 32;

	__shared__ T level_index[warpsPerBlock][5];
	__shared__ T level_count[warpsPerBlock][5];
	__shared__ T level_prev_index[warpsPerBlock][5];

	__shared__ T current_node_index[warpsPerBlock];
	__shared__ uint64 clique_count[warpsPerBlock];
	__shared__ char l[warpsPerBlock];
	__shared__ char new_level[warpsPerBlock];
	__shared__ uint32_t  sm_id, levelPtr;
	__shared__ T srcStart[warpsPerBlock], srcLen[warpsPerBlock], srcLenBlocks[warpsPerBlock];
	__shared__ T src2Start[warpsPerBlock], src2Len[warpsPerBlock], src2LenBlocks[warpsPerBlock];
	__shared__ T refIndex[warpsPerBlock], refLen[warpsPerBlock];
	__syncthreads();


	if (threadIdx.x == 0)
	{
		sm_id = __mysmid();
		T temp = 0;
		while (atomicCAS(&(levelStats[(sm_id * conc_blocks_per_SM) + temp]), 0, 1) != 0)
		{
			temp++;
		}
		levelPtr = temp;
	}
	__syncthreads();

	for (unsigned long long i = gwx; i < (unsigned long long)current.count[0]; i += warpsPerBlock * gridDim.x) //warp per node
	{
		if (lx < 5)
		{
			level_count[wx][lx] = 0;
			level_index[wx][lx] = 0;
			level_prev_index[wx][lx] = 0;
		}
		else if (lx == 6)
		{
			l[wx] = 2;
			new_level[wx] = 2;
			T src = g.rowInd[current.queue[i]];
			T src2 = g.colInd[current.queue[i]];

			srcStart[wx] = g.rowPtr[src];
			srcLen[wx] = g.rowPtr[src + 1] - srcStart[wx];

			src2Start[wx] = g.rowPtr[src2];
			src2Len[wx] = g.rowPtr[src2 + 1] - src2Start[wx];

			refIndex[wx] = srcLen[wx] < src2Len[wx] ? srcStart[wx] : src2Start[wx];
			refLen[wx] = srcLen[wx] < src2Len[wx] ? srcLen[wx] : src2Len[wx];
			srcLenBlocks[wx] = (refLen[wx] + 32 - 1) / 32;
		}
		else if (lx == 7)
		{

			current_node_index[wx] = UINT_MAX;
			clique_count[wx] = 0;
		}

		T blockOffset = sm_id * conc_blocks_per_SM * (warpsPerBlock * maxDeg)
			+ levelPtr * (warpsPerBlock * maxDeg);

		char* cl = &current_level[blockOffset + wx * maxDeg /*srcStart[wx]*/];
		for (unsigned long long k = lx; k < refLen[wx]; k += 32)
		{
			cl[k] = 2;
		}
		__syncwarp();

		uint64 warpCount = 0;
		if (src2Len[wx] >= kclique - l[wx])
		{
			if (srcLen[wx] < src2Len[wx])
			{
				warpCount += graph::warp_sorted_count_and_set_binary<WARPS_PER_BLOCK, T, true>(0, &g.colInd[srcStart[wx]], srcLen[wx],
					&g.colInd[src2Start[wx]], src2Len[wx], true, srcStart[wx], cl, l[wx] + 1, kclique);
			}
			else {
				warpCount += graph::warp_sorted_count_and_set_binary<WARPS_PER_BLOCK, T, true>(0, &g.colInd[src2Start[wx]], src2Len[wx],
					&g.colInd[srcStart[wx]], srcLen[wx], true, srcStart[wx], cl, l[wx] + 1, kclique);
			}

			__syncwarp();
			if (warpCount > 0)
			{
				if (l[wx] + 1 == kclique && lx == 0)
					clique_count[wx] += warpCount;
				else if (l[wx] + 1 < kclique)
				{
					if (lx == 0)
					{
						(l[wx])++;
						(new_level[wx])++;
						level_count[wx][l[wx] - 2] = warpCount;
						level_index[wx][l[wx] - 2] = 0;
						level_prev_index[wx][l[wx] - 2] = 0;
					}
				}

			}
		}


		__syncwarp();
		while (level_count[wx][l[wx] - 2] > level_index[wx][l[wx] - 2])
		{
			for (T k = 0; k < srcLenBlocks[wx]; k++)
			{
				T index = level_prev_index[wx][l[wx] - 2] + k * 32 + lx;
				if (index < refLen[wx] && cl[index] == l[wx])
				{
					atomicMin(&(current_node_index[wx]), index);
				}
				__syncwarp();
				if (current_node_index[wx] != UINT_MAX)
					break;

				__syncwarp();
			}


			// for (T k = 0; k < srcLenBlocks[wx]; k++)
			// {
			// 	T index = level_prev_index[wx][l[wx] - 2] + k * 32 + lx;
			// 	int condition = index < srcLen[wx] && cl[index] == l[wx];
			// 	unsigned int newmask = __ballot_sync(0xFFFFFFFF, condition);
			// 	if (newmask != 0)
			// 	{
			// 		int elected_lane_deq = __ffs(newmask) - 1;
			// 		current_node_index[wx] = __shfl_sync(0xFFFFFFFF, index, elected_lane_deq, 32);
			// 		break;
			// 	}
			// }

			if (lx == 0)
			{
				//current_node_index[0] = finalIndex;
				level_prev_index[wx][l[wx] - 2] = current_node_index[wx] + 1;
				level_index[wx][l[wx] - 2]++;
				new_level[wx] = l[wx];
			}

			__syncwarp();

			uint64 warpCountIn = 0;
			const T dst = g.colInd[current_node_index[wx] + refIndex[wx]];
			const T dstStart = g.rowPtr[dst];
			const T dstLen = g.rowPtr[dst + 1] - dstStart;

			if (dstLen >= kclique - l[wx])
			{
				if (dstLen > refLen[wx])
				{
					warpCountIn = graph::warp_sorted_count_and_set_binary<WARPS_PER_BLOCK, T, true>(0, &g.colInd[refIndex[wx]], refLen[wx],
						&g.colInd[dstStart], dstLen, true, srcStart[wx], cl, l[wx] + 1, kclique);
				}
				else {
					warpCountIn = graph::warp_sorted_count_and_set_binary<WARPS_PER_BLOCK, T, true>(0, &g.colInd[dstStart], dstLen,
						&g.colInd[refIndex[wx]], refLen[wx], false, srcStart[wx], cl, l[wx] + 1, kclique);
				}
				__syncwarp();
				if (lx == 0 && warpCountIn > 0)
				{
					if (l[wx] + 1 == kclique)
						clique_count[wx] += warpCountIn;
					else if (l[wx] + 1 < kclique)
					{
						(l[wx])++;
						(new_level[wx])++;
						level_count[wx][l[wx] - 2] = warpCountIn;
						level_index[wx][l[wx] - 2] = 0;
						level_prev_index[wx][l[wx] - 2] = 0;
					}
				}
			}


			__syncwarp();
			if (lx == 0)
			{
				while (new_level[wx] > 3 && level_index[wx][new_level[wx] - 2] >= level_count[wx][new_level[wx] - 2])
				{
					(new_level[wx])--;
				}
			}

			__syncwarp();
			if (new_level[wx] < l[wx])
			{
				for (T k = 0; k < srcLenBlocks[wx]; k++)
				{
					T index = k * 32 + lx;
					if (index < refLen[wx] && cl[index] > new_level[wx])
						cl[index] = new_level[wx];
				}

				__syncwarp();
			}



			if (lx == 0)
			{
				l[wx] = new_level[wx];
				current_node_index[wx] = UINT_MAX;
			}
			__syncwarp();
		}

		if (lx == 0)
		{
			atomicAdd(counter, clique_count[wx]);
			//cpn[current.queue[i]] = clique_count[wx];
		}
	}

	__syncthreads();
	if (threadIdx.x == 0)
	{
		atomicCAS(&levelStats[sm_id * conc_blocks_per_SM + levelPtr], 1, 0);
	}
}

template <typename T, int BLOCK_DIM_X>
__launch_bounds__(BLOCK_DIM_X, 32)
__global__ void
kckernel_edge_warp_count2(
	uint64* counter,
	graph::COOCSRGraph_d<T> g,
	T kclique,
	T maxDeg,
	graph::GraphQueue_d<T, bool> current,
	char* current_level,
	uint64* cpn,
	T conc_blocks_per_SM,
	T* levelStats
)
{
	constexpr T warpsPerBlock = BLOCK_DIM_X / 32;
	const int wx = threadIdx.x / 32; // which warp in thread block
	const size_t lx = threadIdx.x % 32;
	const T gwx = (BLOCK_DIM_X * blockIdx.x + threadIdx.x) / 32;
	__shared__ T level_index[warpsPerBlock][5];
	__shared__ T level_count[warpsPerBlock][5];
	__shared__ T level_prev_index[warpsPerBlock][5];

	__shared__ T current_node_index[warpsPerBlock];
	__shared__ uint64 clique_count[warpsPerBlock];
	__shared__ char l[warpsPerBlock];
	__shared__ char new_level[warpsPerBlock];
	__shared__ uint32_t  sm_id, levelPtr;
	__shared__ T srcStart[warpsPerBlock], srcLen[warpsPerBlock], srcLenBlocks[warpsPerBlock];
	__shared__ T src2Start[warpsPerBlock], src2Len[warpsPerBlock], src2LenBlocks[warpsPerBlock];
	__shared__ T refIndex[warpsPerBlock], refLen[warpsPerBlock];
	__syncthreads();
	if (threadIdx.x == 0)
	{
		sm_id = __mysmid();
		T temp = 0;
		while (atomicCAS(&(levelStats[(sm_id * conc_blocks_per_SM) + temp]), 0, 1) != 0)
		{
			temp++;
		}
		levelPtr = temp;
	}
	__syncthreads();

	for (unsigned long long i = gwx; i < (unsigned long long)current.count[0]; i += warpsPerBlock * gridDim.x) //warp per node
	{
		if (lx < 5)
		{
			level_count[wx][lx] = 0;
			level_index[wx][lx] = 0;
			level_prev_index[wx][lx] = 0;
		}
		else if (lx == 6)
		{

			T src = g.rowInd[current.queue[i]];
			T src2 = g.colInd[current.queue[i]];

			srcStart[wx] = g.rowPtr[src];
			srcLen[wx] = g.rowPtr[src + 1] - srcStart[wx];

			src2Start[wx] = g.rowPtr[src2];
			src2Len[wx] = g.rowPtr[src2 + 1] - src2Start[wx];

			refIndex[wx] = srcLen[wx] < src2Len[wx] ? srcStart[wx] : src2Start[wx];
			refLen[wx] = srcLen[wx] < src2Len[wx] ? srcLen[wx] : src2Len[wx];
			srcLenBlocks[wx] = (refLen[wx] + 32 - 1) / 32;
		}
		else if (lx == 7)
		{
			l[wx] = 2;
			new_level[wx] = 2;
			current_node_index[wx] = UINT_MAX;
			clique_count[wx] = 0;
		}

		__syncwarp();
		T blockOffset = sm_id * conc_blocks_per_SM * (warpsPerBlock * maxDeg)
			+ levelPtr * (warpsPerBlock * maxDeg);
		char* cl = &current_level[blockOffset + wx * maxDeg /*srcStart[wx]*/];
		for (unsigned long long k = lx; k < refLen[wx]; k += 32)
		{
			cl[k] = 0x01;
		}
		__syncwarp();

		uint64 warpCount = 0;
		if (src2Len[wx] >= kclique - l[wx])
		{
			if (srcLen[wx] < src2Len[wx])
			{
				warpCount += graph::warp_sorted_count_and_set_binary2<WARPS_PER_BLOCK, T, true>(0, &g.colInd[srcStart[wx]], srcLen[wx],
					&g.colInd[src2Start[wx]], src2Len[wx], true, srcStart[wx], cl, l[wx] + 1, kclique);
			}
			else {
				warpCount += graph::warp_sorted_count_and_set_binary2<WARPS_PER_BLOCK, T, true>(0, &g.colInd[src2Start[wx]], src2Len[wx],
					&g.colInd[srcStart[wx]], srcLen[wx], true, srcStart[wx], cl, l[wx] + 1, kclique);
			}

			__syncwarp();
			if (warpCount > 0)
			{
				if (l[wx] + 1 == kclique && lx == 0)
					clique_count[wx] += warpCount;
				else if (l[wx] + 1 < kclique)
				{
					if (lx == 0)
					{
						(l[wx])++;
						(new_level[wx])++;
						level_count[wx][l[wx] - 2] = warpCount;
						level_index[wx][l[wx] - 2] = 0;
						level_prev_index[wx][l[wx] - 2] = 0;
					}
				}

			}
		}


		__syncwarp();
		while (level_count[wx][l[wx] - 2] > level_index[wx][l[wx] - 2])
		{
			for (T k = 0; k < srcLenBlocks[wx]; k++)
			{
				T index = level_prev_index[wx][l[wx] - 2] + k * 32 + lx;
				int condition = index < srcLen[wx] && (cl[index] & (0x01 << (l[wx] - 2)));
				unsigned int newmask = __ballot_sync(0xFFFFFFFF, condition);
				if (newmask != 0)
				{
					uint elected_lane_deq = __ffs(newmask) - 1;
					current_node_index[wx] = __shfl_sync(0xFFFFFFFF, index, elected_lane_deq, 32);
					break;
				}
			}

			if (lx == 0)
			{
				//current_node_index[0] = finalIndex;
				level_prev_index[wx][l[wx] - 2] = current_node_index[wx] + 1;
				level_index[wx][l[wx] - 2]++;
				new_level[wx] = l[wx];
			}

			__syncwarp();

			uint64 warpCountIn = 0;
			const T dst = g.colInd[current_node_index[wx] + refIndex[wx]];
			const T dstStart = g.rowPtr[dst];
			const T dstLen = g.rowPtr[dst + 1] - dstStart;

			bool limit = ((l[wx] - 1 + level_count[wx][l[wx] - 2]) >= kclique) && (dstLen >= kclique - l[wx]);

			if (limit /*dstLen >= kclique - l[wx]*/)
			{
				if (dstLen > refLen[wx])
				{
					warpCountIn = graph::warp_sorted_count_and_set_binary2<WARPS_PER_BLOCK, T, true>(0, &g.colInd[refIndex[wx]], refLen[wx],
						&g.colInd[dstStart], dstLen, true, srcStart[wx], cl, l[wx] + 1, kclique);
				}
				else {
					warpCountIn = graph::warp_sorted_count_and_set_binary2<WARPS_PER_BLOCK, T, true>(0, &g.colInd[dstStart], dstLen,
						&g.colInd[refIndex[wx]], refLen[wx], false, srcStart[wx], cl, l[wx] + 1, kclique);

				}
				__syncwarp();
				if (lx == 0 && warpCountIn > 0)
				{
					if (l[wx] + 1 == kclique)
						clique_count[wx] += warpCountIn;
					else if ((l[wx] + 1 < kclique) /*&& ((l[wx] + warpCountIn) >= kclique)*/)
					{
						//if(warpCountIn >= kclique - l[wx])
						{
							(l[wx])++;
							(new_level[wx])++;
							level_count[wx][l[wx] - 2] = warpCountIn;
							level_index[wx][l[wx] - 2] = 0;
							level_prev_index[wx][l[wx] - 2] = 0;
						}
					}
				}
			}


			__syncwarp();
			if (lx == 0)
			{
				while (new_level[wx] > 3 && level_index[wx][new_level[wx] - 2] >= level_count[wx][new_level[wx] - 2])
				{
					(new_level[wx])--;
				}
			}

			__syncwarp();
			if (new_level[wx] < l[wx])
			{
				char clearMask = ~((1 << (l[wx] - 1)) - (1 << (new_level[wx] - 1)));
				for (auto k = 0; k < srcLenBlocks[wx]; k++)
				{
					T index = k * 32 + lx;
					if (index < refLen[wx])
						cl[index] = cl[index] & clearMask;
				}

				__syncwarp();
			}

			if (lx == 0)
			{
				l[wx] = new_level[wx];
				current_node_index[wx] = UINT_MAX;
			}
			__syncwarp();
		}

		if (lx == 0)
		{
			atomicAdd(counter, clique_count[wx]);
			//cpn[current.queue[i]] = clique_count[wx];
		}
	}

	__syncthreads();
	if (threadIdx.x == 0)
	{
		atomicCAS(&levelStats[sm_id * conc_blocks_per_SM + levelPtr], 1, 0);
	}
}

//not goood
template <typename T, int BLOCK_DIM_X>
__launch_bounds__(BLOCK_DIM_X, 32)
__global__ void
kckernel_edge_warp_count2_shared(
	uint64* counter,
	graph::COOCSRGraph_d<T> g,
	T kclique,
	T maxDeg,
	graph::GraphQueue_d<T, bool> current,
	char* current_level,
	uint64* cpn,
	T conc_blocks_per_SM,
	T* levelStats
)
{
	constexpr T warpsPerBlock = BLOCK_DIM_X / 32;
	const int wx = threadIdx.x / 32; // which warp in thread block
	const size_t lx = threadIdx.x % 32;
	const T gwx = (BLOCK_DIM_X * blockIdx.x + threadIdx.x) / 32;


	__shared__ T srcShared[warpsPerBlock][1 * 32];

	__shared__ T level_index[warpsPerBlock][5];
	__shared__ T level_count[warpsPerBlock][5];
	__shared__ T level_prev_index[warpsPerBlock][5];

	__shared__ T current_node_index[warpsPerBlock];
	__shared__ uint64 clique_count[warpsPerBlock];
	__shared__ char l[warpsPerBlock];
	__shared__ char new_level[warpsPerBlock];
	__shared__ uint32_t  sm_id, levelPtr;
	__shared__ T srcStart[warpsPerBlock], srcLen[warpsPerBlock], srcLenBlocks[warpsPerBlock];
	__shared__ T src2Start[warpsPerBlock], src2Len[warpsPerBlock], src2LenBlocks[warpsPerBlock];
	__shared__ T refIndex[warpsPerBlock], refLen[warpsPerBlock];
	__syncthreads();
	if (threadIdx.x == 0)
	{
		sm_id = __mysmid();
		T temp = 0;
		while (atomicCAS(&(levelStats[(sm_id * conc_blocks_per_SM) + temp]), 0, 1) != 0)
		{
			temp++;
		}
		levelPtr = temp;
	}
	__syncthreads();

	for (unsigned long long i = gwx; i < (unsigned long long)current.count[0]; i += warpsPerBlock * gridDim.x) //warp per node
	{
		if (lx < 5)
		{
			level_count[wx][lx] = 0;
			level_index[wx][lx] = 0;
			level_prev_index[wx][lx] = 0;
		}
		else if (lx == 6)
		{

			T src = g.rowInd[current.queue[i]];
			T src2 = g.colInd[current.queue[i]];

			srcStart[wx] = g.rowPtr[src];
			srcLen[wx] = g.rowPtr[src + 1] - srcStart[wx];

			src2Start[wx] = g.rowPtr[src2];
			src2Len[wx] = g.rowPtr[src2 + 1] - src2Start[wx];

			refIndex[wx] = srcLen[wx] < src2Len[wx] ? srcStart[wx] : src2Start[wx];
			refLen[wx] = srcLen[wx] < src2Len[wx] ? srcLen[wx] : src2Len[wx];
			srcLenBlocks[wx] = (refLen[wx] + 32 - 1) / 32;
		}
		else if (lx == 7)
		{
			l[wx] = 2;
			new_level[wx] = 2;
			current_node_index[wx] = UINT_MAX;
			clique_count[wx] = 0;
		}

		__syncwarp();
		T blockOffset = sm_id * conc_blocks_per_SM * (warpsPerBlock * maxDeg)
			+ levelPtr * (warpsPerBlock * maxDeg);
		char* cl = &current_level[blockOffset + wx * maxDeg /*srcStart[wx]*/];
		for (unsigned long long k = lx; k < refLen[wx]; k += 32)
		{
			cl[k] = 0x01;
		}


		T par = (refLen[wx] + 32 - 1) / (32);
		T numElements = refLen[wx] < 32 ? refLen[wx] : (refLen[wx] + par - 1) / par;

		for (T f = 0; f < 1; f++)
		{
			T sharedIndex = 32 * f + lx;
			T realIndex = refIndex[wx] + (lx + f * 32) * par;
			srcShared[wx][sharedIndex] = (lx + 32 * f) <= numElements ? g.colInd[realIndex] : 0;
		}
		__syncwarp();

		uint64 warpCount = 0;
		if (src2Len[wx] >= kclique - l[wx])
		{
			if (srcLen[wx] < src2Len[wx])
			{
				warpCount += graph::warp_sorted_count_and_set_binary2<WARPS_PER_BLOCK, T, true>(0, &g.colInd[srcStart[wx]], srcLen[wx],
					&g.colInd[src2Start[wx]], src2Len[wx], true, srcStart[wx], cl, l[wx] + 1, kclique);
			}
			else {
				warpCount += graph::warp_sorted_count_and_set_binary2<WARPS_PER_BLOCK, T, true>(0, &g.colInd[src2Start[wx]], src2Len[wx],
					&g.colInd[srcStart[wx]], srcLen[wx], true, srcStart[wx], cl, l[wx] + 1, kclique);
			}

			__syncwarp();
			if (warpCount > 0)
			{
				if (l[wx] + 1 == kclique && lx == 0)
					clique_count[wx] += warpCount;
				else if (l[wx] + 1 < kclique)
				{
					if (lx == 0)
					{
						(l[wx])++;
						(new_level[wx])++;
						level_count[wx][l[wx] - 2] = warpCount;
						level_index[wx][l[wx] - 2] = 0;
						level_prev_index[wx][l[wx] - 2] = 0;
					}
				}

			}
		}


		__syncwarp();
		while (level_count[wx][l[wx] - 2] > level_index[wx][l[wx] - 2])
		{
			for (T k = 0; k < srcLenBlocks[wx]; k++)
			{
				T index = level_prev_index[wx][l[wx] - 2] + k * 32 + lx;
				int condition = index < srcLen[wx] && (cl[index] & (0x01 << (l[wx] - 2)));
				unsigned int newmask = __ballot_sync(0xFFFFFFFF, condition);
				if (newmask != 0)
				{
					uint elected_lane_deq = __ffs(newmask) - 1;
					current_node_index[wx] = __shfl_sync(0xFFFFFFFF, index, elected_lane_deq, 32);
					break;
				}
			}

			if (lx == 0)
			{
				//current_node_index[0] = finalIndex;
				level_prev_index[wx][l[wx] - 2] = current_node_index[wx] + 1;
				level_index[wx][l[wx] - 2]++;
				new_level[wx] = l[wx];
			}

			__syncwarp();

			uint64 warpCountIn = 0;
			const T dst = g.colInd[current_node_index[wx] + refIndex[wx]];
			const T dstStart = g.rowPtr[dst];
			const T dstLen = g.rowPtr[dst + 1] - dstStart;

			bool limit = ((l[wx] - 1 + level_count[wx][l[wx] - 2]) >= kclique) && (dstLen >= kclique - l[wx]);

			if (limit /*dstLen >= kclique - l[wx]*/)
			{
				if (dstLen > refLen[wx])
				{
					warpCountIn = graph::warp_sorted_count_and_set_binary2<WARPS_PER_BLOCK, T, true>(0, &g.colInd[refIndex[wx]], refLen[wx],
						&g.colInd[dstStart], dstLen, true, srcStart[wx], cl, l[wx] + 1, kclique);
				}
				else {


					// warpCountIn = graph::warp_sorted_count_and_set_binary2<WARPS_PER_BLOCK, T, true>(0, &g.colInd[dstStart], dstLen,
					// 	&g.colInd[refIndex[wx]], refLen[wx], false, srcStart[wx], cl, l[wx] + 1, kclique);

					warpCountIn = graph::warp_sorted_count_set_binary_sbd<WARPS_PER_BLOCK, T, true>(&g.colInd[dstStart], dstLen,
						&g.colInd[refIndex[wx]], refLen[wx],
						&(srcShared[wx][0]), par, numElements, 32,
						cl, l[wx] + 1, kclique);
				}
				__syncwarp();
				if (lx == 0 && warpCountIn > 0)
				{
					if (l[wx] + 1 == kclique)
						clique_count[wx] += warpCountIn;
					else if ((l[wx] + 1 < kclique) /*&& ((l[wx] + warpCountIn) >= kclique)*/)
					{
						//if(warpCountIn >= kclique - l[wx])
						{
							(l[wx])++;
							(new_level[wx])++;
							level_count[wx][l[wx] - 2] = warpCountIn;
							level_index[wx][l[wx] - 2] = 0;
							level_prev_index[wx][l[wx] - 2] = 0;
						}
					}
				}
			}


			__syncwarp();
			if (lx == 0)
			{
				while (new_level[wx] > 3 && level_index[wx][new_level[wx] - 2] >= level_count[wx][new_level[wx] - 2])
				{
					(new_level[wx])--;
				}
			}

			__syncwarp();
			if (new_level[wx] < l[wx])
			{
				char clearMask = ~((1 << (l[wx] - 1)) - (1 << (new_level[wx] - 1)));
				for (auto k = 0; k < srcLenBlocks[wx]; k++)
				{
					T index = k * 32 + lx;
					if (index < refLen[wx])
						cl[index] = cl[index] & clearMask;
				}

				__syncwarp();
			}

			if (lx == 0)
			{
				l[wx] = new_level[wx];
				current_node_index[wx] = UINT_MAX;
			}
			__syncwarp();
		}

		if (lx == 0)
		{
			atomicAdd(counter, clique_count[wx]);
			//cpn[current.queue[i]] = clique_count[wx];
		}
	}

	__syncthreads();
	if (threadIdx.x == 0)
	{
		atomicCAS(&levelStats[sm_id * conc_blocks_per_SM + levelPtr], 1, 0);
	}
}


template <typename T, uint BLOCK_DIM_X>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_block_warp_count(
	uint64* counter,
	graph::COOCSRGraph_d<T> g,
	T kclique,
	T maxDeg,
	const  graph::GraphQueue_d<T, bool>  current,
	char* current_level,
	uint64* cpn,
	T conc_blocks_per_SM,
	T* levelStats
)
{
	//will be removed later
	constexpr T warpsPerBlock = BLOCK_DIM_X / 32;
	const int wx = threadIdx.x / 32; // which warp in thread block
	const size_t lx = threadIdx.x % 32;
	__shared__ T level_index[warpsPerBlock][5];
	__shared__ T level_count[warpsPerBlock][5];
	__shared__ T level_prev_index[warpsPerBlock][5];

	__shared__ T current_node_index[warpsPerBlock];
	__shared__ uint64 clique_count[warpsPerBlock];
	__shared__ char l[warpsPerBlock];
	__shared__ char new_level[warpsPerBlock];
	__shared__ uint32_t  sm_id, levelPtr;
	__shared__ T src, srcStart, srcLen;
	__shared__ T src2Start[warpsPerBlock], src2Len[warpsPerBlock], src2LenBlocks[warpsPerBlock];
	__shared__ T refIndex[warpsPerBlock], refLen[warpsPerBlock], srcLenBlocks[warpsPerBlock];

	__syncthreads();

	if (threadIdx.x == 0)
	{
		sm_id = __mysmid();
		T temp = 0;
		while (atomicCAS(&(levelStats[(sm_id * conc_blocks_per_SM) + temp]), 0, 1) != 0)
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
			T src = current.queue[i];
			srcStart = g.rowPtr[src];
			srcLen = g.rowPtr[src + 1] - srcStart;
		}
		__syncthreads();
		//warp loop
		for (unsigned long long j = wx; j < srcLen; j += warpsPerBlock)
		{
			if (lx < 5)
			{
				level_count[wx][lx] = 0;
				level_index[wx][lx] = 0;
				level_prev_index[wx][lx] = 0;
			}
			else if (lx == 6)
			{
				T src2 = g.colInd[srcStart + j];
				src2Start[wx] = g.rowPtr[src2];
				src2Len[wx] = g.rowPtr[src2 + 1] - src2Start[wx];

				refIndex[wx] = srcLen < src2Len[wx] ? srcStart : src2Start[wx];
				refLen[wx] = srcLen < src2Len[wx] ? srcLen : src2Len[wx];
				srcLenBlocks[wx] = (refLen[wx] + 32 - 1) / 32;
			}
			else if (lx == 7)
			{
				l[wx] = 2;
				new_level[wx] = 2;
				current_node_index[wx] = UINT_MAX;
				clique_count[wx] = 0;
			}

			__syncwarp();
			T blockOffset = sm_id * conc_blocks_per_SM * (warpsPerBlock * maxDeg)
				+ levelPtr * (warpsPerBlock * maxDeg);
			char* cl = &current_level[blockOffset + wx * maxDeg /*srcStart[wx]*/];
			for (unsigned long long k = lx; k < refLen[wx]; k += 32)
			{
				cl[k] = 0x01;
			}
			__syncwarp();

			uint64 warpCount = 0;
			if (src2Len[wx] >= kclique - l[wx])
			{
				if (srcLen < src2Len[wx])
				{
					warpCount += graph::warp_sorted_count_and_set_binary2<WARPS_PER_BLOCK, T, true>(0, &g.colInd[srcStart], srcLen,
						&g.colInd[src2Start[wx]], src2Len[wx], true, srcStart, cl, l[wx] + 1, kclique);
				}
				else {
					warpCount += graph::warp_sorted_count_and_set_binary2<WARPS_PER_BLOCK, T, true>(0, &g.colInd[src2Start[wx]], src2Len[wx],
						&g.colInd[srcStart], srcLen, true, srcStart, cl, l[wx] + 1, kclique);
				}

				__syncwarp();
				if (warpCount > 0)
				{
					if (l[wx] + 1 == kclique && lx == 0)
						clique_count[wx] += warpCount;
					else if (l[wx] + 1 < kclique)
					{
						if (lx == 0)
						{
							(l[wx])++;
							(new_level[wx])++;
							level_count[wx][l[wx] - 2] = warpCount;
							level_index[wx][l[wx] - 2] = 0;
							level_prev_index[wx][l[wx] - 2] = 0;
						}
					}

				}
			}
			__syncwarp();
			while (level_count[wx][l[wx] - 2] > level_index[wx][l[wx] - 2])
			{
				for (T k = 0; k < srcLenBlocks[wx]; k++)
				{
					T index = level_prev_index[wx][l[wx] - 2] + k * 32 + lx;
					int condition = index < refLen[wx] && (cl[index] & (0x01 << (l[wx] - 2)));
					unsigned int newmask = __ballot_sync(0xFFFFFFFF, condition);
					if (newmask != 0)
					{
						uint elected_lane_deq = __ffs(newmask) - 1;
						current_node_index[wx] = __shfl_sync(0xFFFFFFFF, index, elected_lane_deq, 32);
						break;
					}
				}

				if (lx == 0)
				{
					//current_node_index[0] = finalIndex;
					level_prev_index[wx][l[wx] - 2] = current_node_index[wx] + 1;
					level_index[wx][l[wx] - 2]++;
					new_level[wx] = l[wx];
				}

				__syncwarp();

				uint64 warpCountIn = 0;
				const T dst = g.colInd[current_node_index[wx] + refIndex[wx]];
				const T dstStart = g.rowPtr[dst];
				const T dstLen = g.rowPtr[dst + 1] - dstStart;

				bool limit = ((l[wx] - 1 + level_count[wx][l[wx] - 2]) >= kclique) && (dstLen >= kclique - l[wx]);

				if (limit /*dstLen >= kclique - l[wx]*/)
				{
					if (dstLen > refLen[wx])
					{
						warpCountIn = graph::warp_sorted_count_and_set_binary2<WARPS_PER_BLOCK, T, true>(0, &g.colInd[refIndex[wx]], refLen[wx],
							&g.colInd[dstStart], dstLen, true, srcStart, cl, l[wx] + 1, kclique);
					}
					else {
						warpCountIn = graph::warp_sorted_count_and_set_binary2<WARPS_PER_BLOCK, T, true>(0, &g.colInd[dstStart], dstLen,
							&g.colInd[refIndex[wx]], refLen[wx], false, srcStart, cl, l[wx] + 1, kclique);

					}
					__syncwarp();
					if (lx == 0 && warpCountIn > 0)
					{
						if (l[wx] + 1 == kclique)
							clique_count[wx] += warpCountIn;
						else if ((l[wx] + 1 < kclique) /*&& ((l[wx] + warpCountIn) >= kclique)*/)
						{
							//if(warpCountIn >= kclique - l[wx])
							{
								(l[wx])++;
								(new_level[wx])++;
								level_count[wx][l[wx] - 2] = warpCountIn;
								level_index[wx][l[wx] - 2] = 0;
								level_prev_index[wx][l[wx] - 2] = 0;
							}
						}
					}
				}


				__syncwarp();
				if (lx == 0)
				{
					while (new_level[wx] > 3 && level_index[wx][new_level[wx] - 2] >= level_count[wx][new_level[wx] - 2])
					{
						(new_level[wx])--;
					}
				}

				__syncwarp();
				if (new_level[wx] < l[wx])
				{
					char clearMask = ~((1 << (l[wx] - 1)) - (1 << (new_level[wx] - 1)));
					for (auto k = 0; k < srcLenBlocks[wx]; k++)
					{
						T index = k * 32 + lx;
						if (index < refLen[wx])
							cl[index] = cl[index] & clearMask;
					}

					__syncwarp();
				}

				if (lx == 0)
				{
					l[wx] = new_level[wx];
					current_node_index[wx] = UINT_MAX;
				}
				__syncwarp();
			}

			if (lx == 0)
			{
				atomicAdd(counter, clique_count[wx]);
				//cpn[current.queue[i]] = clique_count[wx];
			}
		}
	}

	__syncthreads();
	if (threadIdx.x == 0)
	{
		atomicCAS(&levelStats[sm_id * conc_blocks_per_SM + levelPtr], 1, 0);
	}
}


template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_block_warp_binary_count(
	uint64* counter,
	graph::COOCSRGraph_d<T> g,
	const  graph::GraphQueue_d<T, bool>  current,
	T* current_level,
	uint64* cpn,
	T* levelStats,
	T* adj_enc)
{
	//will be removed later
	constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
	const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
	const size_t lx = threadIdx.x % CPARTSIZE;
	__shared__ T level_index[numPartitions][7];
	__shared__ T level_count[numPartitions][7];
	__shared__ T level_prev_index[numPartitions][7];

	__shared__ T  level_offset[numPartitions];
	__shared__ uint64 clique_count[numPartitions];
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

	for (unsigned long long i = blockIdx.x; i < (unsigned long long)current.count[0]; i += gridDim.x)
	{
		//block things
		if (threadIdx.x == 0)
		{
			T src = current.queue[i];
			srcStart = g.rowPtr[src];
			srcLen = g.rowPtr[src + 1] - srcStart;

			//printf("src = %u, srcLen = %u\n", src, srcLen);
		}
		__syncthreads();
		 if (threadIdx.x == 0)
			num_divs_local = (srcLen + 32 - 1) / 32;
		else if (threadIdx.x == 1)
		{
			encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
			encode = &adj_enc[encode_offset  /*srcStart[wx]*/];
		}
		__syncthreads();
		//Encode
		T partMask = (1 << CPARTSIZE) - 1;
		partMask = partMask << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
		T mm = (1 << srcLen) - 1;
		mm = mm << ((wx/numPartitions) * CPARTSIZE);
		for (unsigned long long j = wx; j < srcLen; j += numPartitions)
		{
			for (unsigned long long k = lx; k < num_divs_local; k += CPARTSIZE)
			{
				encode[j * num_divs_local + k] = 0x00;
			}
			__syncwarp(partMask);
			graph::warp_sorted_count_and_encode<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&g.colInd[srcStart], srcLen,
				&g.colInd[g.rowPtr[g.colInd[srcStart + j]]], g.rowPtr[g.colInd[srcStart + j] + 1] - g.rowPtr[g.colInd[srcStart + j]],
				&encode[j * num_divs_local]);
		}

		__syncthreads(); //Done encoding


		for (unsigned long long j = wx; j < srcLen; j += numPartitions)
		{

			level_offset[wx] = sm_id * CBPSM * (numPartitions * NUMDIVS * 7) + levelPtr * (numPartitions * NUMDIVS * 7);
			T* cl = &current_level[level_offset[wx] + wx * (NUMDIVS * 7)];


			if (lx < 7)
			{
				level_count[wx][lx] = 0;
				level_index[wx][lx] = 0;
				level_prev_index[wx][lx] = 0;
			}
			if (lx == 0)
			{
				l[wx] = 3;
				clique_count[wx] = 0;
			}


			// for (unsigned long long k = lx; k < num_divs_local * 7; k += CPARTSIZE)
			// {
			// 	cl[k] = 0x00;
			// }


			//get warp count ??
			uint64 warpCount = 0;
			for (unsigned long long k = lx; k < num_divs_local; k += CPARTSIZE)
			{
				warpCount += __popc(encode[j * num_divs_local + k]);
			}
			reduce_part<T>(partMask, warpCount);

			if (lx == 0 && l[wx] == KCCOUNT)
				clique_count[wx] += warpCount;
			else if (lx == 0 && KCCOUNT > 3 && warpCount >= KCCOUNT - 2)
			{
				level_count[wx][l[wx] - 3] = warpCount;
				level_index[wx][l[wx] - 3] = 0;
				level_prev_index[wx][l[wx] - 3] = 0;
			}
			__syncwarp(partMask);
			while (level_count[wx][l[wx] - 3] > level_index[wx][l[wx] - 3])
			{
				//First Index
				T* from = l[wx] == 3 ? &(encode[num_divs_local * j]) : &(cl[num_divs_local * (l[wx] - 3)]);
				T* to = &(cl[num_divs_local * (l[wx] - 2)]);
				T maskBlock = level_prev_index[wx][l[wx] - 3] / 32;
				T maskIndex = ~((1 << (level_prev_index[wx][l[wx] - 3] & 0x1F)) -1);

				T newIndex = __ffs(from[maskBlock] & maskIndex);
				while(newIndex == 0)
				{
					maskIndex = 0xFFFFFFFF;
					maskBlock++;
					newIndex = __ffs(from[maskBlock] & maskIndex);
				}
				newIndex =  32*maskBlock + newIndex - 1;

				if (lx == 0)
				{
					level_prev_index[wx][l[wx] - 3] = newIndex + 1;
					level_index[wx][l[wx] - 3]++;
				}

				//Intersect
				uint64 warpCount = 0;
				for (T k = lx; k < num_divs_local; k += CPARTSIZE)
				{
					to[k] = from[k] & encode[newIndex* num_divs_local + k];
					warpCount += __popc(to[k]);
				}
				reduce_part<T>(partMask, warpCount);

				if (lx == 0)
				{
					if (l[wx] + 1 == KCCOUNT)
						clique_count[wx] += warpCount;
					else if (l[wx] + 1 < KCCOUNT && warpCount >= KCCOUNT - l[wx])
					{
						(l[wx])++;
						level_count[wx][l[wx] - 3] = warpCount;
						level_index[wx][l[wx] - 3] = 0;
						level_prev_index[wx][l[wx] - 3] = 0;
					}
				
					while (l[wx] > 3 && level_index[wx][l[wx] - 3] >= level_count[wx][l[wx] - 3])
					{
						(l[wx])--;
					}
				}
				__syncwarp(partMask);
			}
			if (lx == 0)
			{
				atomicAdd(counter, clique_count[wx]);
				//cpn[current.queue[i]] = clique_count[wx];
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

template <typename T, uint BLOCK_DIM_X>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_edge_warp_binary_count(
	uint64* counter,
	graph::COOCSRGraph_d<T> g,
	T kclique,
	T maxDeg,
	const  graph::GraphQueue_d<T, bool>  current,
	T* current_level,
	uint64* cpn,
	T conc_blocks_per_SM,
	T* levelStats,
	T* adj_enc,
	T* adj_tri
)
{
	//will be removed later
	const T gwx = (BLOCK_DIM_X * blockIdx.x + threadIdx.x) / 32;
	constexpr T warpsPerBlock = BLOCK_DIM_X / 32;
	const int wx = threadIdx.x / 32; // which warp in thread block
	const size_t lx = threadIdx.x % 32;
	__shared__ T level_index[warpsPerBlock][7];
	__shared__ T level_count[warpsPerBlock][7];
	__shared__ T level_prev_index[warpsPerBlock][7];

	__shared__ T level_offset[warpsPerBlock];
	__shared__ uint64 clique_count[warpsPerBlock];
	__shared__ T l[warpsPerBlock];
	__shared__ uint32_t  sm_id, levelPtr;
	__shared__ T srcStart[warpsPerBlock], srcLen[warpsPerBlock];
	__shared__ T src2Start[warpsPerBlock], src2Len[warpsPerBlock];

	__shared__ T num_divs[warpsPerBlock], num_divs_local[warpsPerBlock], 
	encode_offset[warpsPerBlock], *encode[warpsPerBlock], tri_offset[warpsPerBlock], 
	*tri[warpsPerBlock], scounter[warpsPerBlock];
	


	//__shared__ T scl[896];

	__syncthreads();

	if (threadIdx.x == 0)
	{
		sm_id = __mysmid();
		T temp = 0;
		while (atomicCAS(&(levelStats[(sm_id * conc_blocks_per_SM) + temp]), 0, 1) != 0)
		{
			temp++;
		}
		levelPtr = temp;
	}
	__syncthreads();

	for (unsigned long long i = gwx; i < (unsigned long long)current.count[0]; i += warpsPerBlock * gridDim.x)
	{
		if (lx == 0)
		{
			T src = g.rowInd[current.queue[i]];
			srcStart[wx] = g.rowPtr[src];
			srcLen[wx] = g.rowPtr[src + 1] - srcStart[wx];
			//printf("src = %u, srcLen = %u\n", src, srcLen);
		}
		else if (lx == 1)
		{
			T src2 = g.colInd[current.queue[i]];
			src2Start[wx] = g.rowPtr[src2];
			src2Len[wx] = g.rowPtr[src2 + 1] - src2Start[wx];
		}
		else if(lx == 2)
		{
			tri_offset[wx] = sm_id * conc_blocks_per_SM * (warpsPerBlock * maxDeg) + levelPtr * (warpsPerBlock * maxDeg);
			tri[wx] = &adj_tri[tri_offset[wx] + wx*maxDeg];
			scounter[wx] = 0;
		}

		// //get tri list: by block :!!
		__syncwarp();
		graph::warp_sorted_count_and_set_tri<WARPS_PER_BLOCK, T>(&g.colInd[srcStart[wx]], srcLen[wx], &g.colInd[src2Start[wx]], src2Len[wx],
			tri[wx], &(scounter[wx]));
		
			__syncwarp();
		T mm = (1 << scounter[wx]) - 1;
		if (lx == 0)
			num_divs_local[wx] = (scounter[wx] + 32 - 1) / 32; // 32 here is for div
		else if (lx == 1)
		{
			num_divs[wx] = (maxDeg + 32 - 1) / 32;
			encode_offset[wx] = sm_id * conc_blocks_per_SM * (warpsPerBlock* maxDeg * num_divs[wx]) + levelPtr * (warpsPerBlock *maxDeg * num_divs[wx]);
			encode[wx] = &adj_enc[encode_offset[wx] + wx * maxDeg * num_divs[wx]];
		}

		if(kclique == 3 && lx == 0)
			atomicAdd(counter, scounter[wx]);

	
		__syncwarp(mm);
		//Encode
		for (unsigned long long j = 0; j < scounter[wx]; j++)
		{
			for (unsigned long long k = lx; k < num_divs_local[wx]; k += 32)
			{
				encode[wx][j * num_divs_local[wx] + k] = 0x00;
			}
			__syncwarp(mm);
			graph::warp_sorted_count_and_encode<WARPS_PER_BLOCK, T, true>(tri[wx], scounter[wx],
				&g.colInd[g.rowPtr[tri[wx][j]]], g.rowPtr[tri[wx][j] + 1] - g.rowPtr[tri[wx][j]],
				 &encode[wx][j * num_divs_local[wx]]);
		}

		__syncwarp(mm); //Done encoding
		level_offset[wx] = sm_id * conc_blocks_per_SM * (warpsPerBlock * num_divs[wx] * 7) + levelPtr * (warpsPerBlock * num_divs[wx] * 7);
		T* cl = &current_level[level_offset[wx] + wx * (num_divs[wx] * 7)];
	
		for (unsigned long long j = 0; j < scounter[wx]; j++)
		{
			if (lx < 7)
			{
				level_count[wx][lx] = 0;
				level_index[wx][lx] = 0;
				level_prev_index[wx][lx] = 0;
			}
			else if (lx == 7 + 1)
			{
				l[wx] = 4;
				clique_count[wx] = 0;
			}
			for (unsigned long long k = lx; k < num_divs_local[wx] * 7; k += 32)
			{
				cl[k] = 0x00;
			}
			//get warp count ??
			uint64 warpCount = 0;
			for (unsigned long long k = lx; k < num_divs_local[wx]; k += 32)
			{
				warpCount += __popc(encode[wx][j * num_divs_local[wx] + k]);
			}
			reduce_part<T>(mm, warpCount);

			if (lx == 0 && l[wx] == kclique)
				clique_count[wx] += warpCount;
			else if (lx == 0 && kclique > 4 && warpCount >= kclique - 3)
			{
				level_count[wx][l[wx] - 4] = warpCount;
				level_index[wx][l[wx] - 4] = 0;
				level_prev_index[wx][l[wx] - 4] = 0;
			}
			__syncwarp(mm);
			while (level_count[wx][l[wx] - 4] > level_index[wx][l[wx] - 4])
			{
				//First Index
				T* from = l[wx] == 4 ? &(encode[wx][num_divs_local[wx] * j]) : &(cl[num_divs_local[wx] * (l[wx] - 4)]);
				T* to = &(cl[num_divs_local[wx] * (l[wx] - 3)]);
				T maskBlock = level_prev_index[wx][l[wx] - 4] / 32;
				T maskIndex = ~((1 << (level_prev_index[wx][l[wx] - 4] & 0x1F)) -1);
				T newIndex = __ffs(from[maskBlock] & maskIndex);
				while(newIndex == 0)
				{
					maskIndex = 0xFFFFFFFF;
					maskBlock++;
					newIndex = __ffs(from[maskBlock] & maskIndex);
				}
				newIndex =  32*maskBlock + newIndex - 1;

				if (lx == 0)
				{
					level_prev_index[wx][l[wx] - 4] = newIndex + 1;
					level_index[wx][l[wx] - 4]++;
				}

				__syncwarp(mm);
				//Intersect
				uint64 warpCount = 0;
				for (T k = lx; k < num_divs_local[wx]; k += 32)
				{
					to[k] = from[k] & encode[wx][newIndex * num_divs_local[wx] + k];
					warpCount += __popc(to[k]);
				}
				reduce_part<T>(mm, warpCount);

				if (lx == 0)
				{
					if (l[wx] + 1 == kclique)
						clique_count[wx] += warpCount;
					else if (l[wx] + 1 < kclique && warpCount >= kclique - l[wx])
					{
						(l[wx])++;
						level_count[wx][l[wx] - 4] = warpCount;
						level_index[wx][l[wx] - 4] = 0;
						level_prev_index[wx][l[wx] - 4] = 0;
					}
				
					//Readjust
					while (l[wx] > 4 && level_index[wx][l[wx] - 4] >= level_count[wx][l[wx] - 4])
					{
						(l[wx])--;
					}
				}
				__syncwarp(mm);
			}
			if (lx == 0)
			{
				atomicAdd(counter, clique_count[wx]);
				//cpn[current.queue[i]] = clique_count[wx];
			}

			__syncwarp();
		}
	}

	__syncthreads();
	if (threadIdx.x == 0)
	{
		atomicCAS(&levelStats[sm_id * conc_blocks_per_SM + levelPtr], 1, 0);
	}
}

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
//__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_edge_block_warp_binary_count_o(
	uint64* counter,
	graph::COOCSRGraph_d<T> g,
	const  graph::GraphQueue_d<T, bool>  current,
	T* current_level,
	uint64* cpn,
	T* levelStats,
	T* adj_enc,
	T* adj_tri
)
{
	//will be removed later
	constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
	const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
	const size_t lx = threadIdx.x % CPARTSIZE;
	__shared__ T level_index[numPartitions][7];
	__shared__ T level_count[numPartitions][7];
	__shared__ T level_prev_index[numPartitions][7];

	__shared__ T level_offset[numPartitions];
	__shared__ uint64 clique_count[numPartitions];
	__shared__ T l[numPartitions], tc, wtc[numPartitions];;
	__shared__ uint32_t  sm_id, levelPtr;
	__shared__ T srcStart, srcLen;
	__shared__ T src2Start, src2Len;

	__shared__ T num_divs_local, encode_offset, *encode, tri_offset, *tri, scounter;
	


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

	for (unsigned long long i = blockIdx.x; i < (unsigned long long)current.count[0]; i += gridDim.x)
	{
		//block things
		if (threadIdx.x == 0)
		{
			T src = g.rowInd[current.queue[i]];
			srcStart = g.rowPtr[src];
			srcLen = g.rowPtr[src + 1] - srcStart;
			//printf("src = %u, srcLen = %u\n", src, srcLen);
		}
		else if (threadIdx.x == 1)
		{
			T src2 = g.colInd[current.queue[i]];
			src2Start = g.rowPtr[src2];
			src2Len = g.rowPtr[src2 + 1] - src2Start;
		}
		else if(threadIdx.x == 2)
		{
			tri_offset = sm_id * CBPSM * (MAXDEG) + levelPtr * (MAXDEG);
			tri = &adj_tri[tri_offset  /*srcStart[wx]*/];
			scounter = 0;
			tc = 0;
		}

		// //get tri list: by block :!!
		__syncthreads();
		graph::block_sorted_count_and_set_tri<BLOCK_DIM_X, T>(&g.colInd[srcStart], srcLen, &g.colInd[src2Start], src2Len,
			tri, &scounter);
		
		__syncthreads();

		if (threadIdx.x == 0)
			num_divs_local = (scounter + 32 - 1) / 32;
		else if (threadIdx.x == 1)
		{
			encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
			encode = &adj_enc[encode_offset  /*srcStart[wx]*/];
		}

		if(KCCOUNT == 3 && threadIdx.x == 0)
			atomicAdd(counter, scounter);

	
		__syncthreads();
		//Encode
		T partMask = (1 << CPARTSIZE) - 1;
		partMask = partMask << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
		T mm = (1 << scounter) - 1;
		mm = mm << ((wx/numPartitions) * CPARTSIZE);
		for (unsigned long long j = wx; j < scounter; j += numPartitions)
		{
			for (unsigned long long k = lx; k < num_divs_local; k += CPARTSIZE)
			{
				encode[j * num_divs_local + k] = 0x00;
			}
			__syncwarp(partMask);
			graph::warp_sorted_count_and_encode<WARPS_PER_BLOCK, T, true, CPARTSIZE>(tri, scounter,
				&g.colInd[g.rowPtr[tri[j]]], g.rowPtr[tri[j] + 1] - g.rowPtr[tri[j]],
				 &encode[j * num_divs_local]);
		}

		__syncthreads(); //Done encoding

		if(lx == 0)
			wtc[wx] = atomicAdd(&(tc), 1);
		__syncwarp(partMask);

		while(wtc[wx] < scounter)
		//for (unsigned long long j = wx; j < scounter; j += numPartitions)
		{
			T j = wtc[wx];
			level_offset[wx] = sm_id * CBPSM * (numPartitions * NUMDIVS * 7) + levelPtr * (numPartitions * NUMDIVS * 7);
			T* cl = &current_level[level_offset[wx] + wx * (NUMDIVS * 7)];
			if (lx < 7)
			{
				level_count[wx][lx] = 0;
				level_index[wx][lx] = 0;
				level_prev_index[wx][lx] = 0;
			}
			if (lx == 0)
			{
				l[wx] = 4;
				clique_count[wx] = 0;
			}

			//get warp count ??
			uint64 warpCount = 0;
			for (unsigned long long k = lx; k < num_divs_local; k += CPARTSIZE)
			{
				warpCount += __popc(encode[j * num_divs_local + k]);
			}
			reduce_part<T>(partMask, warpCount);

			if (lx == 0 && l[wx] == KCCOUNT)
				clique_count[wx] += warpCount;
			else if (lx == 0 && KCCOUNT > 4 && warpCount >= KCCOUNT - 3)
			{
				level_count[wx][l[wx] - 4] = warpCount;
				level_index[wx][l[wx] - 4] = 0;
				level_prev_index[wx][l[wx] - 4] = 0;
			}
		 	__syncwarp(partMask);
			while (level_count[wx][l[wx] - 4] > level_index[wx][l[wx] - 4])
			{
			 	//First Index
				T* from = l[wx] == 4 ? &(encode[num_divs_local * j]) : &(cl[num_divs_local * (l[wx] - 4)]);
				T* to = &(cl[num_divs_local * (l[wx] - 3)]);
				T maskBlock = level_prev_index[wx][l[wx] - 4] / 32;
				T maskIndex = ~((1 << (level_prev_index[wx][l[wx] - 4] & 0x1F)) -1);
				T newIndex = __ffs(from[maskBlock] & maskIndex);
				while(newIndex == 0)
				{
					maskIndex = 0xFFFFFFFF;
					maskBlock++;
					newIndex = __ffs(from[maskBlock] & maskIndex);
				}
				newIndex =  32*maskBlock + newIndex - 1;

				if (lx == 0)
				{
					level_prev_index[wx][l[wx] - 4] = newIndex + 1;
					level_index[wx][l[wx] - 4]++;
				}

			 	//Intersect
				uint64 warpCount = 0;
				for (T k = lx; k < num_divs_local; k += CPARTSIZE)
				{
					to[k] = from[k] & encode[newIndex * num_divs_local + k];
					warpCount += __popc(to[k]);
				}
				reduce_part<T>(partMask, warpCount);
				// warpCount += __shfl_down_sync(partMask, warpCount, 4);
				// warpCount += __shfl_down_sync(partMask, warpCount, 2);
				// warpCount += __shfl_down_sync(partMask, warpCount, 1);

				if (lx == 0)
				{
					if (l[wx] + 1 == KCCOUNT)
						clique_count[wx] += warpCount;
					else if (l[wx] + 1 < KCCOUNT && warpCount >= KCCOUNT - l[wx])
					{
						(l[wx])++;
						level_count[wx][l[wx] - 4] = warpCount;
						level_index[wx][l[wx] - 4] = 0;
						level_prev_index[wx][l[wx] - 4] = 0;
					}
				
					//Readjust
					while (l[wx] > 4 && level_index[wx][l[wx] - 4] >= level_count[wx][l[wx] - 4])
					{
						(l[wx])--;
					}
				}
				__syncwarp(partMask);
			}
			if (lx == 0)
			{
				atomicAdd(counter, clique_count[wx]);
				//cpn[current.queue[i]] = clique_count[wx];
			}

			__syncwarp(partMask);

			if(lx == 0)
			wtc[wx] = atomicAdd(&(tc), 1);
			__syncwarp(partMask);
		}
	}

	__syncthreads();
	if (threadIdx.x == 0)
	{
		atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
	}
}




template<typename T>
struct KCTask
{
	T queue_encode_index;
	T level;
	T level_count;
};


template<typename T, uint CPARTSIZE, uint numPartitions>
__device__ __forceinline__ uint64 explore_branch_o(uint start_level, uint num_divs_local, uint *l, uint partMask,
	T startEncodeIndex, T* encode, T* cl, T* level_count, T* level_index, T *level_prev_index)
{
	const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
	const size_t lx = threadIdx.x % CPARTSIZE;
	uint64 globalCount = 0;
	

	while (level_count[l[wx] - start_level] > level_index[l[wx] - start_level])
	{
		T l_mi_sl = l[wx] - start_level;
		T* from = (l_mi_sl==0) ? &(encode[num_divs_local * startEncodeIndex]) : &(cl[num_divs_local * l_mi_sl]);
		T* to = &(cl[num_divs_local * (l_mi_sl + 1)]);
		T maskBlock = level_prev_index[l_mi_sl] / 32;
		T maskIndex = ~((1 << (level_prev_index[l_mi_sl] & 0x1F)) -1);
		T newIndex = __ffs(from[maskBlock] & maskIndex);
		while(newIndex == 0)
		{
			maskIndex = 0xFFFFFFFF;
			maskBlock++;
			newIndex = __ffs(from[maskBlock] & maskIndex);
		}
		newIndex =  32*maskBlock + newIndex - 1;

		if (lx == 0)
		{
			level_prev_index[l_mi_sl] = newIndex + 1;
			level_index[l_mi_sl]++;
		}

		uint64 warpCount = 0;
		for (T k = lx; k < num_divs_local; k += CPARTSIZE)
		{
			to[k] = from[k] & encode[newIndex * num_divs_local + k];
			warpCount += __popc(to[k]);
		}
		reduce_part<T>(partMask, warpCount);

		if(lx == 0)
		{
			if (l[wx] + 1 == KCCOUNT)
				globalCount += warpCount;
			else if (l[wx] + 1 < KCCOUNT && warpCount >= KCCOUNT - l[wx])
			{
				(l[wx])++;
				level_count[l[wx] - start_level] = warpCount;
				level_index[l[wx] - start_level] = 0;
				level_prev_index[l[wx] - start_level] = 0;
			}

		
			//Readjust
			while (l[wx] > start_level && level_index[l[wx] - start_level] >= level_count[l[wx] - start_level])
			{
				(l[wx])--;
			}
			
		}

		__syncwarp(partMask);

	}

	return globalCount;
}



//very bad: sync_shfl is very low performance !!
template<typename T, uint CPARTSIZE, uint numPartitions>
__device__ __forceinline__ uint64 explore_branch_sync(uint start_level, uint num_divs_local, uint *l, uint partMask,
	T startEncodeIndex, T* encode, T* cl, T* level_count, T* level_index, T *level_prev_index)
{
	const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
	const size_t lx = threadIdx.x % CPARTSIZE;
	uint64 globalCount = 0;

	T ll = l[wx];
	
	

	while (level_count[ll- start_level] > level_index[ll - start_level])
	{
		T l_mi_sl = ll - start_level;
		T* from = (l_mi_sl==0) ? &(encode[num_divs_local * startEncodeIndex]) : &(cl[num_divs_local * l_mi_sl]);
		T* to = &(cl[num_divs_local * (l_mi_sl + 1)]);
		T maskBlock = level_prev_index[l_mi_sl] / 32;
		T maskIndex = ~((1 << (level_prev_index[l_mi_sl] & 0x1F)) -1);
		T newIndex = __ffs(from[maskBlock] & maskIndex);
		while(newIndex == 0)
		{
			maskIndex = 0xFFFFFFFF;
			maskBlock++;
			newIndex = __ffs(from[maskBlock] & maskIndex);
		}
		newIndex =  32*maskBlock + newIndex - 1;

		if (lx == 0)
		{
			level_prev_index[l_mi_sl] = newIndex + 1;
			level_index[l_mi_sl]++;
		}

		uint64 warpCount = 0;
		for (T k = lx; k < num_divs_local; k += CPARTSIZE)
		{
			to[k] = from[k] & encode[newIndex * num_divs_local + k];
			warpCount += __popc(to[k]);
		}
		reduce_part<T>(partMask, warpCount);

		warpCount = __shfl_sync(partMask,  warpCount, 0, CPARTSIZE);
	
		if (ll + 1 == KCCOUNT)
			globalCount += warpCount;
		else if (ll + 1 < KCCOUNT && warpCount >= KCCOUNT - ll)
		{
			(ll)++;
			if(lx == 0)
			{
				level_count[ll - start_level] = warpCount;
				level_index[ll - start_level] = 0;
				level_prev_index[ll - start_level] = 0;
			}
		}

		__syncwarp(partMask);
		
		//Readjust
		while (ll > start_level && level_index[ll - start_level] >= level_count[ll - start_level])
		{
			(ll)--;
		}
	}

	return globalCount;
}


template<typename T, uint CPARTSIZE, uint numPartitions>
__device__ __forceinline__ uint64 explore_branch(KCTask<T> *task, T* queue_encode, uint num_divs_local, uint partMask,
	T* encode, T* cl, T* level_count, T* level_index, T *level_prev_index, int j)
{
	const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
	const size_t lx = threadIdx.x % CPARTSIZE;
	uint64 globalCount = 0;


	__shared__ T ll[numPartitions];

	T l = task->level;
	ll[wx] = task->level;
	level_count[0] = task->level_count;
	level_index[0] = 0;
	level_prev_index[0] = 0;
	__syncwarp(partMask);

	while (level_count[ll[wx]- l] > level_index[ll[wx] - l])
	{
		T l_mi_sl = ll[wx] - l;
		T* from = (l_mi_sl==0) ?  /*&(encode[num_divs_local * j])*/ &(queue_encode[task->queue_encode_index]) : &(cl[num_divs_local * l_mi_sl]);
		T* to = &(cl[num_divs_local * (l_mi_sl + 1)]);
		T maskBlock = level_prev_index[l_mi_sl] / 32;
		T maskIndex = ~((1 << (level_prev_index[l_mi_sl] & 0x1F)) -1);
		T newIndex = __ffs(from[maskBlock] & maskIndex);
		while(newIndex == 0)
		{
			maskIndex = 0xFFFFFFFF;
			maskBlock++;
			newIndex = __ffs(from[maskBlock] & maskIndex);
		}
		newIndex =  32*maskBlock + newIndex - 1;

		if (lx == 0)
		{
			level_prev_index[l_mi_sl] = newIndex + 1;
			level_index[l_mi_sl]++;
		}

		uint64 warpCount = 0;
		for (T k = lx; k < num_divs_local; k += CPARTSIZE)
		{
			to[k] = from[k] & encode[newIndex * num_divs_local + k];
			warpCount += __popc(to[k]);
		}
		reduce_part<T>(partMask, warpCount);

		if(lx == 0)
		{
			if (ll[wx] + 1 == KCCOUNT)
				globalCount += warpCount;
			else if (ll[wx] + 1 < KCCOUNT && warpCount >= KCCOUNT - ll[wx])
			{
				(ll[wx])++;
				level_count[ll[wx] - l] = warpCount;
				level_index[ll[wx] - l] = 0;
				level_prev_index[ll[wx] - l] = 0;
			}

		
			//Readjust
			while (ll[wx] > l && level_index[ll[wx] - l] >= level_count[ll[wx] - l])
			{
				(ll[wx])--;
			}
			
		}

		__syncwarp(partMask);
	}

	return globalCount;
}


template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_edge_block_warp_binary_count(
	uint64* counter,
	graph::COOCSRGraph_d<T> g,
	const  graph::GraphQueue_d<T, bool>  current,
	T* current_level,
	uint64* cpn,
	T* levelStats,
	T* adj_enc,
	T* adj_tri,
	//simt::atomic<KCTask<T>, simt::thread_scope_device> *queue_data
	KCTask<T> *queue_data,
	T* queue_encode
)
{
	__shared__ T queue_count;
	 

	//will be removed later
	constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
	const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
	const size_t lx = threadIdx.x % CPARTSIZE;
	__shared__ T level_index[numPartitions][7];
	__shared__ T level_count[numPartitions][7];
	__shared__ T level_prev_index[numPartitions][7];

	__shared__ T level_offset[numPartitions];
	__shared__ uint64 clique_count[numPartitions];
	__shared__ T l[numPartitions], tc, wtc[numPartitions];
	__shared__ uint32_t  sm_id, levelPtr;
	__shared__ T srcStart, srcLen;
	__shared__ T src2Start, src2Len;

	__shared__ T num_divs_local, encode_offset, *encode, tri_offset, *tri, scounter;


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

	for (unsigned long long i = blockIdx.x; i < (unsigned long long)current.count[0]; i += gridDim.x)
	{
		//block things
		if (threadIdx.x == 0)
		{
			T src = g.rowInd[current.queue[i]];
			srcStart = g.rowPtr[src];
			srcLen = g.rowPtr[src + 1] - srcStart;
			//printf("src = %u, srcLen = %u\n", src, srcLen);
		}
		else if (threadIdx.x == 1)
		{
			T src2 = g.colInd[current.queue[i]];
			src2Start = g.rowPtr[src2];
			src2Len = g.rowPtr[src2 + 1] - src2Start;
		}
		else if(threadIdx.x == 2)
		{
			tri_offset = sm_id * CBPSM * (MAXDEG) + levelPtr * (MAXDEG);
			tri = &adj_tri[tri_offset  /*srcStart[wx]*/];
			scounter = 0;
			tc = 0;
		}
		else if(threadIdx.x == 3)
		{
			queue_count = 0;
		}

		// //get tri list: by block :!!
		__syncthreads();
		graph::block_sorted_count_and_set_tri<BLOCK_DIM_X, T>(&g.colInd[srcStart], srcLen, &g.colInd[src2Start], src2Len,
			tri, &scounter);
		
		__syncthreads();

		if (threadIdx.x == 0)
			num_divs_local = (scounter + 32 - 1) / 32;
		else if (threadIdx.x == 1)
		{
			encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
			encode = &adj_enc[encode_offset  /*srcStart[wx]*/];
		}

		if(KCCOUNT == 3 && threadIdx.x == 0)
			atomicAdd(counter, scounter);

	
		__syncthreads();
		//Encode
		T partMask = (1 << CPARTSIZE) - 1;
		partMask = partMask << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
		for (unsigned long long j = wx; j < scounter; j += numPartitions)
		{
		
			for (unsigned long long k = lx; k < num_divs_local; k += CPARTSIZE)
			{
				encode[j * num_divs_local + k] = 0x00;
			}
			__syncwarp(partMask);
			graph::warp_sorted_count_and_encode<WARPS_PER_BLOCK, T, true, CPARTSIZE>(tri, scounter,
				&g.colInd[g.rowPtr[tri[j]]], g.rowPtr[tri[j] + 1] - g.rowPtr[tri[j]],
				 &encode[j * num_divs_local]);
		}

		__syncthreads(); //Done encoding
		level_offset[wx] = sm_id * CBPSM * (numPartitions * NUMDIVS * 7) + levelPtr * (numPartitions * NUMDIVS * 7);
		T* cl = &current_level[level_offset[wx] + wx * (NUMDIVS * 7)];
		KCTask<T>* queue_data_block = &(queue_data[sm_id * CBPSM * (QUEUE_SIZE) + levelPtr * (QUEUE_SIZE)]);
		T* queue_encode_block = &(queue_encode[sm_id * CBPSM * (QUEUE_SIZE*NUMDIVS) + levelPtr * (QUEUE_SIZE*NUMDIVS)]);
		// if(lx == 0)
		// 	wtc[wx] = atomicAdd(&(tc), 1);
		// __syncwarp(partMask);

		//while(wtc[wx] < scounter)
		for (unsigned long long j = wx; j < scounter; j += numPartitions)
		{
			//T j = wtc[wx];
			if (lx < 7)
			{
				level_count[wx][lx] = 0;
				level_index[wx][lx] = 0;
				level_prev_index[wx][lx] = 0;
			}
			if (lx == 0)
			{
				l[wx] = 4;
				clique_count[wx] = 0;
			}

			//get warp count ??
			uint64 warpCount = 0;
			for (unsigned long long k = lx; k < num_divs_local; k += CPARTSIZE)
			{
				warpCount += __popc(encode[j * num_divs_local + k]);
			}
			reduce_part<T>(partMask, warpCount);

			if (lx == 0 && l[wx] == KCCOUNT)
				atomicAdd(counter, warpCount);
			else if (lx == 0 && KCCOUNT > 4 && warpCount >= KCCOUNT - 3)
			{
				level_count[wx][l[wx] - 4] = warpCount;
				level_index[wx][l[wx] - 4] = 0;
				level_prev_index[wx][l[wx] - 4] = 0;
			}
			__syncwarp(partMask);
			// uint64 wc = explore_branch_o<T, CPARTSIZE, numPartitions>(4, num_divs_local, l, partMask,
			// 		j, encode, cl, level_count[wx], level_index[wx], level_prev_index[wx]);

			// if(lx == 0 && KCCOUNT > 4 && warpCount >= KCCOUNT - 3)
			// 	atomicAdd(counter, wc); //clique_count[wx] = wc;
			while (level_count[wx][l[wx] - 4] > level_index[wx][l[wx] - 4])
			{
			 	//First Index
				T* from = l[wx] == 4 ? &(encode[num_divs_local * j]) : &(cl[num_divs_local * (l[wx] - 4)]);
				T* to = &(cl[num_divs_local * (l[wx] - 3)]);
				T maskBlock = level_prev_index[wx][l[wx] - 4] / 32;
				T maskIndex = ~((1 << (level_prev_index[wx][l[wx] - 4] & 0x1F)) -1);
				T newIndex = __ffs(from[maskBlock] & maskIndex);
				while(newIndex == 0)
				{
					maskIndex = 0xFFFFFFFF;
					maskBlock++;
					newIndex = __ffs(from[maskBlock] & maskIndex);
				}
				newIndex =  32*maskBlock + newIndex - 1;

				if (lx == 0)
				{
					level_prev_index[wx][l[wx] - 4] = newIndex + 1;
					level_index[wx][l[wx] - 4]++;
				}

			 	//Intersect
				uint64 warpCount = 0;
				for (T k = lx; k < num_divs_local; k += CPARTSIZE)
				{
					to[k] = from[k] & encode[newIndex * num_divs_local + k];
					warpCount += __popc(to[k]);
				}
				reduce_part<T>(partMask, warpCount);

				if (lx == 0)
				{
					if (l[wx] + 1 == KCCOUNT)
						clique_count[wx] = 1; //+= warpCount;
					else if (l[wx] + 1 < KCCOUNT && warpCount >= KCCOUNT - l[wx])
					{
						(l[wx])++;
						level_count[wx][l[wx] - 4] = warpCount;
						level_index[wx][l[wx] - 4] = 0;
						level_prev_index[wx][l[wx] - 4] = 0;

						// if(warpCount > 100)
						// {
						// 	if(queue_count < QUEUE_SIZE)
						// 	{
						// 		T index = atomicAdd(&queue_count, 1);
						// 		if(index < QUEUE_SIZE)
						// 		{
						// 			level_count[wx][l[wx] - 4] = 0;

						// 			queue_data_block[index].level = l[wx];
						// 			queue_data_block[index].level_count = warpCount;
						// 			queue_data_block[index].queue_encode_index = index * num_divs_local;
						// 			for (unsigned long long k = 0; k < num_divs_local; k++)
						// 			{
						// 				queue_encode_block[queue_data_block[index].queue_encode_index + k] = to[k];
						// 			}
						// 		}
						// 	}
		
						// }
					}
				
					//Readjust
					while (l[wx] > 4 && level_index[wx][l[wx] - 4] >= level_count[wx][l[wx] - 4])
					{
						(l[wx])--;
					}
				}
				__syncwarp(partMask);
			}
			if (lx == 0)
			{
				atomicAdd(counter, clique_count[wx] == 1? 0: 1);
				//cpn[current.queue[i]] = clique_count[wx];
			}

			__syncwarp(partMask);

			// if(lx == 0)
			// 	wtc[wx] = atomicAdd(&(tc), 1);
			// __syncwarp(partMask);
		}
		__syncthreads();
		//dequeue simply

		T limit = (queue_count > QUEUE_SIZE) ? QUEUE_SIZE: queue_count;
		for (unsigned long long j = wx; j < limit; j += numPartitions)
		{
			KCTask<T> t = queue_data_block[j];
			uint64 wc = explore_branch<T, CPARTSIZE, numPartitions>(&t, queue_encode_block, num_divs_local, partMask,
				encode, cl, level_count[wx], level_index[wx], level_prev_index[wx], j);
			if(lx == 0)
				atomicAdd(counter, wc); //clique_count[wx] = wc;
		}

		__syncthreads();
	}


	//search in the queue

	__syncthreads();
	if (threadIdx.x == 0)
	{
		atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
	}
}


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
			reduce_part<T>(partMask[wx], warpCount);

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
		reduce_part<T>(partMask[wx], warpCount);
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

			if(rsize[l-1] - drop[l-1] > KCCOUNT)
			{	
				__syncthreads();
				//printf("Stop Here, %u %u\n", rsize[l-1], drop[l-1]);
				if(threadIdx.x == 0)
				{
					T c = rsize[l-1] - KCCOUNT;
					unsigned long long ncr = nCR[ drop[l-1] * 401 + c  ];
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
						reduce_part<T>(partMask[wx], warpCount);
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
					reduce_part<T>(partMask[wx], warpCount);

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
	__shared__ uint64 maxIntersection;
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
		
		// if(KCCOUNT == 3 && threadIdx.x == 0)
		// 	atomicAdd(counter, scounter);

	
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
			uint64 warpCount = 0;
			for (T k = lx; k < num_divs_local; k += CPARTSIZE)
			{
				warpCount += __popc(encode[j * num_divs_local + k]);
			}
			reduce_part<T>(partMask[wx], warpCount);

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
		reduce_part<T>(partMask[wx], warpCount);
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

			if(rsize[l-2] - drop[l-2] > KCCOUNT)
			{	
				__syncthreads();
				//printf("Stop Here, %u %u\n", rsize[l-1], drop[l-1]);
				if(threadIdx.x == 0)
				{
					T c = rsize[l-2] - KCCOUNT;
					unsigned long long ncr = nCR[ drop[l-2] * 401 + c  ];
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
					uint64 warpCount = 0;
					T bi = j / 32;
					T ii = j & 0x1F;
					if( (to[bi] & (1<<ii)) != 0)
					{
						for (T k = lx; k < num_divs_local; k += CPARTSIZE)
						{
							warpCount += __popc(to[k] & encode[j * num_divs_local + k]);
						}
						reduce_part<T>(partMask[wx], warpCount);
						if(lx == 0 && maxCount[wx] == scounter + 1)
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
						if(rsize[l-2] >= KCCOUNT)
						{
							T c = rsize[l-2] - KCCOUNT;
							unsigned long long ncr = nCR[ drop[l-2] * 401 + c  ];
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

					uint64 warpCount = 0;
					for (T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
					{
						T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
						pl[(l-2)*num_divs_local + j] = ~(encode[level_pivot[l - 2] * num_divs_local + j]) & to[j] & m;
						warpCount += __popc(pl[(l-2)*num_divs_local + j]);
					}
					reduce_part<T>(partMask[wx], warpCount);

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
	if (threadIdx.x == 0)
	{
		atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
	}
}



