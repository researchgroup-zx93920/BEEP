

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


template <typename T, int BLOCK_DIM_X>
__global__ void
kernel_block_level_kclique_count0(
	uint64* counter,
	graph::COOCSRGraph_d<T> g,
	T kclique,
	T level,
	graph::GraphQueue_d<T, bool> current,
	char* current_level,
	T* filter_level,
	T* filter_scan
)
{
	//CUB reduce
	typedef cub::BlockScan<T, BLOCK_DIM_X> BlockScan;
	__shared__ typename BlockScan::TempStorage temp_storage;
	T threadData = 0;
	T aggreagtedData = 0;
	T accumAggData = 0;

	auto tid = threadIdx.x;
	const size_t gbx = (BLOCK_DIM_X * blockIdx.x + threadIdx.x) / BLOCK_DIM_X;

	__shared__ T level_index[BLOCK_DIM_X];
	__shared__ T level_count[BLOCK_DIM_X];

	__shared__ T current_node_index;
	__shared__ uint64 clique_count;
	__shared__ T l;
	__shared__ T new_level;

	__shared__ T block_count_shared;

	if (tid == 0)
		clique_count = 0;

	__syncthreads();

	for (size_t i = gbx; i < current.count[0]; i += gridDim.x)
	{
		const T nodeId = current.queue[i];

		const T srcStart = g.rowPtr[nodeId];
		const T srcStop = g.rowPtr[nodeId + 1];
		const T srcLen = srcStop - srcStart;
		const T srcLenBlocks = (srcLen + BLOCK_DIM_X - 1) / BLOCK_DIM_X;

		level_index[tid] = 0;
		level_count[tid] = 0;
		__syncthreads();

		if (tid == 0)
		{
			l = 2;
			level_count[l - 2] = srcLen;
			current_node_index = UINT_MAX;
		}
		__syncthreads();

		while (level_count[l - 2] > level_index[l - 2])
		{
			/*if (tid == 0)
			{
				printf("Level = %u, Index = %u, %u\n", l, level_index[l - 2], level_count[l - 2]);
			}*/

			//(1) Randomly Select an element from current level: Later
			/*for (int k = 0; k < srcLenBlocks; k++)
			{
				T index = srcStart + k * BLOCK_DIM_X + tid;
				if (index < srcStop)
				{
					T e = current_level[index];
					if (e == l)
					{
						atomicCAS(&current_node_index, UINT_MAX, index);
					}
				}

				__syncthreads();
				if (current_node_index != UINT_MAX)
					break;

			}*/

			//(2) Filter elemenets of current level : This step is over-kill, we might remove it
			aggreagtedData = 0;
			accumAggData = 0;
			for (int k = 0; k < srcLenBlocks; k++)
			{
				T index = srcStart + k * BLOCK_DIM_X + tid;
				threadData = 0;
				aggreagtedData = 0;
				if (index < srcStop && current_level[index] == l)
				{
					threadData += 1;
				}

				__syncthreads();
				BlockScan(temp_storage).ExclusiveSum(threadData, threadData, aggreagtedData);
				__syncthreads();

				if (index < srcStop)
				{
					if (current_level[index] == l)
					{

						//printf("%u, %u, %u\n", index, threadData, g.colInd[index]);

						filter_level[srcStart + accumAggData + threadData] = g.colInd[index];
						filter_scan[srcStart + accumAggData + threadData] = index;
					}
				}

				accumAggData += aggreagtedData;
				__syncthreads();
			}

			__syncthreads();
			if (tid == 0)
			{
				current_node_index = filter_level[srcStart + level_index[l - 2]];
				level_index[l - 2]++;
				new_level = l;
			}
			__syncthreads();

			uint64 blockCount = 0;

			//(3) intesect Adj[current_node] with
			const T filter_srcStop = srcStart + accumAggData; //only filtered
			const T filter_srcLen = accumAggData;

			const T dst = current_node_index;
			const T dstStart = g.rowPtr[current_node_index];
			const T dstStop = g.rowPtr[current_node_index + 1];
			const T dstLen = dstStop - dstStart;

			/*if (tid == 0)
			{
				printf("Adj of %u is: ", dst);
				for (int tt = dstStart; tt < dstStop; tt++)
				{
					printf("%u, ", g.colInd[tt]);
				}
				printf("\n");
			}*/

			if (dstLen >= kclique - l)
			{
				if (dstLen > filter_srcLen)
				{
					blockCount += graph::block_sorted_count_and_set_binary<BLOCK_DIM_X, T, true>(&filter_level[srcStart], filter_srcLen,
						&g.colInd[dstStart], dstLen, true, srcStart, srcStop, current_level, filter_scan, l + 1, kclique);
				}
				else {
					blockCount += graph::block_sorted_count_and_set_binary<BLOCK_DIM_X, T, true>(&g.colInd[dstStart], dstLen,
						&filter_level[srcStart], filter_srcLen, false, srcStart, srcStop, current_level, filter_scan, l + 1, kclique);
				}

				__syncthreads();
				if (tid == 0)
				{

					//1
					if (blockCount > 0 && l + 1 == kclique)
						clique_count += blockCount;
					else if (blockCount > 0 && l + 1 < kclique)
					{
						l++;
						new_level++;
						level_count[l - 2] = blockCount;
						level_index[l - 2] = 0;
					}
				}

			}
			if (tid == 0)
			{

				while (new_level > 2 && level_index[new_level - 2] >= level_count[new_level - 2])
				{
					new_level--;
				}
			}

			__syncthreads();
			if (new_level < l)
			{
				for (int k = 0; k < srcLenBlocks; k++)
				{
					T index = srcStart + k * BLOCK_DIM_X + tid;
					if (index < srcStop && current_level[index] > new_level)
						current_level[index] = new_level;
				}
			}

			__syncthreads();


			if (tid == 0)
				l = new_level;
			__syncthreads();
			/*if (tid == 0)
			{
				printf("Chosen Node Index = %u, Agg = %u of %u: TC Count = %u, level = %u\n", current_node_index, aggreagtedData, srcLen, blockCount, l);
				printf("Now Print all current_level:\n");
				for (int ii = 0; ii < srcLen; ii++)
					printf("%u, ", current_level[srcStart + ii]);
				printf("\n");
			}*/

		}

		if (threadIdx.x == 0)
			atomicAdd(counter, clique_count);

	}
}

template <typename T, uint BLOCK_DIM_X>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kernel_warp_level_kclique_count(
	uint64* counter,
	graph::COOCSRGraph_d<T> g,
	T kclique,
	T level,
	graph::GraphQueue_d<T, bool> current,
	char* current_level,
	uint64* cpn
)
{
	constexpr T warpsPerBlock = BLOCK_DIM_X / 32;
	const int wx = threadIdx.x / 32; // which warp in thread block
	const size_t lx = threadIdx.x % 32;
	const T gwx = (BLOCK_DIM_X * blockIdx.x + threadIdx.x) / 32;

	__shared__ T level_index_all[warpsPerBlock][5];
	__shared__ T level_count_all[warpsPerBlock][5];
	__shared__ T level_prev_index_all[warpsPerBlock][5];
	//__shared__ char current_level_s_all[warpsPerBlock][32 * 8];
	__shared__ T current_node_index_all[warpsPerBlock];
	__shared__ uint64 clique_count_all[warpsPerBlock];
	__shared__ char l_all[warpsPerBlock];
	__shared__ char new_level_all[warpsPerBlock];

	T* level_index = &level_index_all[wx][0];
	T* level_count = &level_count_all[wx][0];
	T* level_prev_index = &level_prev_index_all[wx][0];
	//char* current_level_s = &current_level_s_all[wx][0];
	T* current_node_index = &current_node_index_all[wx];
	uint64* clique_count = &clique_count_all[wx];
	char* l = &l_all[wx];
	char* new_level = &new_level_all[wx];
	__syncthreads();

	for (unsigned long long i = gwx; i < (unsigned long long)current.count[0]; i += warpsPerBlock * gridDim.x) //warp per node
	{
		const T nodeId = current.queue[i];
		const T srcStart = g.rowPtr[nodeId];
		const T srcStop = g.rowPtr[nodeId + 1];
		const T srcLen = srcStop - srcStart;
		const T srcLenBlocks = (srcLen + 32 - 1) / 32;

		if (lx < 5)
		{
			level_index[lx] = 0;
			level_prev_index[lx] = 0;
		}

		char* cl = &current_level[srcStart];
		// if (srcLen <= 8 * 32)
		// {
		// 	cl = current_level_s;
		// 	current_level_s[lx] = 2;
		// 	current_level_s[lx + 32] = 2;
		// 	current_level_s[lx + 32 * 2] = 2;
		// 	current_level_s[lx + 32 * 3] = 2;
		// 	current_level_s[lx + 32 * 4] = 2;
		// 	current_level_s[lx + 32 * 5] = 2;
		// 	current_level_s[lx + 32 * 6] = 2;
		// 	current_level_s[lx + 32 * 7] = 2;
		// }

		if (lx == 0)
		{
			l[0] = 2;
			level_count[l[0] - 2] = srcLen;
			current_node_index[0] = UINT_MAX;
			clique_count[0] = 0;
		}
		__syncwarp();

		while (level_count[l[0] - 2] > level_index[l[0] - 2])
		{
			for (T k = 0; k < srcLenBlocks; k++)
			{
				T index = level_prev_index[l[0] - 2] + k * 32 + lx;
				if (index < srcLen && cl[index] == l[0])
				{
					atomicMin(current_node_index, index);
				}

				__syncwarp();
				if (current_node_index[0] != UINT_MAX)
					break;

				__syncwarp();
			}

			if (lx == 0)
			{
				//current_node_index[0] = finalIndex;
				level_prev_index[l[0] - 2] = current_node_index[0] + 1;
				level_index[l[0] - 2]++;
				new_level[0] = l[0];
			}

			__syncwarp();

			uint64 warpCount = 0;
			const T dst = g.colInd[current_node_index[0] + srcStart];
			const T dstStart = g.rowPtr[dst];
			const T dstStop = g.rowPtr[dst + 1];
			const T dstLen = dstStop - dstStart;

			if (dstLen >= kclique - l[0])
			{
				if (dstLen > srcLen)
				{
					warpCount = graph::warp_sorted_count_and_set_binary<WARPS_PER_BLOCK, T, true>(0, &g.colInd[srcStart], srcLen,
						&g.colInd[dstStart], dstLen, true, srcStart, cl, l[0] + 1, kclique);
				}
				else {
					warpCount = graph::warp_sorted_count_and_set_binary<WARPS_PER_BLOCK, T, true>(0, &g.colInd[dstStart], dstLen,
						&g.colInd[srcStart], srcLen, false, srcStart, cl, l[0] + 1, kclique);
				}

				if (lx == 0 && warpCount > 0)
				{
					if (l[0] + 1 == kclique)
						clique_count[0] += warpCount;
					else if (l[0] + 1 < kclique)
					{
						(l[0])++;
						(new_level[0])++;
						level_count[l[0] - 2] = warpCount;
						level_index[l[0] - 2] = 0;
						level_prev_index[l[0] - 2] = 0;
					}

				}

				__syncwarp();
			}


			if (lx == 0)
			{
				while (new_level[0] > 2 && level_index[new_level[0] - 2] >= level_count[new_level[0] - 2])
				{
					(new_level[0])--;
				}
			}

			__syncwarp();
			if (new_level[0] < l[0])
			{
				for (auto k = 0; k < srcLenBlocks; k++)
				{
					T index = k * 32 + lx;
					if (index < srcLen && cl[index] > new_level[0])
						cl[index] = new_level[0];
				}
			}

			__syncwarp();

			if (lx == 0)
			{
				l[0] = new_level[0];
				current_node_index[0] = UINT_MAX;
			}
			__syncwarp();



		}

		if (lx == 0)
		{
			atomicAdd(counter, clique_count[0]);
			//cpn[nodeId] = clique_count;
		}
	}
}


template <typename T, uint BLOCK_DIM_X>
__launch_bounds__(BLOCK_DIM_X)
__global__ void
kernel_block_level_kclique_count2(
	uint64* counter,
	graph::COOCSRGraph_d<T> g,
	T kclique,
	T level,
	graph::GraphQueue_d<T, bool> current,
	char* current_level,
	T* level_filter,
	uint64* cpn
)
{
	auto tid = threadIdx.x;
	const T gbx = blockIdx.x;

	__shared__ T level_index[10];
	__shared__ T level_count[10];
	__shared__ T level_prev_index[10];

	//__shared__ char current_level_s[BLOCK_DIM_X * 8];

	__shared__ T current_node_index;
	__shared__ uint64 clique_count;
	__shared__ char l;
	__shared__ char new_level;
	bool reset_l2 = true;

	__syncthreads();

	for (unsigned long long i = gbx; i < (unsigned long long)current.count[0]; i += gridDim.x)
	{
		const T nodeId = current.queue[i];
		const T srcStart = g.rowPtr[nodeId];
		const T srcStopOriginal = g.rowPtr[nodeId + 1];

		T srcStop = srcStopOriginal;
		T srcLen = srcStop - srcStart;
		T srcLenBlocks = (srcLen + BLOCK_DIM_X - 1) / BLOCK_DIM_X;

		if (tid < 10)
		{
			level_index[tid] = 0;
			level_prev_index[tid] = 0;
		}

		char* cl = &current_level[srcStart];
		__syncthreads();

		if (tid == 0)
		{
			l = 2;
			level_count[l - 2] = srcLen;
			current_node_index = UINT_MAX;
			clique_count = 0;
		}
		__syncthreads();

		while (level_count[l - 2] > level_index[l - 2])
		{
			if (l == 2 && reset_l2)
			{
				reset_l2 = false;
				srcStop = srcStopOriginal;
				srcLen = srcStop - srcStart;
				srcLenBlocks = (srcLen + BLOCK_DIM_X - 1) / BLOCK_DIM_X;
				for (T w = tid; w < srcLen; w += BLOCK_DIM_X)
				{
					level_filter[srcStart + w] = g.colInd[srcStart + w];
					cl[w] = 2;
				}
			}
			else if (l == 3 && level_index[l - 2] == 0 && (level_count[l - 2] < srcLen / 2))
			{
				// //push data to level_filter: Use cub :(
				typedef cub::BlockScan<T, BLOCK_DIM_X> BlockScan;
				__shared__ typename BlockScan::TempStorage temp_storage;
				T aggreagtedData = 0;
				T accumAggData = 0;
				T threadData = 0;
				for (T k = 0; k < srcLenBlocks; k++)
				{
					T index = k * BLOCK_DIM_X + tid;
					threadData = 0;
					aggreagtedData = 0;
					T prev = 0;
					if (index < srcLen && cl[index] == l)
					{
						threadData = 1;
						prev = level_filter[srcStart + index];
					}

					//__syncthreads();
					BlockScan(temp_storage).ExclusiveSum(threadData, threadData, aggreagtedData);
					//__syncthreads();

					if (index < srcLen && cl[index] == l)
					{
						level_filter[srcStart + accumAggData + threadData] = prev; //g.colInd[srcStart + index];
					}

					accumAggData += aggreagtedData;
					__syncthreads();
				}

				for (T w = tid; w < level_count[l - 2]; w += BLOCK_DIM_X)
					cl[w] = 3;

				reset_l2 = true;
				srcLen = level_count[l - 2];
				srcLenBlocks = (srcLen + BLOCK_DIM_X - 1) / BLOCK_DIM_X;
			}

			__syncthreads();
			//Finding next index in the level
			for (T k = 0; k < srcLenBlocks; k++)
			{
				T index = level_prev_index[l - 2] + k * BLOCK_DIM_X + tid;
				if (index < srcLen && cl[index] == l)
				{
					atomicMin(&current_node_index, index);
				}

				__syncthreads();
				if (current_node_index != UINT_MAX)
					break;

				__syncthreads();
			}
			if (tid == 0)
			{

				level_prev_index[l - 2] = current_node_index + 1;
				level_index[l - 2]++;
				new_level = l;
			}
			__syncthreads();

			//2) Intersect
			uint64 blockCount = 0;
			const T dst = level_filter[current_node_index + srcStart];
			const T dstStart = g.rowPtr[dst];
			const T dstStop = g.rowPtr[dst + 1];
			const T dstLen = dstStop - dstStart;
			if (dstLen >= kclique - l)
			{
				//bool bequal = (srcLen/BLOCK_DIM_X) == (dstLen/BLOCK_DIM_X);
				if ( /*(bequal && srcLen >= dstLen) || (!bequal &&*/ dstLen > srcLen/*)*/)
				{
					blockCount = graph::block_sorted_count_and_set_binary2<BLOCK_DIM_X, T, true>(&level_filter[srcStart], srcLen,
						&g.colInd[dstStart], dstLen, true, srcStart, cl, l + 1, kclique);
				}
				else {
					blockCount = graph::block_sorted_count_and_set_binary2<BLOCK_DIM_X, T, true>(&g.colInd[dstStart], dstLen,
						&level_filter[srcStart], srcLen, false, srcStart, cl, l + 1, kclique);
				}

				__syncthreads();
				//3) Decide whether to count or go deeper
				if (tid == 0 && blockCount > 0)
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

			//Check if we are done with the current level
			__syncthreads();
			//if(level_index[new_level - 2] != 0) not useful
			//{
			if (tid == 0)
			{
				while (new_level > 2 && level_index[new_level - 2] >= level_count[new_level - 2])
				{
					new_level--;
				}
			}

			//If yes, go back
			__syncthreads();
			if (new_level < l)
			{
				for (auto k = 0; k < srcLenBlocks; k++)
				{
					T index = k * BLOCK_DIM_X + tid;
					if (index < srcLen && cl[index] > new_level)
						cl[index] = new_level;
				}
			}

			__syncthreads();
			//}

			if (tid == 0)
			{
				l = new_level;
				current_node_index = UINT_MAX;
			}
			__syncthreads();
		}

		if (threadIdx.x == 0)
		{
			atomicAdd(counter, clique_count);
			//cpn[nodeId] = clique_count;
		}
	}
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

