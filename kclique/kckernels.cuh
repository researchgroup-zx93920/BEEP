

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
	auto tid = threadIdx.x;
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

				__syncwarp();
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
			}

			__syncwarp();
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
