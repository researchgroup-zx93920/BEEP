
#pragma once


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

template<class _A, class _B, class _C> static inline __device__ void __simt_fetch_min_relaxed_32_block(_A _ptr, _B& _dst, _C _op)
{
	asm volatile("atom.min.relaxed.cta.u32 %0,[%1],%2;" : "=r"(_dst) : "l"(_ptr), "r"(_op) : "memory");
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


template <typename T, int BLOCK_DIM_X>
__global__ void init_edge_ptr(graph::COOCSRGraph_d<T> g, T* edgePtr)
{
	const T gtid = (BLOCK_DIM_X * blockIdx.x + threadIdx.x);


	if (gtid < g.numEdges)
	{
		T src = g.rowInd[gtid];
		T dst = g.colInd[gtid];

		T srcDeg = g.rowPtr[src + 1] - g.rowPtr[src];
		T dstDeg = g.rowPtr[dst + 1] - g.rowPtr[dst];

		edgePtr[gtid] = srcDeg < dstDeg ? srcDeg : dstDeg;
	}

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



template <typename T, int BLOCK_DIM_X>
__global__ void
kernel_block_level_kclique_count1(
	uint64* counter,
	graph::COOCSRGraph_d<T> g,
	int kclique,
	int level,
	bool* processed,
	graph::GraphQueue_d<int, bool> current,
	T* a,
	T* b,
	T* c
)
{

	//CUB reduce
	typedef cub::BlockScan<T, BLOCK_DIM_X> BlockScan;
	__shared__ typename BlockScan::TempStorage temp_storage;


	__shared__ uint64 clique_count;
	__shared__ uint64 block_count_shared;

	__shared__ T current_level_s[BLOCK_DIM_X * 2];
	__shared__ T filter_level_s[BLOCK_DIM_X * 2];
	__shared__ T filter_scan_s[BLOCK_DIM_X * 2];



	auto tid = threadIdx.x;
	const size_t gbx = (BLOCK_DIM_X * blockIdx.x + threadIdx.x) / BLOCK_DIM_X;

	T level_index[6];
	T level_count[6];
	T threadData = 0;
	T aggreagtedData = 0;
	T accumAggData = 0;
	T l;
	T new_level;
	T current_node_index;

	clique_count = 0;
	for (size_t i = gbx; i < current.count[0]; i += gridDim.x)
	{
		const T nodeId = current.queue[i];
		const T srcStart = g.rowPtr[nodeId];
		const T srcStop = g.rowPtr[nodeId + 1];
		const T srcLen = srcStop - srcStart;
		const T srcLenBlocks = (srcLen + BLOCK_DIM_X - 1) / BLOCK_DIM_X;

		l = 2;
		level_index[l - 2] = 0;
		level_count[l - 2] = srcLen;
		current_node_index = UINT_MAX;



		current_level_s[tid] = 2;
		current_level_s[tid + BLOCK_DIM_X] = 2;

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
				T index = /*srcStart + */ k * BLOCK_DIM_X + tid;
				threadData = 0;
				aggreagtedData = 0;
				if (index < srcLen/*srcStop*/ && current_level_s[index] == l)
				{
					threadData += 1;
				}

				BlockScan(temp_storage).ExclusiveSum(threadData, threadData, aggreagtedData);
				__syncthreads();

				if (index < srcLen /*srcStop*/ && current_level_s[index] == l)
				{
					//printf("%u, %u, %u\n", index, threadData, g.colInd[index]);

					filter_level_s[/*srcStart + */accumAggData + threadData] = g.colInd[index + srcStart];
					filter_scan_s[/*srcStart + */accumAggData + threadData] = index;
				}

				accumAggData += aggreagtedData;
				__syncthreads();

			}

			current_node_index = filter_level_s[/*srcStart +*/ level_index[l - 2]];
			level_index[l - 2]++;
			new_level = l;

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
					blockCount += graph::block_sorted_count_and_set_binary<BLOCK_DIM_X, T, true>(&filter_level_s[0/*srcStart*/], filter_srcLen,
						&g.colInd[dstStart], dstLen, true, 0, current_level_s, filter_scan_s, l + 1, kclique);
				}
				else {
					blockCount += graph::block_sorted_count_and_set_binary<BLOCK_DIM_X, T, true>(&g.colInd[dstStart], dstLen,
						&filter_level_s[0/*srcStart*/], filter_srcLen, false, 0, current_level_s, filter_scan_s, l + 1, kclique);
				}

				__syncthreads();
				if (tid == 0)
				{
					block_count_shared = blockCount;
					if (block_count_shared > 0 && l + 1 == kclique)
						clique_count += block_count_shared;
				}
				__syncthreads();

				if (block_count_shared > 0 && l + 1 < kclique)
				{
					l++;
					new_level++;
					level_count[l - 2] = block_count_shared;
					level_index[l - 2] = 0;
				}
			}



			while (new_level > 2 && level_index[new_level - 2] >= level_count[new_level - 2])
			{
				new_level--;
			}

			if (new_level < l)
			{
				for (int k = 0; k < srcLenBlocks; k++)
				{
					T index = /*srcStart +*/ k * BLOCK_DIM_X + tid;
					if (index < srcLen /*srcStop*/ && current_level_s[index] > new_level)
						current_level_s[index] = new_level;
				}

				l = new_level;
			}

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
__launch_bounds__(BLOCK_DIM_X)
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
	const int warpIdx = threadIdx.x / 32; // which warp in thread block
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

	T* level_index = &level_index_all[warpIdx][0];
	T* level_count = &level_count_all[warpIdx][0];
	T* level_prev_index = &level_prev_index_all[warpIdx][0];
	//char* current_level_s = &current_level_s_all[warpIdx][0];
	T* current_node_index = &current_node_index_all[warpIdx];
	uint64* clique_count = &clique_count_all[warpIdx];
	char* l = &l_all[warpIdx];
	char* new_level = &new_level_all[warpIdx];
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


			// for (T k = 0; k < srcLenBlocks; k++)
			// {
			// 	T index = level_prev_index[l[0] - 2] + k * 32 + lx;
			// 	if (index < srcLen && cl[index] == l[0])
			// 	{
			// 		atomicMin(current_node_index, index);
			// 	}

			// 	__syncwarp();
			// 	if (current_node_index[0] != UINT_MAX)
			// 		break;

			// 	__syncwarp();
			// }


			//Warp shuffle
			for (T k = 0; k < srcLenBlocks; k++)
			{
				T index = level_prev_index[l[0] - 2] + k * 32 + lx;
				int condition = index < srcLen&& cl[index] == l[0];
				unsigned int newmask = __ballot_sync(0xFFFFFFFF, condition);
				if (newmask > 0)
				{
					int elected_lane_deq = __ffs(newmask) - 1;
					if (lx == elected_lane_deq)
					{
						current_node_index[0] = index;
					}
					break;
				}
			}

			// for (int offset = 16; offset > 0; offset /= 2)
			// {
			// 	T a = __shfl_down_sync(0xFFFFFFFF, finalIndex, offset, 32);
			// 	finalIndex = finalIndex < a ? finalIndex: a;
			// }
			// finalIndex = __shfl_sync(0xFFFFFFFF, finalIndex, 0, 32);

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
__launch_bounds__(BLOCK_DIM_X, 1)
__global__ void
kernel_warp_sync_level_kclique_count(
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
	const int warpIdx = threadIdx.x / 32; // which warp in thread block
	const size_t lx = threadIdx.x % 32;
	const T gwx = (BLOCK_DIM_X * blockIdx.x + threadIdx.x) / 32;

	//__shared__ T level_index_all[warpsPerBlock][5];
	//__shared__ T level_count_all[warpsPerBlock][5];
	//__shared__ T level_prev_index_all[warpsPerBlock][5];
	//__shared__ char current_level_s_all[warpsPerBlock][32 * 4];

	T level_index[5];
	T level_count[5];
	T level_prev_index[5];
	//char* current_level_s = &current_level_s_all[warpIdx][0];
	T current_node_index;
	uint64 clique_count;
	char l;
	char new_level;

	for (unsigned long long i = gwx; i < (unsigned long long)current.count[0]; i += warpsPerBlock * gridDim.x) //warp per node
	{
		const T nodeId = current.queue[i];
		const T srcStart = g.rowPtr[nodeId];
		const T srcStop = g.rowPtr[nodeId + 1];
		const T srcLen = srcStop - srcStart;
		const T srcLenBlocks = (srcLen + 32 - 1) / 32;

		for (T kk = 0; kk < 5; kk++)
		{
			level_index[kk] = 0;
			level_prev_index[kk] = 0;
		}

		char* cl = &current_level[srcStart];
		// if (srcLen <= 4 * 32)
		// {
		// 	cl = current_level_s;
		// 	current_level_s[lx] = 2;
		// 	current_level_s[lx + 32] = 2;
		// 	current_level_s[lx + 32 * 2] = 2;
		// 	current_level_s[lx + 32 * 3] = 2;
		// }


		l = 2;
		clique_count = 0;
		level_count[l - 2] = srcLen;

		while (level_count[l - 2] > level_index[l - 2])
		{
			//Warp shuffle
			current_node_index = srcLen;
			for (T k = 0; k < srcLenBlocks; k++)
			{
				T index = level_prev_index[l - 2] + k * 32 + lx;
				if (current_node_index == srcLen && index < srcLen && cl[index] == l)
				{
					current_node_index = index;
					break;
				}
			}


			for (int offset = 16; offset > 0; offset /= 2)
			{
				T a = __shfl_down_sync(0xFFFFFFFF, current_node_index, offset, 32);
				current_node_index = current_node_index < a ? current_node_index : a;
			}

			current_node_index = __shfl_sync(0xFFFFFFFF, current_node_index, 0, 32);

			__syncwarp();


			level_prev_index[l - 2] = current_node_index + 1;
			level_index[l - 2]++;
			new_level = l;

			uint64 warpCount = 0;
			const T dst = g.colInd[current_node_index + srcStart];
			const T dstStart = g.rowPtr[dst];
			const T dstStop = g.rowPtr[dst + 1];
			const T dstLen = dstStop - dstStart;


			if (dstLen >= kclique - l)
			{
				if (dstLen > srcLen)
				{
					warpCount = graph::warp_sorted_count_and_set_binary<WARPS_PER_BLOCK, T, true>(0, &g.colInd[srcStart], srcLen,
						&g.colInd[dstStart], dstLen, true, srcStart, cl, l + 1, kclique);
				}
				else {
					warpCount = graph::warp_sorted_count_and_set_binary<WARPS_PER_BLOCK, T, true>(0, &g.colInd[dstStart], dstLen,
						&g.colInd[srcStart], srcLen, false, srcStart, cl, l + 1, kclique);
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
				for (auto k = 0; k < srcLenBlocks; k++)
				{
					T index = k * 32 + lx;
					if (index < srcLen && cl[index] > new_level)
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
	}
}

__device__ __inline__ uint32_t __mysmid() {
	unsigned int r;
	asm("mov.u32 %0, %%smid;" : "=r"(r));
	return r;
}

template <typename T, uint BLOCK_DIM_X>
__launch_bounds__(BLOCK_DIM_X)
__global__ void
kernel_block_level_kclique_count(
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
	const T gbx = blockIdx.x;

	__shared__ T level_index[10];
	__shared__ T level_count[10];
	__shared__ T level_prev_index[10];

	//__shared__ char current_level_s[2*BLOCK_DIM_X];

	__shared__ T current_node_index;
	__shared__ uint64 clique_count;
	__shared__ char l;
	__shared__ char new_level;




	__syncthreads();

	for (unsigned long long i = gbx; i < (unsigned long long)current.count[0]; i += gridDim.x)
	{
		const T nodeId = current.queue[i];
		const T srcStart = g.rowPtr[nodeId];
		const T srcStop = g.rowPtr[nodeId + 1];
		const T srcLen = srcStop - srcStart;
		const T srcLenBlocks = (srcLen + BLOCK_DIM_X - 1) / BLOCK_DIM_X;

		if (tid < 10)
		{
			level_index[tid] = 0;
			level_prev_index[tid] = 0;
		}

		char* cl = &current_level[srcStart];
		// if (srcLen > 2*BLOCK_DIM_X)
		// {
		// 	// for (unsigned long long k = 0; k < srcLenBlocks; k++)
		// 	// {
		// 	// 	unsigned long long index = k * BLOCK_DIM_X + tid;
		// 	// 	if (index < srcLen)
		// 	// 		cl[index] = 2;
		// 	// }
		// }
		// else
		// {
		// 	cl = current_level_s;

		// 	current_level_s[tid] = 2;
		// 	current_level_s[tid + BLOCK_DIM_X] = 2;
		// 	// current_level_s[tid + BLOCK_DIM_X * 2] = 2;
		// 	// current_level_s[tid + BLOCK_DIM_X * 3] = 2;
		// 	// current_level_s[tid + BLOCK_DIM_X * 4] = 2;
		// 	// current_level_s[tid + BLOCK_DIM_X * 5] = 2;
		// 	// current_level_s[tid + BLOCK_DIM_X * 6] = 2;
		// 	// current_level_s[tid + BLOCK_DIM_X * 7] = 2;

		// }

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
			// T finalVal = UINT_MAX;
			// for (T k = 0; k < srcLenBlocks; k++)
			// {
			// 	T index = level_prev_index[l - 2] + k * BLOCK_DIM_X + tid;
			// 	if(finalVal == UINT_MAX && index < srcLen && cl[index] == l)
			// 		{
			// 			finalVal = index;
			// 			//break;
			// 		}
			// }

			// if(finalVal != UINT_MAX)
			// 	atomicMin(&current_node_index, finalVal);


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


			//__syncthreads();

			if (tid == 0)
			{
				// for(T k = 0; k<srcLen; k++)
				// {	
				// 	T index = k + level_prev_index[l - 2];
				// 	if(cl[index] == l)
				// 	{
				// 		current_node_index = index;
				// 		break;
				// 	}
				// }

				level_prev_index[l - 2] = current_node_index + 1;
				level_index[l - 2]++;
				new_level = l;
			}

			__syncthreads();

			uint64 blockCount = 0;
			const T dst = g.colInd[current_node_index + srcStart];
			const T dstStart = g.rowPtr[dst];
			const T dstStop = g.rowPtr[dst + 1];
			const T dstLen = dstStop - dstStart;
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

				// if(tid == 0)
				// {
				// 	uint64 ss = 0;
				// 	for(T u=0; u< srcLen; u++)
				// 	{
				// 		if(cl[u] == l + 1)
				// 			ss++;
				// 	}


				// 	if(ss != blockCount && l+1 < kclique)
				// 		printf("SHIT\n");

				// }


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

				//__syncthreads();
			}

			__syncthreads();
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
				for (auto k = 0; k < srcLenBlocks; k++)
				{
					T index = k * BLOCK_DIM_X + tid;
					if (index < srcLen && cl[index] > new_level)
						cl[index] = new_level;
				}
			}

			__syncthreads();

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


template <typename T, uint BLOCK_DIM_X>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kernel_block_mem_level_kclique_count(
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
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kernel_block_mem_level_kclique_edge_count(
	uint64* counter,
	graph::COOCSRGraph_d<T> g,
	T kclique,
	T maxDeg,
	graph::GraphQueue_d<T, bool> current,
	char* __restrict__ current_level,
	uint64* cpn,
	T conc_blocks_per_SM,
	T* levelStats
)
{
	auto tid = threadIdx.x;
	const size_t gbx = blockIdx.x;

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

	for (unsigned long long i = gbx; i < (unsigned long long)current.count[0]; i += gridDim.x)
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
				if (l + 1 == kclique && tid == 0)
					clique_count += blockCount;
				else if (l + 1 < kclique)
				{
					if (tid == 0)
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
				T index = level_prev_index[l - 2] + k * BLOCK_DIM_X + tid;
				if (index < refLen && cl[index] == l)
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
					if (l + 1 == kclique && tid == 0)
						clique_count += blockCountIn;
					else if (l + 1 < kclique)
					{
						if (tid == 0)
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
			if (tid == 0)
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
					T index = k * BLOCK_DIM_X + tid;
					if (index < refLen && cl[index] > new_level)
						cl[index] = new_level;
				}

				__syncthreads();
			}



			if (tid == 0)
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


template <typename T, uint BLOCK_DIM_X>
__global__ void
kernel_block_level_kclique_count_check(
	uint64* counter,
	graph::COOCSRGraph_d<T> g,
	T kclique,
	T level,
	graph::GraphQueue_d<T, bool> current,
	char* current_level,
	uint64* cpn
)
{
	T threadData = 0;
	T aggreagtedData = 0;
	T accumAggData = 0;

	auto tid = threadIdx.x;
	const T gbx = blockIdx.x;

	__shared__ T level_index[10];
	__shared__ T level_count[10];
	__shared__ T level_prev_index[10];

	__shared__ char current_level_s[BLOCK_DIM_X * 10];

	__shared__ T current_node_index;
	__shared__ uint64 clique_count;
	__shared__ char l;
	__shared__ char new_level;



	__syncthreads();

	for (unsigned long long i = gbx; i < (unsigned long long)current.count[0]; i += gridDim.x)
	{
		const T nodeId = current.queue[i];
		const T srcStart = g.rowPtr[nodeId];
		const T srcStop = g.rowPtr[nodeId + 1];
		const T srcLen = srcStop - srcStart;
		const T srcLenBlocks = (srcLen + BLOCK_DIM_X - 1) / BLOCK_DIM_X;

		if (tid < 10)
		{
			level_index[tid] = 0;
			level_prev_index[tid] = 0;
		}

		char* cl;// = &current_level[srcStart];
		// if(srcLen > 8 * BLOCK_DIM_X )
		// {
		// 	for (unsigned long long k = 0; k < srcLenBlocks; k++)
		// 	{
		// 		unsigned long long index = k * BLOCK_DIM_X + tid;
		// 		if (index < srcLen)
		// 			cl[index] = 2;
		// 	}
		// }
		// else
		{
			cl = current_level_s;

			current_level_s[tid] = 2;
			current_level_s[tid + BLOCK_DIM_X] = 2;
			current_level_s[tid + BLOCK_DIM_X * 2] = 2;
			current_level_s[tid + BLOCK_DIM_X * 3] = 2;
			current_level_s[tid + BLOCK_DIM_X * 4] = 2;
			current_level_s[tid + BLOCK_DIM_X * 5] = 2;
			current_level_s[tid + BLOCK_DIM_X * 6] = 2;
			current_level_s[tid + BLOCK_DIM_X * 7] = 2;
			current_level_s[tid + BLOCK_DIM_X * 8] = 2;
			current_level_s[tid + BLOCK_DIM_X * 9] = 2;

		}

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
			// for (T k = 0; k < srcLenBlocks; k++)
			// {
			// 	T index = level_prev_index[l - 2] + k * BLOCK_DIM_X + tid;
			// 	if (index < srcLen && cl[index] == l)
			// 	{
			// 		atomicMin(&current_node_index, index);
			// 	}
			// 	__syncthreads();
			// 	if (current_node_index != UINT_MAX)
			// 		break;
			// }

			__syncthreads();

			if (tid == 0)
			{
				for (T k = 0; k < srcLen; k++)
				{
					T index = k + level_prev_index[l - 2];
					if (cl[index] == l)
					{
						current_node_index = index;
						break;
					}
				}

				level_prev_index[l - 2] = current_node_index + 1;
				level_index[l - 2]++;
				new_level = l;
			}

			__syncthreads();

			uint64 blockCount = 0;
			const T dst = g.colInd[current_node_index + srcStart];
			const T dstStart = g.rowPtr[dst];
			const T dstStop = g.rowPtr[dst + 1];
			const T dstLen = dstStop - dstStart;
			//if (dstLen >= kclique - l)
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

				// if(tid == 0)
				// {
				// 	uint64 ss = 0;
				// 	for(T u=0; u< srcLen; u++)
				// 	{
				// 		if(cl[u] == l + 1)
				// 			ss++;
				// 	}


				// 	if(ss != blockCount && l+1 < kclique)
				// 		printf("SHIT\n");

				// }


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

				__syncthreads();
			}

			__syncthreads();
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
				for (auto k = 0; k < srcLenBlocks; k++)
				{
					T index = k * BLOCK_DIM_X + tid;
					if (index < srcLen && cl[index] > new_level)
						cl[index] = new_level;
				}
			}

			__syncthreads();

			if (tid == 0)
			{
				l = new_level;
				current_node_index = UINT_MAX;
			}
			__syncthreads();
		}

		__syncthreads();
		if (threadIdx.x == 0)
		{
			if (cpn[nodeId] != clique_count)
				printf("nodeId = %u, prev = %llu, now = %llu\n", nodeId, cpn[nodeId], clique_count);

			//atomicAdd(counter, clique_count);
		}

		__syncthreads();
	}
}




template <typename T, int BLOCK_DIM_X>
__global__ void
kernel_block_level_kclique_edge_count(
	uint64* counter,
	graph::COOCSRGraph_d<T> g,
	T kclique,
	T level,
	graph::GraphQueue_d<T, bool> current,
	char* current_level,
	const unsigned long long spaceLen
)
{
	T threadData = 0;
	T aggreagtedData = 0;
	T accumAggData = 0;

	auto tid = threadIdx.x;
	const size_t gbx = blockIdx.x;

	__shared__ T level_index[10];
	__shared__ T level_count[10];
	__shared__ T level_prev_index[10];

	__shared__ char current_level_s[BLOCK_DIM_X * 8];

	__shared__ T current_node_index;
	__shared__ uint64 clique_count;
	__shared__ T l;
	__shared__ T new_level;

	if (tid == 0)
		clique_count = 0;

	__syncthreads();

	for (unsigned long long i = gbx; i < (unsigned long long)current.count[0]; i += gridDim.x)
	{
		const T edgeId = current.queue[i];

		T src = g.rowInd[edgeId];
		T src2 = g.colInd[edgeId];

		const T srcStart = g.rowPtr[src];
		const T srcStop = g.rowPtr[src + 1];
		const T srcLen = srcStop - srcStart;

		const T src2Start = g.rowPtr[src2];
		const T src2Stop = g.rowPtr[src2 + 1];
		const T src2Len = src2Stop - src2Start;

		T refIndex = srcLen < src2Len ? srcStart : src2Start;
		const T refLen = srcLen < src2Len ? srcLen : src2Len;;
		const T srcLenBlocks = (refLen + BLOCK_DIM_X - 1) / BLOCK_DIM_X;


		unsigned long long ii = i * spaceLen;
		char* cl = &(current_level[ii]);

		if (spaceLen > 8 * BLOCK_DIM_X)
		{
			for (unsigned long long k = 0; k < srcLenBlocks; k++)
			{
				unsigned long long index = k * BLOCK_DIM_X + tid;

				if (index < refLen)
					cl[index] = 2;
			}
		}
		else
		{
			cl = current_level_s;

			current_level_s[tid] = 2;
			current_level_s[tid + BLOCK_DIM_X] = 2;
			current_level_s[tid + BLOCK_DIM_X * 2] = 2;
			current_level_s[tid + BLOCK_DIM_X * 3] = 2;
			current_level_s[tid + BLOCK_DIM_X * 4] = 2;
			current_level_s[tid + BLOCK_DIM_X * 5] = 2;
			current_level_s[tid + BLOCK_DIM_X * 6] = 2;
			current_level_s[tid + BLOCK_DIM_X * 7] = 2;

		}

		if (tid == 0)
		{
			l = 2;
			current_node_index = UINT_MAX;
		}
		__syncthreads();

		if (tid < 10)
		{
			level_index[tid] = 0;
			level_count[tid] = 0;
			level_prev_index[tid] = 0;
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
				if (l + 1 == kclique && tid == 0)
					clique_count += blockCount;
				else if (l + 1 < kclique)
				{
					if (tid == 0)
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
				T index = level_prev_index[l - 2] + k * BLOCK_DIM_X + tid;
				if (index < refLen && cl[index] == l)
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
					if (l + 1 == kclique && tid == 0)
						clique_count += blockCountIn;
					else if (l + 1 < kclique)
					{
						if (tid == 0)
						{
							l++;
							new_level++;
							level_count[l - 2] = blockCountIn;
							level_index[l - 2] = 0;
							level_prev_index[l - 2] = 0;
						}
					}

				}

				__syncthreads();
			}

			__syncthreads();
			if (tid == 0)
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
					T index = k * BLOCK_DIM_X + tid;
					if (index < refLen && cl[index] > new_level)
						cl[index] = new_level;
				}
			}

			__syncthreads();

			if (tid == 0)
			{
				l = new_level;
				current_node_index = UINT_MAX;
			}
			__syncthreads();
		}

		if (threadIdx.x == 0)
			atomicAdd(counter, clique_count);
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

				bucket.count.gdata()[0] = CUBSelect(asc.gdata(), bucket.queue.gdata(), bucket.mark.gdata(), node_num);
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

				bucket.count.gdata()[0] = CUBSelect(asc.gdata(), bucket.queue.gdata(), bucket.mark.gdata(), node_num);
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
			identity_arr_asc.initialize("Identity Array Asc", AllocationTypeEnum::unified, n, dev_);
			execKernel(init_asc, grid_size, BLOCK_SIZE, dev_, false, identity_arr_asc.gdata(), n);
		}

	public:


		GPUArray<T> nodeDegree;

		SingleGPU_Kclique(int dev) : dev_(dev) {
			CUDA_RUNTIME(cudaSetDevice(dev_));
			CUDA_RUNTIME(cudaStreamCreate(&stream_));
			CUDA_RUNTIME(cudaStreamSynchronize(stream_));
		}

		SingleGPU_Kclique() : SingleGPU_Kclique(0) {}


		void getNodeDegree(COOCSRGraph_d<T>& g, T* maxDegree,
			const size_t nodeOffset = 0, const size_t edgeOffset = 0)
		{
			const int dimBlock = 256;
			nodeDegree.initialize("Edge Support", unified, g.numNodes, dev_);
			uint dimGridNodes = (g.numNodes + dimBlock - 1) / dimBlock;
			execKernel((getNodeDegree_kernel<T, dimBlock>), dimGridNodes, dimBlock, dev_, false, nodeDegree.gdata(), g, maxDegree);
		}

		void findKclqueIncremental_node_async(int kcount, COOCSRGraph_d<T>& g,
			ProcessingElementEnum pe, const size_t nodeOffset = 0, const size_t edgeOffset = 0)
		{
			CUDA_RUNTIME(cudaSetDevice(dev_));

			T level = 0;
			T span = 1024;
			T bucket_level_end_ = level;

			//Lets apply queues and buckets
			graph::GraphQueue<T, bool> bucket_q;
			bucket_q.Create(unified, g.numNodes, dev_);

			graph::GraphQueue<T, bool> current_q;
			current_q.Create(unified, g.numNodes, dev_);

			GPUArray<T> identity_arr_asc;
			AscendingGpu(g.numNodes, identity_arr_asc);


			GPUArray <uint64> counter("Temp level Counter", unified, 1, dev_);
			GPUArray <uint64> cpn("Temp level Counter", unified, g.numNodes, dev_);
			// cpn.setAll(0, true);
			// GPUArray<T>
			// 	filter_level("Temp filter Counter", unified, g.numEdges, dev_),
			// 	filter_scan("Temp scan Counter", unified, g.numEdges, dev_);


			counter.setSingle(0, 0, true);
			CUDA_RUNTIME(cudaStreamSynchronize(stream_));

			GPUArray <T> maxDegree("Temp Degree", unified, 1, dev_);
			maxDegree.setSingle(0, 0, true);
			getNodeDegree(g, maxDegree.gdata());
			T todo = g.numNodes;
			bucket_scan(nodeDegree, g.numNodes, 0, kcount - 1, current_q, identity_arr_asc, bucket_q, bucket_level_end_);
			todo -= current_q.count.gdata()[0];
			current_q.count.gdata()[0] = 0;
			level = kcount - 1;
			bucket_level_end_ = level;

			//level = 32;
			//bucket_level_end_ = level;
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

					//std::sort(current_q.queue.gdata(), current_q.queue.gdata() + current_q.count.gdata()[0]);
					//current_q.count.gdata()[0] = current_q.count.gdata()[0]< 5000? current_q.count.gdata()[0]: 5000;
					//current_q.count.gdata()[0] = 1;


					if (pe == Warp)
					{
						GPUArray<char> current_level("Temp level Counter", unified, g.numEdges, dev_);
						current_level.setAll(2, true);
						const auto block_size = 128;
						auto grid_block_size = (32 * current_q.count.gdata()[0] + block_size - 1) / block_size;
						execKernel((kernel_warp_level_kclique_count<T, block_size>), grid_block_size, block_size, dev_, false,
							counter.gdata(),
							g,
							kcount,
							level,
							current_q.device_queue->gdata()[0],
							current_level.gdata(), cpn.gdata());
					}
					// else if (pe == Block)
					// {
						// GPUArray<char> current_level("Temp level Counter", unified, g.numEdges, dev_);
						// current_level.setAll(2, true);
					// 	const auto block_size = 64;
					// 	auto grid_block_size = current_q.count.gdata()[0];

					// 	execKernel((kernel_block_level_kclique_count<T, block_size>), grid_block_size, block_size, dev_, false,
					// 	counter.gdata(),
					// 	g,
					// 	kcount,
					// 	level,
					// 	current_q.device_queue->gdata()[0],
					// 	current_level.gdata(), cpn.gdata());


					// 	// execKernel((kernel_block_level_kclique_count2<T, block_size>), grid_block_size, block_size, dev_, false,
					// 	// 	counter.gdata(),
					// 	// 	g,
					// 	// 	kcount,
					// 	// 	level,
					// 	// 	current_q.device_queue->gdata()[0],
					// 	// 	current_level.gdata(), filter_level.gdata(), cpn.gdata());


					// }
					else if (pe == Block)
					{


						const auto block_size = 64;
						auto grid_block_size = current_q.count.gdata()[0];
						CUDAContext context;
						T num_SMs = context.num_SMs;
						T conc_blocks_per_SM = context.GetConCBlocks(block_size);
						GPUArray<T> d_bitmap_states("bmp bitmap stats", AllocationTypeEnum::gpu, num_SMs * conc_blocks_per_SM, dev_);
						d_bitmap_states.setAll(0, true);

						GPUArray<char> current_level("Temp level Counter", unified, num_SMs * conc_blocks_per_SM * maxDegree.gdata()[0], dev_);
						current_level.setAll(2, true);

						execKernel((kernel_block_mem_level_kclique_count<T, block_size>), grid_block_size, block_size, dev_, false,
							counter.gdata(),
							g,
							kcount,
							maxDegree.gdata()[0],
							current_q.device_queue->gdata()[0],
							current_level.gdata(), cpn.gdata(),
							conc_blocks_per_SM, d_bitmap_states.gdata());
					}



					std::cout.imbue(std::locale(""));
					std::cout << "------------- Level = " << level << " Nodes = " << current_q.count.gdata()[0] << " Counter = " << counter.gdata()[0] << "\n";
				}


				level += span;

			}

			current_q.free();
			bucket_q.free();

			k = level;

			printf("Max Degree (+span) = %d\n", k - 1);
		}



		void findKclqueIncremental_edge_async(int kcount, COOCSRGraph_d<T>& g,
			ProcessingElementEnum pe, const size_t nodeOffset = 0, const size_t edgeOffset = 0)
		{
			CUDA_RUNTIME(cudaSetDevice(dev_));

			T level = 0;
			T span = 1024;
			T bucket_level_end_ = level;

			//Lets apply queues and buckets
			graph::GraphQueue<T, bool> bucket_q;
			bucket_q.Create(unified, g.numEdges, dev_);

			graph::GraphQueue<T, bool> current_q;
			current_q.Create(unified, g.numEdges, dev_);

			GPUArray<T> identity_arr_asc;
			AscendingGpu(g.numEdges, identity_arr_asc);


			GPUArray <uint64> counter("Temp level Counter", unified, 1, dev_);

			counter.setSingle(0, 0, true);
			CUDA_RUNTIME(cudaStreamSynchronize(stream_));
			T todo = g.numEdges;

			GPUArray<T> edgePtr("Temp Edge Ptr", unified, g.numEdges, dev_);
			//execKernel((init_edge_ptr<T, 128>), (g.numEdges + 128 - 1) / 128, 128, dev_, false, g, edgePtr.gdata());


			GPUArray <T> maxDegree("Temp Degree", unified, 1, dev_);
			maxDegree.setSingle(0, 0, true);
			execKernel((get_max_degree<T, 128>), (g.numEdges + 128 - 1) / 128, 128, dev_, false, g, edgePtr.gdata(), maxDegree.gdata());

			printf("Max Dgree = %u vs %u\n", maxDegree.gdata()[0], g.numEdges);
			bucket_edge_scan(edgePtr, g.numEdges, 0, kcount - 2, current_q, identity_arr_asc, bucket_q, bucket_level_end_);
			todo -= current_q.count.gdata()[0];
			current_q.count.gdata()[0] = 0;
			level = kcount - 2;
			bucket_level_end_ = level;
			const auto block_size = 64;

			CUDAContext context;
			T num_SMs = context.num_SMs;
			T conc_blocks_per_SM = context.GetConCBlocks(block_size);
			GPUArray<T> d_bitmap_states("bmp bitmap stats", AllocationTypeEnum::gpu, num_SMs * conc_blocks_per_SM, dev_);
			GPUArray<char> current_level("Temp level Counter", unified, num_SMs * conc_blocks_per_SM * maxDegree.gdata()[0], dev_);
			d_bitmap_states.setAll(0, true);
			current_level.setAll(2, true);
			CUDA_RUNTIME(cudaGetLastError());
			cudaDeviceSynchronize();

			while (todo > 0)
			{

				//1 bucket fill
				bucket_edge_scan(edgePtr, g.numEdges, level, span, current_q, identity_arr_asc, bucket_q, bucket_level_end_);
				todo -= current_q.count.gdata()[0];
				if (current_q.count.gdata()[0] > 0)
				{
					//printf("Queue = %d\n", current_q.count.gdata()[0]);
					//current_q.count.gdata()[0] = 1100;


					auto grid_block_size = current_q.count.gdata()[0];
					execKernel((kernel_block_mem_level_kclique_edge_count<T, block_size>), grid_block_size, block_size, dev_, false,
						counter.gdata(),
						g,
						kcount,
						maxDegree.gdata()[0],
						current_q.device_queue->gdata()[0],
						current_level.gdata(),
						NULL,
						conc_blocks_per_SM, d_bitmap_states.gdata());
				}
				level += span;
			}

			std::cout.imbue(std::locale(""));
			std::cout << "Level = " << level << " Nodes = " << current_q.count.gdata()[0] << " Counter = " << counter.gdata()[0] << "\n";

			current_q.free();
			bucket_q.free();

			k = level;

			printf("Max Edge Min Degree = %d\n", k - 1);

		}


		uint findKtrussIncremental_sync(int kmin, int kmax, TcBase<T>* tcCounter, EidGraph_d<T>& g, int* reverseIndex, EncodeDataType* bitMap, const size_t nodeOffset = 0, const size_t edgeOffset = 0)
		{
			findKtrussIncremental_async(kmin, kmax, tcCounter, g, reverseIndex, bitMap, nodeOffset, edgeOffset);
			sync();
			return count();
		}

		void sync() { CUDA_RUNTIME(cudaStreamSynchronize(stream_)); }

		uint count() const { return k - 1; }
		int device() const { return dev_; }
		cudaStream_t stream() const { return stream_; }
	};

} // namespace pangolin
