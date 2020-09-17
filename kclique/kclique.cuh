#pragma once

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

template <typename T, int BLOCK_DIM_X>
__global__ void try_block_scan(T count,  T* output)
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
__global__ void
kernel_block_level_kclique_count(
	uint64 *counter,
	graph::COOCSRGraph_d<T> g,
	int kclique,
	int level,
	bool* processed, 
	graph::GraphQueue_d<int, bool> current,
	T* current_level,
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

	level_index[tid] = 0;
	level_count[tid] = 0;
	if (tid == 0)
		clique_count = 0;
	__syncthreads();

	for (size_t i = gbx; i < current.count[0]; i += gridDim.x)
	{
		T nodeId = current.queue[i];

		T srcStart = g.rowPtr[nodeId];
		T srcStop = g.rowPtr[nodeId + 1];
		T srcLen = srcStop - srcStart;
		T srcLenBlocks = (srcLen + BLOCK_DIM_X - 1) / BLOCK_DIM_X;

		if (tid == 0)
		{
			l = 2;
			level_count[l - 2] = srcLen;
			current_node_index = UINT_MAX;
		}
		__syncthreads();

		while (level_count[l - 2] > level_index[l-2])
		{
			if (tid == 0)
			{
				printf("Level = %u, Index = %u, %u\n", l, level_index[l - 2], level_count[l - 2]);
			}

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

						printf("%u, %u, %u\n", index, threadData, g.colInd[index]);

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
			}
			__syncthreads();

			T blockCount = 0;

			//(3) intesect Adj[current_node] with
			const T filter_srcStop = srcStart + aggreagtedData; //only filtered
			const T filter_srcLen = aggreagtedData;

			const T dst = current_node_index;
			const T dstStart = g.rowPtr[current_node_index];
			const T dstStop = g.rowPtr[current_node_index + 1];
			const T dstLen = dstStop - dstStart;

			if (tid == 0)
			{
				printf("Adj of %u is: ", dst);
				for (int tt = dstStart; tt < dstStop; tt++)
				{
					printf("%u, ", g.colInd[tt]);
				}
				printf("\n");
			}


			
 
			if (dstLen > srcLen) 
			{
				blockCount += graph::block_sorted_count_and_set_binary<BLOCK_DIM_X, T, true>(&filter_level[srcStart], filter_srcLen,
					&g.colInd[dstStart], dstLen,true, srcStart, current_level, filter_scan, l + 1, kclique);
			}
			else {
				blockCount += graph::block_sorted_count_and_set_binary<BLOCK_DIM_X, T, true>(&g.colInd[dstStart], dstLen,
					&filter_level[srcStart], filter_srcLen, false, srcStart, current_level, filter_scan, l + 1, kclique);
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
					level_count[l - 2] = blockCount;
					level_index[l - 2] = 0;
				}
				else if (blockCount == 0)
				{ 
					new_level = l - 1;
				}

				while (new_level > 2 && level_index[new_level - 2] >= level_count[new_level - 2])
				{
					new_level--;
				}
			}

			__syncthreads();
			if (new_level != l)
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
			{
				printf("Chosen Node Index = %u, Agg = %u of %u: TC Count = %u, level = %u\n", current_node_index, aggreagtedData, srcLen, blockCount, l);
				printf("Now Print all current_level:\n");
				for (int ii = 0; ii < srcLen; ii++)
					printf("%u, ", current_level[srcStart + ii]);
				printf("\n");
			}
			
		}

		if (threadIdx.x == 0)
			atomicAdd(counter, clique_count);



		/*for (int j = tid; j < (srcLen + BLOCK_DIM_X - 1) / BLOCK_DIM_X * BLOCK_DIM_X; j += BLOCK_DIM_X)
		{
			__syncthreads();
			if (size >= BLOCK_DIM_X)
			{
				for (int e = tid; e < size; e += BLOCK_DIM_X)
				{
					T e1 = e1_arr[e];
					if (!current.mark[e1])
						process_degree(e1, level, nodeDegree, next, bucket, bucket_level_end_);
				}
				__syncthreads();
				if (tid == 0)
					size = 0;
				__syncthreads();
			}

			if (j < srcLen)
			{
				T affectedNode = g.colInd[j + srcStart];
				auto pos = atomicAdd(&size, 1);
				e1_arr[pos] = affectedNode;
			}
		}

		__syncthreads();
		for (int e = tid; e < size; e += BLOCK_DIM_X)
		{
			T e1 = e1_arr[e];
			if (!current.mark[e1])
				process_degree(e1, level, nodeDegree, next, bucket, bucket_level_end_);
		}*/
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
			GPUArray<int> nodeDegree, uint32_t node_num, int level,
			GraphQueue<int, bool>& current,
			GPUArray<uint> asc,
			GraphQueue<int, bool>& bucket,
			int& bucket_level_end_)
		{
			static bool is_first = true;
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
				execKernel(filter_window, grid_size, BLOCK_SIZE, false,
					nodeDegree.gdata(), node_num, bucket.mark.gdata(), level, bucket_level_end_ + LEVEL_SKIP_SIZE);

				bucket.count.gdata()[0] = CUBSelect(asc.gdata(), bucket.queue.gdata(), bucket.mark.gdata(), node_num);
				bucket_level_end_ += LEVEL_SKIP_SIZE;
			}
			// SCAN the window.
			if (bucket.count.gdata()[0] != 0)
			{
				current.count.gdata()[0] = 0;
				long grid_size = (bucket.count.gdata()[0] + BLOCK_SIZE - 1) / BLOCK_SIZE;
				execKernel(filter_with_random_append, grid_size, BLOCK_SIZE, false,
					bucket.queue.gdata(), bucket.count.gdata()[0], nodeDegree.gdata(), current.mark.gdata(), current.queue.gdata(), &(current.count.gdata()[0]), level);
			}
			else
			{
				current.count.gdata()[0] = 0;
			}
			//Log(LogPriorityEnum::info, "Level: %d, curr: %d/%d", level, current.count.gdata()[0], bucket.count.gdata()[0]);
		}


		void AscendingGpu(int n, GPUArray<uint>& identity_arr_asc)
		{
			long grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
			identity_arr_asc.initialize("Identity Array Asc", AllocationTypeEnum::unified, n, 0);
			execKernel(init_asc, grid_size, BLOCK_SIZE, false, identity_arr_asc.gdata(), n);
		}



		void StreamComapction(
			int n, T& m,
			T*& rowPtr, T*& colInd, T*& eid,
			GPUArray<bool> processed,
			GPUArray<bool>& edge_deleted,
			GPUArray <T>& new_offset, GPUArray<T>& new_eid, GPUArray <T>& new_adj,
			T old_edge_num, T new_edge_num)
		{
			static bool shrink_first_time = true;
			if (shrink_first_time) { //shrink first time, allocate the buffers
				shrink_first_time = false;
				Timer alloc_timer;
				new_adj.initialize("New Adj", gpu, new_edge_num * 2, 0);
				new_eid.initialize("New EID", gpu, new_edge_num * 2, 0);
				new_offset.initialize("New Row Pointer", unified, (n + 1), 0);
				edge_deleted.initialize("Edge deleted", gpu, old_edge_num * 2, 0);
			}


			/*2. construct new CSR (offsets, adj) and rebuild the eid*/
			int block_size = 128;
			// Attention: new_offset gets the histogram.
			execKernel(warp_detect_deleted_edges, GRID_SIZE, block_size, true,
				rowPtr, n, eid, processed.gdata(), new_offset.gdata(), edge_deleted.gdata());

			uint total = CUBScanExclusive<uint, uint>(new_offset.gdata(), new_offset.gdata(), n);
			new_offset.gdata()[n] = total;
			//assert(total == new_edge_num * 2);
			cudaDeviceSynchronize();

			swap_ele(rowPtr, new_offset.gdata());

			/*new adj and eid construction*/
			CUBSelect(colInd, new_adj.gdata(), edge_deleted.gdata(), old_edge_num * 2);
			CUBSelect(eid, new_eid.gdata(), edge_deleted.gdata(), old_edge_num * 2);

			swap_ele(colInd, new_adj.gdata());


			swap_ele(eid, new_eid.gdata());


			m = new_edge_num * 2;
		}

	public:


		GPUArray<int> nodeDegree;

		SingleGPU_Kclique(int dev) : dev_(dev) {
			CUDA_RUNTIME(cudaSetDevice(dev_));
			CUDA_RUNTIME(cudaStreamCreate(&stream_));
			CUDA_RUNTIME(cudaStreamSynchronize(stream_));
		}

		SingleGPU_Kclique() : SingleGPU_Kclique(0) {}


		void getNodeDegree(COOCSRGraph_d<T>& g,
			const size_t nodeOffset = 0, const size_t edgeOffset = 0)
		{
			const int dimBlock = 256;
			nodeDegree.initialize("Edge Support", unified, g.numNodes, 0);
			uint dimGridNodes = (g.numNodes + dimBlock - 1) / dimBlock;
			execKernel(getNodeDegree_kernel<T>, dimGridNodes, dimBlock, false, nodeDegree.gdata(), g);

		}

		void findKclqueIncremental_async(int kcount, COOCSRGraph_d<T>& g,
			const size_t nodeOffset = 0, const size_t edgeOffset = 0)
		{
			CUDA_RUNTIME(cudaSetDevice(dev_));
			constexpr int dimBlock = 128; //For edges and nodes

			GPUArray<BCTYPE> processed; //isDeleted

			int level = 0;
			int bucket_level_end_ = level;


			GPUArray<bool> node_kept;

			//Lets apply queues and buckets
			graph::GraphQueue<int, bool> bucket_q;
			bucket_q.Create(unified, g.numNodes, 0);

			graph::GraphQueue<int, bool> current_q;
			current_q.Create(unified, g.numNodes, 0);

			graph::GraphQueue<int, bool> next_q;
			next_q.Create(unified, g.numNodes, 0);

			GPUArray<T> identity_arr_asc;
			AscendingGpu(g.numNodes, identity_arr_asc);

			GPUArray <uint> newRowPtr;
			GPUArray <uint> newColIndex_csr;
			GPUArray <uint> new_eid;
			GPUArray <uint> new_edge_offset_origin;
			GPUArray<bool> edge_deleted;           // Auxiliaries for shrinking graphs.

			GPUArray <uint64> counter("Temp level Counter", unified, 1, 0) ;

			GPUArray<uint> current_level("Temp level Counter", unified, g.numEdges, 0), 
				filter_level("Temp filter Counter", unified, g.numEdges, 0),
				filter_scan("Temp scan Counter", unified, g.numEdges, 0);

			processed.initialize("is Deleted (Processed)", unified, g.numNodes, 0);

			processed.setAll(false, false);
			
			counter.setSingle(0, 0, true);

			CUDA_RUNTIME(cudaStreamSynchronize(stream_));

			uint numDeleted_l = 0;
			float minPercentage = 0.8;
			float percDeleted_l = 0.0;


		/*	execKernel((try_block_scan<uint, 64>), 1, 64, false, 256, current_level.gdata());

			for (int i = 0; i < 256; i++)
				printf("%u, ", current_level.gdata()[i]);

			printf("\n");*/

			current_level.setAll(2, true);
			getNodeDegree(g);


			int todo = g.numNodes;
			const auto todo_original = g.numNodes;
			while (todo > 0)
			{
				//	//printf("k=%d\n", k);
				numDeleted_l = 0;
				CUDA_RUNTIME(cudaGetLastError());
				cudaDeviceSynchronize();

				//1 bucket fill
				bucket_scan(nodeDegree, todo_original, level, current_q, identity_arr_asc, bucket_q, bucket_level_end_);

				


				todo -= current_q.count.gdata()[0];


				if (0 == todo) {
					break;
				}


				if (level < kcount)
				{
					level++;
					continue;
				}

				

				if (current_q.count.gdata()[0] > 0)
				{
					current_q.count.gdata()[0] = 1;

					const auto block_size = 256;
					auto grid_block_size = current_q.count.gdata()[0];

					execKernel((kernel_block_level_kclique_count<T, block_size>), grid_block_size, block_size, false,
						counter.gdata(),
						g,
						kcount,
						level,
						processed.gdata(),
						current_q.device_queue->gdata()[0],
						current_level.gdata(), filter_level.gdata(), filter_scan.gdata());
				}


				printf("Level = %d, Counter = %ul -------------------------------------------------------------------\n", level, counter.gdata()[0]);

				level++;
			}

			processed.freeGPU();
			current_q.free();
			next_q.free();
			bucket_q.free();

			k = level;

			printf("Max Core = %d\n", k - 1);
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
