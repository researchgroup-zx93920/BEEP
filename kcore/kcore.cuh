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



__inline__ __device__
void process_degree(
	uint32_t nodeId, int level, int* nodeDegree,
	graph::GraphQueue_d<int, bool>& next,
	graph::GraphQueue_d<int, bool>& bucket,
	int bucket_level_end_)
{
	auto cur = atomicSub(&nodeDegree[nodeId], 1);
	if (cur == (level + 1)) 
	{
		add_to_queue_1(next, nodeId);
	}
	if (cur <= level) 
	{
		atomicAdd(&nodeDegree[nodeId], 1);
	}

	// Update the Bucket.
	auto latest = cur - 1;
	if (latest > level && latest < bucket_level_end_) {
		add_to_queue_1_no_dup(bucket, nodeId);
	}

}


template<typename T>
__global__ void getNodeDegree_kernel(int* nodeDegree, graph::COOCSRGraph_d<T> g)
{
	int gtid = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = gtid; i < g.numNodes; i+= blockDim.x*gridDim.x)
	{
		nodeDegree[i] = g.rowPtr[i + 1] - g.rowPtr[i];
	}
}



template <typename T>
__global__ void
kernel_thread_level_next(
	graph::COOCSRGraph_d<T> g,
	int level, bool* processed, int* nodeDegree,
	graph::GraphQueue_d<int, bool> current,
	graph::GraphQueue_d<int, bool>& next,
	graph::GraphQueue_d<int, bool>& bucket,
	int bucket_level_end_
)
{
	size_t gx = blockDim.x * blockIdx.x + threadIdx.x;
	for (size_t i = gx; i < current.count[0]; i += blockDim.x * gridDim.x)
	{
		T nodeId = current.queue[i];

		T srcStart = g.rowPtr[nodeId];
		T srcStop = g.rowPtr[nodeId + 1];

		for (int j = srcStart; j < srcStop; j++)
		{
			T affectedNode = g.colInd[j];
			if(/*!processed[affectedNode] &&*/ !current.mark[affectedNode] )
				process_degree(affectedNode, level, nodeDegree, next, bucket, bucket_level_end_);
		}
	}
}


template <typename T, int BD>
__global__ void
kernel_warp_level_next(
	graph::COOCSRGraph_d<T> g,
	int level, bool* processed, int* nodeDegree,
	graph::GraphQueue_d<int, bool> current,
	graph::GraphQueue_d<int, bool>& next,
	graph::GraphQueue_d<int, bool>& bucket,
	int bucket_level_end_
)
{
	const size_t warpsPerBlock = BD / 32;
	const size_t lx = threadIdx.x % 32;
	const int warpIdx = threadIdx.x / 32; // which warp in thread block
	const size_t gwx = (blockDim.x * blockIdx.x + threadIdx.x) / 32;

	__shared__ T q[warpsPerBlock][32*2];
	__shared__ int sizes[warpsPerBlock];
	T* e1_arr = &q[warpIdx][0];
	int* size = &sizes[warpIdx];

	if (lx == 0)
		*size = 0;

	for (size_t i = gwx; i < current.count[0]; i += blockDim.x * gridDim.x/32)
	{
		T nodeId = current.queue[i];

		T srcStart = g.rowPtr[nodeId];
		T srcStop = g.rowPtr[nodeId + 1];

		for (int j = srcStart + lx; j < (srcStop+31)/32 * 32; j+=32)
		{
			__syncwarp();
			if (*size >= 32)
			{
				for (int e = lx; e < *size; e += 32)
				{
					T e1 = e1_arr[e];
					if (!current.mark[e1])
						process_degree(e1, level, nodeDegree, next, bucket, bucket_level_end_);
				}
				__syncwarp();
				if (lx == 0)
					*size = 0;
				__syncwarp();
			}

			if (j < srcStop)
			{
				T affectedNode = g.colInd[j];
				auto pos = atomicAdd(size, 1);
				e1_arr[pos] = affectedNode;
			}
		}

		__syncwarp();
		for (int e = lx; e < *size; e += 32)
		{
			T e1 = e1_arr[e];
			if (!current.mark[e1])
				process_degree(e1, level, nodeDegree, next, bucket, bucket_level_end_);
		}
	}
}



template <typename T, int BD, int P>
__global__ void
kernel_partition_level_next(
	graph::COOCSRGraph_d<T> g,
	int level, bool* processed, int* nodeDegree,
	graph::GraphQueue_d<int, bool> current,
	graph::GraphQueue_d<int, bool>& next,
	graph::GraphQueue_d<int, bool>& bucket,
	int bucket_level_end_
)
{
	const size_t partitionsPerBlock = BD / P;
	const size_t lx = threadIdx.x % P;
	const int warpIdx = threadIdx.x / P; // which warp in thread block
	const size_t gwx = (blockDim.x * blockIdx.x + threadIdx.x) / P;

	__shared__ T q[partitionsPerBlock][P * 2];
	__shared__ int sizes[partitionsPerBlock];
	T* e1_arr = &q[warpIdx][0];
	int* size = &sizes[warpIdx];

	if (lx == 0)
		*size = 0;

	for (size_t i = gwx; i < current.count[0]; i += blockDim.x * gridDim.x / P)
	{
		T nodeId = current.queue[i];

		T srcStart = g.rowPtr[nodeId];
		T srcStop = g.rowPtr[nodeId + 1];

		for (int j = srcStart + lx; j < (srcStop + P-1) / P * P; j += P)
		{
			__syncwarp();
			if (*size >= P)
			{
				for (int e = lx; e < *size; e += P)
				{
					T e1 = e1_arr[e];
					if (!current.mark[e1])
						process_degree(e1, level, nodeDegree, next, bucket, bucket_level_end_);
				}
				__syncwarp();
				if (lx == 0)
					*size = 0;
				__syncwarp();
			}

			if (j < srcStop)
			{
				T affectedNode = g.colInd[j];
				auto pos = atomicAdd(size, 1);
				e1_arr[pos] = affectedNode;
			}
		}

		__syncwarp();
		for (int e = lx; e < *size; e += P)
		{
			T e1 = e1_arr[e];
			if (!current.mark[e1])
				process_degree(e1, level, nodeDegree, next, bucket, bucket_level_end_);
		}
	}
}



template <typename T, int BLOCK_DIM_X>
__global__ void
kernel_block_level_next(
	graph::COOCSRGraph_d<T> g,
	int level, bool* processed, int* nodeDegree,
	graph::GraphQueue_d<int, bool> current,
	graph::GraphQueue_d<int, bool>& next,
	graph::GraphQueue_d<int, bool>& bucket,
	int bucket_level_end_
)
{
	auto tid = threadIdx.x;
	const size_t gbx = (BLOCK_DIM_X * blockIdx.x + threadIdx.x) / BLOCK_DIM_X;


	__shared__ T q[BLOCK_DIM_X*2];
	T* e1_arr = &q[0];
	__shared__ int size;

	if (tid == 0)
	{
		size = 0;
	}
	__syncthreads();

	for (size_t i = gbx; i < current.count[0]; i += gridDim.x)
	{
		T nodeId = current.queue[i];

		T srcStart = g.rowPtr[nodeId];
		T srcStop = g.rowPtr[nodeId + 1];
		T srcLen = srcStop - srcStart;

		for (int j = tid; j < (srcLen + BLOCK_DIM_X -1) / BLOCK_DIM_X * BLOCK_DIM_X; j += BLOCK_DIM_X)
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
		}
	}
}


namespace graph
{
	template<typename T>
	class SingleGPU_Kcore
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

		SingleGPU_Kcore(int dev) : dev_(dev) {
			CUDA_RUNTIME(cudaSetDevice(dev_));
			CUDA_RUNTIME(cudaStreamCreate(&stream_));
			CUDA_RUNTIME(cudaStreamSynchronize(stream_));
		}

		SingleGPU_Kcore() : SingleGPU_Kcore(0) {}


		void getNodeDegree(COOCSRGraph_d<T>& g,
			const size_t nodeOffset = 0, const size_t edgeOffset = 0)
		{
			const int dimBlock = 256;
			nodeDegree.initialize("Edge Support", unified, g.numNodes, 0);
			uint dimGridNodes = (g.numNodes + dimBlock - 1) / dimBlock;
			execKernel(getNodeDegree_kernel<T>, dimGridNodes, dimBlock, false, nodeDegree.gdata(), g);

		}

		void findKcoreIncremental_async(int kmin, int kmax, COOCSRGraph_d<T>& g,
			const size_t nodeOffset = 0, const size_t edgeOffset = 0)
		{
			CUDA_RUNTIME(cudaSetDevice(dev_));
			constexpr int dimBlock = 32; //For edges and nodes

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
			AscendingGpu(g.numNodes,identity_arr_asc);

			GPUArray <uint> newRowPtr;
			GPUArray <uint> newColIndex_csr;
			GPUArray <uint> new_eid;
			GPUArray <uint> new_edge_offset_origin;
			GPUArray<bool> edge_deleted;           // Auxiliaries for shrinking graphs.


			processed.initialize("is Deleted (Processed)", unified, g.numNodes, 0);
			
			processed.setAll(false, false);
			

			CUDA_RUNTIME(cudaStreamSynchronize(stream_));

			uint numDeleted_l = 0;
			float minPercentage = 0.8;
			float percDeleted_l = 0.0;

			
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

				int iterations = 0;
				while (current_q.count.gdata()[0] > 0)
				{

					todo -= current_q.count.gdata()[0];
					if (0 == todo) {
						break;
					}

					next_q.count.gdata()[0] = 0;
					if (level == 0)
					{
						auto block_size = 256;
						auto grid_size = (current_q.count.gdata()[0] + block_size - 1) / block_size;
						//execKernel(update_processed, grid_size, block_size, false, current_q.queue.gdata(), current_q.count.gdata()[0], current_q.mark.gdata(), processed.gdata());
					}
					else
					{
						//tcCounter->count_moveNext_per_edge_async(g, numEdges,
						//	level, processed,
						//	edgeSupport,
						//	current_q,
						//	next_q,
						//	bucket_q, bucket_level_end_,
						//	0, Warp);
						//tcCounter->sync();
						auto block_size = 256;
						auto grid_size = (current_q.count.gdata()[0] + block_size - 1) / block_size;
						auto grid_warp_size = (32*current_q.count.gdata()[0] + block_size - 1) / block_size;
						auto grid_block_size = current_q.count.gdata()[0];
						if (level < 4)
						{
							execKernel((kernel_thread_level_next<T>), grid_size, block_size, false,
								g,
								level, processed.gdata(), nodeDegree.gdata(),
								current_q.device_queue->gdata()[0],
								next_q.device_queue->gdata()[0],
								bucket_q.device_queue->gdata()[0],
								bucket_level_end_);
						}
						else if (level <= 16)
						{
							const int P = 16;
							auto gridSize = (P * current_q.count.gdata()[0] + block_size - 1) / block_size;

							execKernel((kernel_partition_level_next<T, 256, P>), gridSize, block_size, false,
								g,
								level, processed.gdata(), nodeDegree.gdata(),
								current_q.device_queue->gdata()[0],
								next_q.device_queue->gdata()[0],
								bucket_q.device_queue->gdata()[0],
								bucket_level_end_);
						}
						else if(level <= 32)
						{
							execKernel((kernel_partition_level_next<T, 256, 32>), grid_warp_size, block_size, false,
								g,
								level, processed.gdata(), nodeDegree.gdata(),
								current_q.device_queue->gdata()[0],
								next_q.device_queue->gdata()[0],
								bucket_q.device_queue->gdata()[0],
								bucket_level_end_);
						}
					/*	else if (level <= 64)
						{
							const int P = 64;
							auto gridSize = (P * current_q.count.gdata()[0] + block_size - 1) / block_size;
							execKernel((kernel_partition_level_next<T, 256, P>), gridSize, block_size, false,
								g,
								level, processed.gdata(), nodeDegree.gdata(),
								current_q.device_queue->gdata()[0],
								next_q.device_queue->gdata()[0],
								bucket_q.device_queue->gdata()[0],
								bucket_level_end_);
						}

						else if (level <= 128)
						{
							const int P = 128;
							auto gridSize = (P * current_q.count.gdata()[0] + block_size - 1) / block_size;
							execKernel((kernel_partition_level_next<T, 256, P>), gridSize, block_size, false,
								g,
								level, processed.gdata(), nodeDegree.gdata(),
								current_q.device_queue->gdata()[0],
								next_q.device_queue->gdata()[0],
								bucket_q.device_queue->gdata()[0],
								bucket_level_end_);
						}*/
						else
						{
							execKernel((kernel_block_level_next<T, 256>), grid_block_size, block_size, false,
								g,
								level, processed.gdata(), nodeDegree.gdata(),
								current_q.device_queue->gdata()[0],
								next_q.device_queue->gdata()[0],
								bucket_q.device_queue->gdata()[0],
								bucket_level_end_);
						}

						//execKernel(update_processed, grid_size, block_size, false, current_q.queue.gdata(), current_q.count.gdata()[0], current_q.mark.gdata(), processed.gdata());
					}

					numDeleted_l = g.numNodes - todo;

					swap(current_q, next_q);
					iterations++;
					//swap(inCurr, inNext);
					//current_q.count.gdata()[0] = next_q.count.gdata()[0];
				}
				//printf("Level %d took %d iterations \n", level, iterations);

				percDeleted_l = (numDeleted_l) * 1.0 / (g.numNodes);
				if (percDeleted_l >= 1.0)
				{
					break;
				}
				else
				{
					if (percDeleted_l > 0.2)
					{
						//StreamComapction(g.numNodes, g.numEdges, g.rowPtr_csr, g.colInd_csr, g.eid, processed,
						//	edge_deleted,
						//	newRowPtr, new_eid, newColIndex_csr,
						//	numEdges, todo);

						//numEdges = todo;
					}

				}
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

		uint count() const { return k-1; }
		int device() const { return dev_; }
		cudaStream_t stream() const { return stream_; }
	};

} // namespace pangolin