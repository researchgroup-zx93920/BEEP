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

#include "ourtruss19.cuh"
#include "../include/GraphQueue.cuh"


#include "../include/ScanLarge.cuh"


typedef uint64_t EncodeDataType;


template <size_t BLOCK_DIM_X>
__global__ void PrepareArrays(uint edgeStart, uint numEdges, uint* rowPtr, uint* rowInd, uint* colInd, BCTYPE* processed, uint* reversed, uint* srcKP, uint* destKP)
{
	uint tx = threadIdx.x;
	uint bx = blockIdx.x;

	uint ptx = tx + bx * BLOCK_DIM_X;

	for (uint i = ptx + edgeStart; i < edgeStart + numEdges; i += BLOCK_DIM_X * gridDim.x)
	{
		//node
		uint sn = rowInd[i];
		uint dn = colInd[i];
		//length
		uint sl = rowPtr[sn + 1] - rowPtr[sn];
		uint dl = rowPtr[dn + 1] - rowPtr[dn];

		processed[i] = false;

		reversed[i] = getEdgeId(rowPtr, colInd, dn, sn);
		srcKP[i] = i;
		destKP[i] = i;
	}
}

__global__ void reverse_processed(uint edgeStart, uint numEdges, bool* processed, bool* reversed_processed)
{
	uint tx = threadIdx.x;
	uint bx = blockIdx.x;

	uint ptx = tx + bx * blockDim.x;

	for (uint i = ptx + edgeStart; i < edgeStart + numEdges; i += blockDim.x * gridDim.x)
	{
		reversed_processed[i] = !processed[i];
	}
}




template <size_t BLOCK_DIM_X>
__global__ void RebuildArraysN(uint edgeStart, uint numEdges, uint* rowPtr, uint* rowInd, BCTYPE* processed)
{
	uint tx = threadIdx.x;
	uint bx = blockIdx.x;

	__shared__ uint rows[BLOCK_DIM_X + 1];

	uint ptx = tx + bx * BLOCK_DIM_X;

	for (int i = ptx + edgeStart; i < edgeStart + numEdges; i += BLOCK_DIM_X * gridDim.x)
	{
		rows[tx] = rowInd[ptx];

		processed[i] = false;

		__syncthreads();

		uint end = rows[tx];
		if (i == 0)
		{
			rowPtr[end] = 0;
		}
		else if (tx == 0)
		{
			uint start = rowInd[i - 1];
			for (uint j = start + 1; j <= end; j++)
			{
				rowPtr[j] = i;
			}
		}
		else if (i == numEdges - 1)
		{
			rowPtr[end + 1] = i + 1;

			uint start = rows[tx - 1];
			for (uint j = start + 1; j <= end; j++)
			{
				rowPtr[j] = i;
			}
		}
		else
		{
			uint start = rows[tx - 1];
			for (uint j = start + 1; j <= end; j++)
			{
				rowPtr[j] = i;
			}

		}
	}

}

__global__
void update_queueu(int* curr, uint32_t curr_cnt, int* inCurr) {
	auto gtid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gtid < curr_cnt) {
		auto edge_off = curr[gtid];
		inCurr[edge_off] = 0;
	}
}


__global__ void  update_support(int* count,
	int level, bool* processed,
	int* affected, int *inAffected, int affected_cnt,
	int* next, bool* inNext, int& next_cnt,
	bool* in_bucket_window_, uint* bucket_buf_, uint& window_bucket_buf_size_, int bucket_level_end_,
	uint* reversed
)
{

	size_t gx = blockDim.x * blockIdx.x + threadIdx.x;
	for (size_t i = gx; i < affected_cnt; i += blockDim.x * gridDim.x)
	{
		int edgeId = affected[i];
		
		count[edgeId] -= inAffected[edgeId];
		inAffected[edgeId] = 0;
		auto currCount = count[edgeId];

		if (currCount <= level) {
			count[edgeId] = level;
			auto insert_idx = atomicAdd(&next_cnt, 1);
			next[insert_idx] = edgeId;
			inNext[edgeId] = true;
		}
		else if (currCount > level && currCount < bucket_level_end_)
		{
			auto old_token = atomicCASBool(in_bucket_window_ + edgeId, false, true);
			if (!old_token) {
				auto insert_idx = atomicAdd(&window_bucket_buf_size_, 1);
				bucket_buf_[insert_idx] = edgeId;
			}
		}
		

	}	
}


namespace graph
{
	template<typename T, typename PeelT>
	class SingleGPU_KtrussMod
	{
	private:
		int dev_;
		cudaStream_t stream_;
		uint* selectedOut;

		uint* gnumdeleted;
		uint* gnumaffected;
		bool assumpAffected;

		//Outputs:
		//Max k of a complete ktruss kernel
		int k;

		//Percentage of deleted edges for a specific k
		float percentage_deleted_k;


		void bucket_scan(
			GPUArray<PeelT> edgeSupport, T edge_num, T level,
			GraphQueue<T, bool>& current,
			GPUArray<T> asc,
			GraphQueue<T, bool>& bucket,
			T& bucket_level_end_)
		{

			CUDA_RUNTIME(cudaSetDevice(dev_));
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
				

				auto grid_size = (edge_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
				execKernel((filter_window<T, PeelT>), grid_size, BLOCK_SIZE, dev_, false,
					edgeSupport.gdata(), edge_num, bucket.mark.gdata(), level, bucket_level_end_ + LEVEL_SKIP_SIZE);

				bucket.count.gdata()[0] = CUBSelect(asc.gdata(), bucket.queue.gdata(), bucket.mark.gdata(), edge_num);
				bucket_level_end_ += LEVEL_SKIP_SIZE;
			}
			// SCAN the window.
			if (bucket.count.gdata()[0] != 0)
			{
				current.count.gdata()[0] = 0;
				long grid_size = (bucket.count.gdata()[0] + BLOCK_SIZE - 1) / BLOCK_SIZE;
				execKernel(filter_with_random_append, grid_size, BLOCK_SIZE, dev_, false,
					bucket.queue.gdata(), bucket.count.gdata()[0], edgeSupport.gdata(), current.mark.gdata(), current.queue.gdata(), &(current.count.gdata()[0]), level);
			}
			else
			{
				current.count.gdata()[0] = 0;
			}
			Log(LogPriorityEnum::info, "Level: %d, curr: %d/%d", level, current.count.gdata()[0] , bucket.count.gdata()[0]);
		}


		void AscendingGpu(int n, int m, GPUArray<uint>& identity_arr_asc)
		{
			CUDA_RUNTIME(cudaSetDevice(dev_));

			// 4th: Keep the edge offset mapping.
			long grid_size = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;


			identity_arr_asc.initialize("Identity Array Asc", AllocationTypeEnum::unified, m, 0);

			execKernel(init_asc, grid_size, BLOCK_SIZE, dev_, false, identity_arr_asc.gdata(), m);
		}



		void StreamComapction(
			int n, T& m,
			T*& rowPtr, T*& colInd, T*& eid,
			GPUArray<bool> processed,
			 GPUArray<bool>& edge_deleted,
			GPUArray <T>& new_offset, GPUArray<T>& new_eid, GPUArray <T>& new_adj,
			T old_edge_num, T new_edge_num)
		{

			CUDA_RUNTIME(cudaSetDevice(dev_));
			static bool shrink_first_time = true;
			if (shrink_first_time) { //shrink first time, allocate the buffers
				shrink_first_time = false;
				Timer alloc_timer;
				new_adj.initialize("New Adj", gpu, new_edge_num * 2, dev_);
				new_eid.initialize("New EID", gpu, new_edge_num * 2, dev_);
				new_offset.initialize("New Row Pointer", unified, (n + 1), dev_);
				edge_deleted.initialize("Edge deleted", gpu, old_edge_num * 2, dev_);
			}
			graph::CubLarge<uint> s;

			/*2. construct new CSR (offsets, adj) and rebuild the eid*/
			int block_size = 128;
			// Attention: new_offset gets the histogram.
			execKernel((warp_detect_deleted_edges<T>), GRID_SIZE, block_size, dev_, true,
				rowPtr, n, eid, processed.gdata(), new_offset.gdata(), edge_deleted.gdata());

			uint total = 0;
			if(n < INT_MAX)
				total = CUBScanExclusive<uint, uint>(new_offset.gdata(), new_offset.gdata(), n);
			else
				total = s.ExclusiveSum(new_offset.gdata(), new_offset.gdata(), n);

			new_offset.gdata()[n] = total;
			//assert(total == new_edge_num * 2);
			cudaDeviceSynchronize();

			swap_ele(rowPtr, new_offset.gdata());

			/*new adj and eid construction*/
			if (old_edge_num * 2 < INT_MAX)
			{
				CUBSelect(colInd, new_adj.gdata(), edge_deleted.gdata(), old_edge_num * 2);
				CUBSelect(eid, new_eid.gdata(), edge_deleted.gdata(), old_edge_num * 2);
			}
			else
			{
				s.Select(colInd, new_adj.gdata(), edge_deleted.gdata(), old_edge_num * 2);
				s.Select(eid, new_eid.gdata(), edge_deleted.gdata(), old_edge_num * 2);
			}

			swap_ele(colInd, new_adj.gdata());
			swap_ele(eid, new_eid.gdata());
			
			m = new_edge_num * 2;
		}



	public:
		SingleGPU_KtrussMod(int dev) : dev_(dev) {
			CUDA_RUNTIME(cudaSetDevice(dev_));
			CUDA_RUNTIME(cudaStreamCreate(&stream_));
			CUDA_RUNTIME(cudaMallocManaged(&gnumdeleted, 2 * sizeof(*gnumdeleted)));
			CUDA_RUNTIME(cudaMallocManaged(&selectedOut, sizeof(*selectedOut)));


			CUDA_RUNTIME(cudaStreamSynchronize(stream_));
			//zero_async<2>(gnumdeleted, dev_, stream_); // zero on the device that will do the counting
			//zero_async<1>(gnumaffected, dev_, stream_); // zero on the device that will do the counting

			gnumdeleted[0] = 0;
		
			CUDA_RUNTIME(cudaStreamSynchronize(stream_));
		}

		SingleGPU_KtrussMod() : SingleGPU_KtrussMod(0) {}

		void findKtrussIncremental_async(int kmin, int kmax, TcBase<T>* tcCounter, EidGraph_d<T>& g,
			int* reverseIndex, EncodeDataType* bitMap, const size_t nodeOffset = 0, const size_t edgeOffset = 0)
		{

			T numEdges = g.numEdges / 2;

			CUDA_RUNTIME(cudaSetDevice(dev_));
			constexpr int dimBlock = 32; //For edges and nodes

			bool firstTry = true;
			GPUArray<BCTYPE> processed; //isDeleted

			T level = 0;
			T bucket_level_end_ = level;

			GPUArray<PeelT> edgeSupport;
			GPUArray<bool> edge_kept; 

			//Lets apply queues and buckets
			//GPUArray <bool> in_bucket_window_;
			//GPUArray <uint> bucket_buf_;
			//GPUArray<uint> window_bucket_buf_size_; //should be uint* only
			//PrepareBucket(in_bucket_window_, bucket_buf_, window_bucket_buf_size_, numEdges);


			graph::GraphQueue<T, bool> bucket_q;
			bucket_q.Create(unified, numEdges, dev_);

			graph::GraphQueue<T, bool> current_q;
			current_q.Create(unified, numEdges, dev_);

			graph::GraphQueue<T, bool> next_q;
			next_q.Create(unified, numEdges, dev_);

		
			//Queues
			GPUArray<T> identity_arr_asc;
			AscendingGpu(g.numNodes, numEdges, identity_arr_asc);

			GPUArray <uint> newRowPtr;
			GPUArray <uint> newColIndex_csr;
			GPUArray <uint> new_eid;
			GPUArray <uint> new_edge_offset_origin;
		
			GPUArray<bool> edge_deleted;           // Auxiliaries for shrinking graphs.
		

			processed.initialize("is Deleted (Processed)", unified, numEdges, dev_);
			edgeSupport.initialize("Edge Support", unified, numEdges, dev_);

			uint dimGridEdges = (numEdges + dimBlock - 1) / dimBlock;

			processed.setAll(false, false);

			CUDA_RUNTIME(cudaStreamSynchronize(stream_));

			uint numDeleted_l = 0;
			float minPercentage = 0.8;
			float percDeleted_l = 0.0;
			bool startIndirect = false;

			tcCounter->count_per_edge_eid_async(edgeSupport, g, numEdges, 0, Warp);
			tcCounter->sync();

			int todo = numEdges;
			const auto todo_original = numEdges;
			while (todo > 0)
			{
			//	//printf("k=%d\n", k);
				numDeleted_l = 0;
				CUDA_RUNTIME(cudaGetLastError());
				cudaDeviceSynchronize();

				//1 bucket fill
				bucket_scan(edgeSupport, todo_original, level, current_q, identity_arr_asc, bucket_q, bucket_level_end_);

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
						execKernel((update_processed<T>), grid_size, block_size, dev_, false, current_q.queue.gdata(), current_q.count.gdata()[0], current_q.mark.gdata(), processed.gdata());
					}
					else
					{
						tcCounter->count_moveNext_per_edge_async(g, numEdges,
							level, processed,
							edgeSupport,
							current_q,
							next_q,
							bucket_q, bucket_level_end_,
							0,Block);
						tcCounter->sync();

						auto block_size = 256;
						auto grid_size = (current_q.count.gdata()[0] + block_size - 1) / block_size;
						execKernel(update_processed, grid_size, block_size, dev_, false, current_q.queue.gdata(), current_q.count.gdata()[0], current_q.mark.gdata(), processed.gdata());
					}

					numDeleted_l = numEdges - todo;

					swap(current_q, next_q);
					//swap(inCurr, inNext);
					//current_q.count.gdata()[0] = next_q.count.gdata()[0];


					
				}

				percDeleted_l = (numDeleted_l) * 1.0 / (numEdges);
				if (percDeleted_l >= 1.0)
				{
					break;
				}
				else
				{
					if (percDeleted_l > 0.2)
					{
						StreamComapction(g.numNodes, g.numEdges, g.rowPtr_csr, g.colInd_csr, g.eid, processed,
							edge_deleted,
							newRowPtr, new_eid, newColIndex_csr,
							numEdges, todo);

						numEdges = todo;
					}

				}
				level++;
			}

			k = level + 1;
		}

		uint findKtrussIncremental_sync(int kmin, int kmax, TcBase<T> *tcCounter, EidGraph_d<T>& g, int* reverseIndex, EncodeDataType* bitMap, const size_t nodeOffset = 0, const size_t edgeOffset = 0)
		{
			findKtrussIncremental_async(kmin, kmax, tcCounter, g, reverseIndex, bitMap, nodeOffset, edgeOffset);
			sync();
			return count();
		}

		void sync() { CUDA_RUNTIME(cudaStreamSynchronize(stream_)); }

		uint count() const { return k; }
		int device() const { return dev_; }
		cudaStream_t stream() const { return stream_; }
	};

} // namespace pangolin
