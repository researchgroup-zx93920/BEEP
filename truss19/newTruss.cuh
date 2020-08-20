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




template <size_t BLOCK_DIM_X>
__global__ void core_direct(
	uint *edgeSupport,
	uint* gnumdeleted, uint* gnumaffected,
	const uint k, const size_t edgeStart, const size_t numEdges,
	uint* rowPtr, uint* rowInd, uint* colInd, const size_t numNodes, BCTYPE* keep, bool* affected, uint* reversed, bool firstTry, const int uMax, int* reverseIndex, EncodeDataType* bitMap)

{
	// kernel call
	size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
	uint numberDeleted = 0;
	uint numberAffected = 0;
	__shared__ bool didAffectAnybody[1];
	bool ft = firstTry; //1
	if (0 == threadIdx.x)
		didAffectAnybody[0] = false;

	__syncthreads();
	
	numberDeleted = 0;
	for (size_t i = gx + edgeStart; i < edgeStart + numEdges; i += BLOCK_DIM_X * gridDim.x)
	{
		uint srcNode = rowInd[i];
		uint dstNode = colInd[i];
		if (keep[i] && srcNode < dstNode && (affected[i] || ft))
		{
			affected[i] = false;
			uint triCount = edgeSupport[i];

			//////////////////////////////////////////////////////////////
			if (triCount < (k - 2))
			{
				uint ir = reversed[i];
				keep[i] = false;
				keep[ir] = false;

				uint sp = rowPtr[srcNode];
				uint send = rowPtr[srcNode + 1];

				uint dp = rowPtr[dstNode];
				uint dend = rowPtr[dstNode + 1];

				while (triCount > 0 && sp < send && dp < dend)
				{
					uint sv = colInd[sp];
					uint dv = colInd[dp];

					if ((sv == dv))
					{
						numberAffected += AffectOthers(sp, dp, keep, affected, reversed);
					}
					uint yy = sp + ((sv <= dv) ? 1 : 0);
					dp = dp + ((sv >= dv) ? 1 : 0);
					sp = yy;
				}
			}
		}

		if (!keep[i] && srcNode < dstNode)
			numberDeleted++;
	}

	//Instead of reduction: hope it works
	if (numberAffected > 0)
		didAffectAnybody[0] = true;

	__syncthreads();

	if (0 == threadIdx.x)
	{
		if (didAffectAnybody[0])
			*gnumaffected = 1;
	}

	// Block-wide reduction of threadCount
	typedef cub::BlockReduce<uint, BLOCK_DIM_X> BlockReduce;
	__shared__ typename BlockReduce::TempStorage tempStorage;
	uint deletedByBlock = BlockReduce(tempStorage).Sum(numberDeleted);

	//uint affectedByBlock = BlockReduce(tempStorage).Sum(numberAffected);

	if (0 == threadIdx.x)
	{
		atomicAdd(gnumdeleted, deletedByBlock);
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
	template<typename T>
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
			GPUArray<int> edgeSupport, uint32_t edge_num, int level,
			GPUArray<int>& curr, GPUArray<bool>& inCurr, int& curr_cnt, 
			GPUArray<uint> asc,
			GPUArray<bool>& in_bucket_window_,
			GPUArray<uint>& bucket_buf_, uint*& window_bucket_buf_size_, int& bucket_level_end_)
		{
			static bool is_first = true;
			if (is_first)
			{
				inCurr.setAll(0, true);
				in_bucket_window_.setAll(0, true);
				is_first = false;
			}

			if (level == bucket_level_end_)
			{
				// Clear the bucket_removed_indicator
				

				long grid_size = (edge_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
				execKernel(filter_window, grid_size, BLOCK_SIZE, false,
					edgeSupport.gdata(), edge_num, in_bucket_window_.gdata(), level, bucket_level_end_ + LEVEL_SKIP_SIZE);

				*window_bucket_buf_size_ = CUBSelect(asc.gdata(), bucket_buf_.gdata(), in_bucket_window_.gdata(), edge_num);
				bucket_level_end_ += LEVEL_SKIP_SIZE;
			}
			// SCAN the window.
			if (*window_bucket_buf_size_ != 0)
			{
				curr_cnt = 0;
				long grid_size = (*window_bucket_buf_size_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
				execKernel(filter_with_random_append, grid_size, BLOCK_SIZE, false,
					bucket_buf_.gdata(), *window_bucket_buf_size_, edgeSupport.gdata(), inCurr.gdata(), curr.gdata(), &curr_cnt, level);
			}
			else
			{
				curr_cnt = 0;
			}
			Log(LogPriorityEnum::info, "Level: %d, curr: %d/%d", level, curr_cnt, *window_bucket_buf_size_);
		}


		void PrepareQueues(int n, int m,
			GPUArray<int>& next_cnt, GPUArray<int>& affected_cnt,
			GPUArray<int>& curr, GPUArray<bool>& inCurr, 
			GPUArray<int>& next, GPUArray<bool>& inNext,
			GPUArray<int>& affected, GPUArray<int>& inAffected,
			GPUArray<uint>& identity_arr_asc)
		{
			// 1st: CSR/Eid/Edge List. --> not necessary

			uint32_t edge_num = m;

			// 3rd: Queue Related.

			next_cnt.initialize("Next Count", AllocationTypeEnum::unified, 1, 0);
			affected_cnt.initialize("Affected Count", AllocationTypeEnum::unified, 1, 0);

			curr.initialize("Curr", AllocationTypeEnum::unified, edge_num, 0);
			next.initialize("Next", AllocationTypeEnum::unified, edge_num, 0);
			affected.initialize("Next", AllocationTypeEnum::unified, edge_num, 0);



			inCurr.initialize("In Curr", AllocationTypeEnum::unified, edge_num, 0);
			inNext.initialize("in Next", AllocationTypeEnum::unified, edge_num, 0);
			inAffected.initialize("in Affected", AllocationTypeEnum::unified, edge_num, 0);

			// 4th: Keep the edge offset mapping.
			long grid_size = (edge_num + BLOCK_SIZE - 1) / BLOCK_SIZE;


			identity_arr_asc.initialize("Identity Array Asc", AllocationTypeEnum::unified, edge_num, 0);

			execKernel(init_asc, grid_size, BLOCK_SIZE, false, identity_arr_asc.gdata(), edge_num);
		}



		void StreamComapction(
			int n, T& m,
			GPUArray<T>& rowPtr, GPUArray<T>& colInd, GPUArray<T>& eid,
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
				rowPtr.gdata(), n, eid.gdata(), processed.gdata(), new_offset.gdata(), edge_deleted.gdata());

			uint total = CUBScanExclusive<uint, uint>(new_offset.gdata(), new_offset.gdata(), n);
			new_offset.gdata()[n] = total;
			//assert(total == new_edge_num * 2);
			cudaDeviceSynchronize();

			swap_ele(rowPtr.gdata(), new_offset.gdata());

			/*new adj and eid construction*/
			CUBSelect(colInd.gdata(), new_adj.gdata(), edge_deleted.gdata(), old_edge_num * 2);
			CUBSelect(eid.gdata(), new_eid.gdata(), edge_deleted.gdata(), old_edge_num * 2);

			swap_ele(colInd.gdata(), new_adj.gdata());
			colInd.N = new_adj.N;

			swap_ele(eid.gdata(), new_eid.gdata());
			eid.N = new_eid.N;

			m = new_edge_num * 2;
		}

	public:
		BCTYPE* gKeep, * gPrevKeep;
		bool* gAffected;
		uint* gReveresed;

	public:
		SingleGPU_KtrussMod(int dev) : dev_(dev) {
			CUDA_RUNTIME(cudaSetDevice(dev_));
			CUDA_RUNTIME(cudaStreamCreate(&stream_));
			CUDA_RUNTIME(cudaMallocManaged(&gnumdeleted, 2 * sizeof(*gnumdeleted)));
			CUDA_RUNTIME(cudaMallocManaged(&gnumaffected, sizeof(uint)));
			CUDA_RUNTIME(cudaMallocManaged(&selectedOut, sizeof(*selectedOut)));


			CUDA_RUNTIME(cudaStreamSynchronize(stream_));
			//zero_async<2>(gnumdeleted, dev_, stream_); // zero on the device that will do the counting
			//zero_async<1>(gnumaffected, dev_, stream_); // zero on the device that will do the counting

			gnumdeleted[0] = 0;
			gnumaffected[0] = 0;

			CUDA_RUNTIME(cudaStreamSynchronize(stream_));
		}

		SingleGPU_KtrussMod() : SingleGPU_KtrussMod(0) {}

		void findKtrussIncremental_async(int kmin, int kmax, TcBase<T>* tcCounter, GPUArray<T> rowPtr_csr, GPUArray<T> colIndex_csr,
			GPUArray<T> rowInd, GPUArray<T> colInd, GPUArray<T> eid,
			const size_t numNodes, T totalNumEdges, int* reverseIndex, EncodeDataType* bitMap, const size_t nodeOffset = 0, const size_t edgeOffset = 0)
		{

			T numEdges = totalNumEdges / 2;

			CUDA_RUNTIME(cudaSetDevice(dev_));
			constexpr int dimBlock = 32; //For edges and nodes

			bool firstTry = true;
			GPUArray<BCTYPE> processed; //isDeleted
			GPUArray<T> reversed, srcKP, destKP;


			int level = 0;
			int bucket_level_end_ = level;

			GPUArray<int> edgeSupport;
			GPUArray<bool> edge_kept; 

			//Lets apply queues and buckets
			GPUArray <bool> in_bucket_window_;
			GPUArray <uint> bucket_buf_;
			GPUArray<uint> window_bucket_buf_size_; //should be uint* only
			PrepareBucket(in_bucket_window_, bucket_buf_, window_bucket_buf_size_, numEdges);

			//Queues
			GPUArray<T> identity_arr_asc;
			GPUArray<int> curr, next, affected;
			GPUArray<bool> inCurr, inNext;
			GPUArray<int> inAffedted;
			GPUArray<int> curr_cnt_ptr, next_cnt, affected_cnt;
			curr_cnt_ptr.initialize("Curr Count Pointer", AllocationTypeEnum::unified, 1, 0);
			int*& curr_cnt = curr_cnt_ptr.gdata();
			PrepareQueues(numNodes, numEdges, next_cnt, affected_cnt, curr, inCurr, next, inNext, affected, inAffedted, identity_arr_asc);
			

			GPUArray <uint> newRowPtr;
			GPUArray <uint> newColIndex_csr;
			GPUArray <uint> new_eid;
			GPUArray<int> new_EdgeSupport;
			GPUArray <uint> new_edge_offset_origin;
			GPUArray<bool> reversed_processed;     // Auxiliaries for shrinking graphs.
			GPUArray<bool> edge_deleted;           // Auxiliaries for shrinking graphs.
			GPUArray <uint> scanned_processed;     // Auxiliaries for shrinking graphs.

		

			//Only Pointers: no inialization
			GPUArray<T> ptrSrc, ptrDst;
			GPUArray<T> s1, d1, s2, d2;


			processed.initialize("is Deleted (Processed)", unified, numEdges, 0);
			edgeSupport.initialize("Edge Support", unified, numEdges, 0);

			uint dimGridEdges = (numEdges + dimBlock - 1) / dimBlock;

			processed.setAll(false, false);

			CUDA_RUNTIME(cudaStreamSynchronize(stream_));

			uint numDeleted_l = 0;
			float minPercentage = 0.8;
			float percDeleted_l = 0.0;
			bool startIndirect = false;

			s1.gdata() = rowInd.gdata();
			d1 = colInd;

			s2.gdata() = srcKP.gdata();
			d2 = destKP;

			ptrSrc.gdata() = s1.gdata();
			ptrDst.gdata() = d1.gdata();


			tcCounter->count_per_edge_eid_async(edgeSupport, rowPtr_csr, colIndex_csr, ptrSrc, ptrDst, numEdges, 0, Warp);
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
				bucket_scan(edgeSupport, todo_original, level, curr, inCurr, *curr_cnt, identity_arr_asc, in_bucket_window_, bucket_buf_, window_bucket_buf_size_.gdata(), bucket_level_end_);

				while (*curr_cnt > 0)
				{

					todo -= *curr_cnt;
					if (0 == todo) {
						break;
					}

					*next_cnt.gdata() = 0;
					*affected_cnt.gdata() = 0;
					if (level == 0) 
					{
						auto block_size = 256;
						auto grid_size = (*curr_cnt + block_size - 1) / block_size;
						execKernel(update_processed, grid_size, block_size, false, curr.gdata(), *curr_cnt, inCurr.gdata(), processed.gdata());
					}
					else
					{
						tcCounter->affect_per_edge_level_q_async(rowPtr_csr, colIndex_csr,
							ptrSrc, ptrDst, eid, numEdges,
							level, processed,
							edgeSupport,
							curr, inCurr, *curr_cnt,
							affected, inAffedted, affected_cnt, 
							next, inNext, next_cnt,
							in_bucket_window_, bucket_buf_, window_bucket_buf_size_, bucket_level_end_,
							0,Thread);
						tcCounter->sync();
				
						auto block_size = 256;
						auto grid_size = (*curr_cnt + block_size - 1) / block_size;
						execKernel(update_processed, grid_size, block_size, false, curr.gdata(), *curr_cnt, inCurr.gdata(), processed.gdata());
					}

					numDeleted_l = numEdges - todo;

					swap(curr, next);
					swap(inCurr, inNext);
					*curr_cnt = *next_cnt.gdata();
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
						StreamComapction(numNodes, totalNumEdges, rowPtr_csr, colIndex_csr, eid, processed,
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

		uint findKtrussIncremental_sync(int kmin, int kmax, TcBase<T> *tcCounter, GPUArray<T> rowPtr_csr, GPUArray<T> colIndex_csr,
			GPUArray<T> rowInd, GPUArray<T> colInd, GPUArray<T> eid,
			const size_t numNodes, const size_t numEdges, int* reverseIndex, EncodeDataType* bitMap, const size_t nodeOffset = 0, const size_t edgeOffset = 0)
		{
			findKtrussIncremental_async(kmin, kmax, tcCounter, rowPtr_csr, colIndex_csr, rowInd, colInd, eid, numNodes, numEdges, reverseIndex, bitMap, nodeOffset, edgeOffset);
			sync();
			return count();
		}

		void sync() { CUDA_RUNTIME(cudaStreamSynchronize(stream_)); }

		uint count() const { return k; }
		int device() const { return dev_; }
		cudaStream_t stream() const { return stream_; }
	};

} // namespace pangolin
