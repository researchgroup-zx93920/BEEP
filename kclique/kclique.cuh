
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

#include "kckernels.cuh"

template <typename T, uint BLOCK_DIM_X>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_block_warp_binary_count(
	uint64* counter,
	graph::COOCSRGraph_d<T> g,
	T kclique,
	T maxDeg,
	const  graph::GraphQueue_d<T, bool>  current,
	T* current_level,
	uint64* cpn,
	T conc_blocks_per_SM,
	T* levelStats,
	T* adj_enc
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
	__shared__ T l[warpsPerBlock];
	__shared__ T new_level[warpsPerBlock];
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

			//printf("src = %u, srcLen = %u\n", src, srcLen);
		}
		__syncthreads();
		T num_divs = (maxDeg + 32 - 1) / 32;
		T num_divs_local = (srcLen + 32 - 1) / 32;
		T encode_offset = sm_id * conc_blocks_per_SM * (maxDeg * num_divs) + levelPtr * (maxDeg * num_divs);
		T* encode = &adj_enc[encode_offset  /*srcStart[wx]*/];

		//num_divs = (srcLen + 32 - 1) / 32;
		//Encode
		for (unsigned long long j = wx; j < srcLen; j += warpsPerBlock)
		{
			if (lx == 0)
			{
				T src2 = g.colInd[srcStart + j];
				src2Start[wx] = g.rowPtr[src2];
				src2Len[wx] = g.rowPtr[src2 + 1] - src2Start[wx];
			}

			for (unsigned long long k = lx; k < num_divs_local; k += 32)
			{
				encode[j * num_divs_local + k] = 0x00;
			}
			__syncwarp();
			uint64 warpCount = 0;
			if (srcLen < src2Len[wx])
			{
				warpCount += graph::warp_sorted_count_and_encode<WARPS_PER_BLOCK, T, true>(&g.colInd[srcStart], srcLen,
					&g.colInd[src2Start[wx]], src2Len[wx],
					true, &encode[j * num_divs_local], l[wx] + 1, kclique);
			}
			else {
				warpCount += graph::warp_sorted_count_and_encode<WARPS_PER_BLOCK, T, true>(&g.colInd[src2Start[wx]], src2Len[wx],
					&g.colInd[srcStart], srcLen,
					false, &encode[j * num_divs_local], l[wx] + 1, kclique);
			}
		}

		__syncthreads(); //Done encoding

		for (unsigned long long j = wx; j < srcLen; j += warpsPerBlock)
		{
			if (lx < 5)
			{
				level_count[wx][lx] = 0;
				level_index[wx][lx] = 0;
				level_prev_index[wx][lx] = 0;
			}
			else if (lx == 7)
			{
				l[wx] = 2;
				new_level[wx] = 2;
				current_node_index[wx] = UINT_MAX;
				clique_count[wx] = 0;
			}

			__syncwarp();
			T level_offset = sm_id * conc_blocks_per_SM * (warpsPerBlock * num_divs * 5) + levelPtr * (warpsPerBlock * num_divs * 5);
			T* cl = &current_level[level_offset + wx * (num_divs_local * 5) /*srcStart[wx]*/];
			for (unsigned long long k = lx; k < num_divs_local * 5; k += 32)
			{
				cl[k] = 0x00;
			}


			//get warp count ??
			uint64 warpCount = 0;
			for (unsigned long long k = lx; k < num_divs_local; k += 32)
			{
				warpCount += __popc(encode[j * num_divs_local + lx]);
			}
			warpCount += __shfl_down_sync(0xFFFFFFFF, warpCount, 16);
			warpCount += __shfl_down_sync(0xFFFFFFFF, warpCount, 8);
			warpCount += __shfl_down_sync(0xFFFFFFFF, warpCount, 4);
			warpCount += __shfl_down_sync(0xFFFFFFFF, warpCount, 2);
			warpCount += __shfl_down_sync(0xFFFFFFFF, warpCount, 1);

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


			__syncwarp();
			while (level_count[wx][l[wx] - 2] > level_index[wx][l[wx] - 2])
			{
				/*if (lx == 0)
				{
					printf("HERE : #DIVS=%u, j = %llu, Level = %u, index = %u of %u, curr=%u\n", num_divs, j, l[wx], level_index[wx][l[wx] - 2], level_count[wx][l[wx] - 2], current_node_index[wx]);
				}
				*/

				//First Index
				T* from = l[wx] == 3 ? &(encode[num_divs_local * j]) : &(cl[num_divs_local * (l[wx] - 2)]);
				T* to = &(cl[num_divs_local * (l[wx] - 1)]);



				T startDiv = level_prev_index[wx][l[wx] - 2] / 32;
				T first_mask = ~((1 << (level_prev_index[wx][l[wx] - 2] & 0x1F)) - 1);
				T warpsDiv = (num_divs_local + 31) / 32;

				//printf("StartDiv = %u, firstMask = %u, warpDiv = %u\n", startDiv, first_mask, warpsDiv);

				for (T k = 0; k < warpsDiv; k++)
				{
					T index = startDiv + k * 32 + lx;
					T val = index < num_divs_local ? from[index] : 0;
					unsigned int mask = (index == startDiv) ? first_mask : 0xFFFFFFFF;
					unsigned int oneIndex = mask & val;
					unsigned int newmask = __ballot_sync(0xFFFFFFFF, oneIndex);
					if (newmask != 0)
					{
						uint elected_lane_deq = __ffs(newmask) - 1;
						current_node_index[wx] = __shfl_sync(0xFFFFFFFF, index * 32 + __ffs(oneIndex) - 1, elected_lane_deq, 32);
						break;
					}
				}

				/*if (lx == 0)
				{
					printf("j = %llu, Level = %u, index = %u of %u, curr=%u\n", j, l[wx], level_index[wx][l[wx] - 2], level_count[wx][l[wx] - 2], current_node_index[wx]);
				}*/

				if (lx == 0)
				{
					//current_node_index[0] = finalIndex;
					level_prev_index[wx][l[wx] - 2] = current_node_index[wx] + 1;
					level_index[wx][l[wx] - 2]++;
					new_level[wx] = l[wx];
				}

				__syncwarp();

				if (current_node_index[wx] >= srcLen)
					printf("Wrong src = %u, srcLen = %u, level = %u  \n", src, srcLen, l[wx]);

				//Intersect
				uint64 warpCount = 0;
				for (T k = lx; k < num_divs_local; k += 32)
				{
					to[k] = from[k] & encode[current_node_index[wx] * num_divs_local + k];
					warpCount += __popc(to[k]);
				}
				warpCount += __shfl_down_sync(0xFFFFFFFF, warpCount, 16);
				warpCount += __shfl_down_sync(0xFFFFFFFF, warpCount, 8);
				warpCount += __shfl_down_sync(0xFFFFFFFF, warpCount, 4);
				warpCount += __shfl_down_sync(0xFFFFFFFF, warpCount, 2);
				warpCount += __shfl_down_sync(0xFFFFFFFF, warpCount, 1);

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
				//Readjust
				__syncwarp();
				if (lx == 0)
				{
					while (new_level[wx] > 3 && level_index[wx][new_level[wx] - 2] >= level_count[wx][new_level[wx] - 2])
					{
						(new_level[wx])--;
					}
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

			__syncwarp();
		}

		__syncthreads();
	}

	__syncthreads();
	if (threadIdx.x == 0)
	{
		atomicCAS(&levelStats[sm_id * conc_blocks_per_SM + levelPtr], 1, 0);
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
			const int dimBlock = 256;
			nodeDegree.initialize("Edge Support", unified, g.numNodes, dev_);
			uint dimGridNodes = (g.numNodes + dimBlock - 1) / dimBlock;
			execKernel((getNodeDegree_kernel<T, dimBlock>), dimGridNodes, dimBlock, dev_, false, nodeDegree.gdata(), g, maxDegree);
		}

		void findKclqueIncremental_node_async(int kcount, COOCSRGraph_d<T>& g,
			ProcessingElementEnum pe, const size_t nodeOffset = 0, const size_t edgeOffset = 0)
		{
			CUDA_RUNTIME(cudaSetDevice(dev_));
			const auto block_size = 256;
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
					//current_q.count.gdata()[0] = 64;
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
						const uint dv = 32;
						const uint max_level = 5;
						uint num_divs = (maxDegree.gdata()[0] + dv - 1) / dv;
						const uint64 level_size = num_SMs * conc_blocks_per_SM * factor * max_level * num_divs;
						const uint64 encode_size = num_SMs * conc_blocks_per_SM * maxDegree.gdata()[0] * num_divs;
						printf("Level Size = %llu, Encode Size = %llu\n", level_size, encode_size);
						GPUArray<T> current_level2("Temp level Counter", unified, level_size, dev_);
						GPUArray<T> node_be("Temp level Counter", unified, encode_size, dev_);
						current_level2.setAll(0, true);
						node_be.setAll(0, true);



						auto grid_block_size = current_q.count.gdata()[0];
						execKernel((kckernel_node_block_warp_binary_count<T, block_size>), grid_block_size, block_size, dev_, false,
							counter.gdata(),
							g,
							kcount,
							maxDegree.gdata()[0],
							current_q.device_queue->gdata()[0],
							current_level2.gdata(), cpn.gdata(),
							conc_blocks_per_SM, d_bitmap_states.gdata(), node_be.gdata());


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
					//current_q.count.gdata()[0] = 2000;

					if (pe == Warp)
					{
						auto grid_block_size = (32 * current_q.count.gdata()[0] + block_size - 1) / block_size;
						execKernel((kckernel_edge_warp_count2_shared<T, block_size>), grid_block_size, block_size, dev_, false,
							counter.gdata(),
							g,
							kcount,
							maxDegree.gdata()[0],
							current_q.device_queue->gdata()[0],
							current_level.gdata(),
							NULL,
							conc_blocks_per_SM, d_bitmap_states.gdata());

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
