
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


template<typename T, uint CPARTSIZE>
__device__ __forceinline__ uint64 explore_branch(uint start_level, uint num_divs_local, uint* l, uint partMask,
	T startEncodeIndex, T* encode, T* cl, T* level_count, T* level_index, T* level_prev_index)
{
	const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
	const size_t lx = threadIdx.x % CPARTSIZE;
	uint64 globalCount = 0;


	while (level_count[l[wx] - start_level] > level_index[l[wx] - start_level])
	{
		T l_mi_sl = l[wx] - start_level;
		T* from = (l_mi_sl == 0) ? &(encode[num_divs_local * startEncodeIndex]) : &(cl[num_divs_local * l_mi_sl]);
		T* to = &(cl[num_divs_local * (l_mi_sl + 1)]);
		T maskBlock = level_prev_index[l_mi_sl] / 32;
		T maskIndex = ~((1 << (level_prev_index[l_mi_sl] & 0x1F)) - 1);
		T newIndex = __ffs(from[maskBlock] & maskIndex);
		while (newIndex == 0)
		{
			maskIndex = 0xFFFFFFFF;
			maskBlock++;
			newIndex = __ffs(from[maskBlock] & maskIndex);
		}
		newIndex = 32 * maskBlock + newIndex - 1;

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
		//warpCount += __shfl_down_sync(mm, warpCount, 16);
		//warpCount += __shfl_down_sync(partMask, warpCount, 8);
		//warpCount += __shfl_down_sync(partMask, warpCount, 4);
		warpCount += __shfl_down_sync(partMask, warpCount, 2);
		warpCount += __shfl_down_sync(partMask, warpCount, 1);

		if (lx == 0)
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
	__shared__ T l[numPartitions], tc, wtc[numPartitions];
	__shared__ uint32_t  sm_id, levelPtr;
	__shared__ T srcStart, srcLen;
	__shared__ T src2Start, src2Len;

	__shared__ T num_divs_local, encode_offset, * encode, tri_offset, * tri, scounter;



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
		else if (threadIdx.x == 2)
		{
			tri_offset = sm_id * CBPSM * (MAXDEG)+levelPtr * (MAXDEG);
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

		if (KCCOUNT == 3 && threadIdx.x == 0)
			atomicAdd(counter, scounter);


		__syncthreads();
		//Encode
		T partMask = (1 << CPARTSIZE) - 1;
		partMask = partMask << ((wx % (32 / CPARTSIZE)) * CPARTSIZE);
		T mm = (1 << scounter) - 1;
		mm = mm << ((wx / numPartitions) * CPARTSIZE);
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
		if (lx == 0)
			wtc[wx] = atomicAdd(&(tc), 1);
		__syncwarp(partMask);

		while (wtc[wx] < scounter)
		{

			T j = wtc[wx];


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

			for (unsigned long long k = lx; k < num_divs_local * 7; k += CPARTSIZE)
			{
				cl[k] = 0x00;
			}

			//get warp count ??
			uint64 warpCount = 0;
			for (unsigned long long k = lx; k < num_divs_local; k += CPARTSIZE)
			{
				warpCount += __popc(encode[j * num_divs_local + k]);
			}
			//warpCount += __shfl_down_sync(partMask, warpCount, 16);
			//warpCount += __shfl_down_sync(partMask, warpCount, 8);
			//warpCount += __shfl_down_sync(partMask, warpCount, 4);
			warpCount += __shfl_down_sync(partMask, warpCount, 2);
			warpCount += __shfl_down_sync(partMask, warpCount, 1);

			if (lx == 0 && l[wx] == KCCOUNT)
				atomicAdd(counter, warpCount);
			else if (lx == 0 && KCCOUNT > 4 && warpCount >= KCCOUNT - 3)
			{
				level_count[wx][l[wx] - 4] = warpCount;
				level_index[wx][l[wx] - 4] = 0;
				level_prev_index[wx][l[wx] - 4] = 0;
			}
			__syncwarp(partMask);

			uint64 wc = explore_branch<T, CPARTSIZE>(4, num_divs_local, l, partMask,
				j, encode, cl, level_count[wx], level_index[wx], level_prev_index[wx]);

			if (lx == 0 && KCCOUNT > 4 && warpCount >= KCCOUNT - 3)
				atomicAdd(counter, wc); //clique_count[wx] = wc;


			__syncwarp(partMask);

			if (lx == 0)
				wtc[wx] = atomicAdd(&(tc), 1);
			__syncwarp(partMask);
		}
	}


	//search in the queue

	__syncthreads();
	if (threadIdx.x == 0)
	{
		atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
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
			const auto block_size = 128;
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
					//current_q.count.gdata()[0] = 1; 
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

						const T partitionSize = 4;
						factor = (block_size / partitionSize);

						const uint dv = 32;
						const uint max_level = 7;
						uint num_divs = (maxDegree.gdata()[0] + dv - 1) / dv;
						const uint64 level_size = num_SMs * conc_blocks_per_SM * factor * max_level * num_divs;
						const uint64 encode_size = num_SMs * conc_blocks_per_SM * maxDegree.gdata()[0] * num_divs;
						printf("Level Size = %llu, Encode Size = %llu\n", level_size, encode_size);
						GPUArray<T> current_level2("Temp level Counter", unified, level_size, dev_);
						GPUArray<T> node_be("Temp level Counter", unified, encode_size, dev_);
						current_level2.setAll(0, true);
						node_be.setAll(0, true);



						const T numPartitions = block_size / partitionSize;
						cudaMemcpyToSymbol(KCCOUNT, &kcount, sizeof(KCCOUNT));
						cudaMemcpyToSymbol(PARTSIZE, &partitionSize, sizeof(PARTSIZE));
						cudaMemcpyToSymbol(NUMPART, &numPartitions, sizeof(NUMPART));
						cudaMemcpyToSymbol(MAXLEVEL, &max_level, sizeof(MAXLEVEL));
						cudaMemcpyToSymbol(NUMDIVS, &num_divs, sizeof(NUMDIVS));
						cudaMemcpyToSymbol(MAXDEG, &(maxDegree.gdata()[0]), sizeof(MAXDEG));
						cudaMemcpyToSymbol(CBPSM, &(conc_blocks_per_SM), sizeof(CBPSM));

						auto grid_block_size = current_q.count.gdata()[0];
						execKernel((kckernel_node_block_warp_binary_count<T, block_size, partitionSize>), grid_block_size, block_size, dev_, false,
							counter.gdata(),
							g,
							current_q.device_queue->gdata()[0],
							current_level2.gdata(), cpn.gdata(),
							d_bitmap_states.gdata(), node_be.gdata());


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

			const auto block_size = 128;
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
					//current_q.count.gdata()[0] = 1;

					if (pe == Warp)
					{
						// auto grid_block_size = (32 * current_q.count.gdata()[0] + block_size - 1) / block_size;
						// execKernel((kckernel_edge_warp_count2_shared<T, block_size>), grid_block_size, block_size, dev_, false,
						// 	counter.gdata(),
						// 	g,
						// 	kcount,
						// 	maxDegree.gdata()[0],
						// 	current_q.device_queue->gdata()[0],
						// 	current_level.gdata(),
						// 	NULL,
						// 	conc_blocks_per_SM, d_bitmap_states.gdata());



						const uint dv = 32;
						const uint max_level = 7;
						uint num_divs = (maxDegree.gdata()[0] + dv - 1) / dv;
						const uint64 level_size = num_SMs * conc_blocks_per_SM * factor * max_level * num_divs;
						const uint64 encode_size = num_SMs * conc_blocks_per_SM * factor * maxDegree.gdata()[0] * num_divs;
						const uint64 tri_size = num_SMs * conc_blocks_per_SM * factor * maxDegree.gdata()[0];
						printf("Level Size = %llu, Encode Size = %llu\n", level_size, encode_size);
						GPUArray<T> current_level2("Temp level Counter", unified, level_size, dev_);
						GPUArray<T> node_be("Temp level Counter", unified, encode_size, dev_);
						GPUArray<T> tri_list("Temp level Counter", unified, tri_size, dev_);
						current_level2.setAll(0, true);
						node_be.setAll(0, true);
						tri_list.setAll(0, true);



						auto grid_block_size = (32 * current_q.count.gdata()[0] + block_size - 1) / block_size;
						execKernel((kckernel_edge_warp_binary_count<T, block_size>), grid_block_size, block_size, dev_, false,
							counter.gdata(),
							g,
							kcount,
							maxDegree.gdata()[0],
							current_q.device_queue->gdata()[0],
							current_level2.gdata(), NULL,
							conc_blocks_per_SM, d_bitmap_states.gdata(), node_be.gdata(), tri_list.gdata());


						current_level2.freeGPU();


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
					else if (pe == BlockWarp)
					{
						const T partitionSize = 4;
						factor = (block_size / partitionSize);

						const uint dv = 32;
						const uint max_level = 7;
						uint num_divs = (maxDegree.gdata()[0] + dv - 1) / dv;
						const uint64 level_size = num_SMs * conc_blocks_per_SM * factor * max_level * num_divs;
						const uint64 encode_size = num_SMs * conc_blocks_per_SM * maxDegree.gdata()[0] * num_divs;
						const uint64 tri_size = num_SMs * conc_blocks_per_SM * maxDegree.gdata()[0];
						printf("Level Size = %llu, Encode Size = %llu\n", level_size, encode_size);
						GPUArray<T> current_level2("Temp level Counter", unified, level_size, dev_);
						GPUArray<T> node_be("Temp level Counter", unified, encode_size, dev_);
						GPUArray<T> tri_list("Temp level Counter", unified, tri_size, dev_);
						current_level2.setAll(0, true);
						node_be.setAll(0, true);
						tri_list.setAll(0, true);

						const T numPartitions = block_size / partitionSize;
						cudaMemcpyToSymbol(KCCOUNT, &kcount, sizeof(KCCOUNT));
						cudaMemcpyToSymbol(PARTSIZE, &partitionSize, sizeof(PARTSIZE));
						cudaMemcpyToSymbol(NUMPART, &numPartitions, sizeof(NUMPART));
						cudaMemcpyToSymbol(MAXLEVEL, &max_level, sizeof(MAXLEVEL));
						cudaMemcpyToSymbol(NUMDIVS, &num_divs, sizeof(NUMDIVS));
						cudaMemcpyToSymbol(MAXDEG, &(maxDegree.gdata()[0]), sizeof(MAXDEG));
						cudaMemcpyToSymbol(CBPSM, &(conc_blocks_per_SM), sizeof(CBPSM));

						auto grid_block_size = current_q.count.gdata()[0];
						execKernel((kckernel_edge_block_warp_binary_count<T, block_size, partitionSize>), grid_block_size, block_size, dev_, false,
							counter.gdata(),
							g,
							current_q.device_queue->gdata()[0],
							current_level2.gdata(), NULL,
							d_bitmap_states.gdata(), node_be.gdata(), tri_list.gdata());


						current_level2.freeGPU();
						// node_be.freeGPU();
						// tri_list.freeGPU();
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
