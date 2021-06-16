#pragma once
#define QUEUE_SIZE 1024

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

#include "kckernels_local.cuh"

namespace graph
{
	template<typename T>
	class SingleGPU_Kclique_Local
	{
	private:
		int dev_;
		cudaStream_t stream_;

		// Same Function for any comutation
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
			// Log(LogPriorityEnum::info, "Level: %d, curr: %d/%d", level, current.count.gdata()[0], bucket.count.gdata()[0]);
		}


		// Same Function for any comutation
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
			// Log(LogPriorityEnum::info, "Level: %d, curr: %d/%d", level, current.count.gdata()[0], bucket.count.gdata()[0]);
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
        GPUArray <uint64> cpn;
		graph::GraphQueue<T, bool> bucket_q;
		graph::GraphQueue<T, bool> current_q;
		GPUArray<T> identity_arr_asc;

		SingleGPU_Kclique_Local(int dev, COOCSRGraph_d<T>& g) : dev_(dev) {
			CUDA_RUNTIME(cudaSetDevice(dev_));
			CUDA_RUNTIME(cudaStreamCreate(&stream_));
			CUDA_RUNTIME(cudaStreamSynchronize(stream_));

			bucket_q.Create(unified, g.numEdges, dev_);
			current_q.Create(unified, g.numEdges, dev_);
			AscendingGpu(g.numEdges, identity_arr_asc);

			edgePtr.initialize("Edge Support", unified, g.numEdges, dev_);
		}

		SingleGPU_Kclique_Local() : SingleGPU_Kclique_Local(0) {}

		void getNodeDegree(COOCSRGraph_d<T>& g, T* maxDegree,
			const size_t nodeOffset = 0, const size_t edgeOffset = 0)
		{
			const int dimBlock = 128;
			nodeDegree.initialize("Edge Support", unified, g.numNodes, dev_);
			uint dimGridNodes = (g.numNodes + dimBlock - 1) / dimBlock;
			execKernel((getNodeDegree_kernel<T, dimBlock>), dimGridNodes, dimBlock, dev_, false, nodeDegree.gdata(), g, maxDegree);
		}

		template<const int PSIZE>
		void findKclqueIncremental_node_binary_async_local(int kcount, COOCSRGraph_d<T>& g,
            const size_t nodeOffset = 0, const size_t edgeOffset = 0)
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
			cpn = GPUArray <uint64> ("Local clique Counter", gpu, g.numNodes, dev_);
			GPUArray <T> maxDegree("Temp Degree", unified, 1, dev_);

			T conc_blocks_per_SM = context.GetConCBlocks(block_size);
			GPUArray<T> d_bitmap_states("bmp bitmap stats", AllocationTypeEnum::gpu, num_SMs * conc_blocks_per_SM, dev_);

			counter.setSingle(0, 0, true);
			maxDegree.setSingle(0, 0, true);
			d_bitmap_states.setAll(0, true);
            cpn.setAll(0, true);
			getNodeDegree(g, maxDegree.gdata());
			bucket_scan(nodeDegree, g.numNodes, 0, kcount - 1, current_q, identity_arr_asc, bucket_q, bucket_level_end_);
			todo -= current_q.count.gdata()[0];
			current_q.count.gdata()[0] = 0;
			level = kcount - 1;
			bucket_level_end_ = level;

			const T partitionSize = PSIZE; //PART_SIZE;
			T factor = (block_size / partitionSize);

			const uint dv = 32;
			const uint max_level = 9;
			uint num_divs = (maxDegree.gdata()[0] + dv - 1) / dv;
			const uint64 level_size = num_SMs * conc_blocks_per_SM * factor * max_level * num_divs;
			const uint64 encode_size = num_SMs * conc_blocks_per_SM * maxDegree.gdata()[0] * num_divs;
			printf("Level Size = %llu, Encode Size = %llu\n", level_size, encode_size);
			GPUArray<T> current_level2("Temp level Counter", gpu, level_size, dev_);
			GPUArray<T> node_be("Binary Encoding Array", gpu, encode_size, dev_);
			current_level2.setAll(0, true);
			node_be.setAll(0, true);

			const T numPartitions = block_size/partitionSize;
			cudaMemcpyToSymbol(KCCOUNT, &kcount, sizeof(KCCOUNT));
			cudaMemcpyToSymbol(PARTSIZE, &partitionSize, sizeof(PARTSIZE));
			cudaMemcpyToSymbol(NUMPART, &numPartitions, sizeof(NUMPART));
			cudaMemcpyToSymbol(MAXLEVEL, &max_level, sizeof(MAXLEVEL));
			cudaMemcpyToSymbol(NUMDIVS, &num_divs, sizeof(NUMDIVS));
			cudaMemcpyToSymbol(MAXDEG, &(maxDegree.gdata()[0]), sizeof(MAXDEG));
			cudaMemcpyToSymbol(CBPSM, &(conc_blocks_per_SM), sizeof(CBPSM));

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
					auto grid_block_size = current_q.count.gdata()[0];
					execKernel((kckernel_node_block_warp_binary_count_local_globalmem_lazy_loop<T, block_size, partitionSize>), grid_block_size, block_size, dev_, false,
						counter.gdata(),
						g,
						current_q.device_queue->gdata()[0],
						current_level2.gdata(), cpn.gdata(),
						d_bitmap_states.gdata(), node_be.gdata());
				}
				level += span;
			}

            std::cout.imbue(std::locale(""));
			std::cout << " Nodes = " << g.numNodes << " Counter = " << counter.gdata()[0] << "\n";

			current_level2.freeGPU();
			counter.freeGPU();
			node_be.freeGPU();
			d_bitmap_states.freeGPU();
            cpn.copytocpu(0);
            cpn.freeGPU();
		}

		void free()
		{
			current_q.free();
			bucket_q.free();
			identity_arr_asc.freeGPU();
            
			nodeDegree.freeGPU();
			edgePtr.freeGPU();
            cpn.freeCPU();
		}

        void save(const T& n)
        {
            // save the result to file
        }

        void show(const T& n)
        {
            std::cout << "Local clique counter for the first " << n << " Nodes:\n";
            auto cdata = cpn.cdata();
            for (T i = 0; i < n; i ++)
            {
                std::cout << i << '\t' << cdata[i] << '\n';
            }
        }

        ~SingleGPU_Kclique_Local()
        {
            free();
        }

		void sync() { CUDA_RUNTIME(cudaStreamSynchronize(stream_)); }
		int device() const { return dev_; }
		cudaStream_t stream() const { return stream_; }
	};

}
