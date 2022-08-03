#pragma once

#include "utils.cuh"
#include "CGArray.cuh"
template <typename T>
__global__ void map_degree(T *Degree, T len, T *mapping, T *data)
{
	uint gtid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gtid < len)
	{
		mapping[gtid] = Degree[data[gtid]];
	}
}

template <typename T>
__global__ void map_degree_log(T *Degree, T len, T *mapping, T *data)
{
	uint gtid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gtid < len)
	{
		// mapping[gtid] =  Degree[data[gtid]]*((T) (sqrt((float) Degree[data[gtid]])));
		// mapping[gtid] =  Degree[data[gtid]]*((T) (__log2f(Degree[data[gtid]])+1));
		mapping[gtid] = Degree[data[gtid]] * Degree[data[gtid]];
	}
}

namespace graph
{
	template <typename T, typename MarkType = bool>
	struct GraphQueue_d
	{
		T *count;
		T *queue;
		MarkType *mark;
	};

	template <typename T, typename MarkType = bool>
	class GraphQueue
	{

	public:
		GPUArray<T> count;
		GPUArray<T> queue;
		int dev_;
		GPUArray<MarkType> mark; // mark if a node or edge is present in the graph
		GPUArray<GraphQueue_d<T, MarkType>> *device_queue;

		int capacity;
		void Create(AllocationTypeEnum at, uint cap, int devId)
		{
			dev_ = devId;
			capacity = cap;
			count.initialize("Queue Count", at, 1, devId);
			count.setSingle(0, 0, true);
			queue.initialize("Queue data", at, capacity, devId);
			mark.initialize("Queue Mark", at, capacity, devId);

			device_queue = new GPUArray<GraphQueue_d<T, MarkType>>();
			device_queue->initialize("Device Queue", unified, 1, devId);

			count.switch_to_gpu();
			queue.switch_to_gpu();
			mark.switch_to_gpu();

			device_queue->gdata()[0].count = count.gdata();
			device_queue->gdata()[0].queue = queue.gdata();
			device_queue->gdata()[0].mark = mark.gdata();

			device_queue->switch_to_gpu();
		}

		void CreateQueueStruct(GraphQueue_d<T, MarkType> *&d)
		{
			d = device_queue->gdata();
		}

		void free()
		{
			device_queue->freeGPU();
			count.freeGPU();
			queue.freeGPU();
			mark.freeGPU();
		}

		void map_n_key_sort(T *degree)
		{
			T *mapping;
			T *aux_mapping;
			T *queue;
			T *aux_queue;
			T len = count.gdata()[0];
			cudaMalloc((void **)&mapping, len * sizeof(T));
			cudaMalloc((void **)&aux_mapping, len * sizeof(T));
			cudaMalloc((void **)&queue, len * sizeof(T));
			cudaMalloc((void **)&aux_queue, len * sizeof(T));
			auto block_dim = 512;
			auto gridsize = (len + block_dim - 1) / block_dim;

			CUDA_RUNTIME(cudaMemcpy(queue, device_queue->gdata()->queue, len * sizeof(T), cudaMemcpyDeviceToDevice));
			execKernel(map_degree<uint>, gridsize, block_dim, dev_, false, degree, len, mapping, queue);

			void *d_temp_storage = NULL;
			size_t temp_storage_bytes = 0;
			cub::DoubleBuffer<T> d_keys(mapping, aux_mapping);
			cub::DoubleBuffer<T> d_values(queue, aux_queue);
			// Dummy run to estimate memory
			CUDA_RUNTIME(cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys, d_values, len));
			// CUDA_RUNTIME(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, len));
			CUDA_RUNTIME(cudaMalloc((void **)&d_temp_storage, temp_storage_bytes));

			// Final run to sort
			CUDA_RUNTIME(cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys, d_values, len));
			// CUDA_RUNTIME(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, len));
			CUDA_RUNTIME(cudaMemcpy(device_queue->gdata()->queue, d_values.Current(), len * sizeof(T), cudaMemcpyDeviceToDevice));
			cudaFree(mapping);
			cudaFree(aux_mapping);
			cudaFree(aux_queue);
			cudaFree(queue);
			cudaFree(d_temp_storage);
		}
		void key_sort_ascending(T *mapping)
		{
			// T *mapping;
			T *aux_mapping;
			T *queue;
			T *aux_queue;
			T len = count.gdata()[0];
			// CUDA_RUNTIME(cudaMalloc((void **)&mapping, len * sizeof(T)));
			CUDA_RUNTIME(cudaMalloc((void **)&aux_mapping, len * sizeof(T)));
			CUDA_RUNTIME(cudaMalloc((void **)&queue, len * sizeof(T)));
			CUDA_RUNTIME(cudaMalloc((void **)&aux_queue, len * sizeof(T)));
			CUDA_RUNTIME(cudaMemcpy(queue, device_queue->gdata()->queue, len * sizeof(T), cudaMemcpyDeviceToDevice));

			void *d_temp_storage = NULL;
			size_t temp_storage_bytes = 0;
			cub::DoubleBuffer<T> d_keys(mapping, aux_mapping);
			cub::DoubleBuffer<T> d_values(queue, aux_queue);
			CUDA_RUNTIME(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, len));
			CUDA_RUNTIME(cudaMalloc((void **)&d_temp_storage, temp_storage_bytes));
			CUDA_RUNTIME(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, len));
			CUDA_RUNTIME(cudaMemcpy(device_queue->gdata()->queue, d_values.Current(), len * sizeof(T), cudaMemcpyDeviceToDevice));
			// cudaFree(mapping);
			cudaFree(aux_mapping);
			cudaFree(aux_queue);
			cudaFree(queue);
			cudaFree(d_temp_storage);
		}
		void i_scan(uint64 *scanned, T *degree)
		{
			T len = count.gdata()[0];
			T *mapping;
			T *queue;
			cudaMalloc((void **)&mapping, len * sizeof(T));
			cudaMalloc((void **)&queue, len * sizeof(T));
			auto block_dim = 512;
			auto gridsize = (len + block_dim - 1) / block_dim;

			CUDA_RUNTIME(cudaMemcpy(queue, device_queue->gdata()->queue, len * sizeof(T), cudaMemcpyDeviceToDevice));
			execKernel(map_degree_log<uint>, gridsize, block_dim, dev_, false, degree, len, mapping, queue);

			void *d_temp_storage = NULL;
			size_t temp_storage_bytes = 0;
			cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, mapping, scanned, len);
			cudaMalloc(&d_temp_storage, temp_storage_bytes);
			cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, mapping, scanned, len);
			CUDA_RUNTIME(cudaDeviceSynchronize());

			cudaFree(mapping);
			cudaFree(queue);
			cudaFree(d_temp_storage);
		}
	};
};