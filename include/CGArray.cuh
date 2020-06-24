#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include "utils.cuh"

namespace graph
{

	
	template<class T>
	class GPUArray 
	{
	public:
		GPUArray(std::string s, AllocationTypeEnum at, int size, int devId)
		{
			N = size;
			name = s;
			_at = at;
			_deviceId = devId;
			CUDA_RUNTIME(cudaSetDevice(_deviceId));
			CUDA_RUNTIME(cudaStreamCreate(&stream_));

			switch (at)
			{
			case gpu:
				cpu_data = (T*)malloc(size * sizeof(T));
				CUDA_RUNTIME(cudaMalloc(&gpu_data, size * sizeof(T)));
				break;
			case unified:
				CUDA_RUNTIME(cudaMallocManaged(&gpu_data, size * sizeof(T)));
				break;
			case zerocopy:
				break;
			default:
				break;
			}
		}

		GPUArray(std::string s, AllocationTypeEnum at) {
			name = s;
			_at = at;
		}

		void free()
		{
			if (!freed)
			{
				freed = true;
				if (_at == AllocationTypeEnum::gpu)
				{
					std::free(cpu_data);
				}
				CUDA_RUNTIME(cudaFree(gpu_data));
			}
		}

		void allocate_cpu(int size)
		{
			if (_at == AllocationTypeEnum::cpuonly)
			{
				N = size;
				cpu_data = (T*)malloc(size * sizeof(T));
			}
			else if (_at == AllocationTypeEnum::unified)
			{
				N = size;
				CUDA_RUNTIME(cudaMallocManaged(&gpu_data, size * sizeof(T)));
			}
			else
			{
				Log(LogPriorityEnum::critical, "At allocate_cpu: Only CPU allocation\n");
			}
		}
		void switch_to_gpu(int devId, int size)
		{

			if (_at == AllocationTypeEnum::cpuonly)
			{
				N = size;
				_at = AllocationTypeEnum::gpu;
				_deviceId = devId;
				CUDA_RUNTIME(cudaSetDevice(_deviceId));
				CUDA_RUNTIME(cudaStreamCreate(&_stream));
				CUDA_RUNTIME(cudaMalloc(&gpu_data, N * sizeof(T)));
				CUDA_RUNTIME(cudaMemcpy(gpu_data, cpu_data, N * sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice));
			}
		}


		void zero(bool sync)
		{
			if (_at == AllocationTypeEnum::cpuonly)
			{
				memset(cpu_Data, 0, N * sizeof(T));
			}
			else
			{
				CUDA_RUNTIME(cudaSetDevice(_deviceId));
				CUDA_RUNTIME(cudaMemset(gpu_data, 0, N * sizeof(T)));
				if (sync)
					CUDA_RUNTIME(cudaStreamSynchronize(stream_));
			}
		}

		T* copytocpu(int devId, int startIndex, int count, bool newAlloc)
		{
			if (_at == AllocationTypeEnum::unified)
				return &(gpu_data[startIndex]);
			
			if (newAlloc)
			{
				_deviceId = devId;
				T *temp = (T*)malloc(count * sizeof(T));
				CUDA_RUNTIME(cudaSetDevice(_deviceId));
				CUDA_RUNTIME(cudaMemcpy(temp, &(gpu_data[startIndex]), count*sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost));
				return temp;
			}

			CUDA_RUNTIME(cudaSetDevice(_deviceId));
			CUDA_RUNTIME(cudaMemcpy(cpu_data, &(gpu_data[startIndex]), count*sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost));
			return cpu_data;
		}

		T*& gdata()
		{
			return gpu_data;
		}

		T*& cdata()
		{
			if (_at == AllocationTypeEnum::unified)
				return gpu_data;

			return cpu_data;
		}

		std::size_t N;
		std::string name;
	private:
		T* cpu_data;
		T* gpu_data;
		AllocationTypeEnum _at;
		cudaStream_t _stream;
		int _deviceId;
		bool freed = false;

	};
}