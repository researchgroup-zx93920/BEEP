#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include "utils.cuh"

template<typename T>
__global__ void setelements(T* arr, int count, T val)
{
	int gtx = threadIdx.x + blockDim.x * blockIdx.x;
	for (int i = gtx; i < count; i += blockDim.x * gridDim.x)
	{
		arr[i] = val;
	}
}

namespace graph
{

	
	template<class T>
	class GPUArray 
	{
	public:
		GPUArray(std::string s, AllocationTypeEnum at, uint size, int devId)
		{
			N = size;
			name = s;
			_at = at;
			_deviceId = devId;
			CUDA_RUNTIME(cudaSetDevice(_deviceId));
			CUDA_RUNTIME(cudaStreamCreate(&_stream));

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

		void allocate_cpu(uint size)
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
		void switch_to_gpu(int devId, uint size=0)
		{
			if (_at == AllocationTypeEnum::cpuonly)
			{
				N = (size==0) ? N : size;
				_at = AllocationTypeEnum::gpu;
				_deviceId = devId;
				CUDA_RUNTIME(cudaSetDevice(_deviceId));
				CUDA_RUNTIME(cudaStreamCreate(&_stream));
				CUDA_RUNTIME(cudaMalloc(&gpu_data, N * sizeof(T)));
				CUDA_RUNTIME(cudaMemcpy(gpu_data, cpu_data, N * sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice));
			}
			else if (_at == AllocationTypeEnum::gpu) //memory is already allocated
			{
				if (size > N)
				{
					Log(LogPriorityEnum::critical, "Memory needed is more than allocated-Nothing is done\n");
					return;
				}

				N = (size == 0) ? N : size;
				CUDA_RUNTIME(cudaSetDevice(_deviceId));
				CUDA_RUNTIME(cudaStreamCreate(&_stream));
				CUDA_RUNTIME(cudaMemcpy(gpu_data, cpu_data, N * sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice));
			}
		}


		void zero(bool sync)
		{
			if (N < 1)
				return;

			if (_at == AllocationTypeEnum::cpuonly)
			{
				memset(cpu_data, 0, N * sizeof(T));
			}
			else if (_at == AllocationTypeEnum::gpu)
			{
				
				memset(cpu_data, 0, N * sizeof(T));
				CUDA_RUNTIME(cudaSetDevice(_deviceId));
				CUDA_RUNTIME(cudaMemset(gpu_data, 0, N * sizeof(T)));
				if (sync)
					CUDA_RUNTIME(cudaStreamSynchronize(_stream));
			}
			else if (_at == AllocationTypeEnum::unified)
			{
				CUDA_RUNTIME(cudaSetDevice(_deviceId));
				CUDA_RUNTIME(cudaMemset(gpu_data, 0, N * sizeof(T)));
				if (sync)
					CUDA_RUNTIME(cudaStreamSynchronize(_stream));
			}
		}

		void set(T index, T val, bool sync)
		{
			if (N < 1)
				return;

			if (_at == AllocationTypeEnum::cpuonly)
			{
				memset(&cpu_data[index], val, 1 * sizeof(T));
			}
			else if (_at == AllocationTypeEnum::gpu)
			{
				memset(&cpu_data[index], val, 1 * sizeof(T));
				CUDA_RUNTIME(cudaSetDevice(_deviceId));
				setelements<T> << <1, 1, 0, _stream >> > (&gpu_data[index], 1, val);
				if (sync)
					CUDA_RUNTIME(cudaStreamSynchronize(_stream));
			}
			else if(_at == AllocationTypeEnum::unified)
			{
				CUDA_RUNTIME(cudaSetDevice(_deviceId));
				setelements<T> << <1, 1, 0, _stream >> > (&gpu_data[index], 1, val);
				if (sync)
					CUDA_RUNTIME(cudaStreamSynchronize(_stream));
			}
		}

		T* copytocpu(int startIndex, uint count=0, bool newAlloc=false)
		{
			int c = count == 0 ? N : count;


			if (_at == AllocationTypeEnum::unified)
				return &(gpu_data[startIndex]);
			
			CUDA_RUNTIME(cudaSetDevice(_deviceId));
			if (newAlloc)
			{
				T *temp = (T*)malloc(c * sizeof(T));
				CUDA_RUNTIME(cudaMemcpy(temp, &(gpu_data[startIndex]), c *sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost));
				return temp;
			}

			CUDA_RUNTIME(cudaMemcpy(cpu_data, &(gpu_data[startIndex]), c *sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost));
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