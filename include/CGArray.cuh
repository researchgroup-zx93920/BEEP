#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include "utils.cuh"

template<typename T>
__global__ void setelements(T* arr, uint64 count, T val)
{
	uint64 gtx = threadIdx.x + blockDim.x * blockIdx.x;
	for (uint64 i = gtx; i < count; i += blockDim.x * gridDim.x)
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


		GPUArray()
		{
			N = 0;
			name = "Unknown";
		}

		void initialize(std::string s, AllocationTypeEnum at, uint size, int devId, bool cpu_data=true, bool pinned=false)
		{
			N = size;
			name = s;
			_at = at;
			_deviceId = devId;
			_cpu_data = cpu_data;
			_pinned = pinned;
			CUDA_RUNTIME(cudaSetDevice(_deviceId));
			CUDA_RUNTIME(cudaStreamCreate(&_stream));

			switch (at)
			{
			case gpu:

				if(_cpu_data)
				{
					if(_pinned)
					{
						cudaMallocHost((void**)&cpu_data, size * sizeof(T));
					}
					else
					{
						cpu_data = (T*)malloc(size * sizeof(T));
					}
				}
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

		void initialize(std::string s, AllocationTypeEnum at)
		{
			name = s;
			_at = at;
		}

		GPUArray(std::string s, AllocationTypeEnum at, uint size, int devId, bool pinned)
		{
			N = size;
			name = s;
			_at = at;
			_deviceId = devId;
			CUDA_RUNTIME(cudaSetDevice(_deviceId));
			CUDA_RUNTIME(cudaStreamCreate(&_stream));

			// if(pinned)
			// {
			// 	cudaMallocHost((void**)&cpu_data, size * sizeof(T));
			// }
			// else
			// {
			// 	cpu_data = (T*)malloc(size * sizeof(T));
			// }
			

			cpu_data = (T*)malloc(size * sizeof(T));
			CUDA_RUNTIME(cudaMalloc(&gpu_data, size * sizeof(T)));
			
		}



		GPUArray(std::string s, AllocationTypeEnum at, uint size, int devId)
		{
			initialize(s, at, size, devId, cpu_data, pinned);
		}

		GPUArray(std::string s, AllocationTypeEnum at) {
			initialize(s, at);
		}

		void freeGPU()
		{
			if (N != 0)
			{
				freed = true;
				CUDA_RUNTIME(cudaSetDevice(_deviceId));
				CUDA_RUNTIME(cudaFree(gpu_data));
				CUDA_RUNTIME(cudaStreamDestroy(_stream));
				N = 0;
			}
		}
		void freeCPU()
		{
			if(_cpu_data)
			{
				if(_pinned)
				{
					cudaFreeHost(cpu_data);
				}
				else
				{
					delete cpu_data;
				}
			}
		}

		void allocate_cpu(uint size, bool pinned=false)
		{

			_pinned = pinned;

			if (_at == AllocationTypeEnum::cpuonly)
			{
				N = size;
				if(_pinned)
				{
					cudaMallocHost((void**)&cpu_data, size * sizeof(T));
				}
				else
				{
					cpu_data = (T*)malloc(size * sizeof(T));
				}
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
		void switch_to_gpu(int devId=0, uint size=0)
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
				CUDA_RUNTIME(cudaMemcpy(gpu_data, cpu_data, N * sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice));
			}
		}

		void switch_to_unified(int devId, uint size = 0)
		{
			if (_at == AllocationTypeEnum::cpuonly)
			{
				N = (size == 0) ? N : size;
				_at = AllocationTypeEnum::unified;
				_deviceId = devId;
				CUDA_RUNTIME(cudaSetDevice(_deviceId));
				CUDA_RUNTIME(cudaStreamCreate(&_stream));
				CUDA_RUNTIME(cudaMallocManaged(&gpu_data, N * sizeof(T)));
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
				CUDA_RUNTIME(cudaMemcpy(gpu_data, cpu_data, N * sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice));
			}

		}

		void copyCPUtoGPU(int devId, uint size = 0)
		{
			if (_at != cpuonly)
			{
				N = (size == 0) ? N : size;
				CUDA_RUNTIME(cudaSetDevice(_deviceId));
				CUDA_RUNTIME(cudaStreamCreate(&_stream));
				CUDA_RUNTIME(cudaMemcpy(gpu_data, cpu_data, N * sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice));
			}
		}


		void setAll(T val, bool sync)
		{
			if (N < 1)
				return;

			if (_at == AllocationTypeEnum::cpuonly)
			{
				memset(cpu_data, val, N * sizeof(T));
			}
			else if (_at == AllocationTypeEnum::gpu)
			{
				
				memset(cpu_data, val, N * sizeof(T));
				CUDA_RUNTIME(cudaSetDevice(_deviceId));
				setelements<T> << <(N+512-1)/512, 512, 0, _stream >> > (gpu_data, N, val);
				if (sync)
					CUDA_RUNTIME(cudaStreamSynchronize(_stream));
			}
			else if (_at == AllocationTypeEnum::unified)
			{
				CUDA_RUNTIME(cudaSetDevice(_deviceId));
				setelements<T> << <(N + 512 - 1)/512, 512, 0, _stream >> > (gpu_data, N, val);
				if (sync)
					CUDA_RUNTIME(cudaStreamSynchronize(_stream));
			}
		}

		void setSingle(uint64 index, T val, bool sync)
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


		T getSingle(uint64 index)
		{
			CUDA_RUNTIME(cudaSetDevice(_deviceId));
			T val = 0;
			if (_at == AllocationTypeEnum::unified)
				return (gpu_data[index]);
			
			CUDA_RUNTIME(cudaMemcpy(&val, &(gpu_data[index]), sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost));
			return val;
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


		void advicePrefetch(bool sync)
		{
			if (_at == AllocationTypeEnum::unified)
			{
				CUDA_RUNTIME(cudaSetDevice(_deviceId));

		#ifndef __VS__
				CUDA_RUNTIME(cudaMemPrefetchAsync (gpu_data, N*sizeof(T), _deviceId, _stream));
		#endif // !__VS__

				
				if (sync)
					CUDA_RUNTIME(cudaStreamSynchronize(_stream));
			}
		}

		T*& gdata()
		{
			return gpu_data;
		}

		T*& cdata()
		{
			if (_at == unified)
				return gpu_data;

			return cpu_data;
		}

		uint64 N;
		std::string name;
	private:
		T* cpu_data;
		T* gpu_data;
		AllocationTypeEnum _at;
		cudaStream_t _stream;
		int _deviceId;
		bool freed = false;

		bool _cpu_data;
		bool _pinned;

	};
}