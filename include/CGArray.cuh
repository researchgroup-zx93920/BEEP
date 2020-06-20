#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include "utils.cuh"

enum AllocationType {gpu, unified, zerocopy};
template<class T>
class GPUArray {
public:
	GPUArray(AllocationType at, int size, int devId)
	{
		N = size;
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

	GPUArray() {}

	~GPUArray()
	{
		if (_at == AllocationType::gpu)
		{
			free(cpu_data);
		}
		CUDA_RUNTIME(cudaFree(gpu_data));
	}

	void cpu(int size)
	{
		N = size;
		cpu_data = (T*)malloc(size * sizeof(T));
	}
	void switch_to_gpu(int devId, int size)
	{
		N = size;
		_at = AllocationType::gpu;
		_deviceId = devId;
		CUDA_RUNTIME(cudaSetDevice(_deviceId));
		CUDA_RUNTIME(cudaStreamCreate(&_stream));
		CUDA_RUNTIME(cudaMalloc(&gpu_data, N * sizeof(T)));
		CUDA_RUNTIME(cudaMemcpy(gpu_data, cpu_data, N, cudaMemcpyKind::cudaMemcpyHostToDevice));
	}


	void zero(bool sync)
	{
		CUDA_RUNTIME(cudaSetDevice(_deviceId));
		CUDA_RUNTIME(cudaMemset(gpu_data, 0, N));
		if(sync)
			CUDA_RUNTIME(cudaStreamSynchronize(stream_));
	}

	T*& copytocpu(int startIndex, int count)
	{
		if (unified)
			return &(gpu_data[startIndex]);

		CUDA_RUNTIME(cudaSetDevice(_deviceId));
		CUDA_RUNTIME(cudaMemcpy(cpu_data, &(gpu_data[startIndex]), count, cudaMemcpyKind::cudaMemcpyDeviceToHost));
		return cpu_data;
	}

	T*& gdata()
	{
		return gpu_data;
	}

	T*& cdata(AllocationType at)
	{
		if (at == AllocationType::unified)
			return gpu_data;

		return cpu_data;
	}

	std::size_t N;
private:
	T* cpu_data;
	T* gpu_data;
	AllocationType _at;
	cudaStream_t _stream;
	int _deviceId;
	
};