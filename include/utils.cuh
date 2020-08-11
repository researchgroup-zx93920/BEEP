#pragma once
#include <mutex>
#include <cassert>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <numeric>
#include <tuple>
#include <stdio.h>
#include <stdarg.h>

#include <cstdio>
#include <string>
//freestanding specific

#include "defs.cuh"
#include "cuda.h" 
#include <cuda_runtime_api.h>
#include <cub/cub.cuh>



#define CUDA_RUNTIME(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false) {

    if(code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(1);
    }
}


#define PRINT_ERROR \
    do { \
        fprintf(stderr, "Error at line %d, file %s (%d) [%s]\n", \
        __LINE__, __FILE__, errno, strerror(errno)); exit(1); \
    } while(0)


static std::chrono::time_point<std::chrono::high_resolution_clock> now() {
    return std::chrono::high_resolution_clock::now();
}

static timepoint stime()
{
    return std::chrono::system_clock::now();
}

static double elapsedSec(timepoint start)
{
    return (std::chrono::system_clock::now() - start).count() / 1e9;
}

/*Device function that returns how many SMs are there in the device/arch - it can be more than the maximum readable SMs*/
__device__ __forceinline__ unsigned int getnsmid(){
    unsigned int r;
    asm("mov.u32 %0, %%nsmid;" : "=r"(r));
    return r;
}

/*Device function that returns the current SMID of for the block being run*/
__device__ __forceinline__ unsigned int getsmid(){
    unsigned int r;
    asm("mov.u32 %0, %%smid;" : "=r"(r));
    return r;
}

/*Device function that returns the current warpid of for the block being run*/
__device__ __forceinline__ unsigned int getwarpid(){
    unsigned int r;
    asm("mov.u32 %0, %%warpid;" : "=r"(r));
    return r;
}

/*Device function that returns the current laneid of for the warp in the block being run*/
__device__ __forceinline__ unsigned int getlaneid(){
    unsigned int r;
    asm("mov.u32 %0, %%laneid;" : "=r"(r));
    return r;
}


template<typename T>
__host__ __device__
void swap_ele(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}



template<typename T>
void MatrixStats(int nnz, int nr, int nc, T* csrRowPointers, T* csrColumns)
{
	std::map<int, int> dictionary;

	int minRow = 0;
	int maxRow = 1;
	int totalRow = 0;
	int median = 0;

	int countMin = 0;
	int countMax = 0;
	int countMedian = 0;
	double sq_sum = 0;

	int _nr = nr;

	for (int i = 0; i < nr; i++)
	{
		int start = csrRowPointers[i] - 1;
		int end = csrRowPointers[i + 1] - 1;
		int rowCount = (end - start);

		if (dictionary.count(rowCount) < 1)
			dictionary[rowCount] = 1;
		else dictionary[rowCount]++;


		if (rowCount == 0)
			_nr--;
	}

	typedef std::map<int, int>::iterator it_type;
	for (it_type iterator = dictionary.begin(); iterator != dictionary.end(); iterator++) 
	{
		if (iterator->first == 0)
			continue;

		if (minRow == 0)
		{
			minRow = iterator->first;
			countMin = iterator->second;
		}

		if (countMedian < iterator->second)
		{
			median = iterator->first;
			countMedian = iterator->second;
		}

		totalRow += iterator->second * iterator->first;

		maxRow = iterator->first;
		countMax = iterator->second;
	}

	double mean = 1.0 * totalRow / nr;
	//double sd = std::sqrt( (sq_sum / nr) - (mean * mean));
	double sd = 0;
	for (int i = 0; i < nr; i++)
	{
		int start = csrRowPointers[i];
		int end = csrRowPointers[i + 1];
		int rowCount = (end - start);

		double diff = rowCount - mean;
		sd += diff * diff;
	}

	double variance = sd / nr;
	sd = std::sqrt(variance);


	printf("%d,%d,%d,%d,%.1f,%.1f,%d,%d,%d,%d,%d\n",
		nnz, nr, nc, minRow, mean, sd, median, maxRow, countMin, countMedian, countMax);
}


template<typename T>
void PrintMtarixStruct(int nnz, int nr, int nc, T* csrRowPointer, T* csrColumns)
{
	const int resolution = 20;
	float st[resolution][resolution];
	for (int i = 0; i < resolution; i++)
		for (int j = 0; j < resolution; j++)
			st[i][j] = 0.0;

	int rowUnit = (int)std::ceil(nr / resolution);
	int colUnit = (int)std::ceil(nc / resolution);

	//#pragma omp parallel for
	for (int i = 0; i < nr; i++)
	{
		int start = csrRowPointer[i];
		int end = csrRowPointer[i + 1];
		int j;

		int rowIndex = (int)std::floor(i / rowUnit);

		for (j = start; j < end; j++)
		{
			int colIndex = (int)std::floor(csrColumns[j] / rowUnit);
			st[rowIndex][colIndex]++;
		}
	}

	for (int i = 0; i < resolution; i++)
	{
		for (int j = 0; j < resolution; j++)
		{
			float perc = st[i][j] / nnz;

			if (perc >= 0.001)
				printf("%d,", (int)(1000 * st[i][j] / nnz));
			else if (perc > 0)
				printf("+,");
			else
				printf("-,");

		}
		printf("\n");
	}

} // referenceSpMV
