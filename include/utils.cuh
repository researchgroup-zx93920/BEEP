#pragma once
#include <mutex>
#include <cassert>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>
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


#define __VERBOSE__

typedef unsigned long long int uint64;

//Enums
enum ProcessingElementEnum { Thread, Warp, Block, Grid, Test };
enum AllocationTypeEnum { cpuonly, gpu, unified, zerocopy };
enum LogPriorityEnum { critical, warn, error, info, debug, none };


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

