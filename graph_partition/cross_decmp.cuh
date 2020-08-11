#pragma once

#include <fstream>
#include <iostream>
#include <cmath>
#include <vector>
#include <stdlib.h>
#include <cstdint>
#include <random>
#include <algorithm>
#include <string>
#include <sstream>
#include <iterator>
#include <map>
#include <string.h>
#include <ctime>
#include <chrono>
#include <math.h>
#include <assert.h>
#include <cstring>

#include "../include/utils.cuh"
#include "../include/CGArray.cuh"
#include <cuda_runtime.h>

using namespace std::chrono;
using namespace std;

/* change the number if you want to partition with different size*/
#define PARTITION 4


/* !Kernel code that runs the cross-decomposition algorithm
*/

template<typename T>
__global__ void Decide_Keep_Node(int num_nodes, T* row_ptr, bool* keep_node, int hard_limit)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < num_nodes; i+= gridDim.x * blockDim.x)
    {
        int rowLen = row_ptr[i + 1] - row_ptr[i];
        keep_node[i] = rowLen < hard_limit;
    }
}

template<typename T>
__global__ void CrossDecomposition_kernel(uint8_t* orig_P, uint8_t* new_P,
    uint64_t* cardi, uint64_t* new_cardi,
    const uint64_t num_nodes, const float h, uint64_t capacity,
    T* coo_row, T* coo_col,
    T* row_ptr, bool is_divisible, bool *keep_node, int f_num_nodes)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;


    if (idx < num_nodes) {
        uint64_t cur_node = idx;

      


        if (!keep_node[cur_node])
            return;

        uint64_t connected_and_in_curpart[PARTITION] = { 0 };
        uint64_t degree = (row_ptr[cur_node + 1] - row_ptr[cur_node]);
        int newDegree = degree;

        for (int j = 0; j < degree; j++) {
            uint64_t cur_col_ptr = row_ptr[cur_node] + j;
            uint64_t cur_col = coo_col[cur_col_ptr];

            if (keep_node[cur_col])
            {
                for (int cur_part = 0; cur_part < PARTITION; cur_part++) {
                    if (orig_P[cur_col] == cur_part)
                        connected_and_in_curpart[cur_part] += 1;
                }
            }
            else
                newDegree--;
        }

        float cost[PARTITION] = { 0 };
        for (int i = 0; i < PARTITION; i++) {
            cost[i] = h * connected_and_in_curpart[i] +
                (1 - h) * (f_num_nodes - (cardi[i] + newDegree - connected_and_in_curpart[i]));
        }

        //initialize arg_sort array
        uint8_t arg_sort[PARTITION];
        for (uint8_t i = 0; i < PARTITION; i++)
            arg_sort[i] = i;

        for (int i = 0; i < PARTITION - 1; i++) {
            for (int j = 0; j < PARTITION - i - 1; j++) {
                if (cost[j] < cost[j + 1]) {
                    float temp = cost[j];
                    cost[j] = cost[j + 1];
                    cost[j + 1] = temp;
                    int temp2 = arg_sort[j];
                    arg_sort[j] = arg_sort[j + 1];
                    arg_sort[j + 1] = temp2;
                }
            }
        }

        unsigned long long int old_size;
        for (int i = 0; i < PARTITION; i++) {
            if (!is_divisible) {
                if (arg_sort[i] == PARTITION - 1) {
                    old_size = atomicAdd((unsigned long long int*) & new_cardi[arg_sort[i]], (unsigned long long int) 1);
                    if (old_size >= capacity + (num_nodes % PARTITION)) {
                        old_size = atomicSub((unsigned int*)&new_cardi[arg_sort[i]], (unsigned int)1);
                    }
                    else 
                    {
                        new_P[cur_node] = arg_sort[i];
                        break;
                    }
                }
                else {
                    old_size = atomicAdd((unsigned long long int*) & new_cardi[arg_sort[i]], (unsigned long long int) 1);
                    if (old_size >= capacity) 
                    {
                        old_size = atomicSub((unsigned int*)&new_cardi[arg_sort[i]], (unsigned int)1);
                    }
                    else 
                    {
                        new_P[cur_node] = arg_sort[i];
                        break;
                    }
                }
            }
            else {
                old_size = atomicAdd((unsigned long long int*) & new_cardi[arg_sort[i]], (unsigned long long int) 1);
                if (old_size >= capacity) {
                    old_size = atomicSub((unsigned int*)&new_cardi[arg_sort[i]], (unsigned int)1);
                }
                else 
                {
                    new_P[cur_node] = arg_sort[i];
                    break;
                }
            }
        }
    }
}

/* !Kernel to count the number of edges to evaluate the quality of partition
*/
template<typename T>
__global__ void evalEdges(uint8_t* P,
    T* coo_row, T* coo_col,
    uint64_t* edges_per_part, uint64_t coo_size, bool *keep_node)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < coo_size) {
        uint32_t src = coo_row[idx];
        uint32_t dest = coo_col[idx];
        if (src != dest && keep_node[src] && keep_node[dest]) {
            uint8_t src_part = P[src];
            uint8_t dest_part = P[dest];
            atomicAdd((unsigned long long int*) & edges_per_part[src_part * PARTITION + dest_part], (unsigned long long int) 1);
        }
    }
    return;
}

/* !Simple function to check CUDA runtime error
*/
void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n",
            cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
}

/* !Thanos class definition
*/

template<typename T>
class Thanos {
private:
    float h = 0.9;
    bool is_divisible = true;
    uint64_t num_nodes, edgecount, coo_size, capacity, rem, filtered_num_nodes;
    bool* keep_node;

    uint8_t* row_P, * col_P; //row and column partition arrays
    T* coo_row, *coo_col; //COO format
    T* row_ptr; //row pointer of CSR format
    uint64_t* row_cardi, * row_new_cardi, * col_cardi, * col_new_cardi; //cardinality arrays
    uint64_t* edges_per_part; //this var is for evaluating the partition quality

    cudaError_t err;

  
    void AllocateGPUMem();
    void initMem();
    void initParts(uint8_t* P, uint64_t* cardi, const uint64_t num_nodes, bool *keep_nodes);
    void CrossDecomposition();
    void evaluatePartition();
    void printEdgesPerPar(uint64_t* edges_per_part);

public:
    Thanos() {};
    Thanos(graph::GPUArray<T> rowPtr, graph::GPUArray<T> rowInd, graph::GPUArray<T> colInd, const size_t numEdges, const int numNodes);
    ~Thanos();

};

/* !Constructor for Thanos
    Construct Thanos with tsv file will run everyting for you
*/

template<typename T>
Thanos<T>::Thanos(graph::GPUArray<T> rowPtr, graph::GPUArray<T> rowInd, graph::GPUArray<T> colInd, const size_t numEdges, const int numNodes) 
{


    row_ptr = rowPtr.gdata();
    coo_row = rowInd.gdata();
    coo_col = colInd.gdata();
    coo_size = numEdges;
    num_nodes = numNodes;


    AllocateGPUMem();
    initMem();
    evaluatePartition();
    CrossDecomposition();
    evaluatePartition();
}

/* !Destructor for Thanos.
    Deallocates all the GPU memories
*/

template<typename T>
Thanos<T>::~Thanos() {

    cudaFree(row_P);
    cudaFree(col_P);
    cudaFree(row_cardi);
    cudaFree(row_new_cardi);
    cudaFree(col_cardi);
    cudaFree(col_new_cardi);
    cudaFree(edges_per_part);
}

/* !Host function to call the cross-decomposition kernel.
    You can change the boundary of for loop to control the number of iterations
*/
template<typename T>
void Thanos<T>::CrossDecomposition() {
    dim3 dimGrid(ceil(((float)num_nodes) / 512), 1, 1);
    dim3 dimBlock(512, 1, 1);

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    for (int i = 0; i < 3; i++) {
        CrossDecomposition_kernel << <dimGrid, dimBlock >> > (row_P, col_P, col_cardi, col_new_cardi,
            num_nodes, h, capacity,
            coo_row, coo_col,
            row_ptr, is_divisible, keep_node, filtered_num_nodes);

        CrossDecomposition_kernel << <dimGrid, dimBlock >> > (col_P, row_P, row_cardi, row_new_cardi,
            num_nodes, h, capacity,
            coo_row, coo_col,
            row_ptr, is_divisible, keep_node, filtered_num_nodes);

        cudaDeviceSynchronize();
        CUDA_RUNTIME(cudaGetLastError());

        checkCuda(cudaMemcpy(row_cardi, row_new_cardi, PARTITION * sizeof(uint64_t), cudaMemcpyHostToHost));
        std::fill(row_new_cardi, row_new_cardi + PARTITION, 0);
        checkCuda(cudaMemcpy(col_cardi, col_new_cardi, PARTITION * sizeof(uint64_t), cudaMemcpyHostToHost));
        std::fill(col_new_cardi, col_new_cardi + PARTITION, 0);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error1: %s\n", cudaGetErrorString(err));
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(t2 - t1).count();
    cout << "RUN Time in Sec(only kernel): " << duration * pow(10, -6) << endl;
}

/* !Host function to call evalEdges kernel. After kernel call,
    it calls printEdgesPerPar function to show the result on the terminal
*/

template<typename T>
void Thanos<T>::evaluatePartition() {
    std::fill(edges_per_part, edges_per_part + PARTITION * PARTITION, 0);
    dim3 dimGrid0(ceil(((float)coo_size) / 512), 1, 1);
    dim3 dimBlock0(512, 1, 1); //1024
    evalEdges << <dimGrid0, dimBlock0 >> > (row_P, coo_row, coo_col,
        edges_per_part, coo_size, keep_node);
    err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error1: %s\n", cudaGetErrorString(err));
    checkCuda(cudaDeviceSynchronize());
    printEdgesPerPar(edges_per_part);
}


/* !Function to allocate all the memories required for Thanos on GPU.
*/

template<typename T>
void Thanos<T>::AllocateGPUMem() {
    checkCuda(cudaMallocManaged((void**)&row_P, num_nodes * sizeof(uint8_t)));
    checkCuda(cudaMallocManaged((void**)&col_P, num_nodes * sizeof(uint8_t)));

    checkCuda(cudaMallocManaged((void**)&row_cardi, PARTITION * sizeof(uint64_t)));
    checkCuda(cudaMallocManaged((void**)&row_new_cardi, PARTITION * sizeof(uint64_t)));
    checkCuda(cudaMallocManaged((void**)&col_cardi, PARTITION * sizeof(uint64_t)));
    checkCuda(cudaMallocManaged((void**)&col_new_cardi, PARTITION * sizeof(uint64_t)));

    checkCuda(cudaMallocManaged((void**)&edges_per_part, PARTITION * PARTITION * sizeof(uint64_t)));


    checkCuda(cudaMallocManaged((void**)&keep_node, num_nodes * sizeof(bool)));

}

/* !Initialize the memories that are allocated in function Allocate GPU Mem
*/

template<typename T>
void Thanos<T>::initMem() 
{
    std::fill(row_cardi, row_cardi + PARTITION, 0);
    std::fill(row_new_cardi, row_new_cardi + PARTITION, 0);
    std::fill(col_cardi, col_cardi + PARTITION, 0);
    std::fill(col_new_cardi, col_new_cardi + PARTITION, 0);
   

    std::fill(keep_node, keep_node + num_nodes, true);
    int blockSize = 512;
    int gridSize = (num_nodes + blockSize - 1) / blockSize;
    //execKernel(Decide_Keep_Node, gridSize, blockSize, false, num_nodes, row_ptr, keep_node, 100);
    filtered_num_nodes = CUBSum<bool,int>(keep_node, num_nodes);

    initParts(row_P, row_cardi, num_nodes, keep_node);
    initParts(col_P, col_cardi, num_nodes, keep_node);

    capacity = floor(filtered_num_nodes / PARTITION);


}

/* !Initialize the partition with uniform distribution and
    update the cardinality array
*/

template<typename T>
void Thanos<T>::initParts(uint8_t* P, uint64_t* cardi,
    const uint64_t num_nodes, bool *keep_node)
{
    random_device rd;
    mt19937 mt(rd());
    uniform_int_distribution<int> dist(0, PARTITION - 1);
    for (size_t i = 0; i < num_nodes; i++) {
        if (keep_node[i])
        {
            uint64_t part = dist(mt);
            P[i] = part;
            cardi[part]++;
        }
    }
}

/* !This function nicely outputs the result of evaluation to the terminal
    if you want to see the number of edges in each partition and between partitions,
    uncomment the couts
*/
template<typename T>
void Thanos<T>::printEdgesPerPar(uint64_t* edges_per_part)
{
    uint32_t total_internal_edges = 0, total_external_edges = 0;
    cout << "********************************************************" << endl;
    map<pair<uint8_t, uint8_t>, bool> track;
    for (uint8_t i = 0; i < PARTITION; i++) {
        for (uint8_t j = 0; j < PARTITION; j++) {
            if (i == j) {
                cout << "Internal Edges for Partition " << (int)i << " :" << (edges_per_part[i * PARTITION + j]) / 2 << endl;
                total_internal_edges += edges_per_part[i * PARTITION + j] / 2;
            }
            else if (track.find(make_pair(i, j)) == track.end()) {
                cout << "Between "<< "PARTITION " << (int)i << " and " << (int)j << " Edges: " << edges_per_part[i * PARTITION + j] << endl;
                total_external_edges += (edges_per_part[i * PARTITION + j]);
                track[make_pair(i, j)] = true;
                track[make_pair(j, i)] = true;
            }
        }
    }
    // cout << "--------------------------------------------------------" << endl;
    cout << "Total Internal Edges: " << total_internal_edges << endl;
    cout << "Total External Edges: " << total_external_edges << endl;
    cout << "********************************************************" << endl;
    return;
}
