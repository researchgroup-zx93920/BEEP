#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include<string>
#include <fstream>
#include <map>

#include "omp.h"
#include<vector>

#include "../include/Logger.cuh"
#include "../include/FIleReader.cuh"
#include "../include/CGArray.cuh"

using namespace std;

__global__ void add(int *data, int count)
{
	int thread = threadIdx.x;
	
	if (thread < count)
	{
		printf("Hello\n");
	}
}



bool compareInterval(Edge left, Edge right)
{

	if (std::get<0>(left) < std::get<0>(right))
		return true;
	else if (std::get<0>(left) == std::get<0>(right)
		&& std::get<1>(left) < std::get<1>(right))
		return true;
	return false;
}

bool compareIntervalW(WEdge left, WEdge right)
{

	if (std::get<0>(left) < std::get<0>(right))
		return true;
	else if (std::get<0>(left) == std::get<0>(right)
		&& std::get<1>(left) < std::get<1>(right))
		return true;
	return false;
}

void ReadTsv(char* filename, int* nr, int* nc, int* nnz, int** cooRows, int** cooColumns,
	int** rowPointer, int oneIndex)
{
	bool symmetric = false;
	// open file
	std::ifstream mfs(filename);
	std::string line;

	// read COO values
	int index = 0;
	EdgeList edges;
	while (!getline(mfs, line).eof())
	{
		int row, column;
		double value;
		// read the value from file
		//sscanf(line.c_str(), "%d %d %lg", &row, &column, &value);
		sscanf_s(line.c_str(), "%d %d", &row, &column);
		edges.push_back(pair<int, int>(row - oneIndex, column - oneIndex));

		if (symmetric)
		{
			edges.push_back(pair<int, int>(column - oneIndex, row - oneIndex));
			index++;
		}

		index++;
	}

	std::sort(edges.begin(), edges.end(), compareInterval);

	*nnz = index;
	// alloc space for COO matrix
	*cooRows = (int*)malloc(*nnz * sizeof(int));
	*cooColumns = (int*)malloc(*nnz * sizeof(int));
	printf("Done reading edge list %d \n", *nnz);

	*nr = 0;
	*nc = 0;

	int maxr = 0;
	int maxc = 0;

#pragma omp parallel for reduction(max:maxr) reduction(max:maxc)
	for (int i = 0; i < *nnz; i++)
	{
		(*cooRows)[i] = std::get<0>(edges[i]);
		(*cooColumns)[i] = std::get<1>(edges[i]);
		if (maxr < (*cooRows)[i])
			maxr = (*cooRows)[i];

		if (maxc < (*cooColumns)[i])
			maxc = (*cooColumns)[i];
	}

	(*nr) = maxr + oneIndex;
	(*nc) = maxc + oneIndex;



	printf("Done reading edge list %d, %d, %d \n", *nnz, *nr, *nc);


	// alloc space for CSR matrix
	*rowPointer = (int*)malloc((*nr + 1) * sizeof(int));

	//Override the coo source
	(*rowPointer)[0] = 0;
	int startRowIndex = 1;
	for (int i = 1; i < *nnz; i++)
	{
		if ((*cooRows)[i] != (*cooRows)[i - 1])
		{
			(*rowPointer)[startRowIndex] = i;
			startRowIndex++;
		}

	}
	(*rowPointer)[*nr] = *nnz;


	printf("Created row pointer\n");


	mfs.close();
}


void EdgesToCSRCOO(std::vector<EdgeTy<uint>> edges, int oneIndex, int* nr, int* nc, int* nnz, uint** cooRows, uint** cooColumns, uint** rowPointer)
{
	std::sort(edges.begin(), edges.end(), compareInterval);

	*nnz = edges.size();
	// alloc space for COO matrix
	*cooRows = (uint*)malloc(*nnz * sizeof(uint));
	*cooColumns = (uint*)malloc(*nnz * sizeof(uint));
	printf("Done reading edge list %d \n", *nnz);

	*nr = 0;
	*nc = 0;

	int maxr = 0;
	int maxc = 0;

	#pragma omp parallel for reduction(max:maxr) reduction(max:maxc)
	for (int i = 0; i < *nnz; i++)
	{
		(*cooRows)[i] = std::get<0>(edges[i]);
		(*cooColumns)[i] = std::get<1>(edges[i]);
		if (maxr < (*cooRows)[i])
			maxr = (*cooRows)[i];

		if (maxc < (*cooColumns)[i])
			maxc = (*cooColumns)[i];
	}

	(*nr) = maxr + oneIndex;
	(*nc) = maxc + oneIndex;

	printf("Done reading edge list %d, %d, %d \n", *nnz, *nr, *nc);


	// alloc space for CSR matrix
	*rowPointer = (uint*)malloc((*nr + 1) * sizeof(uint));

	//Override the coo source
	(*rowPointer)[0] = 0;
	int startRowIndex = 1;
	for (int i = 1; i < *nnz; i++)
	{
		if ((*cooRows)[i] != (*cooRows)[i - 1])
		{
			(*rowPointer)[startRowIndex] = i;
			startRowIndex++;
		}

	}
	(*rowPointer)[*nr] = *nnz;

	printf("Created row pointer\n");
}

void EdgesToWCSRCOO(std::vector<WEdgeTy<uint, wtype>> edges, int oneIndex, int* nr, int* nc, int* nnz, int** cooRows, int** cooColumns, int** rowPointer, wtype** weights)
{
	std::sort(edges.begin(), edges.end(), compareIntervalW);

	*nnz = edges.size();
	// alloc space for COO matrix
	*cooRows = (int*)malloc(*nnz * sizeof(int));
	*cooColumns = (int*)malloc(*nnz * sizeof(int));
	*weights = (wtype*)malloc(*nnz * sizeof(wtype));
	printf("Done reading edge list %d \n", *nnz);

	*nr = 0;
	*nc = 0;

	int maxr = 0;
	int maxc = 0;

#pragma omp parallel for reduction(max:maxr) reduction(max:maxc)
	for (int i = 0; i < *nnz; i++)
	{
		(*cooRows)[i] = std::get<0>(edges[i]);
		(*cooColumns)[i] = std::get<1>(edges[i]);
		(*weights)[i] = std::get<2>(edges[i]);
		if (maxr < (*cooRows)[i])
			maxr = (*cooRows)[i];

		if (maxc < (*cooColumns)[i])
			maxc = (*cooColumns)[i];
	}

	(*nr) = maxr + oneIndex;
	(*nc) = maxc + oneIndex;

	printf("Done reading edge list %d, %d, %d \n", *nnz, *nr, *nc);


	// alloc space for CSR matrix
	*rowPointer = (int*)malloc((*nr + 1) * sizeof(int));

	//Override the coo source
	(*rowPointer)[0] = 0;
	int startRowIndex = 1;
	for (int i = 1; i < *nnz; i++)
	{
		if ((*cooRows)[i] != (*cooRows)[i - 1])
		{
			(*rowPointer)[startRowIndex] = i;
			startRowIndex++;
		}

	}
	(*rowPointer)[*nr] = *nnz;

	printf("Created row pointer\n");
}


int main(int argc, char **argv){


	int nnz, nr, nc;

	//int* source, * destination;
	//int* neighborlist;

	char matr[] = "D:\\graphs\\as20000102_adj.tsv";

	EdgeListFile f(matr);

	std::vector<EdgeTy<uint>> edges;
	std::vector<EdgeTy<uint>> fileEdges;
	while (f.get_edges(fileEdges, 10)) {
		edges.insert(edges.end(), fileEdges.begin(), fileEdges.end());
	}

	GPUArray<uint> sl, dl, neil;

	uint*& source = sl.cdata();
	uint*& destination = dl.cdata();
	uint*& neighborlist = neil.cdata();


	EdgesToCSRCOO(edges, 1, &nr, &nc, &nnz, &source, &destination, &neighborlist);

	sl.switch_to_gpu(0, nnz);
	dl.switch_to_gpu(0, nnz);
	neil.switch_to_gpu(0, nr+1);


	//For weighted edges
	//std::vector<WEdgeTy<uint, wtype>> wedges;
	//std::vector<WEdgeTy<uint,wtype>> wfileEdges;
	//while (f.get_weighted_edges(wfileEdges, 10)) {
	//	wedges.insert(wedges.end(), wfileEdges.begin(), wfileEdges.end());
	//}

	printf("File is read %d, %d, %d \n", nr, nc, nnz);


	int count = 10;
	int *d_A;
	
	int h_A[] = {1,2,3,4,5,6,7,8,9,10};
	
	CUDA_RUNTIME(cudaMalloc((void**) &d_A, count * sizeof(int)));
	CUDA_RUNTIME(cudaMemcpy(d_A, h_A, count * sizeof(int), cudaMemcpyHostToDevice));
	
	add<<<1, count>>> (d_A, count);
	
	cudaDeviceSynchronize();

    printf("Testing if things work\n");
    return 0;
}


