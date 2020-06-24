#include <cuda_runtime.h>
#include <iostream>
#include<string>
#include <fstream>
#include <map>

#include "omp.h"
#include<vector>

#include "../include/Logger.cuh"
#include "../include/FIleReader.cuh"
#include "../include/CGArray.cuh"
#include "../include/TriCountPrim.cuh"
#include "../include/TcBase.cuh"
#include "../include/TcSerial.cuh"
#include "../include/TcBinary.cuh"
#include "../include/CSRCOO.cuh"

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
	Log(LogPriorityEnum::info,"Done reading edge list %d \n", *nnz);

	*nr = 0;
	*nc = 0;

	int maxr = 0;
	int maxc = 0;

	#pragma omp parallel for reduction(max:maxr) reduction(max:maxc)
	for (int i = 0; i < *nnz; i++)
	{
		(*cooRows)[i] = std::get<0>(edges[i]) - oneIndex;
		(*cooColumns)[i] = std::get<1>(edges[i]) - oneIndex;
		if (maxr < (*cooRows)[i])
			maxr = (*cooRows)[i];

		if (maxc < (*cooColumns)[i])
			maxc = (*cooColumns)[i];
	}

	(*nr) = maxr + 1;
	(*nc) = maxc + 1;

	Log(LogPriorityEnum::info,"Done reading edge list %d, %d, %d \n", *nnz, *nr, *nc);


	// alloc space for CSR matrix
	*rowPointer = (uint*)malloc((*nr + 1) * sizeof(uint));

	//Override the coo source
	int lastIndex = 0;
	(*rowPointer)[0] = lastIndex;
	int currentRow = lastIndex;
	for (int i = 1; i < *nnz; i++)
	{
		int prev = (*cooRows)[i - 1];
		int now = (*cooRows)[i];

		if (now != prev)
		{
			while (now - currentRow > 0)
			{
				(*rowPointer)[currentRow] = lastIndex;
				currentRow++;
			}
			(*rowPointer)[currentRow] = i;
			lastIndex = i;
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

template<typename T>
void CountTriangles(graph::TcBase<T> *tc, graph::GPUArray<T> rowPtr, graph::GPUArray<T> rowInd, graph::GPUArray<T> colInd,
	const size_t numEdges, const size_t numRows, const size_t edgeOffset = 0, char kernelType = 1, int limit = 0)
{
	tc->count_async(rowPtr, rowInd, colInd, numEdges, edgeOffset, 1, 0);
	tc->sync();
	CUDA_RUNTIME(cudaGetLastError());
	printf("TC = %d\n", tc->count());
	double secs = tc->kernel_time();
	int dev = tc->device();
	Log(LogPriorityEnum::info, "gpu %d kernel time %f (%f teps) \n", dev, secs, numEdges / secs);
	cudaDeviceSynchronize();
}

int main(int argc, char **argv){

	printf("\033[0m");
	//int nnz, nr, nc;

	//int* source, * destination;
	//int* neighborlist;

	char matr[] = "D:\\graphs\\as20000102_adj.bel";

	graph::EdgeListFile f(matr);

	std::vector<EdgeTy<uint>> edges;
	std::vector<EdgeTy<uint>> fileEdges;
	auto upperTriangular = [](const Edge& e) { return e.first < e.second; };
	while (f.get_edges(fileEdges, 100)) 
	{
		edges.insert(edges.end(), fileEdges.begin(), fileEdges.end());
	}

	graph::CSRCOO<uint> csrcoo = graph::CSRCOO<uint>::from_edgelist(edges, upperTriangular);


	graph::GPUArray<uint> sl("source", AllocationTypeEnum::cpuonly), 
		dl("destination", AllocationTypeEnum::cpuonly), 
		neil("row pointer", AllocationTypeEnum::cpuonly);

	sl.cdata() = csrcoo.row_ind();
	dl.cdata() = csrcoo.col_ind();
	neil.cdata() = csrcoo.row_ptr();


	//EdgesToCSRCOO(edges, 1, &nr, &nc, &nnz, &source, &destination, &neighborlist);
	

	/*for (int i = 0; i < nr + 1; i++)
	{
		if (neighborlist[i] > nnz)
			printf("%d, %u\n", i, neighborlist[i]);
	}*/


	sl.switch_to_gpu(0, csrcoo.nnz());
	dl.switch_to_gpu(0, csrcoo.nnz());
	neil.switch_to_gpu(0, csrcoo.num_rows()+1);

	uint* t = neil.copytocpu(0,0, 10, true);

	//usability example
	graph::GPUArray<uint> d("Test", AllocationTypeEnum::cpuonly);
	d.cdata() = new uint[10]{ 1,2,3,4,7,12,25,30,33,37 };
	d.switch_to_gpu(0, 10);
	//graph::binary_search<uint> << <1, 1>> > (d.gdata(), 0, 10, 25);

	cudaDeviceSynchronize();
	graph::TcBase<uint> *tc = new graph::TcSerial<uint>(0, csrcoo.nnz(), csrcoo.num_rows());
	CountTriangles<uint>(tc, neil, sl, dl, csrcoo.nnz(), csrcoo.num_rows(), 0, 1, 0);


	graph::TcBase<uint> *tcb = new graph::TcBinary<uint>(0, csrcoo.nnz(), csrcoo.num_rows());
	CountTriangles<uint>(tcb, neil, sl, dl, csrcoo.nnz(), csrcoo.num_rows(), 0, 1, 0);

	//For weighted edges
	//std::vector<WEdgeTy<uint, wtype>> wedges;
	//std::vector<WEdgeTy<uint,wtype>> wfileEdges;
	//while (f.get_weighted_edges(wfileEdges, 10)) {
	//	wedges.insert(wedges.end(), wfileEdges.begin(), wfileEdges.end());
	//}


    printf("Testing if things work\n");
	sl.free();
	dl.free();
	neil.free();
	d.free();

    return 0;
}


