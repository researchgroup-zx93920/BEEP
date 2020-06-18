#include "../include/utils.h"


__global__ void add(int *data, int count)
{
	int thread = threadIdx.x;
	
	if (thread < count)
	{
		printf("Hello\n");
	}
}


int main(int argc, char **argv){

	int count = 10;
	int *d_A;
	
	int h_A[] = {1,2,3,4,5,6,7,8,9,10};
	
	cuda_err_chk(cudaMalloc((void**) &d_A, count * sizeof(int)));
	cuda_err_chk(cudaMemcpy(d_A, h_A, count * sizeof(int), cudaMemcpyHostToDevice));
	
	add<<<1, count>>> (d_A, count);
	
	cudaDeviceSynchronize();

    printf("Testing if things work\n");
    return 0;
}


