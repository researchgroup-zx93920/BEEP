#include <iostream>
#include <cuda_runtime.h>
#include "utils.h"

template<class T>
class GPUArray {

private:
	bool gpu_is_updated = false;
	bool cpu_is_updated = true;
	std::size_t N;
	T* cpu_data;
	T* gpu_data;
	bool cpu_array_owned;
};