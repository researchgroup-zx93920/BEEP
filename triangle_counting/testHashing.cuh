#pragma once
#include "cuda.h"
#include "../include/CGArray.cuh"
#include "../include/utils.cuh"

namespace graph
{
	class Hashing
	{
	public:

		template<typename T>
		static void goldenSerialIntersectionCPU(graph::GPUArray<T> A, graph::GPUArray<T> B, T inputSize)
		{
			//Sanity check
			uint ap = 0;
			uint bp = 0;
			uint a, b;
			int count = 0;
			while (ap < inputSize && bp < inputSize)
			{
				a = A.cdata()[ap];
				b = B.cdata()[bp];
				if (a == b) {
					++count;
					//printf("%u, ", a);
					++ap;
					++bp;
				}
				else if (a < b) {
					++ap;

				}
				else {
					++bp;
				}
			}
			printf("\nTrue Count = %d\n", count);

		}


		template<typename T>
		static void goldenBinaryGPU(graph::GPUArray<T> A, graph::GPUArray<T> B, T inputSize)
		{
			const auto start = stime();
			graph::GPUArray<uint> countBinary("Test binary search: Out", AllocationTypeEnum::unified, 1, 0);
			graph::binary_search_2arr_g<uint, 32> << <1, 32 >> > (countBinary.gdata(), A.gdata(), inputSize, B.gdata(), inputSize);
			cudaDeviceSynchronize();
			double elapsed = elapsedSec(start);
			printf("Binary Elapsed time = %f, Count = %u\n", elapsed, countBinary.cdata()[0]);

		}

		template<typename T>
		static void test1levelHashing(graph::GPUArray<T> A, graph::GPUArray<T> B, T inputSize, const int binSize)
		{
			graph::GPUArray<uint> BH("Hashing test B", gpu, inputSize + inputSize / 3, 0);
			const uint numBins = (inputSize + binSize - 1) / binSize;
			const uint stashStart = binSize * numBins;
			const uint stashLimit = inputSize / 3;
			int* binOcc = new int[numBins];
			int stashSize = 0;
			for (int i = 0; i < numBins; i++)
			{
				binOcc[i] = 0;
				BH.cdata()[i * binSize] = 0xFFFFFFFF;
			}

			for (int i = 0; i < inputSize; i++)
			{
				uint v = B.cdata()[i];
				uint b = (v/11) % numBins;

				if (binOcc[b] < binSize)
				{
					BH.cdata()[b * binSize + binOcc[b]] = v;
					binOcc[b]++;
					if (binOcc[b] < binSize)
						BH.cdata()[b * binSize + binOcc[b]] = 0xFFFFFFFF;
				}
				else if (stashSize < stashLimit)
				{
					BH.cdata()[stashStart + stashSize] = v;
					stashSize++;
				}
				else
				{
					printf("Shit\n");
				}
			}


			BH.switch_to_gpu(0);

			const auto startHash = stime();
			graph::GPUArray<T> countHash("Test hash search: Out", AllocationTypeEnum::unified, 1, 0);
			graph::hash_search_g<T, 32><< <1,32>> >(countHash.gdata(), A.gdata(), inputSize, BH.gdata(), inputSize, binSize, stashSize);
			cudaDeviceSynchronize();
			double elapsedHash = elapsedSec(startHash);
			printf("Hash Elapsed time = %f, Count = %u Stash Size=%d\n", elapsedHash, countHash.cdata()[0], stashSize);

			delete binOcc;
			//BH.freeCPU();
			BH.freeGPU();
		}


		template<typename T>
		static void testHashNosStash(graph::GPUArray<T> A, graph::GPUArray<T> B, T inputSize, const int div)
		{
			auto hash1 = [](T val, T div) { return (val / 11) % div; };
			auto hash2 = [](T val, T div) { return (val / 11) % div; };

			const T numBins = inputSize / div;

			graph::GPUArray<T> HP("Hash Pointer", gpu, numBins + 1, 0);
			graph::GPUArray<T> HD("Hashing test B", gpu, inputSize, 0);

			//I hate maps, but just for now
			std::map<T, std::vector<T>> hash;
			for (int i = 0; i < inputSize; i++)
			{
				T v = B.cdata()[i];
				//try hash1
				T b1 = hash1(v, numBins);
				hash[b1].push_back(v);
			}

			//reshape
			HP.cdata()[0] = 0;
			int lastHi = 1;
			int lastP = 0;
			for (auto i : hash)
			{
				T hi = i.first;
				while (lastHi <= hi)
				{
					HP.cdata()[lastHi] = HP.cdata()[lastHi - 1];
					lastHi++;
				}
				HP.cdata()[lastHi] = i.second.size() + HP.cdata()[lastHi - 1];
				lastHi++;

				for (auto v : i.second)
				{
					HD.cdata()[lastP++] = v;
				}
			}

			for (int i = lastHi; i <= numBins; i++)
			{
				HP.cdata()[i] = HP.cdata()[lastHi - 1];
			}
			int maxBucket = 0;
			int max = 0;
			for (auto i : hash)
			{
				int a = i.second.size();
				if (a > max)
				{
					maxBucket = i.first;
					max = a;
				}
			}

			printf("%u, %d\n", maxBucket, hash[maxBucket].size());

			HD.switch_to_gpu(0);
			HP.switch_to_gpu(0);

			const auto startHash = stime();
			graph::GPUArray<T> countHash("Test hash search: Out", AllocationTypeEnum::unified, 1, 0);
			graph::hash_search_nostash_g<T, 32> << <1, 32 >> > (countHash.gdata(), A.gdata(), inputSize, HP.gdata(), HD.gdata(), numBins);
			cudaDeviceSynchronize();
			double elapsedHash = elapsedSec(startHash);
			printf("Hash Elapsed time = %f, Count = %u \n", elapsedHash, countHash.cdata()[0]);

			HP.freeCPU();
			HP.freeGPU();

			HD.freeCPU();
			HD.freeGPU();
		}

	};
};