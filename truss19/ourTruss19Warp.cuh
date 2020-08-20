#pragma once

#include "../include/utils.cuh"
#include "../include/Logger.cuh"
#include "../include/CGArray.cuh"

#include "ourtruss19.cuh"



template <size_t BLOCK_DIM_X>
__global__ void core_direct_warp(
	uint* gnumdeleted, uint* gnumaffected,
	const uint k, const size_t edgeStart, const size_t numEdges,
	uint* rowPtr, uint* rowInd, uint* colInd, const size_t numNodes, BCTYPE* keep, bool* affected, uint* reversed, bool firstTry, const int uMax, unsigned short* reverseIndex, EncodeDataType* bitMap)

{
	// kernel call
	size_t gwx = (BLOCK_DIM_X * blockIdx.x + threadIdx.x)/32;
	constexpr size_t warpsPerBlock = BLOCK_DIM_X / 32;
	const size_t lx = threadIdx.x % 32;
	const size_t wx = threadIdx.x / 32;

	uint numberDeleted = 0;
	uint numberAffected = 0;
	__shared__ bool didAffectAnybody[1];
	bool ft = firstTry; //1
	if (0 == threadIdx.x)
		didAffectAnybody[0] = false;

	__syncthreads();
	numberDeleted = 0;
	for (size_t i = gwx + edgeStart; i < numEdges; i += BLOCK_DIM_X * gridDim.x/32)
	{
		uint srcNode = rowInd[i];
		uint dstNode = colInd[i];
		uint warpCount = 0;
		if (keep[i] && srcNode < dstNode && (affected[i] || ft))
		{
			uint src = srcNode;
			uint dst = dstNode;

			if(lx==0)
				affected[i] = false;

			uint srcStart = rowPtr[src];
			uint srcStop = rowPtr[src + 1];

			uint dstStart = rowPtr[dst];
			uint dstStop = rowPtr[dst + 1];

			uint dstLen = dstStop - dstStart;
			uint srcLen = srcStop - srcStart;

			if (srcLen > dstLen)
			{
				swap_ele(src, dst);
				swap_ele(srcStart, dstStart);
				swap_ele(srcStop, dstStop);
				swap_ele(srcLen, dstLen);
			}

			// FIXME: remove warp reduction from this function call
			warpCount = graph::warp_sorted_count_binary_upto<warpsPerBlock>(k-2, keep, &colInd[srcStart], srcStart, srcLen,
					&colInd[dstStart], dstStart, dstLen);

			unsigned int writemask_deq = __activemask();
			warpCount = __shfl_sync(writemask_deq, warpCount, 0);
			
			//////////////////////////////////////////////////////////////
			if (warpCount < (k - 2))
			{
				uint ir = reversed[i];

				if (lx == 0)
				{
					keep[i] = false;
					keep[ir] = false;
				}


				/////////////////////////////////////////////////
				uint lastIndex = 0;
				for (size_t i = lx; i < srcLen; i += 32)
				{
					// one element of A per thread, just search for A into B


					const uint searchVal = colInd[srcStart + i];
					const uint leftValue = colInd[dstStart + lastIndex];
					if (searchVal >= leftValue)
					{
						const uint lb = graph::binary_search<uint>(&colInd[dstStart], lastIndex, dstLen, searchVal);
						if (lb < dstLen)
						{
							if (colInd[dstStart + lb] == searchVal)
							{
								//Should be regularized
								numberAffected += AffectOthers(srcStart + i, dstStart + lb, keep, affected, reversed);
							}
						}

						lastIndex = lb;
					}
				}
				/////////////////////////////////////////////////
			}
		}

		if (lx == 0 && !keep[i] && srcNode < dstNode)
			numberDeleted++;
	}

	//Instead of reduction: hope it works
	if (numberAffected > 0)
		didAffectAnybody[0] = true;

	__syncthreads();

	if (0 == threadIdx.x)
	{
		if (didAffectAnybody[0])
			*gnumaffected = 1;
	}

	// Block-wide reduction of threadCount
	typedef cub::BlockReduce<uint, BLOCK_DIM_X> BlockReduce;
	__shared__ typename BlockReduce::TempStorage tempStorage;
	uint deletedByBlock = BlockReduce(tempStorage).Sum(numberDeleted);

	//uint affectedByBlock = BlockReduce(tempStorage).Sum(numberAffected);

	if (0 == threadIdx.x)
	{
		atomicAdd(gnumdeleted, deletedByBlock);
	}

}



namespace graph
{
	template<typename T>
	class SingleGPU_KtrussWarp
	{
	private:
		int dev_;
		cudaStream_t stream_;
		uint* selectedOut;

		uint* gnumdeleted;
		uint* gnumaffected;
		bool assumpAffected;

		//Outputs:
		//Max k of a complete ktruss kernel
		int k;

		//Percentage of deleted edges for a specific k
		float percentage_deleted_k;

	public:
		BCTYPE* gKeep, * gPrevKeep;
		bool* gAffected;
		uint* gReveresed;

	public:
		SingleGPU_KtrussWarp(int dev) : dev_(dev) {
			CUDA_RUNTIME(cudaSetDevice(dev_));
			CUDA_RUNTIME(cudaStreamCreate(&stream_));
			CUDA_RUNTIME(cudaMallocManaged(&gnumdeleted, 2 * sizeof(*gnumdeleted)));
			CUDA_RUNTIME(cudaMallocManaged(&gnumaffected, sizeof(uint)));
			CUDA_RUNTIME(cudaMallocManaged(&selectedOut, sizeof(*selectedOut)));


			CUDA_RUNTIME(cudaStreamSynchronize(stream_));
			//zero_async<2>(gnumdeleted, dev_, stream_); // zero on the device that will do the counting
			//zero_async<1>(gnumaffected, dev_, stream_); // zero on the device that will do the counting

			gnumdeleted[0] = 0;
			gnumaffected[0] = 0;

			CUDA_RUNTIME(cudaStreamSynchronize(stream_));
		}

		SingleGPU_KtrussWarp() : SingleGPU_KtrussWarp(0) {}

		void findKtrussIncremental_async(int kmin, int kmax, GPUArray<T> rowPtr, GPUArray<T> rowInd, GPUArray<T> colInd,
			const size_t numNodes, uint numEdges, unsigned short* reverseIndex, EncodeDataType* bitMap, const size_t nodeOffset = 0, const size_t edgeOffset = 0)
		{

			CUDA_RUNTIME(cudaSetDevice(dev_));
			constexpr int dimBlock = 32; //For edges and nodes

			bool firstTry = true;
			GPUArray<BCTYPE> keep_l;
			GPUArray<bool> affected_l;
			GPUArray<T> reversed, srcKP, destKP;
			uint* byNodeElim;


			GPUArray<T> ptrSrc, ptrDst;
			GPUArray<T> s1, d1, s2, d2;


			keep_l.initialize("Keep", gpu, numEdges, 0);
			affected_l.initialize("Affetced", gpu, numEdges, 0);
			reversed.initialize("Reversed", gpu, numEdges, 0);
			srcKP.initialize("Src", gpu, numEdges, 0);
			destKP.initialize("Dst", gpu, numEdges, 0);

			CUDA_RUNTIME(cudaMallocManaged((void**)&byNodeElim, sizeof(uint)));

			uint dimGridEdges = (numEdges + dimBlock - 1) / dimBlock;
			const int dimGridWarp = (32 * (numEdges - edgeOffset) + (dimBlock)-1) / (dimBlock);


			//KTRUSS skeleton
			//Initialize Private Data
			InitializeArrays<dimBlock> << <dimGridEdges, dimBlock, 0, stream_ >> > (edgeOffset, numEdges, rowPtr.gdata(), rowInd.gdata(), colInd.gdata(), keep_l.gdata(),
				affected_l.gdata(), reversed.gdata(), srcKP.gdata(), destKP.gdata());

			CUDA_RUNTIME(cudaStreamSynchronize(stream_));

			uint numDeleted_l = 0;
			float minPercentage = 0.8;
			float percDeleted_l = 0.0;
			bool startIndirect = false;

			s1.gdata() = rowInd.gdata();
			d1 = colInd;

			s2.gdata() = srcKP.gdata();
			d2 = destKP;

			ptrSrc.gdata() = s1.gdata();
			ptrDst.gdata() = d1.gdata();


			k = 3;
			int cc = 2;
			while (true)
			{
				//printf("k=%d\n", k);
				numDeleted_l = 0;
				firstTry = true;
				gnumaffected[0] = 0;
				assumpAffected = true;
				CUDA_RUNTIME(cudaGetLastError());
				cudaDeviceSynchronize();

				while (assumpAffected)
				{
					assumpAffected = false;

					core_direct_warp<dimBlock> << <dimGridWarp, dimBlock, 0, stream_ >> > (gnumdeleted,
						gnumaffected, k, edgeOffset, numEdges,
						rowPtr.gdata(), ptrSrc.gdata(), ptrDst.gdata(), numNodes, keep_l.gdata(), affected_l.gdata(), reversed.gdata(), firstTry, 1, reverseIndex, bitMap);

					CUDA_RUNTIME(cudaGetLastError());
					cudaDeviceSynchronize();
					firstTry = false;

					if (gnumaffected[0] > 0)
						assumpAffected = true;

					numDeleted_l = gnumdeleted[0];

					gnumdeleted[0] = 0;
					gnumaffected[0] = 0;
					cudaDeviceSynchronize();
				}

				percDeleted_l = (numDeleted_l) * 1.0 / (numEdges / 2);
				if (percDeleted_l >= 1.0)
				{
					break;
				}
				else
				{
					k++;
					Log(info, "K = %d", k);
					if (percDeleted_l > 0.1)
					{

						CUBSelect(s1.gdata(), s2.gdata(), keep_l.gdata(), numEdges);
						T newNumEdges = CUBSelect(d1.gdata(), d2.gdata(), keep_l.gdata(), numEdges);

						ptrSrc.gdata() = s2.gdata();
						s2.gdata() = s1.gdata();
						s1.gdata() = ptrSrc.gdata();

						ptrDst.gdata() = d2.gdata();
						d2.gdata() = d1.gdata();
						d1.gdata() = ptrDst.gdata();

						dimGridEdges = (newNumEdges + dimBlock - 1) / dimBlock;
						numEdges = newNumEdges;


						RebuildArrays<dimBlock> << <dimGridEdges, dimBlock, 0, stream_ >> > (0, numEdges, rowPtr.gdata(), ptrSrc.gdata(), keep_l.gdata(), affected_l.gdata());
						RebuildReverse<dimBlock> << <dimGridEdges, dimBlock, 0, stream_ >> > (0, numEdges, rowPtr.gdata(), ptrSrc.gdata(), ptrDst.gdata(), reversed.gdata());



						cudaDeviceSynchronize();
						CUDA_RUNTIME(cudaGetLastError());
					}
					assumpAffected = true;
				}
				cudaDeviceSynchronize();
			}
		}

		uint findKtrussIncremental_sync(int kmin, int kmax, GPUArray<T> rowPtr, GPUArray<T> rowInd, GPUArray<T> colInd, const size_t numNodes, const size_t numEdges, unsigned short* reverseIndex, EncodeDataType* bitMap, const size_t nodeOffset = 0, const size_t edgeOffset = 0)
		{
			findKtrussIncremental_async(kmin, kmax, rowPtr, rowInd, colInd, numNodes, numEdges, reverseIndex, bitMap, nodeOffset, edgeOffset);
			sync();
			return count();
		}

		void sync() { CUDA_RUNTIME(cudaStreamSynchronize(stream_)); }

		uint count() const { return k - 1; }
		int device() const { return dev_; }
	};

} // namespace pangolin
