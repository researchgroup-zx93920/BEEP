#pragma once

#include "../include/utils.cuh"
#include "../include/utils_cuda.cuh"
#include "../include/Logger.cuh"
#include "../include/CGArray.cuh"


typedef uint64_t EncodeDataType;




template <size_t BLOCK_DIM_X>
__global__ void InitializeArrays(uint edgeStart, uint numEdges, uint* rowPtr, uint* rowInd, uint* colInd, BCTYPE* keep_l,
	bool* affected_l, uint* reversed, uint* srcKP, uint* destKP)
{
	uint tx = threadIdx.x;
	uint bx = blockIdx.x;

	uint ptx = tx + bx * BLOCK_DIM_X;

	for (uint i = ptx + edgeStart; i < edgeStart + numEdges; i += BLOCK_DIM_X * gridDim.x)
	{
		//node
		uint sn = rowInd[i];
		uint dn = colInd[i];
		//length
		uint sl = rowPtr[sn + 1] - rowPtr[sn];
		uint dl = rowPtr[dn + 1] - rowPtr[dn];

		bool val = sl > 1 && dl > 1;
		keep_l[i] = val;
		affected_l[i] = false;

		reversed[i] = getEdgeId(rowPtr, colInd, dn, sn);
		srcKP[i] = i;
		destKP[i] = i;
	}
}


template <size_t BLOCK_DIM_X>
__global__ void RebuildArrays(uint edgeStart, uint numEdges, uint* rowPtr, uint* rowInd, BCTYPE* keep_l, bool* affected_l)
{
	uint tx = threadIdx.x;
	uint bx = blockIdx.x;

	__shared__ uint rows[BLOCK_DIM_X + 1];

	uint ptx = tx + bx * BLOCK_DIM_X;

	for (int i = ptx + edgeStart; i < edgeStart + numEdges; i += BLOCK_DIM_X * gridDim.x)
	{
		rows[tx] = rowInd[ptx];

		keep_l[i] = true;
		affected_l[i] = false;

		__syncthreads();

		uint end = rows[tx];
		if (i == 0)
		{
			rowPtr[end] = 0;
		}
		else if (tx == 0)
		{
			uint start = rowInd[i - 1];
			for (uint j = start + 1; j <= end; j++)
			{
				rowPtr[j] = i;
			}
		}
		else if (i == numEdges - 1)
		{
			rowPtr[end + 1] = i + 1;

			uint start = rows[tx - 1];
			for (uint j = start + 1; j <= end; j++)
			{
				rowPtr[j] = i;
			}
		}
		else
		{
			uint start = rows[tx - 1];
			for (uint j = start + 1; j <= end; j++)
			{
				rowPtr[j] = i;
			}

		}
	}

}

template <size_t BLOCK_DIM_X>
__global__ void RebuildReverse(uint edgeStart, uint numEdges, uint* rowPtr, uint* rowInd, uint* colInd, uint* reversed)
{
	uint tx = threadIdx.x;
	uint bx = blockIdx.x;

	uint ptx = tx + bx * BLOCK_DIM_X;

	for (uint i = ptx + edgeStart; i < edgeStart + numEdges; i += BLOCK_DIM_X * gridDim.x)
	{
		//node
		uint sn = rowInd[i];
		uint dn = colInd[i];
		//length
		uint sl = rowPtr[sn + 1] - rowPtr[sn];
		uint dl = rowPtr[dn + 1] - rowPtr[dn];
		reversed[i] = getEdgeId(rowPtr, colInd, dn, sn);
	}
}


template <size_t BLOCK_DIM_X>
__global__ void NodeEliminate(uint w, uint edgeStart, uint numEdges, uint* rowPtr, uint* rowInd, uint* colInd, BCTYPE* keep, uint* gnumkept)
{
	uint tx = threadIdx.x;
	uint bx = blockIdx.x;

	uint ptx = tx + bx * BLOCK_DIM_X;
	uint numKept = 0;

	for (uint i = ptx + edgeStart; i < edgeStart + numEdges; i += BLOCK_DIM_X * gridDim.x)
	{
		//node
		uint sn = rowInd[i];
		uint dn = colInd[i];
		//length
		uint sl = rowPtr[sn + 1] - rowPtr[sn];
		uint dl = rowPtr[dn + 1] - rowPtr[dn];

		keep[i] = (sl >= w) && (dl >= w);
		numKept += keep[i];
	}

	// Block-wide reduction of threadCount
	typedef cub::BlockReduce<uint, BLOCK_DIM_X> BlockReduce;
	__shared__ typename BlockReduce::TempStorage tempStorage;
	uint keptByBlock = BlockReduce(tempStorage).Sum(numKept);
	if (0 == threadIdx.x)
	{
		atomicAdd(gnumkept, keptByBlock);
	}
}


__device__ inline int AffectOthers(uint sp, uint dp, BCTYPE* keep, bool* affected, uint* reversed)
{
	int numberAffected = 0;
	int y1 = reversed[sp];
	int y2 = reversed[dp];

	if (!affected[sp] /*&& keep[sp]*/)
	{
		affected[sp] = true;
		numberAffected++;
	}
	if (!affected[dp] /*&& keep[dp]*/)
	{
		affected[dp] = true;
		numberAffected++;
	}
	if (!affected[y1] /*&& keep[y1]*/)
	{
		affected[y1] = true;
		numberAffected++;
	}
	if (!affected[y2] /*&& keep[y2]*/)
	{
		affected[y2] = true;
		numberAffected++;
	}

	return numberAffected;
}

template <size_t BLOCK_DIM_X>
__global__ void core_indirect(uint* keepPointer, uint* gnumdeleted, uint* gnumaffected,
	const uint k, const size_t edgeStart, const size_t numEdges,
	uint* rowPtr, uint* rowInd, uint* colInd, BCTYPE* keep, bool* affected, uint* reversed, bool firstTry, const int uMax)
{
	// kernel call
	size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
	uint numberDeleted = 0;
	uint numberAffected = 0;
	__shared__ bool didAffectAnybody[1];
	bool ft = firstTry; //1
	if (0 == threadIdx.x)
		didAffectAnybody[0] = false;

	__syncthreads();
	int startS = 0, startD = 0, endS = 0, endD = 0;
	for (int z = 0; z < uMax; z++)
	{
		numberDeleted = 0;
		for (size_t ii = gx + edgeStart; ii < edgeStart + numEdges; ii += BLOCK_DIM_X * gridDim.x)
		{
			size_t i = keepPointer[ii];

			int srcNode = rowInd[i];
			int dstNode = colInd[i];

			if (keep[i] && srcNode < dstNode && (affected[i] || ft))
			{
				affected[i] = false;
				int edgeCount = 0;

				uint sp = startS == 0 ? rowPtr[rowInd[i]] : startS;
				uint send = rowPtr[rowInd[i] + 1];

				uint dp = startD == 0 ? rowPtr[colInd[i]] : startD;
				uint dend = rowPtr[colInd[i] + 1];

				bool firstHit = true;
				while (edgeCount < k - 2 && sp < send && dp < dend)
				{
					uint sv = /*sp <limit? source[sp -  spBase]:*/ colInd[sp];
					uint dv = colInd[dp];

					if (sv == dv)
					{
						if (keep[sp] && keep[dp])
						{
							edgeCount++;
							if (firstHit)
							{
								startS = sp;
								startD = dp;
								firstHit = false;
							}

							bool cond = ((dend - dp) < (k - 2 - edgeCount)) || ((send - sp) < (k - 2 - edgeCount)); //fact
							if (!cond)
							{
								endS = sp + 1;
								endD = dp + 1;
							}
							else
							{
								numberAffected += AffectOthers(sp, dp, keep, affected, reversed);
							}

						}
					}
					int yy = sp + ((sv <= dv) ? 1 : 0);
					dp = dp + ((sv >= dv) ? 1 : 0);
					sp = yy;
				}

				//////////////////////////////////////////////////////////////
				if (edgeCount < (k - 2))
				{
					uint ir = reversed[i];
					keep[i] = false;
					keep[ir] = false;

					uint sp = startS;
					uint dp = startD;

					while (edgeCount > 0 && sp < endS && dp < endD)
					{
						uint sv = /*sp < limit? source[sp -  spBase]:*/ colInd[sp];
						uint dv = colInd[dp];

						if ((sv == dv))
						{
							numberAffected += AffectOthers(sp, dp, keep, affected, reversed);
						}
						int yy = sp + ((sv <= dv) ? 1 : 0);
						dp = dp + ((sv >= dv) ? 1 : 0);
						sp = yy;
					}
				}
			}

			if (!keep[i])
				numberDeleted++;
		}
		ft = false;
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
		//atomicAdd(gnumaffected, affectedByBlock);
	}

}


template <size_t BLOCK_DIM_X>
__global__ void core_directA(
	uint* gnumdeleted, uint* gnumaffected,
	const uint k, const size_t edgeStart, const size_t numEdges,
	uint* rowPtr, uint* rowInd, uint* colInd, const size_t numNodes, BCTYPE* keep, bool* affected, uint* reversed, bool firstTry, const int uMax, unsigned short* reverseIndex, EncodeDataType* bitMap)

{
	// kernel call
	size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
	uint numberDeleted = 0;
	uint numberAffected = 0;
	__shared__ bool didAffectAnybody[1];
	bool ft = firstTry; //1
	if (0 == threadIdx.x)
		didAffectAnybody[0] = false;

	__syncthreads();
	uint startS = 0, startD = 0, endS = 0, endD = 0;

	numberDeleted = 0;
	for (size_t i = gx + edgeStart; i < edgeStart + numEdges; i += BLOCK_DIM_X * gridDim.x)
	{
		uint srcNode = rowInd[i];
		uint dstNode = colInd[i];
		if (keep[i] && srcNode < dstNode && (affected[i] || ft))
		{
			affected[i] = false;
			uint triCount = 0;

			uint sp = startS == 0 ? rowPtr[rowInd[i]] : startS;
			uint send = rowPtr[rowInd[i] + 1];

			uint dp = startD == 0 ? rowPtr[colInd[i]] : startD;
			uint dend = rowPtr[colInd[i] + 1];

			bool firstHit = true;
			while (triCount < k - 2 && sp < send && dp < dend)
			{
				uint sv = colInd[sp];
				uint dv = colInd[dp];

				if (sv == dv)
				{
					if (keep[sp] && keep[dp])
					{
						triCount++;
						if (firstHit)
						{
							startS = sp;
							startD = dp;
							firstHit = false;
						}

						bool cond = ((dend - dp) < (k - 2 - triCount)) || ((send - sp) < (k - 2 - triCount)); //fact
						if (!cond)
						{
							endS = sp + 1;
							endD = dp + 1;
						}
						else
						{
							numberAffected += AffectOthers(sp, dp, keep, affected, reversed);
						}

					}
				}
				uint yy = sp + ((sv <= dv) ? 1 : 0);
				dp = dp + ((sv >= dv) ? 1 : 0);
				sp = yy;
			}

			//////////////////////////////////////////////////////////////
			if (triCount < (k - 2))
			{
				uint ir = reversed[i];
				keep[i] = false;
				keep[ir] = false;

				uint sp = startS;
				uint dp = startD;

				while (triCount > 0 && sp < endS && dp < endD)
				{
					uint sv = colInd[sp];
					uint dv = colInd[dp];

					if ((sv == dv))
					{
						numberAffected += AffectOthers(sp, dp, keep, affected, reversed);
					}
					uint yy = sp + ((sv <= dv) ? 1 : 0);
					dp = dp + ((sv >= dv) ? 1 : 0);
					sp = yy;
				}
			}
		}

		if (!keep[i] && srcNode < dstNode)
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

template <size_t BLOCK_DIM_X>
__global__ void core_full_direct(uint* gnumdeleted, uint* gnumaffected,
	const uint k, const size_t edgeStart, const size_t numEdges,
	uint* rowPtr, uint* rowInd, uint* colInd, BCTYPE* keep, bool* affected, uint* reversed, bool firstTry, const int uMax)
{
	// kernel call
	size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
	uint numberDeleted = 0;
	uint numberAffected = 0;
	__shared__ bool didAffectAnybody[1];
	bool ft = firstTry; //1
	if (0 == threadIdx.x)
		didAffectAnybody[0] = false;

	__syncthreads();
	uint startS = 0, startD = 0, endS = 0, endD = 0;

	numberDeleted = 0;
	for (size_t i = gx + edgeStart; i < edgeStart + numEdges; i += BLOCK_DIM_X * gridDim.x)
	{
		uint srcNode = rowInd[i];
		uint dstNode = colInd[i];
		if (keep[i] && srcNode < dstNode && (affected[i] || ft))
		{
			affected[i] = false;
			uint triCount = 0;


			uint sp = rowPtr[rowInd[i]];
			uint send = rowPtr[rowInd[i] + 1];

			uint dp = rowPtr[colInd[i]];
			uint dend = rowPtr[colInd[i] + 1];

			bool firstHit = true;
			while (triCount < k - 2 && sp < send && dp < dend)
			{
				uint sv = colInd[sp];
				uint dv = colInd[dp];

				if (sv == dv)
				{
					if (keep[sp] && keep[dp])
					{
						triCount++;
						if (firstHit)
						{
							startS = sp;
							startD = dp;
							firstHit = false;
						}

						bool cond = ((dend - dp) < (k - 2 - triCount)) || ((send - sp) < (k - 2 - triCount)); //fact
						if (!cond)
						{
							endS = sp + 1;
							endD = dp + 1;
						}
						else
						{
							numberAffected += AffectOthers(sp, dp, keep, affected, reversed);
						}

					}
				}
				uint yy = sp + ((sv <= dv) ? 1 : 0);
				dp = dp + ((sv >= dv) ? 1 : 0);
				sp = yy;
			}

			//////////////////////////////////////////////////////////////
			if (triCount < (k - 2))
			{
				keep[i] = false;
				uint ir = reversed[i];
				keep[ir] = false;

				uint sp = startS;
				uint dp = startD;

				while (triCount > 0 && sp < endS && dp < endD)
				{
					uint sv = colInd[sp];
					uint dv = colInd[dp];

					if ((sv == dv))
					{
						numberAffected += AffectOthers(sp, dp, keep, affected, reversed);
					}
					uint yy = sp + ((sv <= dv) ? 1 : 0);
					dp = dp + ((sv >= dv) ? 1 : 0);
					sp = yy;
				}
			}
		}

		if (!keep[i] && srcNode < dstNode)
			numberDeleted += 2;//numberDeleted++;
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
template <size_t N, size_t GRID_DIM_X, size_t BLOCK_DIM_X, typename T>
__global__ void zero(T* ptr //!< [in] pointer to array
) {
	const size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
	for (size_t i = gx; i < N; i += GRID_DIM_X * BLOCK_DIM_X) {
		ptr[i] = static_cast<T>(0);
	}
}

template <size_t N, typename T> void zero_async(T* ptr, const int dev, cudaStream_t stream) {
	CUDA_RUNTIME(cudaSetDevice(dev));
	constexpr size_t dimBlock = 512;
	constexpr size_t dimGrid = (dimBlock + N - 1) / dimBlock;
	Log(debug, "launch zero: device = %d, blocks = %d, threads = %d stream = %ul", dev, dimGrid, dimBlock,uintptr_t(stream));
	zero<N, dimGrid, dimBlock> << <dimGrid, dimBlock, 0, stream >> > (ptr);
	CUDA_RUNTIME(cudaGetLastError());
}

namespace graph 
{
	template<typename T>
	class SingleGPU_Ktruss 
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
		SingleGPU_Ktruss(int dev) : dev_(dev) {
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

		SingleGPU_Ktruss() : SingleGPU_Ktruss(0) {}

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

			cudaDeviceSynchronize();

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

					core_directA<dimBlock> << <dimGridEdges, dimBlock, 0, stream_ >> > (gnumdeleted,
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

						/*CUBSelect(s1.gdata(), s2.gdata(), keep_l.gdata(), numEdges);
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
						CUDA_RUNTIME(cudaGetLastError());*/
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
