#pragma once
#define QUEUE_SIZE 1024

#include <driver_types.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include "../include/utils.cuh"
#include "../include/Logger.cuh"
#include "../include/CGArray.cuh"

#include "../triangle_counting/TcBase.cuh"
#include "../triangle_counting/TcSerial.cuh"
#include "../triangle_counting/TcBinary.cuh"
#include "../triangle_counting/TcVariablehash.cuh"
#include "../triangle_counting/testHashing.cuh"
#include "../triangle_counting/TcBmp.cuh"

#include "../include/GraphQueue.cuh"

#include "mckernels.cuh"
#include "kckernels.cuh"

// namespace graph
// {
//     template<typename T>
//     class SingleGPU_Maximal_Clique
//     {
//     private:
//         int dev_;
//         cudaStream_t stream_;

//         void AscendingGpu(T n, GPUArray<T>& identity_arr_asc)
//         {
//             long grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
//             identity_arr_asc.initialize("Identity Array Asc", AllocationTypeEnum::gpu, n, dev_);
//             execKernel(init_asc, grid_size, BLOCK_SIZE, dev_, false, identity_arr_asc.gdata(), n);
//         }

//     public:
//         GPUArray<T> nodeDegree;
//         GPUArray<T> edgePtr;
//         GPUArray <uint64> cpn;
//         GPUArray<T> identity_arr_asc;

//         SingleGPU_Maximal_Clique(int dev, COOCSRGraph_d<T>& g) : dev_(dev) {
//             CUDA_RUNTIME(cudaSetDevice(dev_));
//             CUDA_RUNTIME(cudaStreamCreate(&stream_));
//             CUDA_RUNTIME(cudaStreamSynchronize(stream_));

//             AscendingGpu(g.numEdges, identity_arr_asc);

//             edgePtr.initialize("Edge Support", unified, g.numEdges, dev_);
//         }

//         SingleGPU_Maximal_Clique() : SingleGPU_Maximal_Clique(0) {}

//         void getNodeDegree(COOCSRGraph_d<T>& g, T* maxDegree,
//             const size_t nodeOffset = 0, const size_t edgeOffset = 0)
//         {
//             const int dimBlock = 128;
//             nodeDegree.initialize("Edge Support", unified, g.numNodes, dev_);
//             uint dimGridNodes = (g.numNodes + dimBlock - 1) / dimBlock;
//             execKernel((getNodeDegree_kernel<T, dimBlock>), dimGridNodes, dimBlock, dev_, false, nodeDegree.gdata(), g, maxDegree);
//         }

//         template<const int PSIZE>
//         void find_maximal_clique_node_pivot_async_local(COOCSRGraph_d<T>& gd,
//             const size_t nodeOffset = 0, const size_t edgeOffset = 0)
//         {
//             CUDA_RUNTIME(cudaSetDevice(dev_));
//             const auto block_size = 128;
//             CUDAContext context;
//             T num_SMs = context.num_SMs;

//             cpn = GPUArray <uint64> ("clique Counter", gpu, gd.numNodes, dev_);
//             GPUArray <T> maxDegree("Max Degree", unified, 1, dev_);

//             T conc_blocks_per_SM = context.GetConCBlocks(block_size);
//             GPUArray<T> d_bitmap_states("bmp bitmap stats", AllocationTypeEnum::gpu, num_SMs * conc_blocks_per_SM, dev_);

//             maxDegree.setSingle(0, 0, true);
//             d_bitmap_states.setAll(0, true);
//             cpn.setAll(0, true);

//             getNodeDegree(gd, maxDegree.gdata());

//             const T partitionSize = PSIZE; 
//             T factor = (block_size / partitionSize);

//             const uint dv = 32;
//             const uint max_level = maxDegree.gdata()[0];
//             uint num_divs = (maxDegree.gdata()[0] + dv - 1) / dv;
            
//             const uint64 encode_size = num_SMs * conc_blocks_per_SM * maxDegree.gdata()[0] * num_divs; //per block
//             GPUArray<T> node_be("Binary Encoding Array", gpu, encode_size, dev_);

//             const uint64 level_size = num_SMs * conc_blocks_per_SM * /*factor **/ max_level * num_divs; //per partition
//             const uint64 level_item_size = num_SMs * conc_blocks_per_SM * /*factor **/ max_level; //per partition

//             GPUArray<T> current_level2("Temp level Counter", gpu, level_size, dev_);
//             GPUArray<T> possible("Possible", gpu, level_size, dev_);
//             GPUArray<T> x_level("X", gpu, level_size, dev_);
            
//             GPUArray<T> level_count("Level Count", gpu, level_item_size, dev_);
//             GPUArray<T> level_prev("Level Prev", gpu, level_item_size, dev_);

//             printf("Level Size = %llu, Encode Size = %llu\n", 3 *level_size + 2 * level_item_size, encode_size);

//             current_level2.setAll(0, true);
//             node_be.setAll(0, true);
//             const T numPartitions = block_size/partitionSize;
//             cudaMemcpyToSymbol(PARTSIZE, &partitionSize, sizeof(PARTSIZE));
//             cudaMemcpyToSymbol(NUMPART, &numPartitions, sizeof(NUMPART));
//             cudaMemcpyToSymbol(MAXLEVEL, &max_level, sizeof(MAXLEVEL));
//             cudaMemcpyToSymbol(NUMDIVS, &num_divs, sizeof(NUMDIVS));
//             cudaMemcpyToSymbol(MAXDEG, &(maxDegree.gdata()[0]), sizeof(MAXDEG));
//             cudaMemcpyToSymbol(CBPSM, &(conc_blocks_per_SM), sizeof(CBPSM));

//             CUDA_RUNTIME(cudaGetLastError());
//             cudaDeviceSynchronize();

//             auto grid_block_size = (gd.numNodes + block_size - 1) / block_size;
//             execKernel((mckernel_node_block_warp_binary<T, block_size, partitionSize>), grid_block_size, block_size, dev_, false,
//                 gd,
//                 gd.numNodes,
//                 current_level2.gdata(), cpn.gdata(),
//                 d_bitmap_states.gdata(), node_be.gdata(),

//                 possible.gdata(),
//                 x_level.gdata(),
//                 level_count.gdata(),
//                 level_prev.gdata()
//             );

//             std::cout.imbue(std::locale(""));
//             // std::cout << "Nodes = " << g.numNodes << ", Edges = " << g.numEdges << ", Counter = " << counter.gdata()[0] << "\n";

//             current_level2.freeGPU();
//             d_bitmap_states.freeGPU();
//             maxDegree.freeGPU();
//             node_be.freeGPU();
//             possible.freeGPU();
//             x_level.freeGPU();
//             level_count.freeGPU();
//             level_prev.freeGPU();
//             cpn.copytocpu(0);
//             cpn.freeGPU();
//         }

//         void free_memory()
//         {
//             identity_arr_asc.freeGPU();
            
//             nodeDegree.freeGPU();
//             edgePtr.freeGPU();
//             cpn.freeCPU();
//         }

//         void save(const T& n)
//         {
//             // save the result to file
//         }

//         void show(const T& n)
//         {
//             uint64_t tot = 0;
//             uint64_t mx = 0;
//             auto cdata = cpn.cdata();
//             for (T i = 0; i < n; i ++)
//             {
//                 tot += cdata[i];   
//                 mx = mx < cdata[i] ? cdata[i] : mx;
//             }
//             std::cout << "Total: " << tot << ' ' << "Max: " << mx << '\n';
//         }

//         ~SingleGPU_Maximal_Clique()
//         {
//             free_memory();
//         }

//         void sync() { CUDA_RUNTIME(cudaStreamSynchronize(stream_)); }
//         int device() const { return dev_; }
//         cudaStream_t stream() const { return stream_; }
//     };

// }

namespace graph
{
    template<typename T>
    class SingleGPU_Maximal_Clique
    {
    private:
        int dev_;
        cudaStream_t stream_;

    public:
        GPUArray<T> nodeDegree;
        GPUArray <uint64> cpn;
        GPUArray<T> tmpNode;

        SingleGPU_Maximal_Clique(int dev, COOCSRGraph_d<T>& g) : dev_(dev) {
            CUDA_RUNTIME(cudaSetDevice(dev_));
            CUDA_RUNTIME(cudaStreamCreate(&stream_));
            CUDA_RUNTIME(cudaStreamSynchronize(stream_));
        }

        SingleGPU_Maximal_Clique() : SingleGPU_Maximal_Clique(0) {}

        void getNodeDegree(COOCSRGraph_d<T>& g, T* maxDegree,
            const size_t nodeOffset = 0, const size_t edgeOffset = 0)
        {
            const int dimBlock = 128;
            nodeDegree.initialize("Edge Support", unified, g.numNodes, dev_);
            uint dimGridNodes = (g.numNodes + dimBlock - 1) / dimBlock;
            execKernel((getNodeDegree_kernel<T, dimBlock>), dimGridNodes, dimBlock, dev_, false, nodeDegree.gdata(), g, maxDegree);
        }

        template<const int PSIZE>
        void find_maximal_clique_node_pivot_async_local(COOCSRGraph_d<T>& g_dir, COOCSRGraph_d<T>& g_undir,
            const size_t nodeOffset = 0, const size_t edgeOffset = 0)
        {
            CUDA_RUNTIME(cudaSetDevice(dev_));
            const auto block_size = 128;
            CUDAContext context;
            T num_SMs = context.num_SMs;

            cpn = GPUArray <uint64> ("clique Counter", gpu, g_dir.numNodes, dev_);
            GPUArray <T> maxDegree("Max Degree", unified, 1, dev_);
            GPUArray <T> maxUndirectedDegree("Max Undirected Degree", unified, 1, dev_);

            T conc_blocks_per_SM = context.GetConCBlocks(block_size);
            GPUArray<T> d_bitmap_states("bmp bitmap stats", AllocationTypeEnum::gpu, num_SMs * conc_blocks_per_SM, dev_);

            maxDegree.setSingle(0, 0, true);
            maxUndirectedDegree.setSingle(0, 0, true);
            d_bitmap_states.setAll(0, true);
            cpn.setAll(0, true);

            getNodeDegree(g_dir, maxDegree.gdata());
            getNodeDegree(g_undir, maxUndirectedDegree.gdata());

            const T partitionSize = PSIZE; 
            T factor = (block_size / partitionSize);

            const uint dv = 32;
            const uint max_level = maxDegree.gdata()[0];
            uint num_divs = (maxDegree.gdata()[0] + dv - 1) / dv;
            
            const uint64 encode_size = num_SMs * conc_blocks_per_SM * maxDegree.gdata()[0] * num_divs; //per block
            GPUArray<T> node_be("Binary Encoding Array", gpu, encode_size, dev_);

            const uint64 level_size = num_SMs * conc_blocks_per_SM * /*factor **/ max_level * num_divs; //per partition
            const uint64 level_item_size = num_SMs * conc_blocks_per_SM * /*factor **/ max_level; //per partition

            GPUArray<T> current_level2("Temp level Counter", gpu, level_size, dev_);
            GPUArray<T> possible("Possible", gpu, level_size, dev_);
            GPUArray<T> x_level("X", gpu, level_size, dev_);
            
            GPUArray<T> level_count("Level Count", gpu, level_item_size, dev_);
            GPUArray<T> level_prev("Level Prev", gpu, level_item_size, dev_);

            printf("Level Size = %llu, Encode Size = %llu\n", 3 *level_size + 2 * level_item_size, encode_size);

            current_level2.setAll(0, true);
            node_be.setAll(0, true);
            const T numPartitions = block_size/partitionSize;
            cudaMemcpyToSymbol(PARTSIZE, &partitionSize, sizeof(PARTSIZE));
            cudaMemcpyToSymbol(NUMPART, &numPartitions, sizeof(NUMPART));
            cudaMemcpyToSymbol(MAXLEVEL, &max_level, sizeof(MAXLEVEL));
            cudaMemcpyToSymbol(NUMDIVS, &num_divs, sizeof(NUMDIVS));
            cudaMemcpyToSymbol(MAXDEG, &(maxDegree.gdata()[0]), sizeof(MAXDEG));
            cudaMemcpyToSymbol(CBPSM, &(conc_blocks_per_SM), sizeof(CBPSM));

            CUDA_RUNTIME(cudaGetLastError());
            cudaDeviceSynchronize();

            tmpNode = GPUArray<T> ("Temp Node", gpu, g_undir.numEdges, dev_);
            tmpNode.setAll(0, true);
            
            auto grid_block_size = (g_dir.numNodes + block_size - 1) / block_size;
            execKernel((mckernel_node_block_warp_binary<T, block_size, partitionSize>), grid_block_size, block_size, dev_, false,
                g_undir,
                g_dir,
                g_dir.numNodes,
                current_level2.gdata(), cpn.gdata(),
                d_bitmap_states.gdata(), node_be.gdata(),

                possible.gdata(),
                x_level.gdata(),
                level_count.gdata(),
                level_prev.gdata(),
                tmpNode.gdata()
            );

            std::cout.imbue(std::locale(""));

            current_level2.freeGPU();
            d_bitmap_states.freeGPU();
            maxDegree.freeGPU();
            node_be.freeGPU();
            possible.freeGPU();
            x_level.freeGPU();
            level_count.freeGPU();
            level_prev.freeGPU();
            cpn.copytocpu(0);
            cpn.freeGPU();
        }

        void free_memory()
        {        
            nodeDegree.freeGPU();
            tmpNode.freeGPU();
            cpn.freeCPU();
        }

        void save(const T& n)
        {
            // save the result to file
        }

        void show(const T& n)
        {
            uint64_t tot = 0;
            uint64_t mx = 0;
            auto cdata = cpn.cdata();
            for (T i = 0; i < n; i ++)
            {
                tot += cdata[i];   
                mx = mx < cdata[i] ? cdata[i] : mx;
            }
            std::cout << "Total: " << tot << ' ' << "Max: " << mx << '\n';
        }

        ~SingleGPU_Maximal_Clique()
        {
            free_memory();
        }

        void sync() { CUDA_RUNTIME(cudaStreamSynchronize(stream_)); }
        int device() const { return dev_; }
        cudaStream_t stream() const { return stream_; }
    };

}