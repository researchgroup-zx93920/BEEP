
#pragma once

#include "sgm_kernels.cuh"

namespace graph
{
    /*******************
    * CLASS DEFINITION *
    *******************/
    template<typename T>
    class SG_Match
    {
    private:
        // GPU info
        int dev_;
        cudaStream_t stream_;
        AllocationTypeEnum alloc_;

        // Processed query graphs
        GPUArray<T>* query_sequence;
        GPUArray<uint>* query_edges;
        GPUArray<uint>* query_edge_ptr;        

    public:
        // Constructors
        SG_Match(int dev, AllocationTypeEnum alloc = AllocationTypeEnum::cpuonly, cudaStream_t stream = 0) :
            dev_(dev),
            alloc_(alloc),
            stream_(stream) 
        {
            query_sequence = new GPUArray<T>("Query Sequence", alloc_);
            query_edges = new GPUArray<uint>("Query edges", alloc_);
            query_edge_ptr = new GPUArray<uint>("Query edge ptr", alloc_); 
        }

        SG_Match(): SG_Match(0) {}

        // Destructor
        ~SG_Match() {
            query_sequence->freeCPU();
            delete query_sequence;

            query_edges->freeCPU();
            delete query_edges;

            query_edge_ptr->freeCPU();
            delete query_edge_ptr;
        }

        void run(graph::COOCSRGraph_d<T>& dataGraph, graph::COOCSRGraph<T>& patGraph)
        {
            CUDA_RUNTIME(cudaSetDevice(dev_));
		    CUDA_RUNTIME(cudaStreamCreate(&stream_));
            CUDA_RUNTIME(cudaStreamSynchronize(stream_));
            
            double time;

            //Timer qp;
            preprocess_query(patGraph);
            //time = qp.elapsed();
            //Log(info, "Template preprocessing Time: %f ms", time*1000);

            Timer t;
            count_subgraphs(dataGraph);
            time = t.elapsed();
		    Log(info, "count time %f s", time);
        }

    protected:
        // Function declarations. Definitions outside class.
        void preprocess_query(graph::COOCSRGraph<T>& patGraph);
        void count_subgraphs(graph::COOCSRGraph_d<T>& dataGraph);
    };


    /***********************
    * FUNCTION DEFINITIONS *
    ***********************/
    
    template<typename T>
    void SG_Match<T>::preprocess_query(graph::COOCSRGraph<T>& query)
    {
        // Allocate and initialize sequence array, tree and non-tree edge arrays
        query_sequence->allocate_cpu(query.numNodes);
        query_edges->allocate_cpu(query.numEdges);
        query_edge_ptr->allocate_cpu(query.numNodes + 1);

        query_sequence->setAll(0, false);
        query_edges->setAll(0, false);
        query_edge_ptr->setAll(0, false);

        // Initialise "Neighborhood Encoding" for query graph
        // For unlabeled graph = degree (Current implementation)
        uint* degree = new uint[query.numNodes];
           
        // Generate Query node sequence based on "Maxmimum Likelihood Estimation" (MLE)
        // First look for node with highest node-mapping-degree d_M 
        //      (i.e., degree with nodes already in query_sequence)
        // Nodes with same d_M are sorted with their likelihood P_f
        //      (This computation is ignored here, check VF3 for details)
        // Nodes with same d_M and p_f are sorted with their degree
        //
        // These conditions can be combined into one as follows:
        //      d_M * max_degree + node_degree

        // Initialise d_M with 0 for all nodes
        int* d_M = new int[query.numNodes];
        memset(d_M, 0, query.numNodes * sizeof(int));

        // For ith node in the query sequence
        for ( int i = 0; i < query.numNodes; i++ ) {
            uint ml = 0; // Maximum likelihood 
            uint idx = 0; // Index of node with Max Likelihood

            // Traverse all nodes to find ml and idx
            for ( int j = 0; j < query.numNodes; j++ ) {
                if (d_M[j] >= 0) {  // d_M = -1 denotes node already in sequence
                    int likelihood = d_M[j] * query.numNodes + degree[j];
                    if ( likelihood > ml) {
                        ml = likelihood;
                        idx = j;
                    }
                }   
            }

            // Append node to sequence
            query_sequence->cdata()[i] = idx;
            
            // Mark idx as done in d_M
            d_M[idx] = -1;
            for ( int j = query.rowPtr->cdata()[idx]; j < query.rowPtr->cdata()[idx+1]; j++ ) {
                uint neighbor = query.colInd->cdata()[j];
                if ( d_M[neighbor] != -1 ) d_M[neighbor]++;
            }

            // Populate query edges
            query_edge_ptr->cdata()[i+1] = query_edge_ptr->cdata()[i];

            // For all previous nodes in the sequence
            for ( int j = 0; j < i; j++ ) {
                // For all neighbors of node_i
                for ( int n = query.rowPtr->cdata()[idx]; n < query.rowPtr->cdata()[idx+1]; n++) {
                    // If neighbor is same as node_j, it's an edge
                    if ( query.colInd->cdata()[n] == query_sequence->cdata()[j]) {
                        query_edges->cdata()[query_edge_ptr->cdata()[i + 1]++] = j;
                    }
                }
            }
        }
        
        // Clean Up
        delete[] d_M;
        delete[] degree;

        /*
        // Print statements to check results.
        printf("Node Sequence:\n");
        for (int i = 0; i < query.numNodes; i++) {
            printf("i: %d;\tnode: %d\n", i, query_sequence->cdata()[i]);
        }

        printf("Query edges:\n");
        for (int i = 0; i < query.numNodes; i++) {
            printf("i: %d\t", i);
            for (int j = query_edge_ptr->cdata()[i]; j < query_edge_ptr->cdata()[i+1]; j++ ) {
                printf("%d,", query_edges->cdata()[j]);
            }
            printf("\n");
        }
        */
    }

    template<typename T>
    void SG_Match<T>::count_subgraphs(graph::COOCSRGraph_d<T>& dataGraph)
    {
        // Initialise Kernel Dims
        const auto block_size = 128;
        const T partitionSize = 4;
        const T numPartitions = block_size / partitionSize;
        const uint dv = 32;

        // CUDA Initialise, gather runtime Info.
        CUDAContext context;
        T num_SMs = context.num_SMs;
        T conc_blocks_per_SM = context.GetConCBlocks(block_size);

        // Initialise Arrays
        GPUArray<uint64> counter("Temp level Counter", unified, 1, dev_);
        GPUArray<T> maxDegree("Temp Degree", unified, 1, dev_);
        GPUArray<T> d_bitmap_states("bmp bitmap stats", AllocationTypeEnum::gpu, num_SMs * conc_blocks_per_SM, dev_);

        counter.setSingle(0, 0, true);
        maxDegree.setSingle(0, 0, true);
        d_bitmap_states.setAll(0, true);

        // Compute Max Degree
        const int dimBlock = 256;
        GPUArray<T> nodeDegree;
		nodeDegree.initialize("Edge Support", unified, dataGraph.numNodes, dev_);
		uint dimGridNodes = (dataGraph.numNodes + dimBlock - 1) / dimBlock;
		execKernel((getNodeDegree_kernel<T, dimBlock>), dimGridNodes, dimBlock, dev_, false, nodeDegree.gdata(), dataGraph, maxDegree.gdata());

        // Array Sizes
        uint num_divs = (maxDegree.gdata()[0] + dv - 1) / dv;
        const uint64 level_size = num_SMs * conc_blocks_per_SM * numPartitions * DEPTH * num_divs;
        const uint64 encode_size = num_SMs * conc_blocks_per_SM * maxDegree.gdata()[0] * num_divs;
        //printf("Level Size = %llu, Encode Size = %llu\n", level_size, encode_size);
        GPUArray<T> current_level("Temp level Counter", unified, level_size, dev_);
        GPUArray<T> node_be("Temp level Counter", unified, encode_size, dev_);
        current_level.setAll(0, true);
        node_be.setAll(0, true);

        // GPU Constant memory
        cudaMemcpyToSymbol(KCCOUNT, &(query_sequence->N), sizeof(KCCOUNT));
        cudaMemcpyToSymbol(PARTSIZE, &partitionSize, sizeof(PARTSIZE));
        cudaMemcpyToSymbol(NUMPART, &numPartitions, sizeof(NUMPART));
        cudaMemcpyToSymbol(MAXLEVEL, &DEPTH, sizeof(MAXLEVEL));
        cudaMemcpyToSymbol(CBPSM, &(conc_blocks_per_SM), sizeof(CBPSM));  
        cudaMemcpyToSymbol(NUMDIVS, &num_divs, sizeof(NUMDIVS));
        cudaMemcpyToSymbol(MAXDEG, &(maxDegree.gdata()[0]), sizeof(MAXDEG));
        cudaMemcpyToSymbol(QEDGE, &(query_edges->cdata()[0]), query_edges->N * sizeof(QEDGE[0]));
        cudaMemcpyToSymbol(QEDGE_PTR, &(query_edge_ptr->cdata()[0]), query_edge_ptr->N * sizeof(QEDGE_PTR[0]));

        // Kernel Launch
        auto grid_block_size = dataGraph.numNodes;
        execKernel((sgm_kernel_central_node_base_binary<T, block_size, partitionSize>), grid_block_size, block_size, dev_, false,
            counter.gdata(),
            dataGraph,
            current_level.gdata(),
            d_bitmap_states.gdata(), node_be.gdata());

        
        // Cleanup
        current_level.freeGPU();
        node_be.freeGPU();

        counter.freeGPU();
        d_bitmap_states.freeGPU();
			
        std::cout << "------------- Counter = " << counter.gdata()[0] << "\n";
    }
}