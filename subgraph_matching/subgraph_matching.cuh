
#pragma once

#include "sgm_kernels.cuh"

namespace graph
{
    
    // MAXN definition needed by NAUTY to allow static allocation
    // Set to maximum number of nodes in the template graph
    // DEPTH declared in sgm_kernels.cuh
    #define MAXN DEPTH 

    // Needs to be included in namespace graph to avoid name conflict
    #include "../nauty/nauty.h"

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
        GPUArray<uint>* sym_nodes;
        GPUArray<uint>* sym_nodes_ptr;
        
        uint min_qDegree, max_qDegree;

        // Processed data graph info
        GPUArray<uint64> counter;
        GPUArray<T> nodeDegree, max_dDegree;

        // Queues
        graph::GraphQueue<T, bool> bucket_q, current_q;

        // Array used by bucket scan
        GPUArray<T> asc;

        // Bucket Scan function from K-Cliques
        void bucket_scan(
			GPUArray<T> nodeDegree, T node_num, T level, T span,
			T& bucket_level_end_,
			GraphQueue<T, bool>& current,
			GraphQueue<T, bool>& bucket)
		{
			static bool is_first = true;
			static int multi = 1;
			if (is_first)
			{
				current.mark.setAll(false, true);
				bucket.mark.setAll(false, true);
				is_first = false;
			}

			if (level == bucket_level_end_)
			{
				// Clear the bucket_removed_indicator


				long grid_size = (node_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
				execKernel((filter_window<T, T>), grid_size, BLOCK_SIZE, dev_, false,
					nodeDegree.gdata(), node_num, bucket.mark.gdata(), level, bucket_level_end_ + span);

				multi++;

				bucket.count.gdata()[0] = CUBSelect(asc.gdata(), bucket.queue.gdata(), bucket.mark.gdata(), node_num, dev_);
				bucket_level_end_ += span;
			}
			// SCAN the window.
			if (bucket.count.gdata()[0] != 0)
			{
				current.count.gdata()[0] = 0;
				long grid_size = (bucket.count.gdata()[0] + BLOCK_SIZE - 1) / BLOCK_SIZE;
				execKernel((filter_with_random_append<T, T>), grid_size, BLOCK_SIZE, dev_, false,
					bucket.queue.gdata(), bucket.count.gdata()[0], nodeDegree.gdata(), current.mark.gdata(), current.queue.gdata(), &(current.count.gdata()[0]), level, span);
			}
			else
			{
				current.count.gdata()[0] = 0;
			}
			//Log(LogPriorityEnum::info, "Level: %d, curr: %d/%d", level, current.count.gdata()[0], bucket.count.gdata()[0]);
        }

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
            sym_nodes = new GPUArray<uint>("Symmetrical nodes", alloc_);
            sym_nodes_ptr = new GPUArray<uint>("Symmetrical node pointer", alloc_);

            counter.initialize("Temp level Counter", unified, 1, dev_);
            max_dDegree.initialize("Temp Degree", unified, 1, dev_);

            counter.setSingle(0, 0, false);
            max_dDegree.setSingle(0, 0, false);
            
            max_qDegree = 0;
            min_qDegree = INT_MAX;
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

            sym_nodes->freeCPU();
            delete sym_nodes;

            sym_nodes_ptr->freeCPU();
            delete sym_nodes_ptr;
        }

        void run(graph::COOCSRGraph_d<T>& dataGraph, graph::COOCSRGraph<T>& patGraph)
        {
            CUDA_RUNTIME(cudaSetDevice(dev_));
		    CUDA_RUNTIME(cudaStreamCreate(&stream_));
            CUDA_RUNTIME(cudaStreamSynchronize(stream_));
            
            double time;

            Timer qp;
            preprocess_query(patGraph);
            time = qp.elapsed();
            Log(info, "Template preprocessing Time: %f ms", time*1000);

            Timer dp;
            initialize(dataGraph);
            peel_data(dataGraph);
            time = dp.elapsed();
            Log(info, "Data graph preprocessing time: %f ms", time*1000);
            
            Timer t;
            count_subgraphs(dataGraph);
            time = t.elapsed();
		    Log(info, "count time %f s", time);
        }

    protected:
        // Function declarations. Definitions outside class.
        void preprocess_query(graph::COOCSRGraph<T>& patGraph);
        void detect_symmetry(graph::COOCSRGraph<T>& graph, int orbits[MAXN]);
        void initialize(graph::COOCSRGraph_d<T>& dataGraph);
        void peel_data(graph::COOCSRGraph_d<T>& dataGraph);
        void count_subgraphs(graph::COOCSRGraph_d<T>& dataGraph);
    };


    /***********************
    * FUNCTION DEFINITIONS *
    ***********************/
    
    template<typename T>
    void SG_Match<T>::preprocess_query(graph::COOCSRGraph<T>& query)
    {
        // Allocate and initialize arrays
        query_sequence->allocate_cpu(query.numNodes);
        query_edges->allocate_cpu(query.numEdges);
        query_edge_ptr->allocate_cpu(query.numNodes + 1);
        
        sym_nodes->allocate_cpu(query.numNodes * (query.numNodes - 1) / 2);
        sym_nodes_ptr->allocate_cpu(query.numNodes + 1);

        query_sequence->setAll(0, false);
        query_edges->setAll(0, false);
        query_edge_ptr->setAll(0, false);
        
        sym_nodes->setAll(0, false);
        sym_nodes_ptr->setAll(0, false);

        // Initialise "Neighborhood Encoding" for query graph
        // For unlabeled graph = degree (Current implementation)
        uint* degree = new uint[query.numNodes];
        for (int i = 0; i < query.numNodes; i++) {
            degree[i] = query.rowPtr->cdata()[i+1] - query.rowPtr->cdata()[i];
            if ( degree[i] > max_qDegree ) max_qDegree = degree[i];
            if ( degree[i] < min_qDegree ) min_qDegree = degree[i]; 
        }
           
        // Detect symmetrical nodes in query graph
        int orbits[MAXN];
        detect_symmetry(query, orbits);
        
        // Generate Query node sequence based on "Maxmimum Likelihood Estimation" (MLE)
        // First look for node with highest node-mapping-degree d_M 
        //      (i.e., degree with nodes already in query_sequence)
        // Nodes with same d_M are sorted with the highest symmetrical degree s_M
        //      (i.e., most number of symmetrical nodes already in the sequence)
        // Nodes with same d_M and p_f are sorted with their degree
        //
        // These conditions can be combined into one as follows:
        //      d_M * num_nodes^2 + s_M * num_nodes + node_degree

        // Initialise d_M and s_M with 0 for all nodes
        int* d_M = new int[query.numNodes];
        int* s_M = new int[query.numNodes];
        memset(d_M, 0, query.numNodes * sizeof(int));
        memset(s_M, 0, query.numNodes * sizeof(int));

        // For ith node in the query sequence
        for ( int i = 0; i < query.numNodes; i++ ) {
            uint ml = 0; // Maximum likelihood 
            uint idx = 0; // Index of node with Max Likelihood

            // Traverse all nodes to find ml and idx
            for ( int j = 0; j < query.numNodes; j++ ) {
                if (d_M[j] >= 0) {  // d_M = -1 denotes node already in sequence
                    uint likelihood = d_M[j] * query.numNodes * query.numNodes + s_M[j] * query.numNodes + degree[j];
                    if ( likelihood > ml) {
                        ml = likelihood;
                        idx = j;
                    }
                }   
            }

            // Append node to sequence
            query_sequence->cdata()[i] = idx;
            
            // Mark idx as done in d_M and s_M
            d_M[idx] = -1;
            s_M[idx] = -1;

            // Update d_M of other nodes
            for ( int j = query.rowPtr->cdata()[idx]; j < query.rowPtr->cdata()[idx+1]; j++ ) {
                uint neighbor = query.colInd->cdata()[j];
                if ( d_M[neighbor] != -1 ) d_M[neighbor]++;
            }

            // Update s_M of other nodes
            for (int j = 0; j < query.numNodes; j++) {
                if (orbits[j] == orbits[idx] && s_M[j] != -1) s_M[j]++;
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

            // Populate symmetrical nodes
            sym_nodes_ptr->cdata()[i+1] = sym_nodes_ptr->cdata()[i];
            
            for ( int j = 0; j < i; j++ ) {
                if ( orbits[query_sequence->cdata()[j]] == orbits[idx] ) {
                    sym_nodes->cdata()[sym_nodes_ptr->cdata()[i + 1]++] = j;
                }
            }
        }
        
        // Clean Up
        delete[] d_M;
        delete[] s_M;
        delete[] degree;

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

        printf("Orbits (as reported by NAUTY): ");
        for (int i = 0; i < query.numNodes; i++) {
            printf("%d, ", orbits[i]);
        }
        printf("\n");

        printf("Symmetrical nodes:\n");
        for (int i = 0; i < query.numNodes; i++) {
            printf("i: %d\t", i);
            for (int j = sym_nodes_ptr->cdata()[i]; j < sym_nodes_ptr->cdata()[i+1]; j++ ) {
                printf("%d,", sym_nodes->cdata()[j]);
            }
            printf("\n");
        }
    }

    template<typename T>
    void SG_Match<T>::detect_symmetry(graph::COOCSRGraph<T>& patGraph, int orbits[MAXN])
    {
        // Define required variables for NAUTY
        graph g[MAXN];
        int lab[MAXN], ptn[MAXN];
        static DEFAULTOPTIONS_GRAPH(opt);
        statsblk stats;
        int m = 1;
        int n = patGraph.numNodes;

        // Populate graph
        EMPTYGRAPH(g, m, n);
        for (int i = 0; i < patGraph.numNodes; i++) {
            for (int j = patGraph.rowPtr->cdata()[i]; j < patGraph.rowPtr->cdata()[i+1]; j++) {
                ADDONEEDGE(g, i, patGraph.colInd->cdata()[j], m);
            }
        }

        // Call NAUTY to get orbit
        densenauty(g, lab, ptn, orbits, &opt, &stats, m, n, NULL);
    }

    template<typename T>
    void SG_Match<T>::initialize(graph::COOCSRGraph_d<T>& dataGraph)
    {
        const auto block_size = 256;
        bucket_q.Create(unified, dataGraph.numEdges, dev_);
        current_q.Create(unified, dataGraph.numEdges, dev_);

        asc.initialize("Identity array asc", AllocationTypeEnum::gpu, dataGraph.numEdges, dev_);
        execKernel(init_asc, (dataGraph.numEdges + BLOCK_SIZE - 1)/ BLOCK_SIZE, BLOCK_SIZE, dev_,false, 
                    asc.gdata(), dataGraph.numEdges);

        CUDA_RUNTIME(cudaGetLastError());
        cudaDeviceSynchronize();

        // Compute Max Degree
		nodeDegree.initialize("Edge Support", unified, dataGraph.numNodes, dev_);
		uint dimGridNodes = (dataGraph.numNodes + block_size - 1) / block_size;
        execKernel((getNodeDegree_kernel<T, block_size>), dimGridNodes, block_size, dev_, false, 
                    nodeDegree.gdata(), dataGraph, max_dDegree.gdata());
    }

    template<typename T>
    void SG_Match<T>::peel_data(graph::COOCSRGraph_d<T>& dataGraph)
    {
        // CubLarge
        graph::CubLarge<uint> cl(dev_);
        const uint block_size = 512;

        // Initialise Arrays
        GPUArray<bool> keep;
        GPUArray<T> new_ptr;
        GPUArray<T> new_adj; 

        // Keep peeling till you can.
        do
        {
            // Find nodes with degree < min_qDegree
            T bucket_level_end_ = 1;
            bucket_scan(nodeDegree, dataGraph.numNodes, 1, min_qDegree, bucket_level_end_, current_q, bucket_q);
            
            // Populate keep array
            keep.initialize("Keep edges", unified, dataGraph.numEdges, dev_);
            keep.setAll(true, true);
            uint grid_size = (current_q.count.gdata()[0] - 1) / block_size + 1;

            execKernel((remove_edges_connected_to_node<T>), grid_size, block_size, dev_, true,
                        dataGraph, current_q.device_queue->gdata()[0], keep.gdata());

            // Compute new number of edges per node
            grid_size = (dataGraph.numEdges - 1) / block_size + 1;
            execKernel((warp_detect_deleted_edges<T>), grid_size, block_size, dev_, true,
                        dataGraph.rowPtr, dataGraph.numNodes, keep.gdata(), nodeDegree.gdata());

            // Perform scan to get new row pointer array
            new_ptr.initialize("New row pointer", unified, dataGraph.numNodes + 1, dev_);
            new_ptr.setAll(0, false);

            uint total = 0;
            if (dataGraph.numNodes < INT_MAX) 
                total = CUBScanExclusive<uint, uint>(nodeDegree.gdata(), new_ptr.gdata(), dataGraph.numNodes, dev_);
            else
                total = cl.ExclusiveSum(nodeDegree.gdata(), new_ptr.gdata(), dataGraph.numNodes);
            new_ptr.gdata()[dataGraph.numNodes] = total;
            
            // Select marked edges from ColInd
            new_adj.initialize("New column index", unified, total, dev_);
            if (dataGraph.numEdges < INT_MAX)
                CUBSelect(dataGraph.colInd, new_adj.gdata(), keep.gdata(), dataGraph.numEdges, dev_);
            else
                cl.Select(dataGraph.colInd, new_adj.gdata(), keep.gdata(), dataGraph.numEdges);
            
            // Update dataGraph
            swap_ele(dataGraph.rowPtr, new_ptr.gdata());
            swap_ele(dataGraph.colInd, new_adj.gdata());
            printf("Edges removed: %d\n", dataGraph.numEdges - total);
            dataGraph.numEdges = total;

            // Print Stats
            printf("Nodes filtered: %d\n", current_q.count.gdata()[0]);

            // Clean up
            keep.freeGPU();
            new_ptr.freeGPU();
            new_adj.freeGPU();
        } while (current_q.count.gdata()[0] > 0);

        // Recompute max degree
        uint grid_size = (dataGraph.numNodes - 1) / block_size + 1;
        execKernel((getNodeDegree_kernel<T, block_size>), grid_size, block_size, dev_, false, 
                    nodeDegree.gdata(), dataGraph, max_dDegree.gdata());

        printf("New max degree: %d\n", max_dDegree.gdata()[0]);
    }

    template<typename T>
    void SG_Match<T>::count_subgraphs(graph::COOCSRGraph_d<T>& dataGraph)
    {
        // Initialise Kernel Dims
        const auto block_size = 128;
        const T partitionSize = 8;
        const T numPartitions = block_size / partitionSize;
        const uint dv = 32;

        // CUDA Initialise, gather runtime Info.
        CUDAContext context;
        T num_SMs = context.num_SMs;
        T conc_blocks_per_SM = context.GetConCBlocks(block_size);

        // Initialise Arrays
        GPUArray<T> d_bitmap_states("bmp bitmap stats", AllocationTypeEnum::gpu, num_SMs * conc_blocks_per_SM, dev_);
        d_bitmap_states.setAll(0, true);

        // GPU Constant memory
        cudaMemcpyToSymbol(KCCOUNT, &(query_sequence->N), sizeof(KCCOUNT));
        cudaMemcpyToSymbol(PARTSIZE, &partitionSize, sizeof(PARTSIZE));
        cudaMemcpyToSymbol(NUMPART, &numPartitions, sizeof(NUMPART));
        cudaMemcpyToSymbol(MAXLEVEL, &DEPTH, sizeof(MAXLEVEL));
        cudaMemcpyToSymbol(CBPSM, &(conc_blocks_per_SM), sizeof(CBPSM));        
        cudaMemcpyToSymbol(QEDGE, &(query_edges->cdata()[0]), query_edges->N * sizeof(QEDGE[0]));
        cudaMemcpyToSymbol(QEDGE_PTR, &(query_edge_ptr->cdata()[0]), query_edge_ptr->N * sizeof(QEDGE_PTR[0]));
        cudaMemcpyToSymbol(SYMNODE, &(sym_nodes->cdata()[0]), sym_nodes->N * sizeof(SYMNODE[0]));
        cudaMemcpyToSymbol(SYMNODE_PTR, &(sym_nodes_ptr->cdata()[0]), sym_nodes_ptr->N * sizeof(SYMNODE_PTR[0]));

        // Initialise Queueing
        T todo = dataGraph.numNodes;
        T span = 8192;
        T level = 0;
        T bucket_level_end_ = level;

        // Ignore starting nodes with degree < max_qDegree.
        bucket_scan(nodeDegree, dataGraph.numNodes, level, max_qDegree, bucket_level_end_, current_q, bucket_q);
        todo -= current_q.count.gdata()[0];
        current_q.count.gdata()[0] = 0;
        level = max_qDegree;
        bucket_level_end_ = level;

        // Loop over different buckets
        while(todo > 0)
        {
            // Compute bucket
            bucket_scan(nodeDegree, dataGraph.numNodes, level, span, bucket_level_end_, current_q, bucket_q);

            if ( current_q.count.gdata()[0] > 0 )
            {
                todo -= current_q.count.gdata()[0];
        
                // Array Sizes
                uint maxDeg = level + span < max_dDegree.gdata()[0] ? level + span : max_dDegree.gdata()[0];
                uint64 numBlock = num_SMs * conc_blocks_per_SM;
                bool persistant = true;
                if ( current_q.count.gdata()[0] < num_SMs * conc_blocks_per_SM ) 
                {
                    numBlock = current_q.count.gdata()[0];
                    persistant = false;
                }

                uint num_divs = (maxDeg + dv - 1) / dv;
                uint64 level_size = numBlock * numPartitions * DEPTH * num_divs;
                uint64 encode_size = numBlock * maxDeg * num_divs;
                uint64 mask_size = numBlock * num_divs;
                //printf("Level Size = %llu, Encode Size = %llu\n", level_size, encode_size);
                GPUArray<T> current_level("Temp level Counter", unified, level_size, dev_);
                GPUArray<T> node_be("Temp level Counter", unified, encode_size, dev_);
                GPUArray<T> orient_mask("Orientation mask", unified, mask_size, dev_);
                current_level.setAll(0, true);
                node_be.setAll(0, true);
                orient_mask.setAll(0, true);

                // Constant memory
                cudaMemcpyToSymbol(NUMDIVS, &num_divs, sizeof(NUMDIVS));
                cudaMemcpyToSymbol(MAXDEG, &maxDeg, sizeof(MAXDEG));

                // Kernel Launch
                auto grid_block_size = dataGraph.numNodes;
                if (persistant) {
                    execKernel((sgm_kernel_central_node_base_binary_persistant<T, block_size, partitionSize>), grid_block_size, block_size, dev_, false,
                        counter.gdata(),
                        dataGraph,
                        current_q.device_queue->gdata()[0],
                        current_level.gdata(),
                        d_bitmap_states.gdata(), node_be.gdata(),
                        orient_mask.gdata());
                }
                else {
                    execKernel((sgm_kernel_central_node_base_binary<T, block_size, partitionSize>), grid_block_size, block_size, dev_, false,
                        counter.gdata(),
                        dataGraph,
                        current_q.device_queue->gdata()[0],
                        current_level.gdata(),
                        node_be.gdata(), orient_mask.gdata());
                }
                
                // Cleanup
                current_level.freeGPU();
                node_be.freeGPU();
                orient_mask.freeGPU();

                // Print bucket stats:
                std::cout << "Bucket levels: " << level << " to " << maxDeg
                            << ", nodes: " << current_q.count.gdata()[0]
                            << ", Counter: " << counter.gdata()[0] << std::endl;
            }
            level += span;
            if (current_q.count.gdata()[0] < num_SMs * conc_blocks_per_SM) span *= 4;
        }
        std::cout << "------------- Counter = " << counter.gdata()[0] << "\n";

        counter.freeGPU();
        d_bitmap_states.freeGPU();			
    }
}