#pragma once

#include "sgm_kernels.cuh"
#include "thrust/sort.h"

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
    template <typename T>
    class SG_Match
    {
    private:
        // GPU info
        int dev_;

        // Configuration
        MAINTASK task_;
        ProcessBy by_;

        // Processed query graphs
        GPUArray<T> *query_sequence;
        GPUArray<T> *query_degree;
        GPUArray<uint> *query_edges;
        GPUArray<uint> *query_edge_ptr;
        GPUArray<uint> *sym_nodes;
        GPUArray<uint> *sym_nodes_ptr;
        uint min_qDegree, max_qDegree;
        uint unmat_level;

        // Processed data graph info
        GPUArray<uint64> counter;
        GPUArray<T> nodeDegree, max_dDegree;
        GPUArray<T> edgeDegree, max_eDegree;
        GPUArray<T> oriented_nodeDegree, oriented_max_dDegree;
        GPUArray<T> oriented_edgeDegree, oriented_max_eDegree;

        // Queues
        graph::GraphQueue<T, bool> bucket_q, current_q;

        // Array used by bucket scan
        GPUArray<T> asc;

        // Bucket Scan function from K-Cliques
        void bucket_scan(
            GPUArray<T> nodeDegree, T node_num, T level, T span,
            T &bucket_level_end_,
            GraphQueue<T, bool> &current,
            GraphQueue<T, bool> &bucket)
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
                           bucket.queue.gdata(), bucket.count.gdata()[0], nodeDegree.gdata(),
                           current.mark.gdata(), current.queue.gdata(), &(current.count.gdata()[0]), level, span);
            }
            else
            {
                current.count.gdata()[0] = 0;
            }
            // Log(LogPriorityEnum::info, "Level: %d, curr: %d/%d", level, current.count.gdata()[0], bucket.count.gdata()[0]);
        }

    public:
        // Constructors
        SG_Match(MAINTASK task = GRAPH_MATCH, ProcessBy by = ByNode, int dev = 0) : task_(task),
                                                                                    by_(by),
                                                                                    dev_(dev)
        {
            query_sequence = new GPUArray<T>("Query Sequence", cpuonly);
            query_degree = new GPUArray<T>("Query degree", cpuonly);
            query_edges = new GPUArray<uint>("Query edges", cpuonly);
            query_edge_ptr = new GPUArray<uint>("Query edge ptr", cpuonly);
            sym_nodes = new GPUArray<uint>("Symmetrical nodes", cpuonly);
            sym_nodes_ptr = new GPUArray<uint>("Symmetrical node pointer", cpuonly);

            counter.initialize("Temp level Counter", unified, 1, dev_);
            max_dDegree.initialize("Temp Degree", unified, 1, dev_);
            max_eDegree.initialize("Temp Degree", unified, 1, dev_);
            oriented_max_dDegree.initialize("Oriented Degree", unified, 1, dev_);
            oriented_max_eDegree.initialize("Oriented Degree", unified, 1, dev_);

            counter.setSingle(0, 0, false);
            max_dDegree.setSingle(0, 0, false);
            max_eDegree.setSingle(0, 0, false);
            oriented_max_dDegree.setSingle(0, 0, false);
            oriented_max_eDegree.setSingle(0, 0, false);

            max_qDegree = 0;
            min_qDegree = INT_MAX;

            unmat_level = 0;
        }

        // Destructor
        ~SG_Match()
        {
            query_sequence->freeCPU();
            delete query_sequence;

            query_degree->freeCPU();
            delete query_degree;

            query_edges->freeCPU();
            delete query_edges;

            query_edge_ptr->freeCPU();
            delete query_edge_ptr;

            sym_nodes->freeCPU();
            delete sym_nodes;

            sym_nodes_ptr->freeCPU();
            delete sym_nodes_ptr;
        }

        void run(graph::COOCSRGraph_d<T> dataGraph, graph::COOCSRGraph<T> &patGraph)
        {
            CUDA_RUNTIME(cudaSetDevice(dev_));

            // CUDA_RUNTIME(cudaStreamCreate(&stream_));
            // CUDA_RUNTIME(cudaStreamSynchronize(stream_));

            double time;

            Timer p;

            preprocess_query(patGraph);
            Log(debug, "Parsing template succesful");
            initialize(dataGraph);
            Log(debug, "Initializing datagraph succesful");
            peel_data(dataGraph); // dataGraph transfers to unified memory here
            Log(debug, "datagraph peeling succesful");
// Degeneracy_orientation(dataGraph);
// Log(debug, "orientation succesfull\n");
#ifdef DEGENERACY
            Log(info, "Degeneracy Ordering");
            degeneracy_ordering(dataGraph);
#elif DEGREE
            Log(info, "Degree Ordering");
            degree_ordering(dataGraph);
#else
            Log(info, "lexicographic Ordering");
            lexicographic_ordering(dataGraph);
#endif
            Log(debug, "orientation succesfull");
            initialize1(dataGraph);
            Log(debug, "initialize 1 succesfull");
            // print_graph(dataGraph);
            time = p.elapsed();

            Log(info, "Preprocessing time: %f ms", time * 1000);
            Log(debug, "Undirected graph maxdegree: %lu\n", max_dDegree.gdata()[0]);
            Log(debug, "Directed graph maxdegree %lu\n", oriented_max_dDegree.gdata()[0]);
            Timer t;
            count_subgraphs(dataGraph);
            time = t.elapsed();
            Log(info, "count time %f s", time);
        }

    protected:
        // Function declarations. Definitions outside class.
        void preprocess_query(graph::COOCSRGraph<T> &query);
        void detect_symmetry();
        void check_unmat();
        void peel_data(graph::COOCSRGraph_d<T> &dataGraph);
        void initialize(graph::COOCSRGraph_d<T> &dataGraph);
        void initialize1(graph::COOCSRGraph_d<T> &oriented_dataGraph);
        void count_subgraphs(graph::COOCSRGraph_d<T> &dataGraph);
        void print_graph(graph::COOCSRGraph_d<T> &dataGraph);
        void degeneracy_orientation(graph::COOCSRGraph_d<T> &dataGraph);
        void degeneracy_ordering(graph::COOCSRGraph_d<T> &g);
        void degree_ordering(graph::COOCSRGraph_d<T> &g);
        void lexicographic_ordering(graph::COOCSRGraph_d<T> &g);
    };

    /***********************
     * FUNCTION DEFINITIONS *
     ***********************/

    template <typename T>
    void SG_Match<T>::preprocess_query(graph::COOCSRGraph<T> &query)
    {
        // Allocate and initialize arrays
        query_sequence->allocate_cpu(query.numNodes);
        query_degree->allocate_cpu(query.numNodes);
        query_edges->allocate_cpu(query.numEdges / 2);
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
        uint *degree = new uint[query.numNodes];
        for (int i = 0; i < query.numNodes; i++)
        {
            degree[i] = query.rowPtr->cdata()[i + 1] - query.rowPtr->cdata()[i];
            if (degree[i] > max_qDegree)
                max_qDegree = degree[i];
            if (degree[i] < min_qDegree)
                min_qDegree = degree[i];
        }

        // Generate Query node sequence based on "Maxmimum Likelihood Estimation" (MLE)
        // First look for node with highest node-mapping-degree d_M
        //      (i.e., degree with nodes already in query_sequence)
        // Nodes with same d_M are sorted with their degree
        //
        // These conditions can be combined into one as follows:
        //      d_M * num_nodes + node_degree

        // Initialise d_M and s_M with 0 for all nodes
        int *d_M = new int[query.numNodes];
        memset(d_M, 0, query.numNodes * sizeof(int));

        // For ith node in the query sequence
        for (int i = 0; i < query.numNodes; i++)
        {
            uint ml = 0;  // Maximum likelihood
            uint idx = 0; // Index of node with Max Likelihood

            // Traverse all nodes to find ml and idx
            for (int j = 0; j < query.numNodes; j++)
            {
                if (d_M[j] >= 0)
                { // d_M = -1 denotes node already in sequence
                    uint likelihood = d_M[j] * query.numNodes + degree[j];
                    if (likelihood > ml)
                    {
                        ml = likelihood;
                        idx = j;
                    }
                }
            }

            // Append node to sequence
            query_sequence->cdata()[i] = idx;
            query_degree->cdata()[i] = degree[idx];

            // Mark idx as done in d_M and s_M
            d_M[idx] = -1;

            // Update d_M of other nodes
            for (int j = query.rowPtr->cdata()[idx]; j < query.rowPtr->cdata()[idx + 1]; j++)
            {
                uint neighbor = query.colInd->cdata()[j];
                if (d_M[neighbor] != -1)
                    d_M[neighbor]++;
            }

            // Populate query edges
            query_edge_ptr->cdata()[i + 1] = query_edge_ptr->cdata()[i];

            // For all previous nodes in the sequence
            for (int j = 0; j < i; j++)
            {
                // For all neighbors of node_i
                for (int n = query.rowPtr->cdata()[idx]; n < query.rowPtr->cdata()[idx + 1]; n++)
                {
                    // If neighbor is same as node_j, it's an edge
                    if (query.colInd->cdata()[n] == query_sequence->cdata()[j])
                    {
                        query_edges->cdata()[query_edge_ptr->cdata()[i + 1]++] = j;
                    }
                }
            }
        }

        // Detect symmetrical nodes in query graph
        detect_symmetry();

        // Check for unmaterialized nodes
        if (task_ == GRAPH_COUNT)
            check_unmat();

        // Clean Up
        delete[] d_M;
        delete[] degree;

        // Print statements to check results.

        printf("Node Sequence:\n");
        for (int i = 0; i < query.numNodes; i++)
        {
            printf("i: %d;\tnode: %d (Degree: %d)\n", i, query_sequence->cdata()[i], query_degree->cdata()[i]);
        }

        printf("Query edges:\n");
        for (int i = 0; i < query.numNodes; i++)
        {
            printf("i: %d\t", i);
            for (int j = query_edge_ptr->cdata()[i]; j < query_edge_ptr->cdata()[i + 1]; j++)
            {
                printf("%d,", query_edges->cdata()[j]);
            }
            printf("\n");
        }

        printf("Number of levels (other than last) to unmaterialize: %d\n", unmat_level);

        printf("Symmetrical nodes:\n");
        for (int i = 0; i < query.numNodes; i++)
        {
            printf("i: %d\t", i);
            for (int j = sym_nodes_ptr->cdata()[i]; j < sym_nodes_ptr->cdata()[i + 1]; j++)
            {
                printf("%d,", sym_nodes->cdata()[j]);
            }
            printf("\n");
        }
    }

    template <typename T>
    void SG_Match<T>::detect_symmetry()
    {
        // Define required variables for NAUTY
        graph g[MAXN];
        int lab[MAXN], ptn[MAXN];
        int orbits[MAXN];
        static DEFAULTOPTIONS_GRAPH(opt);
        statsblk stats;
        int m = 1;
        int n = query_sequence->N;

        opt.defaultptn = FALSE;
        for (int i = 0; i < n; i++)
        {
            lab[i] = i;
            ptn[i] = 1;
        }
        ptn[n - 1] = 0;

        // Populate graph
        EMPTYGRAPH(g, m, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = query_edge_ptr->cdata()[i]; j < query_edge_ptr->cdata()[i + 1]; j++)
            {
                ADDONEEDGE(g, i, query_edges->cdata()[j], m);
                ADDONEEDGE(g, query_edges->cdata()[j], i, m);
            }
        }

        // Vector to hold symmetrical nodes
        vector<int> sym[MAXN];
        vector<T> history;
        while (true)
        {
            // Call NAUTY to get orbit
            densenauty(g, lab, ptn, orbits, &opt, &stats, m, n, NULL);

            // Find the biggest orbit
            int cnts[MAXN] = {0};
            for (int i = 0; i < n; i++)
                cnts[orbits[i]]++;
            int maxLen = 0;
            int maxIdx = -1;
            for (int i = 0; i < n; i++)
            {
                if (cnts[i] > maxLen)
                {
                    maxLen = cnts[i];
                    maxIdx = i;
                }
            }

            if (maxLen == 1)
                break;

            // Add symmetrical nodes from biggest orbit
            for (int i = 0; i < n; i++)
            {
                if (orbits[i] == maxIdx && i != maxIdx)
                    sym[i].push_back(maxIdx);
            }

            // Fix maxIdx node in a separate partition
            history.push_back(maxIdx);
            for (int i = 0; i < history.size(); i++)
            {
                lab[i] = history[i];
                ptn[i] = 0;
            }

            int ctr = history.size();
            for (int i = 0; i < n; i++)
            {
                bool found = false;
                for (int j = 0; j < history.size(); j++)
                {
                    if (history[j] == i)
                    {
                        found = true;
                        break;
                    }
                }

                if (!found)
                {
                    lab[ctr] = i;
                    ptn[ctr] = 1;
                    ctr++;
                }
            }

            /*
            printf("Orbits:\n");
            for (int i = 0; i < query_sequence->N; i++) {
                printf("i: %d;\tnode: %d (Degree: %d)\n", i, query_sequence->cdata()[i], orbits[i]);
            }
            */
        }

        // Populate symmetrical nodes from sym
        for (int i = 0; i < n; i++)
        {
            sym_nodes_ptr->cdata()[i + 1] = sym_nodes_ptr->cdata()[i];
            for (int j = 0; j < sym[i].size(); j++)
                sym_nodes->cdata()[sym_nodes_ptr->cdata()[i + 1]++] = sym[i][j];
        }
    }

    template <typename T>
    void SG_Match<T>::check_unmat()
    {
        // For now we only check for last two levels to be unmat
        // Last level is always unmat
        unmat_level = 0;
        // 2nd to last level is unmat if not used by last level
        bool found = false;
        for (int i = query_edge_ptr->cdata()[query_sequence->N - 1]; i < query_edge_ptr->cdata()[query_sequence->N]; i++)
        {
            if (query_edges->cdata()[i] == query_sequence->N - 2)
            {
                found = true;
                break;
            }
        }

        if (!found)
            unmat_level++;
    }

    template <typename T>
    void SG_Match<T>::initialize(graph::COOCSRGraph_d<T> &dataGraph)
    {
        const auto block_size = 256;
        size_t qSize = (by_ == ByEdge) ? dataGraph.numEdges : dataGraph.numNodes;
        bucket_q.Create(unified, qSize, dev_);
        current_q.Create(unified, qSize, dev_);
        asc.initialize("Identity array asc", unified, qSize, dev_);
        execKernel(init_asc, (qSize + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, dev_, false,
                   asc.gdata(), qSize);

        CUDA_RUNTIME(cudaGetLastError());
        cudaDeviceSynchronize();
        // Compute node Degrees and max degree
        nodeDegree.initialize("Edge Support", unified, dataGraph.numNodes, dev_);
        uint dimGridNodes = (dataGraph.numNodes + block_size - 1) / block_size;
        execKernel((getNodeDegree_kernel<T, block_size>), dimGridNodes, block_size, dev_, false,
                   nodeDegree.gdata(), dataGraph, max_dDegree.gdata());

        if (by_ == ByEdge)
        {
            edgeDegree.initialize("Edge Support", unified, dataGraph.numEdges, dev_);
            uint dimGridEdges = (dataGraph.numEdges + block_size - 1) / block_size;
            execKernel((get_max_degree<T, block_size>), dimGridEdges, block_size, dev_, false,
                       dataGraph, edgeDegree.gdata(), max_eDegree.gdata());
        }
    }

    template <typename T>
    void SG_Match<T>::initialize1(graph::COOCSRGraph_d<T> &oriented_dataGraph)
    {
        const auto block_size = 256;
        size_t qSize = (by_ == ByEdge) ? oriented_dataGraph.numEdges : oriented_dataGraph.numNodes;

        asc.initialize("Identity array asc", unified, qSize, dev_);
        execKernel(init_asc, (qSize + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, dev_, false,
                   asc.gdata(), qSize);

        CUDA_RUNTIME(cudaGetLastError());
        cudaDeviceSynchronize();

        // Compute Max Degrees and max degree
        oriented_nodeDegree.initialize("Edge Support", unified, oriented_dataGraph.numNodes, dev_);
        uint dimGridNodes = (oriented_dataGraph.numNodes + block_size - 1) / block_size;
        execKernel((getNodeDegree_split_kernel<T, block_size>), dimGridNodes, block_size, dev_, false,
                   oriented_nodeDegree.gdata(), oriented_dataGraph, oriented_max_dDegree.gdata());

        if (by_ == ByEdge)
        {
            oriented_edgeDegree.initialize("Edge Support", unified, oriented_dataGraph.numEdges, dev_);
            uint dimGridEdges = (oriented_dataGraph.numEdges + block_size - 1) / block_size;
            execKernel((get_max_degree<T, block_size>), dimGridEdges, block_size, dev_, false,
                       oriented_dataGraph, oriented_edgeDegree.gdata(), oriented_max_eDegree.gdata());
        }
    }

    template <typename T>
    void SG_Match<T>::peel_data(graph::COOCSRGraph_d<T> &dataGraph)
    {
        // CubLarge
        graph::CubLarge<uint> cl(dev_);
        const uint block_size = 128;

        // Initialise Arrays
        GPUArray<bool> keep;
        GPUArray<T> new_ptr;
        GPUArray<T> new_adj;
        GPUArray<T> new_row;

        // Keep peeling till you can.
        do
        {
            // Find nodes with degree < min_qDegree
            T bucket_level_end_ = 1;
            bucket_scan(nodeDegree, dataGraph.numNodes, 1, min_qDegree - 1, bucket_level_end_, current_q, bucket_q);

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
            // printf("Total value %u\n", total);
            new_adj.initialize("New column index", unified, total, dev_);
            new_row.initialize("New row index", unified, total, dev_);
            if (dataGraph.numEdges < INT_MAX)
            {
                CUBSelect(dataGraph.colInd, new_adj.gdata(), keep.gdata(), dataGraph.numEdges, dev_);
                CUBSelect(dataGraph.rowInd, new_row.gdata(), keep.gdata(), dataGraph.numEdges, dev_);
            }
            else
            {
                cl.Select(dataGraph.colInd, new_adj.gdata(), keep.gdata(), dataGraph.numEdges);
                cl.Select(dataGraph.rowInd, new_row.gdata(), keep.gdata(), dataGraph.numEdges);
            }

            // Update dataGraph
            swap_ele(dataGraph.rowPtr, new_ptr.gdata());
            swap_ele(dataGraph.colInd, new_adj.gdata());
            swap_ele(dataGraph.rowInd, new_row.gdata());
            // printf("Edges removed: %d\n", dataGraph.numEdges - total);
            dataGraph.numEdges = total;

            // Print Stats
            // printf("Nodes filtered: %d\n", current_q.count.gdata()[0]);

            // Clean up
            keep.freeGPU();
            new_ptr.freeGPU();
            new_adj.freeGPU();
            new_row.freeGPU();
        } while (current_q.count.gdata()[0] > 0.05 * dataGraph.numNodes);

        // Recompute max degree
        uint grid_size = (dataGraph.numNodes - 1) / block_size + 1;
        execKernel((getNodeDegree_kernel<T, block_size>), grid_size, block_size, dev_, false,
                   nodeDegree.gdata(), dataGraph, max_dDegree.gdata());

        if (by_ == ByEdge)
        {
            uint edge_grid = (dataGraph.numEdges - 1) / block_size + 1;
            execKernel((get_max_degree<T, block_size>), edge_grid, block_size, dev_, false,
                       dataGraph, edgeDegree.gdata(), max_eDegree.gdata());
        }
    }

    template <typename T>
    void SG_Match<T>::count_subgraphs(graph::COOCSRGraph_d<T> &dataGraph)
    {
        GPUArray<uint64> intersection_count;
        intersection_count.initialize("Temp level Counter", unified, 1, dev_);
        intersection_count.setSingle(0, 0, false);

        // Initialise Kernel Dims
        const auto block_size_LD = 1024; // Block size for low degree nodes
        const T partitionSize_LD = 32;
        const T numPartitions_LD = block_size_LD / partitionSize_LD;
        const auto block_size_HD = 1024; // Block size for high degree nodes
        const T partitionSize_HD = 32;
        const T numPartitions_HD = block_size_HD / partitionSize_HD;
        const T bound_LD = 2048;
        const T bound_HD = 32768 + 16384;
        const uint dv = 32;
        Log(debug, "Partition size: %u", partitionSize_LD);
        // CUDA Initialise, gather runtime Info.
        CUDAContext context;
        T num_SMs = context.num_SMs;
        T conc_blocks_per_SM_LD = context.GetConCBlocks(block_size_LD);
        T conc_blocks_per_SM_HD = context.GetConCBlocks(block_size_HD);

        // Initialise Arrays
        GPUArray<T> d_bitmap_states("bmp bitmap stats", AllocationTypeEnum::gpu, num_SMs * conc_blocks_per_SM_LD, dev_);
        cudaMemset(d_bitmap_states.gdata(), 0, num_SMs * conc_blocks_per_SM_LD * sizeof(T));
        GPUArray<T> d_count_per_node("per node count", AllocationTypeEnum::gpu, dataGraph.numNodes, dev_);
        cudaMemset(d_count_per_node.gdata(), 0, dataGraph.numNodes * sizeof(T));

        // GPU Constant memory
        cudaMemcpyToSymbol(KCCOUNT, &(query_sequence->N), sizeof(KCCOUNT));
        Log(debug, "query size: %u", query_sequence->N);
        cudaMemcpyToSymbol(LUNMAT, &(unmat_level), sizeof(LUNMAT));
        cudaMemcpyToSymbol(MAXLEVEL, &(max_qDegree), sizeof(MAXLEVEL));
        cudaMemcpyToSymbol(MINLEVEL, &(min_qDegree), sizeof(MINLEVEL));
        cudaMemcpyToSymbol(QEDGE, &(query_edges->cdata()[0]), query_edges->N * sizeof(QEDGE[0]));
        cudaMemcpyToSymbol(QEDGE_PTR, &(query_edge_ptr->cdata()[0]), query_edge_ptr->N * sizeof(QEDGE_PTR[0]));
        cudaMemcpyToSymbol(SYMNODE, &(sym_nodes->cdata()[0]), sym_nodes->N * sizeof(SYMNODE[0]));
        cudaMemcpyToSymbol(SYMNODE_PTR, &(sym_nodes_ptr->cdata()[0]), sym_nodes_ptr->N * sizeof(SYMNODE_PTR[0]));
        cudaMemcpyToSymbol(QDEG, &(query_degree->cdata()[0]), query_degree->N * sizeof(QDEG[0]));

        // Initialise Queueing
        T todo = (by_ == ByEdge) ? dataGraph.numEdges : dataGraph.numNodes;
        T span = bound_HD;
        T level = 0;
        T bucket_level_end_ = level;

        // Ignore starting nodes with degree < max_qDegree.
        if (by_ == ByEdge)
            bucket_scan(edgeDegree, dataGraph.numEdges, level, max_qDegree, bucket_level_end_, current_q, bucket_q);
        else
            bucket_scan(nodeDegree, dataGraph.numNodes, level, max_qDegree, bucket_level_end_, current_q, bucket_q);
        todo -= current_q.count.gdata()[0];
        current_q.count.gdata()[0] = 0;
        level = max_qDegree;
        bucket_level_end_ = level;
        span -= level;

        // Loop over different buckets
        while (todo > 0)
        {
            // Compute bucket
            if (by_ == ByEdge)
                bucket_scan(edgeDegree, dataGraph.numEdges, level, span, bucket_level_end_, current_q, bucket_q);
            else
                bucket_scan(nodeDegree, dataGraph.numNodes, level, span, bucket_level_end_, current_q, bucket_q);

            if (current_q.count.gdata()[0] > 0)
            {
                todo -= current_q.count.gdata()[0];

                // Array Sizes
                // maxDeg from oriented dataGraph
                uint oriented_maxDeg = (by_ == ByEdge) ? oriented_max_eDegree.gdata()[0] : oriented_max_dDegree.gdata()[0];
                uint maxDeg = (by_ == ByEdge) ? max_eDegree.gdata()[0] : max_dDegree.gdata()[0];
                Log(debug, "unoriented max degree %lu", maxDeg);
                Log(debug, "oriented max degree %lu", oriented_maxDeg);

                maxDeg = level + span < maxDeg ? level + span : maxDeg;
                uint64 numBlock = maxDeg >= bound_LD ? num_SMs * conc_blocks_per_SM_HD : num_SMs * conc_blocks_per_SM_LD;
                bool persistant = true;
                if (current_q.count.gdata()[0] < numBlock)
                {
                    numBlock = current_q.count.gdata()[0];
                    persistant = false;
                }

                uint num_divs = (maxDeg + dv - 1) / dv;
                uint64 level_size = numBlock * max_qDegree * num_divs;
                level_size *= (maxDeg >= bound_LD ? numPartitions_HD : numPartitions_LD);
                uint64 encode_size = numBlock * maxDeg * num_divs;

                Log(debug, "Level Size = %llu, Encode Size = %llu", level_size, encode_size);
                GPUArray<T> node_be("Binary Encoded data", AllocationTypeEnum::gpu, encode_size, dev_);
                GPUArray<T> current_level("Temp level Counter", AllocationTypeEnum::gpu, level_size, dev_);

                cudaMemset(node_be.gdata(), 0, encode_size * sizeof(T));
                cudaMemset(current_level.gdata(), 0, level_size * sizeof(T));

                // Constant memory
                if (maxDeg >= bound_LD)
                {
                    cudaMemcpyToSymbol(NUMPART, &numPartitions_HD, sizeof(NUMPART));
                    cudaMemcpyToSymbol(PARTSIZE, &partitionSize_HD, sizeof(PARTSIZE));
                    cudaMemcpyToSymbol(CBPSM, &(conc_blocks_per_SM_HD), sizeof(CBPSM));
                }
                else
                {
                    cudaMemcpyToSymbol(NUMPART, &numPartitions_LD, sizeof(NUMPART));
                    cudaMemcpyToSymbol(PARTSIZE, &partitionSize_LD, sizeof(PARTSIZE));
                    cudaMemcpyToSymbol(CBPSM, &(conc_blocks_per_SM_LD), sizeof(CBPSM));
                }
                cudaMemcpyToSymbol(NUMDIVS, &num_divs, sizeof(NUMDIVS));
                cudaMemcpyToSymbol(MAXDEG, &maxDeg, sizeof(MAXDEG));
                cudaMemcpyToSymbol(ORIENTED_MAXDEG, &oriented_maxDeg, sizeof(ORIENTED_MAXDEG));

                // Kernel Launch
                Timer a;

                auto grid_block_size = current_q.count.gdata()[0];
                if (maxDeg >= bound_LD)
                {
                    if (persistant)
                    {
                        execKernel((sgm_kernel_central_node_base_binary_persistant<T, block_size_HD, partitionSize_HD>),
                                   grid_block_size, block_size_HD, dev_, false,
                                   counter.gdata(), d_count_per_node.gdata(), intersection_count.gdata(),
                                   dataGraph, current_q.device_queue->gdata()[0],
                                   current_level.gdata(), d_bitmap_states.gdata(),
                                   node_be.gdata(),
                                   by_ == ByNode);
                    }
                    else
                    {
                        execKernel((sgm_kernel_central_node_base_binary<T, block_size_HD, partitionSize_HD>),
                                   grid_block_size, block_size_HD, dev_, false,
                                   counter.gdata(), d_count_per_node.gdata(), intersection_count.gdata(),
                                   dataGraph, current_q.device_queue->gdata()[0],
                                   current_level.gdata(),
                                   node_be.gdata(),
                                   by_ == ByNode);
                    }
                }
                else
                {
                    if (persistant)
                    {
                        execKernel((sgm_kernel_central_node_base_binary_persistant<T, block_size_LD, partitionSize_LD>),
                                   grid_block_size, block_size_LD, dev_, false,
                                   counter.gdata(), d_count_per_node.gdata(), intersection_count.gdata(),
                                   dataGraph, current_q.device_queue->gdata()[0],
                                   current_level.gdata(), d_bitmap_states.gdata(),
                                   node_be.gdata(),
                                   by_ == ByNode);
                    }
                    else
                    {
                        execKernel((sgm_kernel_central_node_base_binary<T, block_size_LD, partitionSize_LD>),
                                   grid_block_size, block_size_LD, dev_, false,
                                   counter.gdata(), d_count_per_node.gdata(), intersection_count.gdata(),
                                   dataGraph, current_q.device_queue->gdata()[0],
                                   current_level.gdata(),
                                   node_be.gdata(),
                                   by_ == ByNode);
                    }
                }
                double time = a.elapsed();
                Log(info, "kernel time %f s", time);
                // Cleanup
                current_level.freeGPU();
                node_be.freeGPU();
                d_count_per_node.freeGPU();

                // Print bucket stats:
                std::cout << "Bucket levels: " << level << " to " << maxDeg
                          << ", nodes/edges: " << current_q.count.gdata()[0]
                          << ", Counter: " << counter.gdata()[0] << std::endl;
            }
            level += span;
        }

        std::cout << "------------- Counter = " << counter.gdata()[0] << "\n";
        std::cout << "------------- Intersection count = " << intersection_count.gdata()[0] << "\n";
        counter.freeGPU();
        intersection_count.freeGPU();
        d_bitmap_states.freeGPU();
    }

    template <typename T>
    void SG_Match<T>::print_graph(graph::COOCSRGraph_d<T> &dataGraph)
    {
#define ORIENTED_PRINT

        Log(info, "Printing graph as read");
        Log(info, "num Nodes %lu", dataGraph.numNodes);
        graph::COOCSRGraph<uint> temp_g;
        uint m = dataGraph.numEdges, n = dataGraph.numNodes;
        temp_g.capacity = m;
        temp_g.numEdges = m;
        temp_g.numNodes = n;

        uint *rp, *ri, *ci, *oci, *sp;
        cudaMallocHost((void **)&rp, (n + 1) * sizeof(uint));
        cudaMallocHost((void **)&ri, (m) * sizeof(uint));
        cudaMallocHost((void **)&ci, (m) * sizeof(uint));
        cudaMallocHost((void **)&oci, (m) * sizeof(uint));
        cudaMallocHost((void **)&sp, (n + 1) * sizeof(uint));

        temp_g.rowPtr = new graph::GPUArray<uint>("Temp row ptr", AllocationTypeEnum::cpuonly);
        temp_g.rowInd = new graph::GPUArray<uint>("Temp row indices", AllocationTypeEnum::cpuonly);
        temp_g.colInd = new graph::GPUArray<uint>("Temp col indices", AllocationTypeEnum::cpuonly);

        CUDA_RUNTIME(cudaMemcpy(rp, dataGraph.rowPtr, (n + 1) * sizeof(uint), cudaMemcpyKind::cudaMemcpyDeviceToHost));
        CUDA_RUNTIME(cudaMemcpy(ri, dataGraph.rowInd, (m) * sizeof(uint), cudaMemcpyKind::cudaMemcpyDeviceToHost));
        CUDA_RUNTIME(cudaMemcpy(ci, dataGraph.colInd, (m) * sizeof(uint), cudaMemcpyKind::cudaMemcpyDeviceToHost));
        CUDA_RUNTIME(cudaMemcpy(oci, dataGraph.oriented_colInd, (m) * sizeof(uint), cudaMemcpyKind::cudaMemcpyDeviceToHost));
        CUDA_RUNTIME(cudaMemcpy(sp, dataGraph.splitPtr, (n) * sizeof(uint), cudaMemcpyKind::cudaMemcpyDeviceToHost));

        temp_g.rowPtr->cdata() = rp;
        temp_g.rowInd->cdata() = ri;
        temp_g.colInd->cdata() = ci;

#ifndef ORIENTED_PRINT
        for (T src = temp_g.rowInd->cdata()[0]; src < temp_g.numNodes; src++)
        {
            T srcStart = temp_g.rowPtr->cdata()[src];
            T srcEnd = temp_g.rowPtr->cdata()[src + 1];

            printf("%u\t: ", src);
            for (T j = srcStart; j < srcEnd; j++)
            {
                printf("%u, ", temp_g.colInd->cdata()[j]);
            }
            printf("\t split after: %u\n", sp[src] - srcStart);
            printf("\n");
        }
#else
        for (T src = temp_g.rowInd->cdata()[0]; src < temp_g.numNodes; src++)
        {
            T srcStart = temp_g.rowPtr->cdata()[src];
            T srcEnd = temp_g.rowPtr->cdata()[src + 1];
            printf("%u\t: ", src);
            for (T j = srcStart; j < srcEnd; j++)
            {
                printf("%u, ", oci[j]);
            }
            printf("\t split after: %u", sp[src] - srcStart);
            printf("\n");
        }
#endif
        printf("\n");
        printf("***end of graph***\n");
        cudaFreeHost(rp);
        cudaFreeHost(ri);
        cudaFreeHost(ci);
        cudaFreeHost(sp);
        // graph::free_csrcoo_host(temp_g);
    }

    template <typename T>
    void SG_Match<T>::degeneracy_orientation(graph::COOCSRGraph_d<T> &g)
    {
        T m = g.numEdges;
        T n = g.numNodes;

        graph::SingleGPU_Kcore<uint, PeelType> mohacore(dev_);
        Timer degeneracy_time;
        mohacore.findKcoreIncremental_async(3, 1000, g, 0, 0);
        Log(info, "Degeneracy ordering time: %f s", degeneracy_time.elapsed());

        Timer csr_recreation_time;
        graph::GPUArray<uint> split_col("Split Column", gpu, m, dev_);
        graph::GPUArray<uint> tmp_row("Temp Row", gpu, m / 2, dev_);
        graph::GPUArray<uint> tmp_col("Temp Column", gpu, m / 2, dev_);
        graph::GPUArray<uint> split_ptr("Split Pointer", gpu, n + 1, dev_);
        graph::GPUArray<uint> asc("ASC temp", AllocationTypeEnum::unified, m, dev_);
        graph::GPUArray<bool> keep("Keep temp", AllocationTypeEnum::unified, m, dev_);
        execKernel((init<uint, PeelType>), ((m - 1) / 51200) + 1, 512, dev_, false, g, asc.gdata(), keep.gdata(), mohacore.nodeDegree.gdata(), mohacore.nodePriority.gdata());

        graph::CubLarge<uint> s(dev_);
        if (m < INT_MAX)
        {
            CUBSelect(g.rowInd, tmp_row.gdata(), keep.gdata(), m, dev_);
            CUBSelect(g.colInd, tmp_col.gdata(), keep.gdata(), m, dev_);
        }
        else
        {
            s.Select2(g.rowInd, g.colInd, tmp_row.gdata(), tmp_col.gdata(), keep.gdata(), m);
        }
        execKernel((warp_detect_deleted_edges<uint>), (32 * n + 128 - 1) / 128, 128, dev_, false, g.rowPtr, n, keep.gdata(), split_ptr.gdata());
        uint total = CUBScanExclusive<uint, uint>(split_ptr.gdata(), split_ptr.gdata(), n, dev_, 0, AllocationTypeEnum::gpu);
        split_ptr.setSingle(n, total, true);
        execKernel((split_child<uint>), ((m - 1) / 51200) + 1, 512, dev_, false, g, tmp_row.gdata(), tmp_col.gdata(), split_col.gdata(), split_ptr.gdata());
        execKernel((split_inverse<uint>), ((m - 1) / 51200) + 1, 512, dev_, false, keep.gdata(), m);
        if (m < INT_MAX)
        {
            CUBSelect(g.rowInd, tmp_row.gdata(), keep.gdata(), m, dev_);
            CUBSelect(g.colInd, tmp_col.gdata(), keep.gdata(), m, dev_);
        }
        else
        {
            s.Select2(g.rowInd, g.colInd, tmp_row.gdata(), tmp_col.gdata(), keep.gdata(), m);
        }
        execKernel((warp_detect_deleted_edges<uint>), (32 * n + 128 - 1) / 128, 128, dev_, false, g.rowPtr, n, keep.gdata(), split_ptr.gdata());
        total = CUBScanExclusive<uint, uint>(split_ptr.gdata(), split_ptr.gdata(), n, dev_, 0, AllocationTypeEnum::gpu);
        split_ptr.setSingle(n, total, true);
        execKernel((split_parent<uint>), ((m - 1) / 51200) + 1, 512, dev_, false, g, tmp_row.gdata(), tmp_col.gdata(), split_col.gdata(), split_ptr.gdata());
        execKernel((warp_detect_deleted_edges<uint>), (32 * n + 128 - 1) / 128, 128, dev_, false, g.rowPtr, n, keep.gdata(), split_ptr.gdata());
        execKernel((split_acc<uint>), ((n - 1) / 51200) + 1, 512, dev_, false, g, split_ptr.gdata());

        cudaDeviceSynchronize();
        asc.freeGPU();
        keep.freeGPU();
        tmp_row.freeGPU();
        tmp_col.freeGPU();
        CUDA_RUNTIME(cudaFree(g.colInd));
        g.colInd = split_col.gdata();
        g.splitPtr = split_ptr.gdata();

        Log(info, "CSR Recreation time: %f s", csr_recreation_time.elapsed());
    }

    template <typename T>
    void SG_Match<T>::degeneracy_ordering(graph::COOCSRGraph_d<T> &g)
    {

        T m = g.numEdges;
        T n = g.numNodes;
        auto blockSize = 256;
        auto gridSize = (m + blockSize - 1) / blockSize;
        graph::SingleGPU_Kcore<uint, PeelType> mohacore(dev_);
        Timer degeneracy_time;
        mohacore.findKcoreIncremental_async(3, 1000, g, 0, 0);
        Log(info, "Degeneracy ordering time: %f s", degeneracy_time.elapsed());

        CUDA_RUNTIME(cudaMallocManaged((void **)&g.splitPtr, n * sizeof(T)));
        CUDA_RUNTIME(cudaMemcpy(g.splitPtr, g.rowPtr, n * sizeof(T), cudaMemcpyDeviceToDevice));

        execKernel((set_priority<T>), gridSize, blockSize, dev_, false, g, mohacore.nodePriority.gdata()); // get splitptr data

        graph::GPUArray<triplet<T>> triplet_array("Array paired with partitions to ", unified, m, dev_);
        execKernel((map_and_gen_triplet_array<T>), gridSize, blockSize, dev_, false, triplet_array.gdata(), g, mohacore.nodePriority.gdata());

        thrust::stable_sort(thrust::device, triplet_array.gdata(), triplet_array.gdata() + m, comparePriority<T>());
        thrust::stable_sort(thrust::device, triplet_array.gdata(), triplet_array.gdata() + m, comparePartition<T>());
        CUDA_RUNTIME(cudaDeviceSynchronize()); // sorts entries with

        CUDA_RUNTIME(cudaMalloc(&g.oriented_colInd, m * sizeof(T)));
        execKernel((map_back<T>), gridSize, blockSize, dev_, false, triplet_array.gdata(), g);
        triplet_array.freeGPU();
    }

    template <typename T>
    void SG_Match<T>::degree_ordering(graph::COOCSRGraph_d<T> &g)
    {
        T m = g.numEdges;
        T n = g.numNodes;
        const auto blockSize = 256;
        auto node_gridsize = (n + blockSize - 1) / blockSize;
        auto edge_gridSize = (m + blockSize - 1) / blockSize;

        execKernel((getNodeDegree_kernel<T, blockSize>), node_gridsize, blockSize, dev_, false,
                   nodeDegree.gdata(), g, max_dDegree.gdata());
        CUDA_RUNTIME(cudaMallocManaged((void **)&g.splitPtr, n * sizeof(T)));
        CUDA_RUNTIME(cudaMemcpy(g.splitPtr, g.rowPtr, n * sizeof(T), cudaMemcpyDeviceToDevice));

        execKernel((set_priority<T>), edge_gridSize, blockSize, dev_, false, g, nodeDegree.gdata()); // get split ptr data

        graph::GPUArray<triplet<T>> triplet_array("Array paired with partitions to ", unified, m, dev_);
        execKernel((map_and_gen_triplet_array<T>), edge_gridSize, blockSize, dev_, false, triplet_array.gdata(), g, nodeDegree.gdata());

        thrust::stable_sort(thrust::device, triplet_array.gdata(), triplet_array.gdata() + m, comparePriority<T>());
        thrust::stable_sort(thrust::device, triplet_array.gdata(), triplet_array.gdata() + m, comparePartition<T>());
        CUDA_RUNTIME(cudaDeviceSynchronize());

        CUDA_RUNTIME(cudaMalloc(&g.oriented_colInd, m * sizeof(T)));
        execKernel((map_back<T>), edge_gridSize, blockSize, dev_, false, triplet_array.gdata(), g);
        triplet_array.freeGPU();
    }

    template <typename T>
    void SG_Match<T>::lexicographic_ordering(graph::COOCSRGraph_d<T> &g)
    {
        T m = g.numEdges;
        T n = g.numNodes;
        const auto blockSize = 256;
        auto node_gridsize = (n + blockSize - 1) / blockSize;
        auto edge_gridSize = (m + blockSize - 1) / blockSize;

        CUDA_RUNTIME(cudaMallocManaged((void **)&g.splitPtr, n * sizeof(T)));
        CUDA_RUNTIME(cudaMemcpy(g.splitPtr, g.rowPtr, n * sizeof(T), cudaMemcpyDeviceToDevice));

        execKernel((set_priority_l<T>), edge_gridSize, blockSize, dev_, false, g);

        CUDA_RUNTIME(cudaMalloc(&g.oriented_colInd, m * sizeof(T)));
        CUDA_RUNTIME(cudaMemcpy(g.oriented_colInd, g.colInd, m * sizeof(T), cudaMemcpyDeviceToDevice));
    }
}
