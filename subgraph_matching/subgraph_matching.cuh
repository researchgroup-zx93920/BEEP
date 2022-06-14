
#pragma once
#include "config.cuh"

#ifdef TIMER
#include "sgm_kernelsLA_timer.cuh"
#include "sgm_kernelsLD_timer.cuh"
#else
#include "sgm_kernelsLD.cuh"
#include "sgm_kernelsLA.cuh"
#endif
#include "thrust/sort.h"
// #include <stdint.h>

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
        GPUArray<uint> *reuse_level;
        GPUArray<bool> *q_reusable;
        GPUArray<uint> *reuse_ptr;

        uint min_qDegree, max_qDegree;
        uint unmat_level, first_sym_level;

        // Processed data graph info
        GPUArray<uint64> counter;
        GPUArray<T> nodeDegree, max_dDegree;
        GPUArray<T> edgeDegree, max_eDegree;
        GPUArray<T> Degree_Scan;
        GPUArray<T> oriented_nodeDegree, oriented_max_dDegree;
        GPUArray<T> oriented_edgeDegree, oriented_max_eDegree;

        // Queues
        graph::GraphQueue<T, bool> bucket_nq, current_nq;
        graph::GraphQueue<T, bool> bucket_eq, current_eq;

        // Array used by bucket scan
        GPUArray<T> asc;

        // Bucket Scan function from K-Cliques
        void bucket_scan(
            GPUArray<T> &Degree, T node_num, T level, T span,
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
                           Degree.gdata(), node_num, bucket.mark.gdata(), level, bucket_level_end_ + span);

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
                           bucket.queue.gdata(), bucket.count.gdata()[0], Degree.gdata(),
                           current.mark.gdata(), current.queue.gdata(), &(current.count.gdata()[0]),
                           level, span);
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
            reuse_level = new GPUArray<uint>("Reuse info", cpuonly);
            q_reusable = new GPUArray<bool>("Is reusable", cpuonly);
            reuse_ptr = new GPUArray<uint>("Reuse ptr", cpuonly);

            counter.initialize("Temp level Counter", unified, 1, dev_);
            max_dDegree.initialize("Max Node Degree", unified, 1, dev_);
            max_eDegree.initialize("Max Edge Degree", unified, 1, dev_);
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

            reuse_level->freeCPU();
            delete reuse_level;

            q_reusable->freeCPU();
            delete q_reusable;

            reuse_ptr->freeCPU();
            delete reuse_ptr;
        }

        void run(graph::COOCSRGraph_d<T> &dataGraph, graph::COOCSRGraph<T> &patGraph)
        {
            CUDA_RUNTIME(cudaSetDevice(dev_));

            // CUDA_RUNTIME(cudaStreamCreate(&stream_));
            // CUDA_RUNTIME(cudaStreamSynchronize(stream_));

            double time;

            Timer p;
            preprocess_query(patGraph);
            Log(debug, "Parsing template succesful");

            detect_reuse(patGraph);
            Log(debug, "detecting reuse succesful");

            initialize(dataGraph);
            Log(debug, "Initializing datagraph succesful");

            peel_data(dataGraph);
            Log(debug, "datagraph peeling succesful");

            Log(info, "Degree Ordering");
            degree_ordering(dataGraph);
            Log(debug, "ordering succesfull");

            // initialize1(dataGraph);
            // Log(debug, "initialize 1 succesfull");

            time = p.elapsed();
            Log(info, "Preprocessing TIME: %f ms", time * 1000);
            Log(debug, "Undirected graph maxdegree: %lu", max_dDegree.gdata()[0]);
            Log(debug, "Directed graph maxdegree %lu", oriented_max_dDegree.gdata()[0]);

            Timer t;

// print_graph(dataGraph);
#ifdef TIMER
            count_subgraphs_timer(dataGraph);
#else
            count_subgraphs(dataGraph);
#endif
            time = t.elapsed();
            Log(info, "final count time %f s", time);
        }

    protected:
        // Function declarations. Definitions outside class.
        void preprocess_query(graph::COOCSRGraph<T> &patGraph);
        void detect_symmetry();
        void detect_reuse(graph::COOCSRGraph<T> &query);
        void check_unmat();

        void peel_data(graph::COOCSRGraph_d<T> &dataGraph);
        void initialize(graph::COOCSRGraph_d<T> &dataGraph);
        void initialize1(graph::COOCSRGraph_d<T> &oriented_dataGraph);
#ifdef TIMER
        void count_subgraphs_timer(graph::COOCSRGraph_d<T> &dataGraph);
#else
        void count_subgraphs(graph::COOCSRGraph_d<T> &dataGraph);
#endif
        void degree_ordering(graph::COOCSRGraph_d<T> &g);
        void print_graph(graph::COOCSRGraph_d<T> &dataGraph);
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
    void SG_Match<T>::detect_reuse(graph::COOCSRGraph<T> &query)
    {
        T numNodes = query.numNodes;
        reuse_level->allocate_cpu(numNodes);
        q_reusable->allocate_cpu(numNodes);
        reuse_ptr->allocate_cpu(numNodes);

        reuse_level->setAll(0, false);
        q_reusable->setAll(false, false);
        reuse_ptr->setAll(0, false);

        const T wrows = numNodes - 2;
        const T wsize = wrows * wrows;
        // GPUArray<T> w("weights", AllocationTypeEnum::cpuonly, wsize, dev_);
        // w.setAll(0, false);
        uint *w = new uint[wsize];
        std::fill(w, w + wsize, 0);

        Log(debug, "reached here");

        // build W
        for (uint i = 2; i < numNodes; i++)
        {
            uint *A = &query_edges->cdata()[query_edge_ptr->cdata()[i]];
            uint A_len = query_edge_ptr->cdata()[i + 1] - query_edge_ptr->cdata()[i];
            for (int j = 2; j < i; j++)
            {
                uint *B = &query_edges->cdata()[query_edge_ptr->cdata()[j]];
                uint B_len = query_edge_ptr->cdata()[j + 1] - query_edge_ptr->cdata()[j];
                uint Ahead = 2, Bhead = 2; // since everything is connected to zero
                while (Ahead < A_len && Bhead < B_len)
                {
                    if (A[Ahead] < B[Bhead])
                        Ahead++;
                    else if (A[Ahead] > B[Bhead])
                        Bhead++;
                    else
                    {
                        w[(i - 2) * wrows + j - 2] = w[(i - 2) * wrows + j - 2] + 1;
                        Ahead++;
                        Bhead++;
                    }
                }
            }
        }

        // first symmetric level
        for (uint i = 1; i < numNodes + 1; i++)
        {
            if (sym_nodes_ptr->cdata()[i] > 0)
            {
                first_sym_level = i;
                break;
            }
        }
        Log(debug, "First symmetric level: %u", first_sym_level);
        cudaMemcpyToSymbol(FIRST_SYM_LEVEL, &first_sym_level, sizeof(uint));

// Find max W
#ifdef REUSE
        for (uint i = 0; i < numNodes; i++)
        {
            uint max = 0, maxid = 0;
            for (uint j = 2; j < i; j++)
            {
                if (max < w[(i - 2) * wrows + j - 2])
                {
                    max = w[(i - 2) * wrows + j - 2];
                    maxid = j;
                }
            }
            if (max > 0)
            {
                q_reusable->cdata()[maxid] = true;
                reuse_level->cdata()[i] = maxid;
                reuse_ptr->cdata()[i] = query_edge_ptr->cdata()[i] + max + 2;
            }
            else
            {
                reuse_ptr->cdata()[i] = query_edge_ptr->cdata()[i + 1];
            }
        }
#endif
        for (uint i = 0; i < numNodes; i++)
        {
            if (reuse_level->cdata()[i] == 0)
                reuse_ptr->cdata()[i] = query_edge_ptr->cdata()[i] + 1;
        }

        Log(critical, "ID: Reusable: Reuse:\n");
        for (int i = 0; i < numNodes; i++)
        {
            Log(critical, "%d\t %u\t %u", i, q_reusable->cdata()[i], reuse_level->cdata()[i]);
        }
        Log(info, "ID: qedge_ptr: Reuse ptr: qedge_ptr_end");
        for (uint i = 0; i < numNodes; i++)
        {
            Log(info, "%u,\t %u,\t %u,\t %u", i, query_edge_ptr->cdata()[i], reuse_ptr->cdata()[i], query_edge_ptr->cdata()[i + 1]);
        }

        // cleanup
        delete[] w;
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
        // size_t qSize = (by_ == ByEdge) ? dataGraph.numEdges : dataGraph.numNodes;
        size_t qSize = max(dataGraph.numNodes, dataGraph.numEdges);
        bucket_nq.Create(unified, dataGraph.numNodes, dev_);
        current_nq.Create(unified, dataGraph.numNodes, dev_);

        bucket_eq.Create(unified, dataGraph.numEdges, dev_);
        current_eq.Create(unified, dataGraph.numEdges, dev_);

        asc.initialize("Identity array asc", unified, qSize, dev_);
        execKernel(init_asc, (qSize + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, dev_, false,
                   asc.gdata(), qSize);

        CUDA_RUNTIME(cudaGetLastError());
        cudaDeviceSynchronize();

        // Compute Max Degree
        edgeDegree.initialize("Edge Support", unified, dataGraph.numEdges, dev_);
        uint dimGridEdges = (dataGraph.numEdges + block_size - 1) / block_size;
        cudaDeviceSynchronize();
        execKernel((get_max_degree<T, block_size>), dimGridEdges, block_size, dev_, false,
                   dataGraph, edgeDegree.gdata(), max_eDegree.gdata());

        nodeDegree.initialize("Node Support", unified, dataGraph.numNodes, dev_);
        uint dimGridNodes = (dataGraph.numNodes + block_size - 1) / block_size;
        execKernel((getNodeDegree_kernel<T, block_size>), dimGridNodes, block_size, dev_, false,
                   nodeDegree.gdata(), dataGraph, max_dDegree.gdata());
        Log(debug, "Max Edge Degree: %u\t Max Node Degree: %u", max_eDegree.gdata()[0], max_dDegree.gdata()[0]);
    }

    template <typename T>
    void SG_Match<T>::initialize1(graph::COOCSRGraph_d<T> &oriented_dataGraph)
    {
        const auto block_size = 256;
        size_t qSize = max(oriented_dataGraph.numEdges, oriented_dataGraph.numNodes);

        // asc.initialize("Identity array asc", unified, qSize, dev_);
        execKernel(init_asc, (qSize + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, dev_, false,
                   asc.gdata(), qSize);

        CUDA_RUNTIME(cudaGetLastError());
        cudaDeviceSynchronize();

        // Compute Max Degrees and max degree
        oriented_nodeDegree.initialize("Edge Support", unified, oriented_dataGraph.numNodes, dev_);
        uint dimGridNodes = (oriented_dataGraph.numNodes + block_size - 1) / block_size;
        execKernel((getNodeDegree_split_kernel<T, block_size>), dimGridNodes, block_size, dev_, false,
                   oriented_nodeDegree.gdata(), oriented_dataGraph, oriented_max_dDegree.gdata());
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
            bucket_scan(nodeDegree, dataGraph.numNodes, 1, min_qDegree - 1, bucket_level_end_, current_nq, bucket_nq);
            // bucket_level_end_ -= min_qDegree - 1;
            // bucket_scan(edgeDegree, dataGraph.numEdges, 1, min_qDegree - 1, bucket_level_end_, current_eq, bucket_eq);
            // printf("edge count: %u\n", current_eq.count.gdata()[0]);
            // Populate keep array
            keep.initialize("Keep edges", unified, dataGraph.numEdges, dev_);
            keep.setAll(true, true);
            uint grid_size = (current_nq.count.gdata()[0] - 1) / block_size + 1;

            execKernel((remove_edges_connected_to_node<T>), grid_size, block_size, dev_, true,
                       dataGraph, current_nq.device_queue->gdata()[0], keep.gdata());

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
        } while (current_nq.count.gdata()[0] > 0.05 * dataGraph.numNodes);

        // Recompute max degree
        uint grid_size = (dataGraph.numNodes - 1) / block_size + 1;
        execKernel((getNodeDegree_kernel<T, block_size>), grid_size, block_size, dev_, false,
                   nodeDegree.gdata(), dataGraph, max_dDegree.gdata());

        // if (by_ == ByEdge)
        // {
        uint edge_grid = (dataGraph.numEdges - 1) / block_size + 1;
        execKernel((get_max_degree<T, block_size>), edge_grid, block_size, dev_, false,
                   dataGraph, edgeDegree.gdata(), max_eDegree.gdata());
        // }
    }

#ifdef TIMER
    template <typename T>
    void SG_Match<T>::count_subgraphs_timer(graph::COOCSRGraph_d<T> &dataGraph)
    {

        // Initialise Kernel Dims
        const auto block_size_LD = BLOCK_SIZE_LD; // Block size for low degree nodes
        const T partitionSize_LD = PARTITION_SIZE_LD;
        const T numPartitions_LD = block_size_LD / partitionSize_LD;
        const auto block_size_HD = BLOCK_SIZE_HD; // Block size for high degree nodes
        const T partitionSize_HD = PARTITION_SIZE_HD;
        const T numPartitions_HD = block_size_HD / partitionSize_HD;
        const T bound_LD = CUTOFF;
        const T bound_HD = 32768 + 16384;
        const uint dv = 32;

        // CUDA Initialise, gather runtime Info.
        CUDAContext context;
        T num_SMs = context.num_SMs;
        T conc_blocks_per_SM_LD = context.GetConCBlocks(block_size_LD);
        T conc_blocks_per_SM_HD = context.GetConCBlocks(block_size_HD);

        GPUArray<Counters> SM_times("SM times", unified, num_SMs, dev_);

        GPUArray<uint64> SM_nodes("nodes processed per SM", unified, num_SMs, dev_);
        CUDA_RUNTIME(cudaMemset(SM_times.gdata(), 0, num_SMs * sizeof(Counters)));
        CUDA_RUNTIME(cudaMemset(SM_nodes.gdata(), 0, num_SMs * sizeof(uint64)));
        GPUArray<uint64> intersection_count;
        intersection_count.initialize("Temp level Counter", unified, 1, dev_);
        intersection_count.setSingle(0, 0, false);

        // Initialise Arrays (always LD block size)
        GPUArray<T> d_bitmap_states("bmp bitmap stats", AllocationTypeEnum::gpu, num_SMs * conc_blocks_per_SM_LD, dev_);
        GPUArray<T> per_node_count("per node count", AllocationTypeEnum::unified, dataGraph.numNodes, dev_);
        cudaMemset(d_bitmap_states.gdata(), 0, num_SMs * conc_blocks_per_SM_LD * sizeof(T));
        cudaMemset(per_node_count.gdata(), 0, dataGraph.numNodes * sizeof(T));

        // template information in GPU Constant memory
        cudaMemcpyToSymbol(KCCOUNT, &(query_sequence->N), sizeof(KCCOUNT));
        cudaMemcpyToSymbol(LUNMAT, &(unmat_level), sizeof(LUNMAT));
        cudaMemcpyToSymbol(MAXLEVEL, &max_qDegree, sizeof(MAXLEVEL));
        cudaMemcpyToSymbol(MINLEVEL, &min_qDegree, sizeof(MINLEVEL));
        cudaMemcpyToSymbol(QEDGE, &(query_edges->cdata()[0]), query_edges->N * sizeof(QEDGE[0]));
        cudaMemcpyToSymbol(QEDGE_PTR, &(query_edge_ptr->cdata()[0]), query_edge_ptr->N * sizeof(QEDGE_PTR[0]));
        cudaMemcpyToSymbol(SYMNODE, &(sym_nodes->cdata()[0]), sym_nodes->N * sizeof(SYMNODE[0]));
        cudaMemcpyToSymbol(SYMNODE_PTR, &(sym_nodes_ptr->cdata()[0]), sym_nodes_ptr->N * sizeof(SYMNODE_PTR[0]));
        cudaMemcpyToSymbol(QDEG, &(query_degree->cdata()[0]), query_degree->N * sizeof(QDEG[0]));

        cudaMemcpyToSymbol(QREUSE, &(reuse_level->cdata()[0]), reuse_level->N * sizeof(uint));
        cudaMemcpyToSymbol(QREUSABLE, &(q_reusable->cdata()[0]), q_reusable->N * sizeof(bool));
        cudaMemcpyToSymbol(REUSE_PTR, &(reuse_ptr->cdata()[0]), reuse_ptr->N * sizeof(uint));

        // Initialise Queueing
        T todo = (by_ == ByEdge) ? dataGraph.numEdges : dataGraph.numNodes;
        T span = bound_LD;
        T level = 0;
        T bucket_level_end_ = level;

        // Ignore starting nodes with degree < max_qDegree.

        double time_;
        Timer t_;

        bucket_scan(nodeDegree, dataGraph.numNodes, level, max_qDegree, bucket_level_end_, current_nq, bucket_nq);

        time_ = t_.elapsed_and_reset();
        Log(debug, "scan1 TIME: %f s", time_);

        todo -= current_nq.count.gdata()[0];
        current_nq.count.gdata()[0] = 0;
        level = max_qDegree;
        bucket_level_end_ = level;
        span -= level;

        // Loop over different buckets
        bool first = true;
        while (todo > 0)
        {

            bucket_scan(nodeDegree, dataGraph.numNodes, level, span, bucket_level_end_, current_nq, bucket_nq);
            time_ = t_.elapsed_and_reset();
            Log(debug, "scan%d TIME: %f s", 3 - (int)first, time_);

            if (current_nq.count.gdata()[0] > 0)
            {
                uint maxDeg = (by_ == ByEdge) ? max_eDegree.gdata()[0] : max_dDegree.gdata()[0];
                maxDeg = level + span < maxDeg ? level + span : maxDeg;
                maxDeg = level + span < maxDeg ? level + span : maxDeg;
                uint num_divs = (maxDeg + dv - 1) / dv;
                size_t free = 0, total = 0;
                cuMemGetInfo(&free, &total);

                if (!first || SCHEDULING)
                    current_nq.map_n_key_sort(nodeDegree.gdata());
                time_ = t_.elapsed_and_reset();
                Log(debug, "sort%d TIME: %f s", 2 - (int)first, time_);

                GPUArray<mapping<T>> src_mapping;
                if (!first)
                {
                    uint count = current_nq.count.gdata()[0]; // grid size for encoding kernel
                    Degree_Scan.initialize("Inclusive Sum scan for edge per block grid", unified, count, dev_);
                    Degree_Scan.setAll(0, current_nq.count.gdata()[0]);
                    current_nq.i_scan(Degree_Scan.gdata(), nodeDegree.gdata());
                    uint len = Degree_Scan.gdata()[count - 1]; // grid size for pre encoded kernel
                    src_mapping.initialize("mapping edges with sources", unified, len, dev_);
                    execKernel((map_src<T>), count, 256, dev_, false, src_mapping.gdata(), current_nq.device_queue->gdata()[0], Degree_Scan.gdata(), nodeDegree.gdata());
                    time_ = t_.elapsed_and_reset();
                    Log(debug, "map1 TIME: %f s", time_);
                }

                if (maxDeg > bound_LD)
                {

                    T nnodes = current_nq.count.gdata()[0];
                    T nedges = Degree_Scan.gdata()[nnodes - 1];
                    GPUArray<int> offset("Encoding offset", AllocationTypeEnum::unified, dataGraph.numNodes, dev_);

                    // // accuracy check
                    T temp_total = 0;
                    for (int i = 0; i < nnodes; i++)
                    {
                        T Node_id = current_nq.device_queue->gdata()->queue[i];
                        T degree = dataGraph.rowPtr[Node_id + 1] - dataGraph.rowPtr[Node_id];
                        T src = (i > 0) ? src_mapping.gdata()[Degree_Scan.gdata()[i - 1]].src : src_mapping.gdata()[0].src;
                        T srcDegree = dataGraph.rowPtr[src + 1] - dataGraph.rowPtr[src];
                        // printf("%u, %u, %u, %u\n", Node_id, degree, src, srcDegree);
                        temp_total += degree;
                        assert(Node_id == src);
                        assert(temp_total == Degree_Scan.gdata()[i]);
                    }
                    assert(temp_total == nedges);

                    time_ = t_.elapsed_and_reset();
                    Log(debug, "check TIME: %f s", time_);

                    // start loop
                    T nq_head = 0;
                    T eq_head = 0;

                    double encode_time = 0, HD_process_time = 0;

                    while (nq_head < current_nq.count.gdata()[0])
                    {

                        T stride_degree, max_blocks;
                        stride_degree = (nq_head > 0) ? Degree_Scan.gdata()[nq_head] - Degree_Scan.gdata()[nq_head - 1] : Degree_Scan.gdata()[0] - 0;

                        num_divs = (stride_degree + dv - 1) / dv;
                        get_max_blocks<T>(stride_degree, max_blocks, num_divs, max_qDegree);

                        cudaMemcpyToSymbol(NUMPART, &numPartitions_HD, sizeof(NUMPART));
                        cudaMemcpyToSymbol(PARTSIZE, &partitionSize_HD, sizeof(PARTSIZE));
                        cudaMemcpyToSymbol(CBPSM, &(conc_blocks_per_SM_HD), sizeof(CBPSM));
                        cudaMemcpyToSymbol(MAXDEG, &stride_degree, sizeof(MAXDEG));
                        cudaMemcpyToSymbol(NUMDIVS, &num_divs, sizeof(NUMDIVS));

                        T node_grid_size = min(nnodes, max_blocks);
                        Log(debug, "processing %u nodes", node_grid_size);
                        size_t encode_size = (size_t)node_grid_size * stride_degree * num_divs;
                        GPUArray<T> node_be("Temp level Counter", AllocationTypeEnum::gpu, encode_size, dev_);

                        cudaMemset(node_be.gdata(), 0, encode_size * sizeof(T));
                        cudaMemset(offset.gdata(), -1, dataGraph.numNodes * sizeof(int));
                        cudaMemset(d_bitmap_states.gdata(), 0, num_SMs * conc_blocks_per_SM_LD * sizeof(T));

                        execKernel((sgm_kernel_compute_encoding<T, block_size_HD, partitionSize_HD>), node_grid_size,
                                   block_size_HD, dev_, false,
                                   dataGraph, current_nq.device_queue->gdata()[0], nq_head,
                                   node_be.gdata(), offset.gdata());

                        encode_time += t_.elapsed_and_reset();

                        CUDA_RUNTIME(cudaMemset(SM_times.gdata(), 0, num_SMs * sizeof(Counters)));
                        CUDA_RUNTIME(cudaMemset(SM_nodes.gdata(), 0, num_SMs * sizeof(uint64)));

                        nnodes -= node_grid_size;
                        nq_head += node_grid_size;
                        T edge_grid_size = (eq_head > 0) ? (Degree_Scan.gdata()[nq_head - 1] - Degree_Scan.gdata()[nq_head - 1 - node_grid_size]) : Degree_Scan.gdata()[nq_head - 1] - 0;

                        uint64 level_size = num_SMs * conc_blocks_per_SM_LD * max_qDegree * num_divs * numPartitions_LD;
                        GPUArray<T> current_level("Temp level Counter", AllocationTypeEnum::gpu, level_size, dev_);
                        cudaMemset(current_level.gdata(), 0, level_size * sizeof(T));

                        cudaMemcpyToSymbol(NUMPART, &numPartitions_LD, sizeof(NUMPART));
                        cudaMemcpyToSymbol(PARTSIZE, &partitionSize_LD, sizeof(PARTSIZE));
                        cudaMemcpyToSymbol(CBPSM, &(conc_blocks_per_SM_LD), sizeof(CBPSM));

                        execKernel((sgm_kernel_pre_encoded_byEdge<T, block_size_LD, partitionSize_LD>),
                                   edge_grid_size, block_size_LD, dev_, false,
                                   counter.gdata(), dataGraph, src_mapping.gdata(), eq_head,
                                   current_level.gdata(),
                                   d_bitmap_states.gdata(), node_be.gdata(), per_node_count.gdata(),
                                   offset.gdata(), SM_times.gdata(), SM_nodes.gdata());

                        nedges -= edge_grid_size;
                        eq_head += edge_grid_size;

                        current_level.freeGPU();
                        node_be.freeGPU();
                        HD_process_time += t_.elapsed_and_reset();
                    }
                    offset.freeGPU();

                    Log(info, "Encode TIME: %f s", encode_time);
                    Log(info, "HD process TIME: %f s", HD_process_time);

                    const uint width = NUM_COUNTERS - 1;
                    GPUArray<uint64> total_times("Total Times", AllocationTypeEnum::unified, num_SMs, dev_);
                    GPUArray<float> SM_frac_times("Fractional SM times", unified, num_SMs * width, dev_);
                    GPUArray<float> total_frac_times("Fractional SM times", unified, width, dev_);
                    for (int i = 0; i < num_SMs; i++)
                    {
                        for (int j = 0; j < width; j++)
                            total_times.gdata()[i] += SM_times.gdata()[i].totalTime[j];

                        for (int j = 0; j < width; j++)
                            SM_frac_times.gdata()[i * width + j] = (SM_times.gdata()[i].totalTime[j] * 1.0) / total_times.gdata()[i];
                    }

                    // get average
                    for (int j = 0; j < width; j++)
                    {
                        for (int i = 0; i < num_SMs; i++)
                        {
                            total_frac_times.gdata()[j] += SM_frac_times.gdata()[width * i + j];
                        }
                        total_frac_times.gdata()[j] = total_frac_times.gdata()[j] / num_SMs;
                    }

                    for (int i = 0; i < width; i++)
                        std::cout << Names[i] << total_frac_times.gdata()[i] << std::endl;
                }
                else
                {
                    uint oriented_maxDeg = oriented_max_dDegree.gdata()[0];
                    maxDeg = level + span < maxDeg ? level + span : maxDeg;

                    cudaMemcpyToSymbol(NUMDIVS, &num_divs, sizeof(NUMDIVS));
                    cudaMemcpyToSymbol(ORIENTED_MAXDEG, &oriented_maxDeg, sizeof(ORIENTED_MAXDEG));
                    cudaMemcpyToSymbol(MAXDEG, &maxDeg, sizeof(MAXDEG));
                    cudaMemcpyToSymbol(NUMPART, &numPartitions_LD, sizeof(NUMPART));
                    cudaMemcpyToSymbol(PARTSIZE, &partitionSize_LD, sizeof(PARTSIZE));
                    cudaMemcpyToSymbol(CBPSM, &(conc_blocks_per_SM_LD), sizeof(CBPSM));

                    auto current_q = (by_ == ByNode) ? current_nq : current_eq;
                    uint64 numBlock = num_SMs * conc_blocks_per_SM_LD;
                    bool persistant = true;
                    if (current_nq.count.gdata()[0] < numBlock)
                    {
                        numBlock = current_nq.count.gdata()[0];
                        persistant = false;
                    }

                    uint64 encode_size = numBlock * maxDeg * num_divs;
                    uint64 level_size = numBlock * max_qDegree * num_divs * numPartitions_LD;

                    GPUArray<T> node_be("Temp level Counter", AllocationTypeEnum::gpu, encode_size, dev_);
                    GPUArray<T> current_level("Temp level Counter", AllocationTypeEnum::gpu, level_size, dev_);
                    GPUArray<T> reuse("Intersection storage for reuse", AllocationTypeEnum::gpu, level_size, dev_);

                    cudaMemset(node_be.gdata(), 0, encode_size * sizeof(T));
                    cudaMemset(current_level.gdata(), 0, level_size * sizeof(T));
                    cudaMemset(reuse.gdata(), 0, level_size * sizeof(T));

                    auto grid_block_size = current_nq.count.gdata()[0];

                    if (persistant)
                    {
                        execKernel((sgm_kernel_central_node_base_binary_persistant<T, block_size_LD, partitionSize_LD>),
                                   grid_block_size, block_size_LD, dev_, false,
                                   counter.gdata(), per_node_count.gdata(), intersection_count.gdata(),
                                   dataGraph, current_q.device_queue->gdata()[0],
                                   current_level.gdata(), reuse.gdata(),
                                   d_bitmap_states.gdata(), node_be.gdata(),
                                   by_ == ByNode);
                    }
                    else
                    {
                        execKernel((sgm_kernel_central_node_base_binary<T, block_size_LD, partitionSize_LD>),
                                   grid_block_size, block_size_LD, dev_, false,
                                   counter.gdata(), per_node_count.gdata(), intersection_count.gdata(),
                                   dataGraph, current_q.device_queue->gdata()[0],
                                   current_level.gdata(), reuse.gdata(), node_be.gdata(),
                                   by_ == ByNode);
                    }

                    // cleanup
                    node_be.freeGPU();
                    reuse.freeGPU();
                    current_level.freeGPU();

                    time_ = t_.elapsed_and_reset();
                    Log(info, "LD process TIME: %f s", time_);
                }

                // Print bucket stats:
                // std::cout << "Bucket levels: " << level << " to " << maxDeg
                //           << ", nodes/edges: " << current_nq.count.gdata()[0]
                //           << ", Counter: " << counter.gdata()[0] << std::endl;
            }
            level += span;
            span = bound_HD;
            todo -= current_nq.count.gdata()[0];
            // todo = 0;
            first = false;
        }

        std::cout << "------------- Counter = " << counter.gdata()[0] << "\n";

        counter.freeGPU();
        per_node_count.freeGPU();
        d_bitmap_states.freeGPU();
        intersection_count.freeGPU();
    }
#else

    template <typename T>
    void SG_Match<T>::count_subgraphs(graph::COOCSRGraph_d<T> &dataGraph)
    {
        // Initialise Kernel Dims
        const auto block_size_LD = BLOCK_SIZE_LD; // Block size for low degree nodes
        const T partitionSize_LD = PARTITION_SIZE_LD;
        const T numPartitions_LD = block_size_LD / partitionSize_LD;
        const auto block_size_HD = BLOCK_SIZE_HD; // Block size for high degree nodes
        const T partitionSize_HD = PARTITION_SIZE_HD;
        const T numPartitions_HD = block_size_HD / partitionSize_HD;
        const T bound_LD = CUTOFF;
        const T bound_HD = 32768 + 16384;
        const uint dv = 32;

        // CUDA Initialise, gather runtime Info.
        CUDAContext context;
        T num_SMs = context.num_SMs;
        T conc_blocks_per_SM_LD = context.GetConCBlocks(block_size_LD);
        T conc_blocks_per_SM_HD = context.GetConCBlocks(block_size_HD);

        // Initialise Arrays (always LD block size)
        GPUArray<T> d_bitmap_states("bmp bitmap stats", AllocationTypeEnum::gpu, num_SMs * conc_blocks_per_SM_LD, dev_);
        cudaMemset(d_bitmap_states.gdata(), 0, num_SMs * conc_blocks_per_SM_LD * sizeof(T));

        // template information in GPU Constant memory
        cudaMemcpyToSymbol(KCCOUNT, &(query_sequence->N), sizeof(KCCOUNT));
        cudaMemcpyToSymbol(LUNMAT, &(unmat_level), sizeof(LUNMAT));
        cudaMemcpyToSymbol(MAXLEVEL, &max_qDegree, sizeof(MAXLEVEL));
        cudaMemcpyToSymbol(MINLEVEL, &min_qDegree, sizeof(MINLEVEL));
        cudaMemcpyToSymbol(QEDGE, &(query_edges->cdata()[0]), query_edges->N * sizeof(QEDGE[0]));
        cudaMemcpyToSymbol(QEDGE_PTR, &(query_edge_ptr->cdata()[0]), query_edge_ptr->N * sizeof(QEDGE_PTR[0]));
        cudaMemcpyToSymbol(SYMNODE, &(sym_nodes->cdata()[0]), sym_nodes->N * sizeof(SYMNODE[0]));
        cudaMemcpyToSymbol(SYMNODE_PTR, &(sym_nodes_ptr->cdata()[0]), sym_nodes_ptr->N * sizeof(SYMNODE_PTR[0]));
        cudaMemcpyToSymbol(QDEG, &(query_degree->cdata()[0]), query_degree->N * sizeof(QDEG[0]));
        cudaMemcpyToSymbol(QREUSE, &(reuse_level->cdata()[0]), reuse_level->N * sizeof(uint));
        cudaMemcpyToSymbol(QREUSABLE, &(q_reusable->cdata()[0]), q_reusable->N * sizeof(bool));
        cudaMemcpyToSymbol(REUSE_PTR, &(reuse_ptr->cdata()[0]), reuse_ptr->N * sizeof(uint));

        // Initialise Queueing
        T todo = (by_ == ByEdge) ? dataGraph.numEdges : dataGraph.numNodes;
        T span = bound_LD;
        T level = 0;
        T bucket_level_end_ = level;

        // Ignore starting nodes with degree < max_qDegree.

        bucket_scan(nodeDegree, dataGraph.numNodes, level, max_qDegree, bucket_level_end_, current_nq, bucket_nq);

        todo -= current_nq.count.gdata()[0];
        current_nq.count.gdata()[0] = 0;
        level = max_qDegree;
        bucket_level_end_ = level;
        span -= level;

        // Loop over different buckets
        bool first = true;
        while (todo > 0)
        {

            bucket_scan(nodeDegree, dataGraph.numNodes, level, span, bucket_level_end_, current_nq, bucket_nq);

            if (current_nq.count.gdata()[0] > 0)
            {
                uint maxDeg = (by_ == ByEdge) ? max_eDegree.gdata()[0] : max_dDegree.gdata()[0];
                maxDeg = level + span < maxDeg ? level + span : maxDeg;
                maxDeg = level + span < maxDeg ? level + span : maxDeg;
                uint num_divs = (maxDeg + dv - 1) / dv;
                size_t free = 0, total = 0;
                cuMemGetInfo(&free, &total);

                if (!first || SCHEDULING)
                    current_nq.map_n_key_sort(nodeDegree.gdata());

                GPUArray<mapping<T>> src_mapping;
                if (!first)
                {
                    uint count = current_nq.count.gdata()[0]; // grid size for encoding kernel
                    Degree_Scan.initialize("Inclusive Sum scan for edge per block grid", unified, count, dev_);
                    Degree_Scan.setAll(0, current_nq.count.gdata()[0]);
                    current_nq.i_scan(Degree_Scan.gdata(), nodeDegree.gdata());
                    uint len = Degree_Scan.gdata()[count - 1]; // grid size for pre encoded kernel
                    src_mapping.initialize("mapping edges with sources", unified, len, dev_);
                    execKernel((map_src<T>), count, 256, dev_, false, src_mapping.gdata(), current_nq.device_queue->gdata()[0], Degree_Scan.gdata(), nodeDegree.gdata());
                }

                if (maxDeg > bound_LD)
                {

                    T nnodes = current_nq.count.gdata()[0];
                    T nedges = Degree_Scan.gdata()[nnodes - 1];
                    GPUArray<int> offset("Encoding offset", AllocationTypeEnum::unified, dataGraph.numNodes, dev_);

                    // // accuracy check
                    // T temp_total = 0;
                    // for (int i = 0; i < nnodes; i++)
                    // {
                    //     T Node_id = current_nq.device_queue->gdata()->queue[i];
                    //     T degree = dataGraph.rowPtr[Node_id + 1] - dataGraph.rowPtr[Node_id];
                    //     T src = (i > 0) ? src_mapping.gdata()[Degree_Scan.gdata()[i - 1]].src : src_mapping.gdata()[0].src;
                    //     T srcDegree = dataGraph.rowPtr[src + 1] - dataGraph.rowPtr[src];
                    //     // printf("%u, %u, %u, %u\n", Node_id, degree, src, srcDegree);
                    //     temp_total += degree;
                    //     assert(Node_id == src);
                    //     assert(temp_total == Degree_Scan.gdata()[i]);
                    // }
                    // assert(temp_total == nedges);

                    // start loop
                    T nq_head = 0;
                    T eq_head = 0;
                    bool first = true;

                    double encode_time = 0, HD_process_time = 0;

                    while (nq_head < current_nq.count.gdata()[0])
                    {
                        T stride_degree, max_blocks;
                        stride_degree = (nq_head > 0) ? Degree_Scan.gdata()[nq_head] - Degree_Scan.gdata()[nq_head - 1] : Degree_Scan.gdata()[0] - 0;

                        num_divs = (stride_degree + dv - 1) / dv;
                        get_max_blocks<T>(stride_degree, max_blocks, num_divs, max_qDegree);

                        cudaMemcpyToSymbol(NUMPART, &numPartitions_HD, sizeof(NUMPART));
                        cudaMemcpyToSymbol(PARTSIZE, &partitionSize_HD, sizeof(PARTSIZE));
                        cudaMemcpyToSymbol(CBPSM, &(conc_blocks_per_SM_HD), sizeof(CBPSM));
                        cudaMemcpyToSymbol(MAXDEG, &stride_degree, sizeof(MAXDEG));
                        cudaMemcpyToSymbol(NUMDIVS, &num_divs, sizeof(NUMDIVS));

                        T node_grid_size = min(nnodes, max_blocks);
                        Log(debug, "processing %u nodes", node_grid_size);
                        size_t encode_size = (size_t)node_grid_size * stride_degree * num_divs;
                        GPUArray<T> node_be("Temp level Counter", AllocationTypeEnum::gpu, encode_size, dev_);

                        cudaMemset(node_be.gdata(), 0, encode_size * sizeof(T));
                        cudaMemset(offset.gdata(), -1, dataGraph.numNodes * sizeof(int));
                        cudaMemset(d_bitmap_states.gdata(), 0, num_SMs * conc_blocks_per_SM_LD * sizeof(T));

                        execKernel((sgm_kernel_compute_encoding<T, block_size_HD, partitionSize_HD>), node_grid_size,
                                   block_size_HD, dev_, false,
                                   dataGraph, current_nq.device_queue->gdata()[0], nq_head,
                                   node_be.gdata(), offset.gdata());

                        encode_time += t_.elapsed_and_reset();

                        nnodes -= node_grid_size;
                        nq_head += node_grid_size;
                        T edge_grid_size = (!first) ? (Degree_Scan.gdata()[nq_head - 1] - Degree_Scan.gdata()[nq_head - 1 - node_grid_size]) : Degree_Scan.gdata()[nq_head - 1] - 0;

                        uint64 level_size = num_SMs * conc_blocks_per_SM_LD * max_qDegree * num_divs * numPartitions_LD;
                        GPUArray<T> current_level("Temp level Counter", AllocationTypeEnum::gpu, level_size, dev_);
                        cudaMemset(current_level.gdata(), 0, level_size * sizeof(T));

                        cudaMemcpyToSymbol(NUMPART, &numPartitions_LD, sizeof(NUMPART));
                        cudaMemcpyToSymbol(PARTSIZE, &partitionSize_LD, sizeof(PARTSIZE));
                        cudaMemcpyToSymbol(CBPSM, &(conc_blocks_per_SM_LD), sizeof(CBPSM));

                        execKernel((sgm_kernel_pre_encoded_byEdge<T, block_size_LD, partitionSize_LD>),
                                   edge_grid_size, block_size_LD, dev_, false,
                                   counter.gdata(), dataGraph, src_mapping.gdata(), eq_head,
                                   current_level.gdata(),
                                   d_bitmap_states.gdata(), node_be.gdata(),
                                   offset.gdata());

                        nedges -= edge_grid_size;
                        eq_head += edge_grid_size;

                        current_level.freeGPU();
                        node_be.freeGPU();
                        first = false;
                        HD_process_time += t_.elapsed_and_reset();
                    }
                    offset.freeGPU();

                    Log(info, "Encode TIME: %f s", encode_time);
                    Log(info, "HD process TIME: %f s", HD_process_time);
                }
                else
                {
                    uint oriented_maxDeg = oriented_max_dDegree.gdata()[0];
                    maxDeg = level + span < maxDeg ? level + span : maxDeg;

                    cudaMemcpyToSymbol(NUMDIVS, &num_divs, sizeof(NUMDIVS));
                    cudaMemcpyToSymbol(ORIENTED_MAXDEG, &oriented_maxDeg, sizeof(ORIENTED_MAXDEG));
                    cudaMemcpyToSymbol(MAXDEG, &maxDeg, sizeof(MAXDEG));
                    cudaMemcpyToSymbol(NUMPART, &numPartitions_LD, sizeof(NUMPART));
                    cudaMemcpyToSymbol(PARTSIZE, &partitionSize_LD, sizeof(PARTSIZE));
                    cudaMemcpyToSymbol(CBPSM, &(conc_blocks_per_SM_LD), sizeof(CBPSM));

                    auto current_q = (by_ == ByNode) ? current_nq : current_eq;
                    uint64 numBlock = num_SMs * conc_blocks_per_SM_LD;
                    bool persistant = true;
                    if (current_nq.count.gdata()[0] < numBlock)
                    {
                        numBlock = current_nq.count.gdata()[0];
                        persistant = false;
                    }

                    uint64 encode_size = numBlock * maxDeg * num_divs;
                    uint64 level_size = numBlock * max_qDegree * num_divs * numPartitions_LD;

                    GPUArray<T> node_be("Temp level Counter", AllocationTypeEnum::gpu, encode_size, dev_);
                    GPUArray<T> current_level("Temp level Counter", AllocationTypeEnum::gpu, level_size, dev_);
                    GPUArray<T> reuse("Intersection storage for reuse", AllocationTypeEnum::gpu, level_size, dev_);

                    cudaMemset(node_be.gdata(), 0, encode_size * sizeof(T));
                    cudaMemset(current_level.gdata(), 0, level_size * sizeof(T));
                    cudaMemset(reuse.gdata(), 0, level_size * sizeof(T));

                    auto grid_block_size = current_nq.count.gdata()[0];

                    if (persistant)
                    {
                        execKernel((sgm_kernel_central_node_base_binary_persistant<T, block_size_LD, partitionSize_LD>),
                                   grid_block_size, block_size_LD, dev_, false,
                                   counter.gdata(),
                                   dataGraph, current_q.device_queue->gdata()[0],
                                   current_level.gdata(), reuse.gdata(),
                                   d_bitmap_states.gdata(), node_be.gdata(),
                                   by_ == ByNode);
                    }
                    else
                    {
                        execKernel((sgm_kernel_central_node_base_binary<T, block_size_LD, partitionSize_LD>),
                                   grid_block_size, block_size_LD, dev_, false,
                                   counter.gdata(),
                                   dataGraph, current_q.device_queue->gdata()[0],
                                   current_level.gdata(), reuse.gdata(), node_be.gdata(),
                                   by_ == ByNode);
                    }

                    // cleanup
                    node_be.freeGPU();
                    reuse.freeGPU();
                    current_level.freeGPU();
                }

                // Print bucket stats:
                // std::cout << "Bucket levels: " << level << " to " << maxDeg
                //           << ", nodes/edges: " << current_nq.count.gdata()[0]
                //           << ", Counter: " << counter.gdata()[0] << std::endl;
            }
            level += span;
            span = bound_HD;
            todo -= current_nq.count.gdata()[0];
            // todo = 0;
            first = false;
        }

        std::cout << "------------- Counter = " << counter.gdata()[0] << "\n";

        counter.freeGPU();
        d_bitmap_states.freeGPU();
    }
#endif

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

        // if there is only one central node and 1 symmetric node:
        graph::GPUArray<triplet<T>> triplet_array("Array paired with partitions to ", unified, m, dev_);

        if (first_sym_level <= 2)
        {
            Log(critical, "Ascending SB");
            execKernel((set_priority<T, true>), edge_gridSize, blockSize, dev_, false, g, nodeDegree.gdata()); // get split ptr data
            execKernel((map_and_gen_triplet_array<T>), edge_gridSize, blockSize, dev_, false, triplet_array.gdata(), g, nodeDegree.gdata());

            thrust::stable_sort(thrust::device, triplet_array.gdata(), triplet_array.gdata() + m, comparePriority<T, true>());
        }
        else
        {
            Log(critical, "Descending SB");
            execKernel((set_priority<T, false>), edge_gridSize, blockSize, dev_, false, g, nodeDegree.gdata()); // get split ptr data
            execKernel((map_and_gen_triplet_array<T>), edge_gridSize, blockSize, dev_, false, triplet_array.gdata(), g, nodeDegree.gdata());
            thrust::stable_sort(thrust::device, triplet_array.gdata(), triplet_array.gdata() + m, comparePriority<T, false>());
        }

        thrust::stable_sort(thrust::device, triplet_array.gdata(), triplet_array.gdata() + m, comparePartition<T>());
        CUDA_RUNTIME(cudaDeviceSynchronize());

        CUDA_RUNTIME(cudaMalloc(&g.oriented_colInd, m * sizeof(T)));
        execKernel((map_back<T>), edge_gridSize, blockSize, dev_, false, triplet_array.gdata(), g);
        triplet_array.freeGPU();
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
}