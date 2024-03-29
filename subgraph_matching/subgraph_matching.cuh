#pragma once
#include "config.cuh"
#include "include/common_utils.cuh"

#ifdef TIMER
#include "sgm_kernelsLD_timer.cuh"
#else
#include "sgm_kernelsLD.cuh"
#endif
#include "thrust/sort.h"
#include "thrust/execution_policy.h"
// #include <stdint.h>

namespace graph
{

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
        int ndev_;

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
        uint cutoff_;

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
        SG_Match(MAINTASK task = GRAPH_MATCH, ProcessBy by = ByNode, int dev = 0, uint cutoff = 768, int ndev = 1) : task_(task),
                                                                                                                     by_(by),
                                                                                                                     dev_(dev),
                                                                                                                     ndev_(ndev)
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
            cutoff_ = cutoff;
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
            Log(debug, "Parsing template succesful!");
            size_t free, total;
            cuMemGetInfo(&free, &total);
            Log(debug, "L:%u free mem : %f GB, total mem %f GB", __LINE__, (free * 1.0) / (1000 * 1000 * 1000), (total * 1.0) / (1000 * 1000 * 1000));

            detect_reuse(patGraph);
            Log(debug, "detecting reuse succesful!");
            cuMemGetInfo(&free, &total);
            Log(debug, "L:%u free mem : %f GB, total mem %f GB", __LINE__, (free * 1.0) / (1000 * 1000 * 1000), (total * 1.0) / (1000 * 1000 * 1000));

            initialize(dataGraph);
            Log(debug, "Initializing datagraph succesful!");
            cuMemGetInfo(&free, &total);
            Log(debug, "L:%u free mem : %.2f GB, total mem %.2f GB", __LINE__, (free * 1.0) / (1000 * 1000 * 1000), (total * 1.0) / (1000 * 1000 * 1000));

            peel_data(dataGraph);
            Log(debug, "datagraph peeling succesful!");
            cuMemGetInfo(&free, &total);
            Log(debug, "L:%u free mem : %.2f GB, total mem %.2f GB", __LINE__, (free * 1.0) / (1000 * 1000 * 1000), (total * 1.0) / (1000 * 1000 * 1000));

            Log(info, "Degree Ordering");
            degree_ordering(dataGraph);
            Log(debug, "ordering succesfull!");
            cuMemGetInfo(&free, &total);
            Log(debug, "L:%u free mem : %.2f GB, total mem %.2f GB", __LINE__, (free * 1.0) / (1000 * 1000 * 1000), (total * 1.0) / (1000 * 1000 * 1000));
            // initialize1(dataGraph);
            // Log(debug, "initialize 1 succesfull");

            time = p.elapsed();
            Log(info, "Preprocessing TIME: %f ms", time * 1000);
            Log(debug, "Undirected graph maxdegree: %lu", max_dDegree.gdata()[0]);

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
        void print_SM_counters(Counters *SM_times, uint64 *SM_nodes);
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
        graph g[DEPTH];
        int lab[DEPTH], ptn[DEPTH];
        int orbits[DEPTH];
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
        vector<int> sym[DEPTH];
        vector<T> history;
        while (true)
        {
            // Call NAUTY to get orbit
            densenauty(g, lab, ptn, orbits, &opt, &stats, m, n, NULL);

            // Find the biggest orbit
            int cnts[DEPTH] = {0};
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
        Log(info, "ID: Reusable: Reuse:\n");
        for (int i = 0; i < numNodes; i++)
        {
            Log(critical, "%d\t %u\t %u", i, q_reusable->cdata()[i], reuse_level->cdata()[i]);
        }
#endif
        for (uint i = 0; i < numNodes; i++)
        {
            if (reuse_level->cdata()[i] == 0)
                reuse_ptr->cdata()[i] = query_edge_ptr->cdata()[i] + 1;
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
        // size_t qSize = max(dataGraph.numNodes, dataGraph.numEdges);
        size_t qSize = dataGraph.numNodes;
        bucket_nq.Create(unified, dataGraph.numNodes, dev_);
        current_nq.Create(unified, dataGraph.numNodes, dev_);

        asc.initialize("Identity array asc", unified, qSize, dev_);
        execKernel(init_asc, (qSize + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, dev_, false,
                   asc.gdata(), qSize);

        CUDA_RUNTIME(cudaGetLastError());
        cudaDeviceSynchronize();

        // edgeDegree.initialize("Edge Support", unified, dataGraph.numEdges, dev_);
        // uint dimGridEdges = (dataGraph.numEdges + block_size - 1) / block_size;
        // cudaDeviceSynchronize();
        // execKernel((get_max_degree<T, block_size>), dimGridEdges, block_size, dev_, false,
        // 					 dataGraph, edgeDegree.gdata(), max_eDegree.gdata());

        // Compute Max Degree
        nodeDegree.initialize("Node Support", unified, dataGraph.numNodes, dev_);
        uint dimGridNodes = (dataGraph.numNodes + block_size - 1) / block_size;
        execKernel((getNodeDegree_kernel<T, block_size>), dimGridNodes, block_size, dev_, false,
                   nodeDegree.gdata(), dataGraph, max_dDegree.gdata());
        Log(debug, "Max Node Degree: %u", max_dDegree.gdata()[0]);
    }

    template <typename T>
    void SG_Match<T>::initialize1(graph::COOCSRGraph_d<T> &oriented_dataGraph)
    {
        const auto block_size = 256;
        // size_t qSize = max(oriented_dataGraph.numEdges, oriented_dataGraph.numNodes);
        size_t qSize = oriented_dataGraph.numNodes;

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
    }

#ifdef TIMER
    template <typename T>
    void SG_Match<T>::count_subgraphs_timer(graph::COOCSRGraph_d<T> &dataGraph)
    {
        const int first_d = dev_; // temp
        uint m = dataGraph.numEdges, n = dataGraph.numNodes;

        // Initialise Kernel Dims
        CUDAContext context;
        const auto block_size = BLOCK_SIZE; // Block size for low degree nodes
        const T partitionSize = PARTITION_SIZE;
        const T numPartitions = block_size / partitionSize;

        const T bound_LD = cutoff_;
        const uint dv = 32;

        // CUDA Initialise, gather runtime Info.

        T num_SMs = context.num_SMs;
        Log(debug, "Num SMs: %u", num_SMs);
        T conc_blocks_per_SM = context.GetConCBlocks(block_size);

        for (int d = first_d; d < first_d + ndev_; d++)
        {
            // template information in GPU Constant memory
            CUDA_RUNTIME(cudaSetDevice(d));
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
        }
        CUDA_RUNTIME(cudaSetDevice(first_d));
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
        GPUArray<T> per_node_count("Template Count per node", AllocationTypeEnum::gpu, dataGraph.numNodes, dev_);
        GPUArray<uint64> intersection_count("", AllocationTypeEnum::unified, 1, dev_);

        while (todo > 0)
        {

            bucket_scan(nodeDegree, dataGraph.numNodes, level, span, bucket_level_end_, current_nq, bucket_nq);
            time_ = t_.elapsed_and_reset();
            Log(debug, "scan%d TIME: %f s", 2, time_);

            if (current_nq.count.gdata()[0] > 0)
            {
                uint maxDeg = (by_ == ByEdge) ? max_eDegree.gdata()[0] : max_dDegree.gdata()[0];
                maxDeg = level + span < maxDeg ? level + span : maxDeg;
                uint num_divs = (maxDeg + dv - 1) / dv;
                size_t free = 0, total = 0;
                cuMemGetInfo(&free, &total);
                Log(debug, "max Bucket degree %u", maxDeg);

                if (SCHEDULING)
                    current_nq.map_n_key_sort(nodeDegree.gdata());

                T nq_len = current_nq.count.gdata()[0];
                GPUArray<T> scanned("scanned array", unified, nq_len, dev_);
                GPUArray<uint64> tails("tails for work lists", unified, ndev_, dev_);
                CUDA_RUNTIME(cudaMemset(scanned.gdata(), 0, nq_len * sizeof(T)));
                CUDA_RUNTIME(cudaMemset(tails.gdata(), 0, ndev_ * sizeof(uint64)));
                current_nq.i_scan(scanned.gdata(), nodeDegree.gdata());
                T temp_scan_total = scanned.gdata()[nq_len - 1];
                T temp_target = temp_scan_total / ndev_;
#pragma omp parallel for
                for (int d = first_d; d < first_d + ndev_; d++)
                {
                    T target = temp_target * (d - first_d + 1);
                    tails.gdata()[d - first_d] = (uint64)scanned.binary_search(temp_target * (d - first_d + 1));
                    if ((ndev_ - (d - first_d)) == 1 || ndev_ == 1) // last device or only device
                    {
                        tails.gdata()[d - first_d] = (uint64)nq_len;
                    }
                    if (ndev_ - d == 0)
                        Log(debug, "device %d takes: %.2f %%", d, (tails.gdata()[d - first_d] * 1.0) / nq_len * 100);
                    else
                    {
                        Log(debug, "device %d takes: %.2f %%", d, ((tails.gdata()[d - first_d] - tails.gdata()[d - first_d - 1]) * 1.0) / nq_len * 100);
                    }
                }
                scanned.freeGPU();
                time_ = t_.elapsed_and_reset();
                Log(debug, "sort%d TIME: %f s", 1, time_);

                GPUArray<mapping<T>> src_mapping;
                int max_active_blocks = 0;

                maxDeg = level + span < maxDeg ? level + span : maxDeg;

                int temp1, temp2;
                cudaOccupancyMaxPotentialBlockSize(&temp1, &temp2,
                                                   sgm_kernel_central_node_function<T, block_size, partitionSize>);
                Log(debug, "%d, %d", temp1, temp2);

                cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &max_active_blocks,
                    sgm_kernel_central_node_function<T, block_size, partitionSize>,
                    block_size, 0);
                cudaMemcpyToSymbol(CBPSM, &(max_active_blocks), sizeof(CBPSM));
                max_active_blocks *= num_SMs;

                uint grid_block_size = min(current_nq.count.gdata()[0], max_active_blocks);
#pragma omp parallel for
                for (int d = first_d; d < first_d + ndev_; d++)
                {
                    CUDA_RUNTIME(cudaSetDevice(d));
                    CUDA_RUNTIME(cudaMemcpyToSymbol(NUMDIVS, &num_divs, sizeof(NUMDIVS)));
                    CUDA_RUNTIME(cudaMemcpyToSymbol(MAXDEG, &maxDeg, sizeof(MAXDEG)));
                    CUDA_RUNTIME(cudaMemcpyToSymbol(NUMPART, &numPartitions, sizeof(NUMPART)));
                    CUDA_RUNTIME(cudaMemcpyToSymbol(PARTSIZE, &partitionSize, sizeof(PARTSIZE)));
                    CUDA_RUNTIME(cudaMemcpyToSymbol(CBPSM, &(max_active_blocks), sizeof(CBPSM)));
                    CUDA_RUNTIME(cudaMemcpyToSymbol(CB, &(grid_block_size), sizeof(CB)));
                }
                CUDA_RUNTIME(cudaSetDevice(first_d));

                cudaMemAdvise(dataGraph.oriented_colInd, (m) * sizeof(uint), cudaMemAdviseSetReadMostly, dev_ /*ignored*/);
                cudaMemAdvise(dataGraph.colInd, (m) * sizeof(uint), cudaMemAdviseSetReadMostly, dev_ /*ignored*/);
                cudaMemAdvise(dataGraph.splitPtr, (n) * sizeof(uint), cudaMemAdviseSetReadMostly, dev_ /*ignored*/);
                cudaMemAdvise(dataGraph.rowPtr, (n + 1) * sizeof(uint), cudaMemAdviseSetReadMostly, dev_ /*ignored*/);
                cudaMemAdvise(dataGraph.rowInd, (m) * sizeof(uint), cudaMemAdviseSetReadMostly, dev_ /*ignored*/);

#pragma omp parallel for
                for (int d = first_d; d < first_d + ndev_; d++)
                {
                    cuda::atomic<uint32_t, cuda::thread_scope_device> *work_ready = nullptr;
                    queue_declare(queue, tickets, head, tail);
                    int d1 = 0;
                    CUDA_RUNTIME(cudaSetDevice(d));
                    CUDA_RUNTIME(cudaGetDevice(&d1));
                    Log(debug, "device %d\n", d1);
                    uint64 encode_size = (uint64)grid_block_size * maxDeg * num_divs;
                    uint64 level_size = (uint64)grid_block_size * max_qDegree * num_divs * numPartitions;
                    GPUArray<uint64> dev_counter("Counter for each device", AllocationTypeEnum::gpu, sizeof(uint64), d);
                    GPUArray<T> node_be("Temp level Counter", AllocationTypeEnum::gpu, encode_size, d);
                    GPUArray<T> current_level("Temp level Counter", AllocationTypeEnum::gpu, level_size, d);
                    GPUArray<MessageBlock> messages("Messages for sharing info", AllocationTypeEnum::gpu, grid_block_size, d);
                    GPUArray<uint64> work_list_head("Global work stealing list", AllocationTypeEnum::unified, 1, d);
                    GPUArray<double> block_util1("block active/inactive ratio", AllocationTypeEnum::unified, grid_block_size, d);
                    GPUArray<double> block_util2("block active/inactive ratio", AllocationTypeEnum::unified, grid_block_size, d);
                    GPUArray<double> block_util3("block active/inactive ratio", AllocationTypeEnum::unified, grid_block_size, d);

                    GPUArray<Counters> SM_times("SM times", unified, num_SMs, d);
                    GPUArray<uint64> SM_nodes("nodes processed per SM", unified, num_SMs, d);
                    CUDA_RUNTIME(cudaMemset(SM_times.gdata(), 0, num_SMs * sizeof(Counters)));
                    CUDA_RUNTIME(cudaMemset(SM_nodes.gdata(), 0, num_SMs * sizeof(uint64)));

                    uint64 wl_head = (first_sym_level > 2) ? 0 : 1;
                    if (d - first_d > 0)
                        wl_head = tails.gdata()[d - first_d - 1];
                    // CUDA_RUNTIME(cudaMemset(work_list_head.gdata(), 0, sizeof(uint64)));
                    CUDA_RUNTIME(cudaMemcpy(work_list_head.gdata(), &wl_head, sizeof(uint64), cudaMemcpyHostToDevice));
                    CUDA_RUNTIME(cudaMemset(node_be.gdata(), 0, encode_size * sizeof(T)));
                    CUDA_RUNTIME(cudaMemset(current_level.gdata(), 0, level_size * sizeof(T)));
                    CUDA_RUNTIME(cudaMemset(dev_counter.gdata(), 0, sizeof(uint64)));
                    CUDA_RUNTIME(cudaMemset(messages.gdata(), 0, grid_block_size * sizeof(MessageBlock)));
                    CUDA_RUNTIME(cudaMemset(block_util1.gdata(), 0, grid_block_size * sizeof(double)));
                    CUDA_RUNTIME(cudaMemset(block_util2.gdata(), 0, grid_block_size * sizeof(double)));
                    CUDA_RUNTIME(cudaMemset(block_util3.gdata(), 0, grid_block_size * sizeof(double)));
                    CUDA_RUNTIME(cudaMemset(SM_times.gdata(), 0, num_SMs * sizeof(Counters)));

                    GLOBAL_HANDLE<T> gh;
                    gh.global_counter = counter.gdata();
                    gh.counter = dev_counter.gdata();
                    gh.g = dataGraph;
                    gh.current = current_nq.device_queue->gdata()[0];
                    gh.work_list_head = work_list_head.gdata();
                    gh.work_list_tail = tails.gdata()[d - first_d];
                    gh.current_level = current_level.gdata();
                    gh.adj_enc = node_be.gdata();
                    gh.Message = messages.gdata();
                    gh.stride = 1;

                    queue_init(queue, tickets, head, tail, grid_block_size, dev_);
                    CUDA_RUNTIME(cudaMalloc((void **)&work_ready, grid_block_size * sizeof(cuda::atomic<uint32_t, cuda::thread_scope_device>)));
                    CUDA_RUNTIME(cudaMemset((void *)work_ready, 0, grid_block_size * sizeof(cuda::atomic<uint32_t, cuda::thread_scope_device>)));
                    gh.work_ready = work_ready;

                    cuMemGetInfo(&free, &total);
                    Log(debug, "Free mem: %f GB, Total mem: %f GB", (free * 1.0) / (1E9), (total * 1.0) / (1E9));
                    Log(debug, "device%d: my head; %lu, my tail: %lu\n", d, gh.work_list_head[0], gh.work_list_tail);

                    Timer devt;
                    execKernel((sgm_kernel_central_node_function<T, block_size, partitionSize>),
                               grid_block_size, block_size, dev_, false,
                               SM_times.gdata(), block_util1.gdata(), block_util2.gdata(), block_util3.gdata(),
                               gh, queue_caller(queue, tickets, head, tail));

                    execKernel(final_counter, 1, 1, d, false, gh.global_counter, gh.counter);
                    Log(debug, "device %d time: %f s", d, (double)devt.elapsed());

                    if (d == 1)
                    {
                        // print_SM_counters(SM_times.gdata(), SM_nodes.gdata());

                        Log(info, "SM count %u", num_SMs);
                        Log(info, "SM Workload");
                        uint64 ttotal = 0;
                        double average;
                        for (int sm = 0; sm < num_SMs; ++sm)
                        {
                            ttotal += SM_times.gdata()[sm].totalTime[STATE1] + SM_times.gdata()[sm].totalTime[STATE2];
                        }
                        average = (ttotal * 1.0) / num_SMs;
                        for (int sm = 0; sm < num_SMs; ++sm)
                        {
                            std::cout << sm << "\t" << ((SM_times.gdata()[sm].totalTime[STATE1] + SM_times.gdata()[sm].totalTime[STATE2]) * 1.0) / average << std::endl;
                        }

                        Log(info, "\n\n block utilization");
                        for (int b = 0; b < grid_block_size; ++b)
                        {
                            std::cout << block_util1.gdata()[b] << "\t" << block_util2.gdata()[b] << "\t" << block_util3.gdata()[b] << std::endl;
                        }
                    }
                    // cleanup
                    node_be.freeGPU();
                    current_level.freeGPU();
                    work_list_head.freeGPU();
                    messages.freeGPU();
                    CUDA_RUNTIME(cudaFree(work_ready));
                    queue_free(queue, tickets, head, tail);
                    block_util1.freeGPU();
                    block_util2.freeGPU();
                    block_util3.freeGPU();
                    SM_times.freeGPU();
                    SM_nodes.freeGPU();
                }
                // Print bucket stats:
                std::cout << "Bucket levels: " << level << " to " << maxDeg
                          << ", nodes/edges: " << current_nq.count.gdata()[0]
                          << ", Counter: " << counter.gdata()[0] << std::endl;
            }
            level += span;
            todo -= current_nq.count.gdata()[0];
            todo = 0;
        }

        std::cout << "------------- Counter = " << counter.gdata()[0] << "\n";

        counter.freeGPU();
    }

#else
    template <typename T>
    void SG_Match<T>::count_subgraphs(graph::COOCSRGraph_d<T> &dataGraph)
    {
        const int first_d = dev_; // temp
        uint m = dataGraph.numEdges, n = dataGraph.numNodes;
        if (n < 30)
            print_graph(dataGraph);

        // Initialise Kernel Dims
        CUDAContext context;
        const auto block_size = BLOCK_SIZE; // Block size for low degree nodes
        const T partitionSize = PARTITION_SIZE;
        const T numPartitions = block_size / partitionSize;

        const T bound_LD = cutoff_ * 100;
        const uint dv = 32;

        // CUDA Initialise, gather runtime Info.

        T num_SMs = context.num_SMs;
        T conc_blocks_per_SM = context.GetConCBlocks(block_size);

#pragma omp parallel for
        for (int d = first_d; d < first_d + ndev_; d++)
        {
            // template information in GPU Constant memory
            CUDA_RUNTIME(cudaSetDevice(d));
            CUDA_RUNTIME(cudaMemcpyToSymbol(KCCOUNT, &(query_sequence->N), sizeof(KCCOUNT)));
            CUDA_RUNTIME(cudaMemcpyToSymbol(LUNMAT, &(unmat_level), sizeof(LUNMAT)));
            CUDA_RUNTIME(cudaMemcpyToSymbol(MAXLEVEL, &max_qDegree, sizeof(MAXLEVEL)));
            CUDA_RUNTIME(cudaMemcpyToSymbol(MINLEVEL, &min_qDegree, sizeof(MINLEVEL)));
            CUDA_RUNTIME(cudaMemcpyToSymbol(QEDGE, &(query_edges->cdata()[0]), query_edges->N * sizeof(QEDGE[0])));
            CUDA_RUNTIME(cudaMemcpyToSymbol(QEDGE_PTR, &(query_edge_ptr->cdata()[0]), query_edge_ptr->N * sizeof(QEDGE_PTR[0])));
            CUDA_RUNTIME(cudaMemcpyToSymbol(SYMNODE, &(sym_nodes->cdata()[0]), sym_nodes->N * sizeof(SYMNODE[0])));
            CUDA_RUNTIME(cudaMemcpyToSymbol(SYMNODE_PTR, &(sym_nodes_ptr->cdata()[0]), sym_nodes_ptr->N * sizeof(SYMNODE_PTR[0])));
            CUDA_RUNTIME(cudaMemcpyToSymbol(QDEG, &(query_degree->cdata()[0]), query_degree->N * sizeof(QDEG[0])));
            CUDA_RUNTIME(cudaMemcpyToSymbol(QREUSE, &(reuse_level->cdata()[0]), reuse_level->N * sizeof(uint)));
            CUDA_RUNTIME(cudaMemcpyToSymbol(QREUSABLE, &(q_reusable->cdata()[0]), q_reusable->N * sizeof(bool)));
            CUDA_RUNTIME(cudaMemcpyToSymbol(REUSE_PTR, &(reuse_ptr->cdata()[0]), reuse_ptr->N * sizeof(uint)));
        }
        CUDA_RUNTIME(cudaSetDevice(first_d));
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
        while (todo > 0)
        {

            bucket_scan(nodeDegree, dataGraph.numNodes, level, span, bucket_level_end_, current_nq, bucket_nq);

            if (current_nq.count.gdata()[0] > 0)
            {
                uint maxDeg = (by_ == ByEdge) ? max_eDegree.gdata()[0] : max_dDegree.gdata()[0];
                maxDeg = (level + span < maxDeg) ? level + span : maxDeg;
                uint num_divs = (maxDeg + dv - 1) / dv;
                int max_active_blocks = 0;

                size_t free = 0, total = 0;
                Log(debug, "max Bucket degree %u", maxDeg);

                if (SCHEDULING)
                    current_nq.map_n_key_sort(nodeDegree.gdata());

                GPUArray<T> sorted_src("Device queue on host", AllocationTypeEnum::cpuonly, current_nq.count.gdata()[0], dev_);
                CUDA_RUNTIME(cudaMemcpy(sorted_src.cdata(), current_nq.device_queue->gdata()[0].queue, current_nq.count.gdata()[0] * sizeof(T), cudaMemcpyDeviceToHost));

                T iter = 0;
                T temp_degree = nodeDegree.gdata()[sorted_src.cdata()[iter]];

                // printf("Degrees:\n");
                while (temp_degree > cutoff_)
                {
                    iter++;
                    temp_degree = nodeDegree.gdata()[sorted_src.cdata()[iter]];
                    // printf("%u, ", temp_degree);
                }
                T boundary = iter; // split nodes till this boundary
                // printf("\n");
                Log(debug, "Boundary: %u", boundary);
                Log(debug, "Cutoff at Boundary: %u", cutoff_);

                sorted_src.freeCPU();

                maxDeg = level + span < maxDeg ? level + span : maxDeg;
                num_divs = (maxDeg + dv - 1) / dv;
                Log(debug, "max-degree: %u", maxDeg);
                Log(debug, "num-divs %u", num_divs);

                int temp1, temp2;
                cudaOccupancyMaxPotentialBlockSize(&temp1, &temp2,
                                                   sgm_kernel_central_node_function<T, block_size, partitionSize>);
                Log(debug, "%d, %d", temp1, temp2);

                cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &max_active_blocks,
                    sgm_kernel_central_node_function<T, block_size, partitionSize>,
                    block_size, 0);
                max_active_blocks *= num_SMs;
                uint grid_block_size = min(current_nq.count.gdata()[0], max_active_blocks);
#pragma omp parallel for
                for (int d = first_d; d < first_d + ndev_; d++)
                {
                    CUDA_RUNTIME(cudaSetDevice(d));
                    CUDA_RUNTIME(cudaMemcpyToSymbol(NUMDIVS, &num_divs, sizeof(NUMDIVS)));
                    CUDA_RUNTIME(cudaMemcpyToSymbol(MAXDEG, &maxDeg, sizeof(MAXDEG)));
                    CUDA_RUNTIME(cudaMemcpyToSymbol(NUMPART, &numPartitions, sizeof(NUMPART)));
                    CUDA_RUNTIME(cudaMemcpyToSymbol(PARTSIZE, &partitionSize, sizeof(PARTSIZE)));
                    CUDA_RUNTIME(cudaMemcpyToSymbol(CBPSM, &(max_active_blocks), sizeof(CBPSM)));
                    CUDA_RUNTIME(cudaMemcpyToSymbol(CB, &(grid_block_size), sizeof(CB)));
                    CUDA_RUNTIME(cudaMemcpyToSymbol(NDEV, &(ndev_), sizeof(NDEV)));
                }
                CUDA_RUNTIME(cudaSetDevice(first_d));

                Log(debug, "current queue count %u\n", current_nq.count.gdata()[0]);
                Log(debug, "grid size: %u", grid_block_size);

                // multidevice implementation with work stealing
                GPUArray<uint64> work_list_head2("Global work stealing head", AllocationTypeEnum::unified, 1, dev_);
                // uint64 wl_head = (first_sym_level > 2) ? 0 : 1;
                uint64 wl_head = boundary;
                work_list_head2.gdata()[0] = wl_head;

                CUDA_RUNTIME(cudaMemAdvise(dataGraph.oriented_colInd, (m) * sizeof(uint), cudaMemAdviseSetReadMostly, dev_ /*ignored*/));
                CUDA_RUNTIME(cudaMemAdvise(dataGraph.colInd, (m) * sizeof(uint), cudaMemAdviseSetReadMostly, dev_ /*ignored*/));
                CUDA_RUNTIME(cudaMemAdvise(dataGraph.splitPtr, (n) * sizeof(uint), cudaMemAdviseSetReadMostly, dev_ /*ignored*/));
                CUDA_RUNTIME(cudaMemAdvise(dataGraph.rowPtr, (n + 1) * sizeof(uint), cudaMemAdviseSetReadMostly, dev_ /*ignored*/));
                CUDA_RUNTIME(cudaMemAdvise(dataGraph.rowInd, (m) * sizeof(uint), cudaMemAdviseSetReadMostly, dev_ /*ignored*/));
                // for (int d = first_d; d < first_d + ndev_; d++)
                // {
                // 	uint64 wl_head = (first_sym_level > 2) ? 0 : 1;
                // 	if (wl_head == tails.gdata()[d - first_d])
                // 		tails.gdata()[d - first_d]++;
                // }
#pragma omp parallel for
                for (int d = first_d; d < first_d + ndev_; d++)
                {
                    cuda::atomic<uint32_t, cuda::thread_scope_device> *work_ready = nullptr;
                    queue_declare(queue, tickets, head, tail);
                    int d1 = 0;
                    CUDA_RUNTIME(cudaSetDevice(d));
                    CUDA_RUNTIME(cudaGetDevice(&d1));
                    uint64 encode_size = (uint64)grid_block_size * maxDeg * num_divs;
                    uint64 level_size = (uint64)grid_block_size * max_qDegree * num_divs * numPartitions;
                    GPUArray<uint64> dev_counter("Counter for each device", AllocationTypeEnum::gpu, sizeof(uint64), d);
                    GPUArray<T> node_be("Temp level Counter", AllocationTypeEnum::gpu, encode_size, d);
                    GPUArray<T> current_level("Temp level Counter", AllocationTypeEnum::gpu, level_size, d);
                    GPUArray<MessageBlock> messages("Messages for sharing info", AllocationTypeEnum::gpu, grid_block_size, d);
                    GPUArray<T> per_node_count("for debugging", AllocationTypeEnum::unified, dataGraph.numNodes, d);
                    GPUArray<uint64> work_list_head1("Global work stealing list", AllocationTypeEnum::gpu, 1, d);
                    // GPUArray<uint64> work_list_head2("Global work stealing list", AllocationTypeEnum::gpu, 1, d);

                    uint64 temp_head1 = (first_sym_level > 2) ? 0 : 1;
                    // if (d - first_d > 0)
                    // 	temp_head = tails.gdata()[d - first_d - 1];
                    // temp_head += d - first_d;
                    // uint64 temp_head2 = boundary + d - first_d;
                    CUDA_RUNTIME(cudaMemcpy(work_list_head1.gdata(), &temp_head1, sizeof(uint64), cudaMemcpyHostToDevice));
                    // CUDA_RUNTIME(cudaMemcpy(work_list_head2.gdata(), &temp_head2, sizeof(uint64), cudaMemcpyHostToDevice));
                    CUDA_RUNTIME(cudaMemset(dev_counter.gdata(), 0, sizeof(uint64)));
                    CUDA_RUNTIME(cudaMemset(node_be.gdata(), 0, encode_size * sizeof(T)));
                    CUDA_RUNTIME(cudaMemset(current_level.gdata(), 0, level_size * sizeof(T)));
                    CUDA_RUNTIME(cudaMemset(messages.gdata(), 0, grid_block_size * sizeof(MessageBlock)));
                    CUDA_RUNTIME(cudaMemset(per_node_count.gdata(), 0, dataGraph.numNodes * sizeof(T)));

                    GLOBAL_HANDLE<T> gh;
                    gh.global_counter = counter.gdata();
                    gh.counter = dev_counter.gdata();
                    gh.g = dataGraph;
                    gh.current = current_nq.device_queue->gdata()[0];
                    gh.work_list_head1 = work_list_head1.gdata();
                    gh.work_list_head2 = work_list_head2.gdata();
                    gh.work_list_tail = gh.current.count[0];
                    // gh.work_list_tail = tails.gdata()[d - first_d];
                    gh.current_level = current_level.gdata();
                    gh.adj_enc = node_be.gdata();
                    gh.Message = messages.gdata();
                    gh.stride = 1;
                    gh.devId = d - first_d;
                    // gh.stride = ndev_;
                    gh.boundary = boundary;
                    gh.cutoff = cutoff_;

                    queue_init(queue, tickets, head, tail, grid_block_size, d);
                    CUDA_RUNTIME(cudaMalloc((void **)&work_ready, grid_block_size * sizeof(cuda::atomic<uint32_t, cuda::thread_scope_device>)));
                    CUDA_RUNTIME(cudaMemset((void *)work_ready, 0, grid_block_size * sizeof(cuda::atomic<uint32_t, cuda::thread_scope_device>)));
                    gh.work_ready = work_ready;

                    cuMemGetInfo(&free, &total);
                    Log(debug, "Free mem: %.2f GB, Total mem: %.2f GB", (free * 1.0) / (1E9), (total * 1.0) / (1E9));
                    // only use with unified
                    // Log(debug, "device%d: my head; %lu, my tail: %lu\n", d, gh.work_list_head[0], gh.work_list_tail);

                    Timer devt;
                    execKernel((sgm_kernel_central_node_function<T, block_size, partitionSize>),
                               grid_block_size, block_size, d, false,
                               gh, /*per_node_count.gdata(),*/
                               queue_caller(queue, tickets, head, tail));

                    execKernel(final_counter, 1, 1, d, false, gh.global_counter, gh.counter);
                    Log(info, "device %d time: %f s", d, (double)devt.elapsed());
                    // execKernel((sgm_kernel_central_node_function_byNode<T, block_size_LD, partitionSize_LD>),
                    //            grid_block_size, block_size_LD, d, false,
                    //            counter.gdata(), work_list_head.gdata(),
                    //            dataGraph, current_nq.device_queue->gdata()[0],
                    //            current_level.gdata(),
                    //            node_be.gdata());

                    // cleanup
                    node_be.freeGPU();
                    current_level.freeGPU();
                    messages.freeGPU();
                    per_node_count.freeGPU();
                    dev_counter.freeGPU();
                    CUDA_RUNTIME(cudaFree(work_ready));
                    queue_free(queue, tickets, head, tail);
                    work_list_head1.freeGPU();
                    // work_list_head2.freeGPU();
                }
                // Print bucket stats:
                // std::cout << "\nBucket levels: " << level << " to " << maxDeg
                // 					<< ", nodes/edges: " << current_nq.count.gdata()[0]
                // 					<< ", Counter: " << counter.gdata()[0] << std::endl;
                level += span;
                todo -= current_nq.count.gdata()[0];
                todo = 0;
            }
        }
        std::cout << "------------- Counter = " << counter.gdata()[0] << "\n";
        counter.freeGPU();
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

        graph::GPUArray<triplet<T>> triplet_array("Array paired with partitions to ", unified, m, dev_);
        Log(debug, "numnodes: %u, numedges: %u", n, m);

        if (first_sym_level <= 2)
        {
            Log(critical, "Ascending SB");
            execKernel((set_priority<T, true>), edge_gridSize, blockSize, dev_, false, g, nodeDegree.gdata()); // get split ptr data
            execKernel((map_and_gen_triplet_array<T>), edge_gridSize, blockSize, dev_, false, triplet_array.gdata(), g, nodeDegree.gdata());

            if (m < 1E9)
            { // thrust-sort
                Log(debug, "performing device sort");
                thrust::stable_sort(thrust::device, triplet_array.gdata(), triplet_array.gdata() + m, comparePriority<T, true>());
            }
            else
            {
                Log(debug, "performing host sort");
                thrust::stable_sort(thrust::host, triplet_array.gdata(), triplet_array.gdata() + m, comparePriority<T, true>());
            }
        }
        else
        {
            Log(critical, "Descending SB");
            execKernel((set_priority<T, false>), edge_gridSize, blockSize, dev_, false, g, nodeDegree.gdata()); // get split ptr data
            execKernel((map_and_gen_triplet_array<T>), edge_gridSize, blockSize, dev_, false, triplet_array.gdata(), g, nodeDegree.gdata());
            if (m < 1E9) // thrust-sort
                thrust::stable_sort(thrust::device, triplet_array.gdata(), triplet_array.gdata() + m, comparePriority<T, false>());
            else
                thrust::stable_sort(thrust::host, triplet_array.gdata(), triplet_array.gdata() + m, comparePriority<T, false>());
        }

        CUDA_RUNTIME(cudaDeviceSynchronize());
        if (m < 1E9)
            thrust::stable_sort(thrust::device, triplet_array.gdata(), triplet_array.gdata() + m, comparePartition<T>());
        else
            thrust::stable_sort(thrust::host, triplet_array.gdata(), triplet_array.gdata() + m, comparePartition<T>());

        CUDA_RUNTIME(cudaMallocManaged(&g.oriented_colInd, m * sizeof(T)));
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

#ifdef TIMER
    template <typename T>
    void SG_Match<T>::print_SM_counters(Counters *SM_times, uint64 *SM_nodes)
    {
        CUDAContext context;
        T num_SMs = context.num_SMs;
        Log(info, "SM count %u", num_SMs);
        const uint width = NUM_COUNTERS - 2; // 2 for wait
        GPUArray<uint64> total_times("Total Times", AllocationTypeEnum::unified, num_SMs, dev_);
        GPUArray<float> SM_frac_times("Fractional SM times", unified, num_SMs * width, dev_);
        GPUArray<float> total_frac_times("Fractional SM times", unified, width, dev_);
        GPUArray<float> average_time("Average workload of an SM", unified, 1, dev_);
        total_times.setAll(0, true);
        SM_frac_times.setAll(0, true);
        total_frac_times.setAll(0, true);
        average_time.setAll(0, true);

        for (int i = 0; i < num_SMs; i++)
        {
            for (int j = 0; j < width; j++)
                total_times.gdata()[i] += SM_times[i].totalTime[j];

            for (int j = 0; j < width; j++)
                SM_frac_times.gdata()[i * width + j] = (SM_times[i].totalTime[j] * 1.0) / total_times.gdata()[i];
        }

        Log(info, "SM Workload");

        // get average
        for (int i = 0; i < num_SMs; i++)
        {
            average_time.gdata()[0] += total_times.gdata()[i];
        }
        average_time.gdata()[0] = average_time.gdata()[0] / num_SMs;

        for (int i = 0; i < num_SMs; i++)
        {
            std::cout << i << "\t" << (total_times.gdata()[i] * 1.0) / average_time.gdata()[0] << "\t" << SM_nodes[i] << std::endl;
        }
        printf("\n\n");

        for (int j = 0; j < width; j++)
        {
            for (int i = 0; i < num_SMs; i++)
            {
                total_frac_times.gdata()[j] += SM_frac_times.gdata()[width * i + j];
            }
            total_frac_times.gdata()[j] = total_frac_times.gdata()[j] / num_SMs;
        }
        Log(info, "Time distribution");
        for (int i = 0; i < width; i++)
            std::cout << Names[i] << "\t" << total_frac_times.gdata()[i] << std::endl;
    }
#endif
}