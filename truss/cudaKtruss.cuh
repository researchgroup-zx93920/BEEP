//
// Created by Bryan on 19/7/2019.
//


#pragma once
#include <omp.h>

#include <functional>
#include <iostream>
#include <sstream> 
#include <unordered_map>
#include <climits>
#include <numeric>
#include <vector>
#include <cuda_runtime.h>

#include "../include/Timer.h"
#include "../include/CGArray.cuh"
#include "../triangle_counting/TcBmp.cuh"
#include "kernels.cuh"
#include "../include/utils_cuda.cuh"
#include "../include/cub_wrappers.cuh"


namespace graph
{

    void PKT_Scan(
        GPUArray<int> EdgeSupport, uint32_t edge_num, int level,
        GPUArray<int>& curr, GPUArray<bool>& inCurr, int& curr_cnt, GPUArray<uint> asc,
        GPUArray<InBucketWinType>& in_bucket_window_,
        GPUArray<int>& bucket_buf_, int* &window_bucket_buf_size_, int& bucket_level_end_) 
    {
        static bool is_first = true;
        if (is_first) 
        {

            inCurr.setAll(0, true);

            #ifndef LEGACY_SCAN
            in_bucket_window_.setAll(0, true);
            #endif
            is_first = false;
        }
        #ifdef LEGACY_SCAN
        /*filter and get the bool vector*/
        long grid_size = (edge_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
        execKernel(filter, grid_size, BLOCK_SIZE, timer, false, EdgeSupport, edge_num, inCurr, level);
        curr_cnt = CUBSelect(asc, curr, inCurr, edge_num, timer));
        #else
        if (level == bucket_level_end_) 
        {
            // Clear the bucket_removed_indicator
            bucket_level_end_ += LEVEL_SKIP_SIZE;

            #ifndef DEBUG_USE_CPU
            long grid_size = (edge_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
            execKernel(filter_window, grid_size, BLOCK_SIZE, false,
                EdgeSupport.gdata(), edge_num, in_bucket_window_.gdata(), level, bucket_level_end_);

            *window_bucket_buf_size_ = CUBSelect(asc.gdata(), bucket_buf_.gdata(), in_bucket_window_.gdata(), edge_num);
            #else
            auto& size = *window_bucket_buf_size_;
            size = 0;
            for (auto i = 0u; i < edge_num; i++) {
                auto sup = EdgeSupport[i];
                if (sup >= level && sup < bucket_level_end_) {
                    in_bucket_window_[i] = true;
                    bucket_buf_[size++] = i;
                }
            }
            #endif
        }
        // SCAN the window.
        if (*window_bucket_buf_size_ != 0) 
        {
            #ifndef DEBUG_USE_CPU
            curr_cnt = 0;
            long grid_size = (*window_bucket_buf_size_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
            execKernel(filter_with_random_append, grid_size, BLOCK_SIZE, false,
                bucket_buf_.gdata(), *window_bucket_buf_size_, EdgeSupport.gdata(), inCurr.gdata(), curr.gdata(), &curr_cnt, level);
            #else
            curr_cnt = 0;
            for (auto i = 0u; i < *window_bucket_buf_size_; i++) {
                auto edge_off = bucket_buf_[i];
                if (EdgeSupport[edge_off] == level) {
                    curr[curr_cnt++] = edge_off;
                    inCurr[edge_off] = true;
                }
            }
            #endif
        }
        else 
        {
            curr_cnt = 0;
        }
        Log(LogPriorityEnum::info, "Level: %d, curr: %d/%d", level, curr_cnt, *window_bucket_buf_size_);
        #endif
    }

    void ShrinkCSREid(
        int n, int& m,
        GPUArray<uint>& rowPtr, GPUArray<uint>& colInd, BmpGpu<uint>& bmp,
        GPUArray<bool> processed,
        GPUArray <uint> new_edge_offset_origin, GPUArray<Edge> new_edge_list,
        GPUArray<bool> reversed_processed, GPUArray<bool>& edge_deleted, GPUArray<uint> scanned_processed,
        GPUArray <uint>& new_offset, GPUArray<uint>& new_eid, GPUArray <uint>& new_adj,
        GPUArray <InBucketWinType> in_bucket_window_, GPUArray <int> bucket_buf_, GPUArray <int> new_bucket_buf_,
        int& window_bucket_buf_size_,
        uint32_t old_edge_num, uint32_t new_edge_num)
    {
        static bool shrink_first_time = true;
        if (shrink_first_time) { //shrink first time, allocate the buffers
            shrink_first_time = false;
            Timer alloc_timer;
            new_adj.initialize("New Adj", gpu, new_edge_num * 2, 0);
            new_eid.initialize("New EID", gpu, new_edge_num * 2, 0);
            new_offset.initialize("New Row Pointer", unified, (n + 1), 0);

            edge_deleted.initialize("Edge deleted", gpu, old_edge_num * 2, 0);
            Log(info, "Shrink Allocation Time: %.9lfs", alloc_timer.elapsed());
        }


        /*2. construct new CSR (offsets, adj) and rebuild the eid*/
        int block_size = 128;
        // Attention: new_offset gets the histogram.
        execKernel(warp_detect_deleted_edges, GRID_SIZE, block_size, true,
            rowPtr.gdata(), n, bmp.eid.gdata(), processed.gdata(), new_offset.gdata(), edge_deleted.gdata());

        uint total = CUBScanExclusive<uint, uint>(new_offset.gdata(), new_offset.gdata(), n);
        new_offset.gdata()[n] = total;
        assert(total == new_edge_num * 2);
        cudaDeviceSynchronize();

        swap_ele(rowPtr.gdata(), new_offset.gdata());

        /*new adj and eid construction*/
        CUBSelect(colInd.gdata(), new_adj.gdata(), edge_deleted.gdata(), old_edge_num * 2);
        CUBSelect(bmp.eid.gdata(), new_eid.gdata(), edge_deleted.gdata(), old_edge_num * 2);

        swap_ele(colInd.gdata(), new_adj.gdata());
        colInd.N = new_adj.N;

        swap_ele(bmp.eid.gdata(), new_eid.gdata());
        bmp.eid.N = new_eid.N;

        m = new_edge_num * 2;
    }
    //
    //    /*
    //     * Shrink the adj, eid, EdgeSupport and edge_list arrays
    //     * */
    //    void PKT_Shrink_all(
    //        graph_t& g, bool*& processed,
    //        int*& EdgeSupport, uint*& edge_offset_origin, CUDA_Edge*& edge_list,
    //        int*& new_EdgeSupport, uint*& new_edge_offset_origin, CUDA_Edge*& new_edge_list,
    //        bool*& reversed_processed, bool*& edge_deleted, uint*& scanned_processed,
    //        uint*& new_offset, uint*& new_eid, uint*& new_adj,
    //        InBucketWinType*& in_bucket_window_, uint*& bucket_buf_, uint*& new_bucket_buf_,
    //        uint32_t& window_bucket_buf_size_,
    //        uint32_t old_edge_num, uint32_t new_edge_num,
    //        ZLCUDAMemStat* mem_stat, ZLCUDATimer* time_stat) {
    //        static bool shrink_first_time = true;
    //        if (shrink_first_time) { //shrink first time, allocate the buffers
    //            shrink_first_time = false;
    //            Timer alloc_timer;
    //            ZLCudaMalloc(&new_EdgeSupport, sizeof(int) * new_edge_num, mem_stat);
    //            ZLCudaMalloc(&new_edge_offset_origin, sizeof(uint) * new_edge_num, mem_stat);
    //            ZLCudaMalloc(&new_edge_list, sizeof(CUDA_Edge) * new_edge_num, mem_stat);
    //            ZLCudaMalloc(&new_offset, sizeof(uint) * (g.n + 1), mem_stat);
    //            ZLCudaMalloc(&new_adj, sizeof(uint) * new_edge_num * 2, mem_stat);
    //            ZLCudaMalloc(&new_eid, sizeof(uint) * new_edge_num * 2, mem_stat);
    //            ZLCudaMalloc(&new_bucket_buf_, sizeof(uint) * new_edge_num, mem_stat);
    //
    //            ZLCudaMalloc(&reversed_processed, sizeof(bool) * old_edge_num, mem_stat);
    //            ZLCudaMalloc(&edge_deleted, sizeof(bool) * old_edge_num * 2, mem_stat);
    //            ZLCudaMalloc(&scanned_processed, sizeof(uint) * old_edge_num, mem_stat);
    //            log_info("Shrink Allocation Time: %.9lfs", alloc_timer.elapsed());
    //        }
    //        auto num_obj = mem_stat->get_num_obj();
    //
    //        /*1. construct new edge list and edge support array*/
    //        int grid_size_reverse_bits = (old_edge_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
    //        execKernel(reverse_bits, grid_size_reverse_bits, BLOCK_SIZE, time_stat, true, processed, reversed_processed,
    //            old_edge_num);
    //
    //        /*write to new arrays*/
    //        CUBSelect(EdgeSupport, new_EdgeSupport, reversed_processed, old_edge_num, time_stat, mem_stat);
    //        CUBSelect(edge_offset_origin, new_edge_offset_origin, reversed_processed, old_edge_num, time_stat, mem_stat);
    //        CUBSelect(edge_list, new_edge_list, reversed_processed, old_edge_num, time_stat, mem_stat);
    //
    //        swap_ele(EdgeSupport, new_EdgeSupport);
    //        swap_ele(edge_list, new_edge_list);
    //        swap_ele(edge_offset_origin, new_edge_offset_origin);
    //
    //        /*2. construct new CSR (offsets, adj) and rebuild the eid*/
    //        int block_size = 128;
    //        execKernel(warp_detect_deleted_edges, GRID_SIZE, block_size, time_stat, true, g.num_edges, g.n, g.eid, processed,
    //            new_offset, edge_deleted);
    //
    //        uint total = CUBScanExclusive(new_offset, new_offset, g.n, time_stat, mem_stat);
    //        new_offset[g.n] = total;
    //
    //        cudaDeviceSynchronize();
    //        swap_ele(g.num_edges, new_offset);
    //
    //        /*new adj and eid construction*/
    //        CUBSelect(g.adj, new_adj, edge_deleted, old_edge_num * 2, time_stat, mem_stat);
    //        CUBSelect(g.eid, new_eid, edge_deleted, old_edge_num * 2, time_stat, mem_stat);
    //        swap_ele(g.adj, new_adj);
    //
    //        CUBScanExclusive(processed, scanned_processed, old_edge_num, time_stat, mem_stat);
    //
    //        int grid_size_update_eid = (new_edge_num * 2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    //        execKernel(update_eid, grid_size_update_eid, BLOCK_SIZE, time_stat, false, new_eid, scanned_processed,
    //            new_edge_num * 2); //do the edge mapping
    //        swap_ele(g.eid, new_eid);
    //
    //#ifndef LEGACY_SCAN
    //        /* Updated: new bucket construction*/
    //        if (window_bucket_buf_size_ > 0) {
    //            auto grid_size = (window_bucket_buf_size_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    //            execKernel(filter_not_processed, grid_size, BLOCK_SIZE, time_stat, false,
    //                bucket_buf_, window_bucket_buf_size_, reversed_processed, processed); //do the edge mapping
    //            window_bucket_buf_size_ = CUBSelect(bucket_buf_, new_bucket_buf_,
    //                reversed_processed, window_bucket_buf_size_, time_stat, mem_stat);
    //
    //            execKernel(update_eid, grid_size, BLOCK_SIZE, time_stat, false,
    //                new_bucket_buf_, scanned_processed, window_bucket_buf_size_); //do the edge mapping
    //            swap_ele(bucket_buf_, new_bucket_buf_);
    //
    //            checkCudaErrors(cudaMemset(in_bucket_window_, 0, sizeof(InBucketWinType) * new_edge_num));
    //            execKernel(random_access_set, grid_size, BLOCK_SIZE, time_stat, false,
    //                bucket_buf_, window_bucket_buf_size_, in_bucket_window_, InBucketTrue);
    //        }
    //        else {
    //            checkCudaErrors(cudaMemset(in_bucket_window_, 0, sizeof(InBucketWinType) * new_edge_num));
    //        }
    //#endif
    //        /*3. update the processed array*/
    //        checkCudaErrors(cudaMemset(processed, 0, sizeof(bool) * new_edge_num));
    //
    //        g.m = new_edge_num * 2;
    //
    //        assert(num_obj == mem_stat->get_num_obj()); //the mem object num should be the same
    //    }


    double set_inter_time = 0;
    double process_update_time = 0;

    void PKT_LevelZeroProcess(
        GPUArray<int> curr, int curr_cnt, GPUArray<bool>& inCurr,
        GPUArray<bool>& processed) {
        int block_size = 256;
        int grid_size = (curr_cnt + block_size - 1) / block_size;
        execKernel(
            update_processed, grid_size, block_size, false,
            curr.gdata(), curr_cnt, inCurr.gdata(), processed.gdata());
    }

    void PKT_SubLevelProcess(
        BmpGpu<uint> bmp, GPUArray<uint> edge_off_origin, uint32_t eids_cnt, int level,
        GPUArray <uint>& rowPtr, GPUArray <uint>& colIdn,
        GPUArray<int>& curr, int curr_cnt, GPUArray<bool>& inCurr,
        GPUArray<int>& next, GPUArray<int>& next_cnt, GPUArray<bool>& inNext,
        GPUArray<bool>& processed, GPUArray<InBucketWinType>& in_bucket_window_,
        GPUArray <int>& bucket_buf_, int* window_bucket_buf_size_, int& bucket_level_end_) 
    {
        int block_size = 256;
        static int shared_memory_size_per_block = block_size * sizeof(int) * 2 * 3;
        int grid_size = curr_cnt;  //each block process an edge
        Timer timer;

        /* 1st: Peeling */
        
        execKernelDynamicAllocation(
            sub_level_process, grid_size, block_size,
            shared_memory_size_per_block, false,
            level, curr.gdata(), curr_cnt, inCurr.gdata(),
            next.gdata(), next_cnt.gdata(), inNext.gdata(),
            rowPtr.gdata(), colIdn.gdata(), bmp.eid.gdata(),
            bmp.idToEdge.gdata(), bmp.edge_sup_gpu.gdata(), processed.gdata(),
            in_bucket_window_.gdata(), bucket_buf_.gdata(), window_bucket_buf_size_, bucket_level_end_);

        set_inter_time += timer.elapsed_and_reset();

        /* 2nd: Update the processed flags */
        grid_size = (curr_cnt + block_size - 1) / block_size;
        execKernel(
            update_processed, grid_size, block_size, false,
            curr.gdata(), curr_cnt, inCurr.gdata(), processed.gdata());
        process_update_time += timer.elapsed_and_reset();
    }

    void PKT_SubLevelTCBased(
        int n, int& m, GPUArray<uint> rowPtr, GPUArray<uint> colInd, BmpGpu<uint> bmp,
        GPUArray<bool> processed,
        GPUArray<uint> edge_off_origin,
        GPUArray<int> new_EdgeSupport, GPUArray <uint> new_edge_offset_origin, GPUArray<Edge> new_edge_list,
        GPUArray<bool> reversed_processed, GPUArray<bool> edge_deleted, GPUArray <uint> scanned_processed,
        GPUArray <uint> new_offset, GPUArray <uint> new_eid, GPUArray <uint> new_adj,
        uint32_t edge_num, uint32_t todo,
        GPUArray<int> curr, int curr_cnt, GPUArray<bool> inCurr,
        GPUArray<int> next, GPUArray<int> next_cnt, GPUArray<bool> inNext,
        int level,
        int num_words_bmp, int num_words_bmp_idx,
        GPUArray <InBucketWinType> in_bucket_window_, GPUArray <int> bucket_buf_, GPUArray <int> new_bucket_buf_,
        int& window_bucket_buf_size_,
        int bucket_level_end_) 
    {
        auto block_size = 256;
        auto grid_size = (curr_cnt + block_size - 1) / block_size;

        /* Mark Processed */
        execKernel(update_processed, grid_size, block_size, false, curr.gdata(), curr_cnt, inCurr.gdata(), processed.gdata());

        /* Shrink Edge Lists, CSR and update eid/edge_off_origin mappings */
        #ifdef SHRINK_ALL
        PKT_Shrink_all(g_cuda, processed, EdgeSupport, edge_off_origin, edge_list,
        #else
        ShrinkCSREid(n, m, rowPtr,colInd,bmp, processed, edge_off_origin,
        #endif
            new_edge_list,
            reversed_processed, edge_deleted, scanned_processed,
            new_offset, new_eid, new_adj,
            in_bucket_window_, bucket_buf_, new_bucket_buf_, window_bucket_buf_size_,
            edge_num, todo);

#ifndef DISABLE_BSR
        // 2nd: BSRs.
       /* ZLCudaMalloc(&bmp_offs, sizeof(uint32_t) * (g_cuda.n + 1), mem_stat);
        execKernel(construct_bsr_row_ptr_per_thread, (g_cuda.n + 127) / 128, 128,
            time_stat, true, g_cuda.num_edges, g_cuda.adj, g_cuda.n, bmp_offs);
        auto word_num = CUBScanExclusive(bmp_offs, bmp_offs, g_cuda.n, time_stat, mem_stat);
        bmp_offs[g_cuda.n] = word_num;
        log_info("Word Num: %d", word_num);
        ZLCudaMalloc(&bmp_word_indices, sizeof(bmp_word_idx_type) * word_num, mem_stat);
        ZLCudaMalloc(&bmp_words, sizeof(bmp_word_type) * word_num, mem_stat);
        execKernel(construct_bsr_content_per_thread, (g_cuda.n + 127) / 128, 128,
            time_stat, true, g_cuda.num_edges, g_cuda.adj, g_cuda.n, bmp_offs, bmp_word_indices, bmp_words);*/
#endif

        /* TC-based Support Updates */
        block_size = 1024;
        dim3 t_dimension(WARP_SIZE, block_size / WARP_SIZE); /*2-D*/
        bmp.d_vertex_count.setAll(0, true);

#ifndef DISABLE_BSR
        execKernelDynamicAllocation(
            bmp_bsr_update_next,
            n, t_dimension,
            num_words_bmp_idx * sizeof(uint32_t), false,
            rowPtr.gdata(), colInd.gdata(), bmp.d_bitmaps.gdata(), bmp.d_bitmap_states.gdata(),
            bmp.d_vertex_count.gdata(), bmp.conc_blocks_per_SM, bmp.eid.gdata(), bmp.edge_sup_gpu.gdata(),
            num_words_bmp, num_words_bmp_idx,
            bmp.bmp_offs.gdata(), bmp.bmp_word_indices, bmp.bmp_words,
            level, next.gdata(), next_cnt.gdata(), inNext.gdata(),
            in_bucket_window_.gdata(), bucket_buf_.gdata(), &window_bucket_buf_size_, bucket_level_end_);
#else
        execKernelDynamicAllocation(
            bmp_update_next,
            g_cuda.n, t_dimension,
            num_words_bmp_idx * sizeof(uint32_t), time_stat, true,
            g_cuda.num_edges, g_cuda.adj, d_bitmaps, d_bitmap_states,
            vertex_count, conc_blocks_per_SM, g_cuda.eid, EdgeSupport,
            num_words_bmp, num_words_bmp_idx,
            level, next, next_cnt, inNext,
            in_bucket_window_, bucket_buf_, &window_bucket_buf_size_, bucket_level_end_);
#endif
    }


    //void InitBMPsBSRs(graph_t& g_cuda, uint32_t*& d_bitmaps, uint32_t*& d_bitmap_states, uint32_t*& d_vertex_count,
    //    uint32_t*& bmp_offs, bmp_word_idx_type*& bmp_word_indices, bmp_word_type*& bmp_words,
    //    ZLCUDAMemStat* mem_stat, ZLCUDATimer* time_stat) 
    //{
    //    // 1st: BMPs.
    //    InitBMP(&g_cuda, d_bitmaps, d_bitmap_states, d_vertex_count, mem_stat);
    //}

    void PrepareCSRELEidQueues(int n, int m,
        GPUArray<uint> rowPtr, GPUArray<uint> colInd, BmpGpu<uint> bmp,
        GPUArray<int>& next_cnt, GPUArray<int>& curr, GPUArray<bool>& inCurr, GPUArray<int>& next, GPUArray<bool>& inNext, GPUArray<bool>& processed,
        uint* edge_off_origin_cpu, GPUArray<uint>& edge_off_origin, GPUArray<uint>& identity_arr_asc) 
    {
        // 1st: CSR/Eid/Edge List. --> not necessary
      
        uint32_t edge_num = m / 2;

        // 2nd: Processed.
        processed.initialize("Processed?", AllocationTypeEnum::gpu, edge_num, 0);

        // 3rd: Queue Related.
       
        next_cnt.initialize("Next Count", AllocationTypeEnum::unified, 1, 0);
        curr.initialize("Curr", AllocationTypeEnum::gpu, edge_num, 0);

        next.initialize("Next", AllocationTypeEnum::gpu, edge_num, 0);
        inCurr.initialize("In Curr", AllocationTypeEnum::gpu, edge_num, 0);
        inNext.initialize("in Next", AllocationTypeEnum::gpu, edge_num, 0);

        // 4th: Keep the edge offset mapping.
        long grid_size = (edge_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
       
        edge_off_origin.initialize("Edge off Origin", AllocationTypeEnum::gpu, edge_num, 0);

        if (edge_off_origin_cpu != nullptr) {
            //cudaMemcpy(edge_off_origin, edge_off_origin_cpu, sizeof(uint) * edge_num, cudaMemcpyHostToDevice);
        }
        else {
            execKernel(init_asc, grid_size, BLOCK_SIZE, false, edge_off_origin.gdata(), edge_num);
        }
        // 5th: Introduce identity_arr_asc for the CUB Select invocations.
       
        identity_arr_asc.initialize("Identity Array Asc", AllocationTypeEnum::gpu, edge_num, 0);

        execKernel(init_asc, grid_size, BLOCK_SIZE, false, identity_arr_asc.gdata(), edge_num);
    }

    void PrepareBucket(GPUArray<InBucketWinType>& in_bucket_window_,
        GPUArray<int>& bucket_buf_, GPUArray<int>& window_bucket_buf_size_, int todo)
    {
        in_bucket_window_.initialize("In bucket window", gpu, (todo + sizeof(long long)), 0);
        bucket_buf_.initialize("Bucket Buffer", gpu, todo, 0);
        window_bucket_buf_size_.initialize("Window Bucket Buffer Size", unified, 1, 0);
    }

    void PKT_cuda(
        int n, int m,
        GPUArray<uint> rowPtr, GPUArray<uint> colInd, BmpGpu<uint> bmp,
        uint* edge_off_origin_cpu, int shrink_factor,
        GPUArray<int> output, uint* level_start_pos, int level, double tc_time)
    {
        Timer scan_timer, sub_process_timer, copy_timer, tc_timer, shrink_timer, prepare_timer, completeTimer;
        double scan_time = 0, sub_process_time = 0, copy_time = 0, shrink_time = 0, prepare_time = 0, penalty_tc_time = 0, completeTime;
        // 1st: Prepare CSR/EL/Eid/Queues.
      

        completeTimer.reset();

        GPUArray<int> curr, next;
        GPUArray<bool> inCurr, inNext;
        GPUArray<int> curr_cnt_ptr, next_cnt;
        GPUArray<bool> processed;
        GPUArray<uint> edge_off_origin, identity_arr_asc;

        curr_cnt_ptr.initialize("Curr Count Pointer", AllocationTypeEnum::unified, 1, 0);
        int*& curr_cnt = curr_cnt_ptr.gdata();

        PrepareCSRELEidQueues(n, m,
            rowPtr, colInd, bmp,
            next_cnt, curr, inCurr, next, inNext, processed,
            edge_off_origin_cpu, edge_off_origin, identity_arr_asc);
        uint32_t edge_num = m / 2;

        /* 2nd: Prepare for double buffered: CSR/EL/Eid/ES/offset-mapping/auxiliaries */
        GPUArray <uint> new_offset;
        GPUArray <uint> new_adj;
        GPUArray <Edge> new_edge_list;
        GPUArray <uint> new_eid;
        GPUArray<int> new_EdgeSupport;
        GPUArray <uint> new_edge_offset_origin;
        GPUArray<bool> reversed_processed;     // Auxiliaries for shrinking graphs.
        GPUArray<bool> edge_deleted;           // Auxiliaries for shrinking graphs.
        GPUArray <uint> scanned_processed;     // Auxiliaries for shrinking graphs.

        

        // Init Buckets.
        // Bucket Related.
        int bucket_level_end_ = level;
        GPUArray <InBucketWinType> in_bucket_window_;
        GPUArray <int> bucket_buf_;
        GPUArray <int> new_bucket_buf_;
        GPUArray<int> window_bucket_buf_size_; //should be uint* only

        #ifndef LEGACY_SCAN
        PrepareBucket(in_bucket_window_, bucket_buf_, window_bucket_buf_size_, edge_num);
        #endif

        // 3rd: Init Triangle-Counting-Based Support Update Data Structures (BMPs and BSRs).
        //Done in bmptc

        // 4th: Init Others.
        double shrink_kernel_time = 0;
        auto todo = edge_num;
        const auto todo_original = edge_num;
        auto deleted_acc = 0;
        auto shrink_cnt = 0;
        bool shrink_first_time = true;  //if true, the identity_arr_asc array should not be freed since it is shared with edge_off_origin
        cudaDeviceSynchronize();

        vector<pair<int, double>> tc_stat;
        vector<pair<int, double>> shrink_stat;
        //Begin of Level-Processing, finding edges in k-truss but not in the (k+1)-truss.
        while (todo > 0) 
        {
            Log(LogPriorityEnum::debug, "Level: %d, todo(origin): %d, todo(cur): %d., have: %d", level, todo_original, todo,
                level_start_pos[level]);
            // 1st: Shrinking.
            if ((deleted_acc * 1.0 / todo_original) > (1.0 / shrink_factor)) 
            { //need to shrink the graph

                shrink_timer.reset();
                #ifdef SHRINK_ALL
                PKT_Shrink_all(g_cuda, processed,
                #else
                ShrinkCSREid(n, m,rowPtr, colInd, bmp, processed,
                #endif
                    edge_off_origin,
                    new_edge_list, reversed_processed, edge_deleted, scanned_processed,
                    new_offset, new_eid, new_adj,
                    in_bucket_window_, bucket_buf_, new_bucket_buf_, *window_bucket_buf_size_.gdata(),
                    edge_num, todo);
                
                edge_num = todo;
                auto temp_shrink_time = shrink_timer.elapsed();

                shrink_stat.emplace_back(level, temp_shrink_time);
                shrink_time += temp_shrink_time;
                shrink_cnt++;
                deleted_acc = 0;

                #ifdef SHRINK_ALL
                CUDA_RUNTIME(cudaFree(identity_arr_asc, mem_stat);
                identity_arr_asc = nullptr;
                ZLCudaMalloc(&identity_arr_asc, sizeof(uint) * edge_num, mem_stat); //now the edge_num is changed.
                auto grid_size = (edge_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
                execKernel(init_asc, grid_size, BLOCK_SIZE, time_stat, false, identity_arr_asc, edge_num);
                #endif
                shrink_first_time = false;
                Log(LogPriorityEnum::debug, "Shrink graph finished");
            }
            cudaDeviceSynchronize();

            // 2nd: Frontier Generation.
            scan_timer.reset();
            PKT_Scan(bmp.edge_sup_gpu,
                #ifdef SHRINK_ALL
                edge_num,
                #else
                todo_original,
                #endif
                level, curr, inCurr, *curr_cnt, identity_arr_asc,
                in_bucket_window_, bucket_buf_, window_bucket_buf_size_.gdata(), bucket_level_end_);
            cudaDeviceSynchronize();
            scan_time += scan_timer.elapsed();

            // 3rd: Iterative Sub-Level Processing.
            int level_acc_cnt = 0;
            while (*curr_cnt > 0) 
            {
                /* 1st: Copy this iteration results (edges to be marked as processed) to the output array */
                copy_timer.reset();
                todo -= *curr_cnt;
                deleted_acc += *curr_cnt;
                auto grid_size = (*curr_cnt + BLOCK_SIZE - 1) / BLOCK_SIZE;
                execKernel(output_edge_support, grid_size, BLOCK_SIZE, false, output.gdata(), curr.gdata(), *curr_cnt,
                    edge_off_origin.gdata(), level_start_pos[level] + level_acc_cnt);
                level_acc_cnt += *curr_cnt;
                copy_time += copy_timer.elapsed();
                // No need to process the last level with PKT_SubLevelProcess.
                if (0 == todo) {
                    break;
                }

                /* 2nd: Sub-Level Processing... */
                *next_cnt.gdata() = 0;
                sub_process_timer.reset();
                cudaDeviceSynchronize();
                if (level == 0) {
                    PKT_LevelZeroProcess(curr, *curr_cnt, inCurr, processed);
                }
                else {
                    size_t task_size = *curr_cnt * (size_t)(level + 1);
                    size_t left_edge_size = todo;
                    double estimated_tc_time = left_edge_size / (m / 2.0) * tc_time + penalty_tc_time;
                    double estimated_process_throughput = 4.0 * pow(10, 9);
                    double estimated_peel_time = task_size / estimated_process_throughput;
                    if (estimated_tc_time > estimated_peel_time) 
                    {
                        //                if (true) {
                        PKT_SubLevelProcess(bmp, edge_off_origin,
                            edge_num, level,
                            rowPtr, colInd,
                            curr, *curr_cnt, inCurr,
                            next, next_cnt, inNext,
                            processed,
                            in_bucket_window_, bucket_buf_, window_bucket_buf_size_.gdata(), bucket_level_end_);
                    }
                    else {
                        shrink_first_time = false;
                        tc_timer.reset();
                        const uint32_t elem_bits = sizeof(uint32_t) * 8; /*#bits in a bitmap element*/
                        const uint32_t num_words_bmp = (n + elem_bits - 1) / elem_bits;
                        const uint32_t num_words_bmp_idx = (num_words_bmp + BITMAP_SCALE - 1) / BITMAP_SCALE;
                        PKT_SubLevelTCBased(n, m, rowPtr, colInd, bmp, 
                            processed,
                            edge_off_origin,
                            new_EdgeSupport, new_edge_offset_origin, new_edge_list,
                            reversed_processed, edge_deleted, scanned_processed,
                            new_offset, new_eid, new_adj,
                            edge_num, todo, curr, *curr_cnt, inCurr, next, next_cnt, inNext,
                            level,
                            num_words_bmp, num_words_bmp_idx,
                            in_bucket_window_, bucket_buf_, new_bucket_buf_, *window_bucket_buf_size_.gdata(),
                            bucket_level_end_);
                        auto cost = tc_timer.elapsed();
                        if (estimated_tc_time * 1.2 < cost) {
                            penalty_tc_time += cost - estimated_tc_time;
                            Log(LogPriorityEnum::info, "Penalty TC-Time: %.9lfs", penalty_tc_time);
                        }
                        tc_stat.emplace_back(level, cost);
                        Log(LogPriorityEnum::info, "TC time: %.9lfs", cost);
                        edge_num = todo;
                        shrink_cnt++;
                        deleted_acc = 0;
                    }
                }
                cudaDeviceSynchronize();

                //            log_info("curr_cnt: %d, next_cnt: %d, todo: %d.", curr_cnt, *next_cnt, todo);
                swap(curr, next);
                swap(inCurr, inNext);
                *curr_cnt = *next_cnt.gdata();
                sub_process_time += sub_process_timer.elapsed();
            }

            level_start_pos[level + 1] = level_start_pos[level] + level_acc_cnt;
            level++;
        }

        completeTime = completeTimer.elapsed();

//
          cudaDeviceSynchronize();
          processed.freeGPU();
          next_cnt.freeGPU();

          curr.freeGPU();
          next.freeGPU();

          inCurr.freeGPU();
          inNext.freeGPU();

          identity_arr_asc.freeGPU();

        if (!shrink_first_time) { //if false, the identity_arr_asc and edge_off_origin point to different array


            edge_off_origin.freeGPU();
            //new_EdgeSupport.freeGPU();
            //new_edge_offset_origin.freeGPU();
            //new_edge_list.freeGPU();
            new_offset.freeGPU();
            new_adj.freeGPU();
            new_eid.freeGPU();
            edge_deleted.freeGPU();

        }
        Log(LogPriorityEnum::info, "Prepare CPU time: %.4f s.", prepare_time);
        Log(LogPriorityEnum::info,"Scan CPU time: %.4f s.", scan_time);

        Log(LogPriorityEnum::info,"Shrink kernel time: %.4f s.", shrink_kernel_time * 1.0 / 1000);
        Log(LogPriorityEnum::info,"Shrink CPU time: %.4f s.", shrink_time);
        Log(LogPriorityEnum::info,"Shrink cnt: %d.", shrink_cnt);
        std::stringstream ss;
        //ss << shrink_stat;
        Log(LogPriorityEnum::info,"Shrink stat: %s.", ss.str().c_str());

        Log(LogPriorityEnum::info,"Sub process CPU time: %.4f s.", sub_process_time);
        Log(LogPriorityEnum::info,"Copy CPU time: %.4f s.", copy_time);
        Log(LogPriorityEnum::info,"Set Intersection time: %.4f s.", set_inter_time);
        std::stringstream ss2;
        //ss2 << tc_stat;
        Log(LogPriorityEnum::info,"TC stat: %s", ss2.str().c_str());
        Log(LogPriorityEnum::info,"Update processed time: %.4f s.", process_update_time);

        Log(LogPriorityEnum::info, "Complete Time: %.4f s.", completeTime);

    }
};