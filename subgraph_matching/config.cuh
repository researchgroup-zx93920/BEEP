#pragma once

// #define IC_COUNT  //make CUTOFF infinity for this to give correct results
// #define TIMER

// #define DEGENERACY
#define DEGREE
// #define LEX

#define BLOCK_SIZE_LD 1024
#define PARTITION_SIZE_LD 32
#define BLOCK_SIZE_HD 32

// #define SCHEDULING true
#define SCHEDULING false

// template <typename T>
// struct SHARED_HANDLE
// {
//     T level_index[DEPTH];
//     T level_count[DEPTH];
//     T level_prec_index[DEPTH];
//     uint64 sg_count;
//     T lvl;
//     T src, srcStart, srcLen, srcSplit, dstIdx;

//     T num_divs_local, *level_offset, *reuse_offset, *encode;
//     T to[BLCOK_DIM_X], newIndex;
// };
