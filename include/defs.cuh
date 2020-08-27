#pragma once

typedef uint64_t EncodeDataType;

typedef std::chrono::system_clock::time_point timepoint;
typedef unsigned int uint;
typedef uint wtype;
typedef std::pair<uint, uint> Edge;
typedef std::tuple<uint, uint, wtype> WEdge;

typedef std::vector<Edge> EdgeList;
typedef std::vector<WEdge> WEdgeList;

template <typename NodeTy> using EdgeTy = std::pair<NodeTy, NodeTy>;
template <typename NodeTy, typename WeightTy> using WEdgeTy = std::tuple<NodeTy, NodeTy, WeightTy>;


#define  BCTYPE bool

#define BITMAP_SCALE_LOG (9)
#define BITMAP_SCALE (1<<BITMAP_SCALE_LOG)  /*#bits in the first-level bitmap indexed by 1 bit in the second-level bitmap*/


#define __VERBOSE__

typedef unsigned long long int uint64;

//Enums
enum ProcessingElementEnum { Thread, Warp, Block, Grid, WarpShared, Test, Queue };
enum AllocationTypeEnum { cpuonly, gpu, unified, zerocopy };
enum LogPriorityEnum { critical, warn, error, info, debug, none };

