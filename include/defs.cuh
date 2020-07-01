#pragma once

typedef std::chrono::system_clock::time_point timepoint;
typedef unsigned int uint;
typedef uint wtype;
typedef std::pair<uint, uint> Edge;
typedef std::tuple<uint, uint, wtype> WEdge;

typedef std::vector<Edge> EdgeList;
typedef std::vector<WEdge> WEdgeList;

template <typename NodeTy> using EdgeTy = std::pair<NodeTy, NodeTy>;
template <typename NodeTy, typename WeightTy> using WEdgeTy = std::tuple<NodeTy, NodeTy, WeightTy>;
