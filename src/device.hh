#pragma once

#include "kdtree.hh"
#include "kdtree_gpu.hh"

KdNodeGpu* upload_kd_tree(const KdTree& kd_tree);
