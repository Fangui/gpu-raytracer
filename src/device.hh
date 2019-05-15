#pragma once

#include "kdtree.hh"
#include "kdtree_gpu.hh"
#include "triangle.hh"

KdNodeGpu* upload_kd_tree(const KdTree& kd_tree, std::vector<Triangle>& triangles);
