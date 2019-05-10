#pragma once

#include "kdtree.hh"
#include "kdtree_gpu.hh"

static KdNodeGpu* upload_kd_node(const KdTree::childPtr& kd_node);
