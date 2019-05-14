extern "C" {
#include <stdio.h>
}

#include "kdtree.hh"
#include "kdtree_gpu.hh"
#include "triangle.hh"

#define cudaCheckError(ans) gpuAssert((ans), __FILE__, __LINE__)
static inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code == cudaSuccess)
        return;
    fprintf(stderr, "CUDA: %s %s %d\n", cudaGetErrorString(code), file, line);
    exit(code);
}

static size_t upload_kd_node(const KdNodeGpu* nodes, const KdNodeGpu* nodes_gpu, const Triangle* triangles, const Triangle* triangles_gpu, const KdTree::childPtr& kd_node, std::size_t& idx)
{
    if (!kd_node)
        return NULL;

    KdNodeGpu& node = nodes[idx++];
    node.left = nodes_gpu + upload_kd_node(nodes, nodes_gpu, kd_node->left, idx);
    node.right = nodes_gpu + upload_kd_node(nodes, nodes_gpu, kd_node->right, idx);
    memcpy(node.box, kd_node->box, sizeof(node.box));
    cudaCheckError(cudaMalloc(&node.beg, sizeof(Triangle) * len));
    node.beg = reinterpret_cast<Triangle*>(reinterpret_cast<char*>(&*kd_node->beg) - reinterpret_cast<char*>(triangles) + reinterpret_cast<char*>(triangles_gpu));
    node.end = reinterpret_cast<Triangle*>(reinterpret_cast<char*>(&*kd_node->end) - reinterpret_cast<char*>(triangles) + reinterpret_cast<char*>(triangles_gpu));
}

KdNodeGpu* upload_kd_tree(const KdTree& kd_tree, const std::vector<Triangle>& triangles)
{
    std::vector<KdNodeGpu> nodes(kd_tree.nodes_count_);
    KdNodeGpu* nodes_gpu;
    cudaCheckError(cudaMalloc(&nodes_gpu, sizeof(*nodes_gpu) * nodes.size()));
    KdNodeGpu* triangles_gpu;
    cudaCheckError(cudaMalloc(&triangles_gpu, sizeof(*triangles_gpu) * triangles.size()));
    size_t idx = 0;
    create_gpu_kd_node(nodes.data(), nodes_gpu, triangles.data(), triangles_gpu, kd_tree.root_, idx);
    cudaCheckError(cudaMemcpy(nodes_gpu, nodes.data(), sizeof(*nodes_gpu) * nodes.size(), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(triangles_gpu, triangles.data(), sizeof(*triangles_gpu) * triangles.size(), cudaMemcpyHostToDevice));

    return nodes_gpu;
}
