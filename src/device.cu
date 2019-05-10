extern "C" {
#include <stdio.h>
}

#include "kdtree.hh"
#include "kdtree_gpu.hh"
#include "triangle_gpu.hh"

#define cudaCheckError(ans) gpuAssert((ans), __FILE__, __LINE__)
static inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code == cudaSuccess)
        return;
    fprintf(stderr, "CUDA: %s %s %d\n", cudaGetErrorString(code), file, line);
    exit(code);
}

static KdNodeGpu* upload_kd_node(const KdTree::childPtr& kd_node)
{
    if (!kd_node)
        return NULL;

    KdNodeGpu node;
    node.left = upload_kd_node(kd_node->left);
    node.right = upload_kd_node(kd_node->right);
    memcpy(node.box, kd_node->box, sizeof(node.box));
    auto len = std::distance(kd_node->beg, kd_node->end);
    cudaCheckError(cudaMalloc(&node.beg, sizeof(Triangle_gpu) * len));
    // Trick to get first elem address: we know a contiguous vector is hidden behind node->beg
    auto& first = *kd_node->beg;
    cudaCheckError(cudaMemcpy(node.beg, &first, sizeof(Triangle_gpu) * len, cudaMemcpyHostToDevice));
    
    KdNodeGpu* node_gpu;
    cudaCheckError(cudaMalloc(&node_gpu, sizeof(*node_gpu)));
    cudaCheckError(cudaMemcpy(node_gpu, &node, sizeof(node), cudaMemcpyHostToDevice));

    return node_gpu;
}

KdNodeGpu* upload_kd_tree(const KdTree& kd_tree)
{
    return upload_kd_node(kd_tree.root_);
}

void device_raytracing(const KdTree& kd_tree)
{
    KdNodeGpu* kd_tree_root = upload_kd_tree(kd_tree);
    // TODO: continue
}
