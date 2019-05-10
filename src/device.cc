#include <cuda_runtime.h>

#include <err.h>

#include "kdtree.hh"
#include "kdtree_gpu.hh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
static inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
        err(code, "CUDA: %s %s %d\n", cudaGetErrorString(code), file, line);
}

static KdNodeGpu* upload_kd_node(const KdTree::childPtr& kd_node)
{
    if (!kd_node)
        return NULL;

    KdNodeGpu* node;
    if (cudaMalloc(&node, sizeof(*node)) != cudaSuccess)
        errx(1, "cudaMalloc failed");

    node->left = upload_kd_node(kd_node->left);
    node->right = upload_kd_node(kd_node->right);
    if (cudaMemcpy(node->box, kd_node->box, sizeof(node->box), cudaMemcpyHostToDevice) != cudaSuccess)
        errx(1, "cudaMemcpy failed");

    cudaMalloc(&node->beg, sizeof(Triangle) * std::distance(kd_node->beg, kd_node->end));
    if (!node)
        errx(1, "cudaMalloc failed");
}

static KdNodeGpu* upload_kd_tree(const KdTree& kd_tree)
{
    return upload_kd_node(kd_tree.root_);
}

void device_raytracing(const KdTree& kd_tree)
{
    KdNodeGpu* kd_tree_root = upload_kd_tree(kd_tree);
    // TODO: continue
}