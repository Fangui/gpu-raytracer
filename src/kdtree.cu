#include "kdtree_gpu.hh"


__global__ void search(struct KdNodeGpu *root, Ray *r, float *dist)
{
    // if inside box 

    KdNodeGpu *stack[64];
    stack[0] = root;

    KdNodeGpu *node = root;
    size_t idx = 1;

    do
    {
        bool has_left = (node->left == nullptr);
        bool has_right = (node->right == nullptr);

        bool intersect = false;
        for (Triangle *tri = node->beg; !intersect && tri < node->end; ++tri)
        {
            // check intersect
        }

        if (!has_left || !has_right) // child
        {
            node = stack[--idx];
        }
        else
        {
            node = has_left ? node->left : node->right;
            if (has_left && has_right)
                stack[idx++] = node->right;
        }
    } while (idx > 1);
}
