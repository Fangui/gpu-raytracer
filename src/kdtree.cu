#include "kdtree_gpu.hh"

__device__ bool is_inside(float *box, const Ray &ray)
{
    const Vector &origin = ray.o;
    float tmin = (box[ray.sign[0]] - origin[0]) * ray.inv[0];
    float tmax = (box[1 - ray.sign[0]] - origin[0]) * ray.inv[0];

    float tymin = (box[2 + ray.sign[1]] - origin[1]) * ray.inv[1];
    float tymax = (box[3 - ray.sign[1]] - origin[1]) * ray.inv[1];

    if (tmin > tymax || tymin > tmax)
        return false;

    if (tymin > tmin)
        tmin = tymin;

    if (tymax < tmax)
        tmax = tymax;

    float tzmin = (box[4 + ray.sign[2]] - origin[2]) * ray.inv[2];
    float tzmax = (box[5 - ray.sign[2]] - origin[2]) * ray.inv[2];

    if (tmin > tzmax || tzmin > tmax)
        return false;

    return true;
}

#include <stdio.h>
__device__ void search(struct KdNodeGpu *root, Ray &r, float *dist)
{
    // if inside box 

    KdNodeGpu *stack[64];
    stack[0] = root;

    size_t idx = 1;

    do
    {
        KdNodeGpu *node = stack[--idx];
        if (is_inside(node->box, r))
        {

           // bool has_left = (node->left != nullptr);
           // bool has_right = (node->right != nullptr);

            bool inter = false;

            for (Triangle_gpu *tri = node->beg; tri < node->end; ++tri)
            {
                float t;
                intersect(tri, &r, &t, &inter);
                if (inter)
                {
                    if (*dist == -1 || *dist > t)
                    {
                        *dist = t;
                        //r.tri = tri;
                    }
                }
            }

            if (node->left != nullptr)
            {
                stack[idx++] = node->left;
                stack[idx++] = node->right;
            }
        }
    } while (idx);
}
