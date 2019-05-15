#include "kdtree_gpu.hh"
#include "light.hh"
#include "material.hh"

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

    /*
    if (tmin > tzmax || tzmin > tmax)
        return false;

    return true;*/
    return !(tmin > tzmax || tzmin > tmax);
}

__device__ Pixel direct_light(struct KdNodeGpu *root, Ray &r, Material *materials,
                               Vector *a_light, Light *d_lights, size_t d_lights_len)
{
    KdNodeGpu *stack[64];
    stack[0] = root;

    size_t idx = 1;
    float dist = -1;

    do
    {
        KdNodeGpu *node = stack[--idx];
        if (is_inside(node->box, r))
        {
            for (Triangle *tri = node->beg; tri < node->end; ++tri)
            {
                float t;
                if (tri->intersect(r, t))
                {
                    if (dist == -1 || dist > t)
                    {
                        dist = t;
                        r.tri = *tri;
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

    if (dist != -1)
    {
        Vector light = *a_light;
        for (size_t i = 0; i < d_lights_len; ++i)
        {
            auto contrib = (d_lights[i].dir * -1).dot_product(r.tri.normal[0]);
            if (contrib > 0)
                light += d_lights[i].color * contrib;
        }
        light *= materials[r.tri.id].kd;
        auto pix = Pixel(light);
        return pix;
    }
    return Pixel(0, 0, 0);
}
