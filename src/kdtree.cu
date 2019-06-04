#include "kdtree_gpu.hh"
#include "light.hh"
#include "material.hh"

__device__ bool is_inside(const float *box, const Ray &ray)
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

__device__ Pixel direct_light(const KdNodeGpu *root, Ray &r, Material *materials,
                               Vector *a_light, Light *d_lights, size_t d_lights_len)
{
    const KdNodeGpu *stack[64];
    stack[0] = root;

    size_t idx = 1;
    float dist = -1;

    do
    {
        const KdNodeGpu *node = stack[--idx];
        if (is_inside(node->box, r))
        {
            for (Triangle *tri = node->beg; tri < node->end; ++tri)
            {
                if (tri->intersect(r, dist)) // dist u v up to date in intersect
                    r.tri = tri;
            }

            if (node->left != nullptr)
                stack[idx++] = node->left;
            if (node->right != nullptr)
                stack[idx++] = node->right;
        }
    } while (idx);

    if (dist != -1)
    {
        Vector normal = r.tri->normal[0] * (1 - r.u - r.v)
                      + r.tri->normal[1] * r.u + r.tri->normal[2] * r.v;
        normal.norm_inplace();

        Vector color = *a_light * materials[r.tri->id].ka;
        for (size_t i = 0; i < d_lights_len; ++i)
        {
            auto contrib = (d_lights[i].dir * -1).dot_product(normal);
            if (contrib > 0)
                color += d_lights[i].color * contrib;
        }
        auto &mat = materials[r.tri->id];
        color *= mat.kd;

        auto pix = Pixel(color);
        return pix;
    }
    return Pixel(0, 0, 0);
}
