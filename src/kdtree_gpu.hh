#pragma once

#include "triangle.hh"

struct KdNodeGpu
{
    KdNodeGpu *left;
    KdNodeGpu *right;

    float box[6];

    Triangle *beg;
    Triangle *end;
};

struct Material;
struct Light;

__device__ Pixel direct_light(const struct KdNodeGpu *root, Ray &r, Material *materials,
                               Vector *a_light, Light *d_lights, size_t d_lights_len);
