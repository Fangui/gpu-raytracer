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

__device__ Vector direct_light(struct KdNodeGpu *root, Ray &r, Material *materials,
                               Vector *a_light, Light *d_lights, std::size_t d_lights_len);
