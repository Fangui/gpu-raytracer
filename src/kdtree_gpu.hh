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

__device__ Vector direct_light(struct KdNodeGpu *root, Ray &r, Material *materials,
                       float *dist);
