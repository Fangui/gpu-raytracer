#pragma once

#include "vector.hh"
#include "triangle.hh"

#define EPSILON 0.00001
#define BIAS    0.001

struct Triangle_gpu
{
    Vector vertices[3];
    Vector normal[3];
    Vector uv_pos[3];
    unsigned char id;
};

__global__
void intersect(struct Triangle_gpu *d_tri, const Ray *ray, bool *is_intersected);
