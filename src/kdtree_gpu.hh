#include "triangle.hh"

struct KdNodeGpu
{
    KdNodeGpu *left;
    KdNodeGpu *right;

    float box[6];

    Triangle *beg;
    Triangle *end;

};

__global__ void search(struct KdNodeGpu *root, Ray *r, float *dist);
