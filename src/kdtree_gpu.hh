#include "triangle_gpu.hh"

struct KdNodeGpu
{
    KdNodeGpu *left;
    KdNodeGpu *right;

    float box[6];

    Triangle_gpu *beg;
    Triangle_gpu *end;
};

__device__ void search(struct KdNodeGpu *root, Ray &r, float *dist);
