#include "triangle.hh"

struct KdNodeGpu
{

    KdNodeGpu *left;
    KdNodeGpu *right;

    float box[6];

    Triangle *beg;
    Triangle *end;
};
