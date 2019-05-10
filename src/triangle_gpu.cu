#include "triangle_gpu.hh"


__global__ void intersect(struct Triangle_gpu *d_tri, Ray *ray, bool *is_intersected)
{
    *is_intersected = false;

    Vector d_vertex0 = d_tri->vertices[0];
    Vector d_vertex1 = d_tri->vertices[1];
    Vector d_vertex2 = d_tri->vertices[2];

    Vector edge1 = d_vertex1 - d_vertex0;
    Vector edge2 = d_vertex2 - d_vertex0;
    Vector h = ray->dir.cross_product(edge2);

    float det = edge1.dot_product(h);
    if (det > -EPSILON && det < EPSILON)
        return;    // This ray is parallel to this triangle.
    float f = 1.f / det;
    Vector s = ray->o - d_vertex0;
    float u = f * (s.dot_product(h));

    if (u < 0.0 || u > 1.0)
        return;

    s = s.cross_product_inplace(edge1);
    float v = f * (ray->dir.dot_product(s));
    if (v < 0.0 || u + v > 1.0)
        return;

    // At this stage we can compute t to find out where the intersection point is on the line.
    float t = f * edge2.dot_product(s);
    if (t > EPSILON) // ray intersection
    {
        ray->u = u;
        ray->v = v;
        //dist = t;
        *is_intersected = true;
    }
}