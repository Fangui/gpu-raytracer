#include "triangle_gpu.hh"

struct Triangle_gpu *init_triangle_gpu(const Triangle &triangle)
{
    Triangle_gpu *triangle_gpu;
    cudaMalloc(&triangle_gpu, sizeof(struct Triangle_gpu));
    cudaMemcpy(triangle_gpu, &triangle, sizeof(struct Triangle_gpu), cudaMemcpyHostToDevice);
    return triangle_gpu;
}

void intersect(struct Triangle_gpu *d_tri, const Ray *ray, bool *is_intersected)
{
    Vector_gpu d_vertex0 = *d_tri->vertices[0];
    Vector_gpu d_vertex1 = *d_tri->vertices[1];
    Vector_gpu d_vertex2 = *d_tri->vertices[2];

    Vector_gpu edge1 = vertex1 - vertex0;
    Vector_gpu edge2 = vertex2 - vertex0;
    Vector_gpu h = ray->dir.cross_product(edge2);

    float det = edge1.dot_product(h);
    if (det > -EPSILON && det < EPSILON)
        return false;    // This ray is parallel to this triangle.
    float f = 1.f / det;
    Vector s = ray.o - vertex0;
    float u = f * (s.dot_product(h));

    if (u < 0.0 || u > 1.0)
        return false;

    s = s.cross_product_inplace(edge1);
    float v = f * (ray.dir.dot_product(s));
    if (v < 0.0 || u + v > 1.0)
        return false;

    // At this stage we can compute t to find out where the intersection point is on the line.
    float t = f * edge2.dot_product(s);
    if (t > EPSILON) // ray intersection
    {
        ray.u = u;
        ray.v = v;
        dist = t;
        return true;
    }
    return false;
}