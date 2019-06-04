#include <algorithm>
#ifdef __GNUC__
#include <parallel/algorithm>
#endif
#include <iostream>
#include <omp.h>

#include "kdtree.hh"

const std::function<bool (const Triangle&, const Triangle&)> func[3] =
{
    [](const Triangle &a, const Triangle &b) { return a.get_mean()[0] < b.get_mean()[0]; },
    [](const Triangle &a, const Triangle &b) { return a.get_mean()[1] < b.get_mean()[1]; },
    [](const Triangle &a, const Triangle &b) { return a.get_mean()[2] < b.get_mean()[2]; }
};

static void get_extremum(float box[6], iterator_v beg,
                                       iterator_v end)
{
    box[0] = beg->vertices[0][0];
    box[1] = beg->vertices[0][0];

    box[2] = beg->vertices[0][1];
    box[3] = beg->vertices[0][1];

    box[4] = beg->vertices[0][2];
    box[5] = beg->vertices[0][2];

    while (beg < end)
    {
        for (unsigned i = 0; i < 3; ++i)
        {
            box[0] = std::min(box[0], beg->vertices[i][0]);
            box[1] = std::max(box[1], beg->vertices[i][0]);

            box[2] = std::min(box[2], beg->vertices[i][1]);
            box[3] = std::max(box[3], beg->vertices[i][1]);


            box[4] = std::min(box[4], beg->vertices[i][2]);
            box[5] = std::max(box[5], beg->vertices[i][2]);
        }
        ++beg;
    }

    for (unsigned i = 0; i < 6; i += 2) // expand bounding box
        box[i] -= 0.01;
    for (unsigned i = 1; i < 6; i += 2)
        box[i] += 0.01;
}

static unsigned get_longest_axis(const float box[6])
{
    float diff_x = box[1] - box[0];
    float diff_y = box[3] - box[2];
    float diff_z = box[5] - box[4];

    if (diff_x > diff_y)
    {
        if (diff_x > diff_z)
            return 0;
        return 2;
    }
    if (diff_y > diff_z)
        return 1;

    return 2;
}

KdTree::KdTree(iterator_v beg, iterator_v end)
{
    root_ = make_child(nodes_count_, beg, end);
}

KdTree::KdNode::KdNode(std::size_t& nodes_count, iterator_v beg, iterator_v end)
{
    unsigned dist = std::distance(beg, end);
    get_extremum(box, beg, end);
    if (dist < 3)
    {
        this->beg = beg;
        this->end = end;
        left = nullptr;
        right = nullptr;
    }
    else
    {
        axis = get_longest_axis(box);
#ifdef __GNUC__
        __gnu_parallel::sort(beg, end, func[axis]);
#else
        sort(beg, end, func[axis]);
#endif

        const iterator_v med = beg + ceil((float)dist / 2.f);

        left = make_child(nodes_count, beg, med);
        right = make_child(nodes_count, med + 1, end);

        this->beg = med;
        this->end = med + 1;
    }
}

bool KdTree::KdNode::inside_box(const Ray &ray) const
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

    if (tmin > tzmax || tzmin > tmax)
        return false;

    return true;
}


bool KdTree::KdNode::search_inter(const Ray &ray) const
{
    if (inside_box(ray))
    {
        for (auto it = beg; it < end; ++it)
        {
            if (it->intersect(ray))
                return true;
        }

        if (left != nullptr && left.get()->search_inter(ray))
            return true;

        if (right != nullptr && right.get()->search_inter(ray))
            return true;
    }
    return false;
}
