#include "sphere_light.hh"

Vector SphereLight::compute_light(const Vector &inter,
        const KdTree &tree,
        float &rat) const
{
    Vector dir = (get_pos() - inter).norm_inplace();

    float dist_obj = (inter - get_pos()).get_dist();
    if (dist_obj > r)
    {
        rat = 0;
        return Vector(0, 0, 0);
    }

    Vector origin = inter + dir * 0.001;
    Ray ray(origin, dir);

    float dist = -1;

    if (tree.search(ray, dist))
    {
        if (dist < dist_obj)  // shadow
        {
            rat = 0;
            return dir;
        }
    }

    // rat = 1 / (4 * M_PI * dist_obj);
    rat = 1 - dist_obj / (float)r;
    rat = pow(rat, 3);

    return dir;
}

