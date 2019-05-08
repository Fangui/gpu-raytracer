#include "compute_light.hh"
#include <random>

Vector reflect(const Vector& incident,
                const Vector& normal)
{
    return incident - 2.0 * normal.dot_product(incident) * normal;
}

Vector cast_ray(const Scene &scene,
                Ray &ray, const KdTree &tree)
{
    float dist = -1;
    if (tree.search(ray, dist))
    {
        const auto material = scene.map.at(scene.mat_names[ray.tri.id]);
        const Vector inter = ray.o + ray.dir * dist;
        Vector normal = ray.tri.normal[0] * (1 - ray.u - ray.v)
          + ray.tri.normal[1] * ray.u +  ray.tri.normal[2] * ray.v;
        normal.norm_inplace();

        return direct_light(scene, material, ray,
                            tree, inter, normal);
    }

    return Vector(0, 0, 0);
}

Vector get_texture(const Ray &ray, const Texture &texture)
{
    auto pos = ray.tri.uv_pos;

    float u = (1 - ray.u - ray.v) * pos[0][0] + ray.u
                                   * pos[1][0] + ray.v * pos[2][0];
    float v = (1 - ray.u - ray.v) * pos[0][1] + ray.u
                                   * pos[1][1] + ray.v * pos[2][1];
    return texture.get_color(u, v);
}

Vector direct_light(const Scene &scene, const Material &material,
                    const Ray &ray, const KdTree &tree,
                    const Vector &inter, const Vector &normal)
{
    Vector color;

    float rat;
    for (const auto *light : scene.lights) // diffuse light
    {
        Vector L = light->compute_light(inter, tree, rat);

        float diff = 0.f;
        if (rat > 0)
        {
            diff = L.dot_product(normal);
            if (diff < 0)
                diff = 0;
        }

        float spec = 0;
        if (L.is_not_null())
        {
            Vector R = reflect(L, normal);
            R.norm_inplace();

            float spec_coef = ray.dir.dot_product(R);
            if (spec_coef < 0)
                spec_coef = 0;
            spec = pow(spec_coef, material.ns);
            if (spec < 0)
                spec = 0;
        }
        if (diff)
        {
            auto kd_map = scene.map_text.find(material.kd_name);
            if (kd_map != scene.map_text.end())
            {
                const Vector &text = get_texture(ray, kd_map->second);
                color += light->color *  text * diff * rat;
            }
            else
                color += light->color * material.kd * diff * rat;
        }
        if (material.illum != 1 && spec)
            color += (light->color * spec * material.ks);
    }

    auto ka_map = scene.map_text.find(material.ka_name);
    if (ka_map != scene.map_text.end())
    {
        const Vector &text = get_texture(ray, ka_map->second);
        color += text * scene.a_light;
    }
    else
        color += material.ka * scene.a_light;

    return color;
}
