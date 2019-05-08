#include "scene.hh"
#include "material.hh"
#include "light.hh"
#include "sphere_light.hh"

Vector direct_light(const Scene &scene, const Material &mat,
                    const Ray &ray, const KdTree &tree,
                    const Vector& inter, const Vector &normal);


Vector cast_ray(const Scene &scene, 
                Ray &ray, const KdTree &tree);
