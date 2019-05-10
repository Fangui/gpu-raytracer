/*
#include <algorithm>
#include <iostream>
#include <vector>
#include <omp.h>
#include <unordered_map>

#include "compute_light.hh"
#include "kdtree.hh"
#include "vector.hh"
*/

#include <fstream>
#include <vector>

#include "parse.hh"
#include "vector.hh"



int write_ppm(const std::string &out_path, Vector *vect,
        int width, int height)
{
    std::ofstream out (out_path);
    unsigned index = 0;

    if (out.is_open())
    {
        out << "P3\n";
        out << width << " " << height << '\n';
        out << 255 << '\n';

        for (int i = 0; i < width; ++i)
        {
            for (int j = 0; j < height; ++j)
            {
                int r = vect[index][0] * 255.0;
                int g = vect[index][1] * 255.0;
                int b = vect[index++][2] * 255.0;

                r = std::fmin(r, 255);
                g = std::fmin(g, 255);
                b = std::fmin(b, 255);
                out << r << " " << g << " " << b << "  ";
            }
            out << '\n';
        }
        std::cout << "Create " + out_path + " file\n";
    }
    else
    {
        std::cerr << "Error while write in " << out_path << '\n';
        return 1;
    }
    return 0;
}

__global__ void render(Vector *d_vect, Vector *d_u, Vector *d_v, 
                       Vector *d_center, Vector *d_cam_pos, 
                       unsigned width, unsigned height)
{
    unsigned i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= width || j >= height)
        return;


    d_vect[j * width + i] = Vector(1, 0.5, 0.5); // cast ray
}

int main(int argc, char *argv[])
{
    std::string path_scene;
    std::string out_file = "out";

    if (argc > 1)
        path_scene = argv[1];
    else
    {
        std::cerr << "Usage: ./main <scene> <nb_ray> <depth> <filter> <out_file>\n";
        return 1;
    }

    if (argc > 2)
        out_file = argv[2];
    //float t1 = omp_get_wtime();

    Scene scene = parse_scene(path_scene);

    Vector u_n = scene.cam_u.norm_inplace();
    Vector v_n = scene.cam_v.norm_inplace();
    Vector w = v_n.cross_product(u_n);

    float val = tan(scene.fov * M_PI / 360);
    val = val == 0.0 ? 0.0001 : val;
    float L = scene.width / 2;
    L /= val; 
    Vector center = scene.cam_pos + (w * L); // center

    Vector *d_vect;
    Vector *d_u;
    Vector *d_v;
    Vector *d_center;
    Vector *d_cam_pos;

    cudaMalloc(&d_vect, scene.width * scene.height * sizeof(struct Vector)); // call wrapper
    cudaMalloc(&d_u, sizeof(struct Vector)); // call wrapper
    cudaMalloc(&d_v, sizeof(struct Vector)); // call wrapper
    cudaMalloc(&d_center, sizeof(struct Vector)); // call wrapper
    cudaMalloc(&d_cam_pos, sizeof(struct Vector)); // call wrapper
    Vector *vect = new Vector[scene.width * scene.height];

    cudaMemcpy(d_u, &u_n, sizeof(struct Vector), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, &v_n, sizeof(struct Vector), cudaMemcpyHostToDevice);
    cudaMemcpy(d_center, &center, sizeof(struct Vector), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cam_pos, &scene.cam_pos, sizeof(struct Vector), cudaMemcpyHostToDevice);

    constexpr int tx = 8;
    constexpr int ty = 8;

    dim3 dim_block(scene.width / 8 + (scene.width % 8 != 0), 
                   scene.height / 8 + (scene.height % 8 != 0));
    dim3 dim_thread(tx, ty);

    render<<<dim_block, dim_thread >>>(d_vect, d_u, d_v, d_center, d_cam_pos,  
                                      scene.width, scene.height);

    cudaMemcpy(vect, d_vect, scene.width * scene.height * sizeof(struct Vector), 
               cudaMemcpyDeviceToHost);

    cudaFree(d_vect);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_center);
    cudaFree(d_cam_pos);
    
    write_ppm(out_file + ".ppm", vect, scene.width, scene.height);
    delete[] vect;

    /*
    // distance between camera and center of screen

    std::vector<Triangle> vertices;
    for (const auto& name : scene.objs)
      obj_to_vertices(name, scene.mat_names, vertices, scene);

    float t2 = omp_get_wtime();
    std::cout << "Time to parse file: " << t2 - t1 << "s\n";

    t1 = omp_get_wtime();
    auto tree = KdTree(vertices.begin(), vertices.end());
    t2 = omp_get_wtime();
    std::cout << "Time build kdTree: " << t2 - t1 << "s\n";

    std::cout << tree.size() << std::endl;

    std::vector<Vector> vect(scene.width * scene.height);


    t1 = omp_get_wtime();

    

    constexpr float gamma = 1. / 2.2;
#pragma omp parallel for schedule (dynamic)
    for (int i = -scene.width / 2; i < scene.width / 2; ++i)
    {
        for (int j = scene.height / 2; j > -scene.height / 2; --j)
        {
            unsigned idx = (i + scene.width / 2) * scene.height + (scene.height / 2 - j);
            Vector o = scene.cam_u * j;
            Vector b = scene.cam_v * i;
            o += C;
            o += b;

            Vector dir = o - scene.cam_pos;
            dir.norm_inplace();
            Ray r(scene.cam_pos, dir);

            vect[idx] = cast_ray(scene, r, tree); // depth
            for (unsigned g = 0; g < 3; ++g) // gamme
            {
                vect[idx][g] = pow(vect[idx][g], gamma);
                if (vect[idx][g] > 1)
                    vect[idx][g] = 1;
            }
        }
    }
    t2 = omp_get_wtime();
    std::cout << "Time raytracing: " << t2 - t1 << "s\n";

    
    write_ppm(out_file + ".ppm", vect, scene.width, scene.height);
    float t4 = omp_get_wtime(); */
    /*
    std::vector<Vector> out;

    for (int i = 0; i < scene.width; i += 2)
    {
        for (int j = 0; j < scene.height; j += 2)
        {
            Vector c = (vect[i * scene.width + j]
                      + vect[i * scene.width + j + 1]
                      + vect[(i + 1) * scene.height + j]
                      + vect[(i + 1) * scene.height + j + 1]) / 4;

            out.push_back(c);
        }
    }*/

    /*
    std::vector<Vector> res;

    float t3 = omp_get_wtime();
    std::cout << "Time applying denoise: " << t3 - t4 << "s\n";

    //return write_ppm("out.ppm", out, scene.width / 2, scene.height / 2);
    return write_ppm(out_file + "_denoise.ppm", res, scene.width, scene.height);
    */
}
