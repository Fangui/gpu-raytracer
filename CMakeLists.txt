cmake_minimum_required(VERSION 3.12)
project(raytracer LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 17)

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang|GNU")
    message("Clang or G++ detected, enabling optimizations")
    #add_compile_options(-Ofast -march=native -fno-trapping-math -fno-signed-zeros)
    add_compile_options(-O0 -g3)
endif()

add_definitions(-D_USE_MATH_DEFINES)

add_executable(raytracer
        src/kdtree.cc src/triangle.cc src/material.cc src/parse.cc src/light.cc
        src/compute_light.cc src/matrix.cc src/texture.cc src/sphere_light.cc
        src/main.cu src/triangle_gpu.cu src/vector.cu src/kdtree.cu
        )
set_target_properties(raytracer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(raytracer SDL2 SDL2_image)
