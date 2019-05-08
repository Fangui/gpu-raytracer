#pragma once

#include <cmath>
#include <iostream>

inline float to_rad(int deg)
{
    return deg * (M_PI / 180.0);
}

class Vector
{
public:
    __host__ __device__ Vector() {}

    __host__ __device__ Vector(float x, float y, float z)
    {
      tab[0] = x;
      tab[1] = y;
      tab[2] = z;
    }

    bool is_not_null()
    {
        return tab[0] == 0 || tab[1] == 0 || tab[2] == 0;
    }
    
    __host__ __device__ Vector operator+(const Vector &rhs) const;
    __host__ __device__ Vector operator+=(const Vector &rhs);
    __host__ __device__ Vector operator-(const Vector &rhs) const;
    __host__ __device__ Vector operator-=(const Vector &rhs);
    __host__ __device__ Vector operator*(float lambda) const;
    __host__ __device__ Vector operator*=(float lambda);
    __host__ __device__ Vector operator*(const Vector &rhs) const;
    __host__ __device__ Vector operator*=(const Vector &rhs);
    __host__ __device__ Vector operator/(const Vector &rhs) const;
    __host__ __device__ Vector operator/=(const Vector &rhs);
    __host__ __device__ Vector operator/(float lambda) const;
    __host__ __device__ Vector operator/=(float lambda);

    __host__ __device__  float operator[](unsigned idx) const { return tab[idx]; };
    __host__ __device__  float& operator[](unsigned idx) { return tab[idx]; };

   // __host__ __device__ Vector operator*(float lambda, const Vector &rhs);

    __host__ __device__ Vector cross_product(const Vector &rhs) const;
    __host__ __device__ Vector cross_product_inplace(const Vector &rhs);
    __host__ __device__ Vector norm(void) const;
    __host__ __device__ Vector norm_inplace(void);

     __host__ __device__ float dot_product(const Vector &rhs) const;

     __host__ __device__ float get_dist() { return sqrtf(tab[0] * tab[0] + tab[1] * tab[1] + tab[2] * tab[2]); };

     __host__ __device__ void set(float x, float y, float z)
    {
        tab[0] = x;
        tab[1] = y;
        tab[2] = z;
    }

    friend std::ostream& operator <<(std::ostream& os, const Vector &v);

    float tab[3];
};

__host__ __device__ Vector operator/(float lambda, const Vector &v);
__host__ __device__ Vector operator*(float lambda, const Vector &v);
