#ifndef UTILITY_HPP
#define UTILITY_HPP

#include "commons.hpp"

const double EPSILON = 1e-8; // A small value to compare floating-point numbers
__host__ __device__
inline bool is_equal(double a, double b) {
    return std::fabs(a - b) < EPSILON;
}   

__host__ __device__
const double INF = std::numeric_limits<double>::infinity();
__host__ __device__
const double PI = 3.14159265358979323846;

__host__ __device__
inline double degrees_to_radians(double degrees) {
    return degrees * PI / 180.0;
}


__host__ __device__
inline bool random_double() {
    return rand() / (RAND_MAX + 1.0);
}

__host__ __device__
inline double random_double(double min, double max) {
    return min + (max - min) * random_double();
}


// some utility functions


#endif // UTILITY_HPP


