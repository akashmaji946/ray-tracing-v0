#ifndef UTILITY_HPP
#define UTILITY_HPP

#include "commons.hpp"

const double EPSILON = 1e-8; // A small value to compare floating-point numbers
inline bool is_equal(double a, double b) {
    return std::fabs(a - b) < EPSILON;
}   

const double INF = std::numeric_limits<double>::infinity();
const double PI = 3.14159265358979323846;

inline double degrees_to_radians(double degrees) {
    return degrees * PI / 180.0;
}


inline bool random_double() {
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}
inline double random_double(double min, double max) {
    return min + (max - min) * random_double();
}


// some utility functions


#endif // UTILITY_HPP


