#include "commons.hpp"
#include "Color.hpp"
#include <iostream>
#include <cstdlib>
#include <cmath>

// Implementation of common utility functions
namespace commons {

double clamp(double x, double min, double max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

void write_color(std::ostream& out, const Color& pixel_color) {
    // Write the translated [0,255] value of each color component.
    out << static_cast<int>(256 * clamp(pixel_color.xx(), 0.0, 0.999)) << ' '
        << static_cast<int>(256 * clamp(pixel_color.yy(), 0.0, 0.999)) << ' '
        << static_cast<int>(256 * clamp(pixel_color.zz(), 0.0, 0.999)) << '\n';
}

} // namespace commons
