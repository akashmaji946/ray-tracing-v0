#ifndef COLOR_H
#define COLOR_H     

#include "Vector.hpp"
using Color = Vector;

inline void write_color(ostream& out, const Color& c) {
    double rr = c.xx();
    double gg = c.yy();
    double bb = c.zz();

    // assert(rr >= 0 && rr <= 1);
    // assert(gg >= 0 && gg <= 1);
    // assert(bb >= 0 && bb <= 1);

    // Clamp the color values to the range [0, 1]
    if (rr < 0) rr = 0;
    if (gg < 0) gg = 0;
    if (bb < 0) bb = 0;
    if (rr > 1) rr = 1;
    if (gg > 1) gg = 1;
    if (bb > 1) bb = 1;

    int r = static_cast<int>(rr * 255.999);
    int g = static_cast<int>(gg * 255.999);
    int b = static_cast<int>(bb * 255.999);

    // Write the color as RGB values
    out << r << ' ' << g << ' ' << b << '\n'; 
}


// inline ostream& operator<<(ostream& out, const Color& v) {
//     return out << "[" << v.e[0] << ", " << v.e[1] << ", " << v.e[2] << "]\n";
// } 

#endif // COLOR_H