#ifndef POINT_H
#define POINT_H     

#include "commons.hpp"
#include "utility.hpp"
#include "Vector.hpp"

using Point = Vector;

__host__ __device__
inline void write_point(ostream& out, const Point& p) {
    out << p.xx() << " " << p.yy() << " " << p.zz() << "\n";
}

// inline ostream& operator<<(ostream& out, const Point& v) {
//     return out << "(" << v.e[0] << ", " << v.e[1] << ", " << v.e[2] << ")\n";
// } 

#endif // POINT_H