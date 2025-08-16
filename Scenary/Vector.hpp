#ifndef VECTOR_H
#define VECTOR_H

#include "commons.hpp"
#include "utility.hpp"

using namespace std;

class Vector{
public:
    // stores the values
    double e[3];
    // default
    __host__ __device__
    Vector(): e{0, 0, 0} {}
    // custom
    __host__ __device__
    Vector(double x, double y, double z): e{x, y, z} {}

    // copy constructor
    __host__ __device__
    Vector(const Vector& v) {
        e[0] = v.e[0];
        e[1] = v.e[1];
        e[2] = v.e[2];
    }

    // getters for components
    __host__ __device__
    double xx() const{
        return e[0];
    }
    __host__ __device__
    double yy() const {
        return e[1];
    }
    __host__ __device__
    double zz() const{
        return e[2];
    }

    __host__ __device__
    double operator[] (int i) const {
        if (i < 0 || i >= 3) {
            // throw std::out_of_range("Index out of bounds");
            return 0;
        }
        return e[i];
    }
    __host__ __device__
    double& operator[](int i) {
        if (i < 0 || i >= 3) {
            // throw std::out_of_range("Index out of bounds");
            return e[0];
        }
        return e[i];
    }

    // addition
    __host__ __device__
    Vector& operator+=(const Vector& v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }
    // subtraction
    __host__ __device__
    Vector& operator-=(const Vector& v) {
        e[0] -= v.e[0];
        e[1] -= v.e[1];
        e[2] -= v.e[2];
        return *this;
    }
    // multiplication by scalar
    __host__ __device__
    Vector& operator*=(double t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }
    // division by scalar
    __host__ __device__
    Vector& operator/=(double t) {
        if (t == 0) {
            // throw std::invalid_argument("Division by zero");
            return *this;
        }
        e[0] /= t;
        e[1] /= t;
        e[2] /= t;
        return *this;
    }   

    __host__ __device__
    Vector operator-() const {
        return Vector(-e[0], -e[1], -e[2]);
    }

    __host__ __device__
    double length() const {
        return std::sqrt(squared_length());
    }   

    __host__ __device__
    double squared_length() const {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }

    __host__ __device__
    void make_unit_vector() {
        double k = 1.0 / length();
        if (k == 0) {
            // throw std::invalid_argument("Cannot normalize a zero vector");
            return;
        }
        e[0] *= k;
        e[1] *= k;
        e[2] *= k;
    }

    __host__ __device__
    Vector unit_vector() const {
        double k = 1.0 / length();
        if (k == 0) {
            // throw std::invalid_argument("Cannot normalize a zero vector");
            return Vector(0, 0, 0);
        }
        return Vector(e[0] * k, e[1] * k, e[2] * k);
    }

    __host__ __device__
    Vector copy() const {
        return Vector(e[0], e[1], e[2]);
    }
};


__host__ __device__
inline void write_vector(ostream& out, const Vector& v) {
    out << v.xx() << " " << v.yy() << " " << v.zz() << "\n";
}   

__host__ __device__
inline ostream& operator<<(ostream& out, const Vector& v) {
    out << std::setprecision(6) << std::fixed; // Set precision for floating-point output
    return out << "< " << v.e[0] << ", " << v.e[1] << ", " << v.e[2] << ">";
    out.clear(); // Clear the stream state
}   

__host__ __device__
inline double dot(const Vector& u, const Vector& v) {
    return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

__host__ __device__
inline Vector cross(const Vector& u, const Vector& v) {
    return Vector(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                  u.e[2] * v.e[0] - u.e[0] * v.e[2],
                  u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__
inline Vector operator+(const Vector& u, const Vector& v) {
    return Vector(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}
__host__ __device__
inline Vector operator-(const Vector& u, const Vector& v) {
    return Vector(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}
__host__ __device__
inline Vector operator*(const Vector& u, const Vector& v) {
    return Vector(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__
inline Vector operator*(const Vector& u, double t) {
    return Vector(u.e[0] * t, u.e[1] * t, u.e[2] * t);
}
__host__ __device__
inline Vector operator*(double t, const Vector& u) {
    return Vector(u.e[0] * t, u.e[1] * t, u.e[2] * t);
}
__host__ __device__
inline Vector operator/(const Vector& u, double t) {
    if (t == 0) {
        // throw std::invalid_argument("Division by zero");
        return Vector(0, 0, 0);
    }
    return Vector(u.e[0] / t, u.e[1] / t, u.e[2] / t);
}

#endif // VECTOR_H