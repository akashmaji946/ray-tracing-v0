#ifndef VECTOR_H
#define VECTOR_H

#include <stdexcept>
#include <cmath>
#include <iostream>
#include <cassert>

using namespace std;

class Vector{
public:
    // stores the values
    double e[3];
    // default
    Vector(): e{0, 0, 0} {}
    // custom
    Vector(double x, double y, double z): e{x, y, z} {}

    // copy constructor
    Vector(const Vector& v) {
        e[0] = v.e[0];
        e[1] = v.e[1];
        e[2] = v.e[2];
    }

    // getters for components
    double xx() const{
        return e[0];
    }
    double yy() const {
        return e[1];
    }
    double zz() const{
        return e[2];
    }

    double operator[] (int i) const {
        if (i < 0 || i >= 3) {
            throw std::out_of_range("Index out of bounds");
        }
        return e[i];
    }
    double& operator[](int i) {
        if (i < 0 || i >= 3) {
            throw std::out_of_range("Index out of bounds");
        }
        return e[i];
    }

    // addition
    Vector& operator+=(const Vector& v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }
    // subtraction
    Vector& operator-=(const Vector& v) {
        e[0] -= v.e[0];
        e[1] -= v.e[1];
        e[2] -= v.e[2];
        return *this;
    }
    // multiplication by scalar
    Vector& operator*=(double t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }
    // division by scalar
    Vector& operator/=(double t) {
        if (t == 0) {
            throw std::invalid_argument("Division by zero");
        }
        e[0] /= t;
        e[1] /= t;
        e[2] /= t;
        return *this;
    }   

    double length() const {
        return std::sqrt(squared_length());
    }   

    double squared_length() const {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }
    void make_unit_vector() {
        double k = 1.0 / length();
        if (k == 0) {
            throw std::invalid_argument("Cannot normalize a zero vector");
        }
        e[0] *= k;
        e[1] *= k;
        e[2] *= k;
    }

    Vector unit_vector() const {
        double k = 1.0 / length();
        if (k == 0) {
            throw std::invalid_argument("Cannot normalize a zero vector");
        }
        return Vector(e[0] * k, e[1] * k, e[2] * k);
    }

    Vector copy() const {
        return Vector(e[0], e[1], e[2]);
    }
};


// some utility functions

inline void write_vector(ostream& out, const Vector& v) {
    out << v.xx() << " " << v.yy() << " " << v.zz() << "\n";
}   

inline ostream& operator<<(ostream& out, const Vector& v) {
    return out << "< " << v.e[0] << ", " << v.e[1] << ", " << v.e[2] << ">";
}   

inline double dot(const Vector& u, const Vector& v) {
    return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

inline Vector operator+(const Vector& u, const Vector& v) {
    return Vector(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}
inline Vector operator-(const Vector& u, const Vector& v) {
    return Vector(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}
inline Vector operator*(const Vector& u, const Vector& v) {
    return Vector(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

inline Vector operator*(const Vector& u, double t) {
    return Vector(u.e[0] * t, u.e[1] * t, u.e[2] * t);
}
inline Vector operator*(double t, const Vector& u) {
    return Vector(u.e[0] * t, u.e[1] * t, u.e[2] * t);
}
inline Vector operator/(const Vector& u, double t) {
    if (t == 0) {
        throw std::invalid_argument("Division by zero");
    }
    return Vector(u.e[0] / t, u.e[1] / t, u.e[2] / t);
}

#endif