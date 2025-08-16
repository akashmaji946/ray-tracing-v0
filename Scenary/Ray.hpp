#ifndef RAY_H
#define RAY_H

#include "commons.hpp"
#include "utility.hpp"
#include "Vector.hpp"
#include "Point.hpp"

class Ray{
public:
    __host__ __device__
    Ray() = default;

    __host__ __device__
    Ray(const Point& origin, const Vector& direction)
        : m_origin(origin), m_direction(direction) {}

    __host__ __device__
    const Point& origin() const { return m_origin; }

    __host__ __device__
    const Vector& direction() const { return m_direction; }

    __host__ __device__
    Point at(double t) const {
        return m_origin + t * m_direction;
    }  
    
    __host__ __device__
    Ray scaled(double t) const {
        return Ray(m_origin, m_direction * t);
    }

    __host__ __device__
    Ray unit_ray() const {
        Vector unit_direction = m_direction.unit_vector();
        return Ray(m_origin, unit_direction);
    }

    // Overload the << operator for easy output
    __host__ __device__
    friend std::ostream& operator<<(std::ostream& out, const Ray& ray) {
        out << "Ray(origin: " << ray.m_origin << ", direction: " << ray.m_direction << ")";
        return out;
    }

    // Copy constructor
    __host__ __device__
    Ray(const Ray& other)
        : m_origin(other.m_origin), m_direction(other.m_direction) {}   

private:
    Point m_origin;
    Vector m_direction;


};


#endif // RAY_H