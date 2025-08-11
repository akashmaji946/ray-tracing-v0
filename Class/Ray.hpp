#ifndef RAY_H
#define RAY_H

#include "Vector.hpp"
#include "Point.hpp"


class Ray{
public:
    Ray() = default;
    Ray(const Point& origin, const Vector& direction)
        : m_origin(origin), m_direction(direction) {}

    const Point& origin() const { return m_origin; }
    const Vector& direction() const { return m_direction; }

    Point at(double t) const {
        return m_origin + t * m_direction;
    }  
    
    Ray scaled(double t) const {
        return Ray(m_origin, m_direction * t);
    }

    // Overload the << operator for easy output
    friend std::ostream& operator<<(std::ostream& out, const Ray& ray) {
        out << "Ray(origin: " << ray.m_origin << ", direction: " << ray.m_direction << ")";
        return out;
    }

    // Copy constructor
    Ray(const Ray& other)
        : m_origin(other.m_origin), m_direction(other.m_direction) {}   

private:
    Point m_origin;
    Vector m_direction;


};


#endif // RAY_H