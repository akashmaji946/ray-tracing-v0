#ifndef SQUARE_HPP
#define SQUARE_HPP

#include "commons.hpp"
#include "utility.hpp"
#include "Point.hpp"
#include "Vector.hpp"
#include "Ray.hpp"
#include "HittableList.hpp"
#include "HitRecord.hpp"
#include "Interval.hpp"
#include "Color.hpp"
#include "Triangle.hpp"

class Square: public Hittable {
public:
    Square() = default;
    Square(const Point& p0, const Point& p1, const Point& p2, const Point& p3)
        : m_p0(p0), m_p1(p1), m_p2(p2), m_p3(p3) {}
    Square(const Square& other)
        : m_p0(other.m_p0), m_p1(other.m_p1), m_p2(other.m_p2), m_p3(other.m_p3) {}
    const Point& p0() const { return m_p0; }
    const Point& p1() const { return m_p1; }
    const Point& p2() const { return m_p2; }
    const Point& p3() const { return m_p3; }

    Square(const Point& top, float width, float height)
        : m_p0(top), m_p1(top + Vector(width, 0, 0)), m_p2(top + Vector(width, height, 0)), m_p3(top + Vector(0, height, 0)) {}

    bool hitted(const Ray& ray, Interval ray_interval, HitRecord& rec) const {
        Triangle t1(m_p0, m_p1, m_p2);
        Triangle t2(m_p0, m_p2, m_p3);
        return t1.hitted(ray, ray_interval, rec) || t2.hitted(ray, ray_interval, rec);
    }

    Point intersection_point(const Ray& ray, double t) const {
        return ray.at(t);
    }

    Vector normal() const {
        Vector e1 = m_p1 - m_p0;
        Vector e2 = m_p2 - m_p0;
        return cross(e1, e2).unit_vector();
    }   

private:
    Point m_p0;
    Point m_p1;
    Point m_p2;
    Point m_p3;
};

#endif