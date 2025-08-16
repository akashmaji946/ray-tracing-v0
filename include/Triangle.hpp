#ifndef TRIANGLE_HPP
#define TRIANGLE_HPP

#include "commons.hpp"
#include "utility.hpp"
#include "Point.hpp"
#include "Vector.hpp"
#include "Ray.hpp"
#include "HittableList.hpp"
#include "HitRecord.hpp"
#include "Interval.hpp"
#include "Color.hpp"

class Triangle: public Hittable {
public:

    __host__ __device__
    Triangle() = default;
    __host__ __device__
    Triangle(const Point& p0, const Point& p1, const Point& p2)
        : m_p0(p0), m_p1(p1), m_p2(p2) {}
    __host__ __device__
    Triangle(const Triangle& other)
        : m_p0(other.m_p0), m_p1(other.m_p1), m_p2(other.m_p2) {}
    __host__ __device__
    const Point& p0() const { return m_p0; }
    __host__ __device__
    const Point& p1() const { return m_p1; }
    __host__ __device__
    const Point& p2() const { return m_p2; }    

    __host__ __device__
    bool hitted(const Ray& ray, Interval ray_interval, HitRecord& rec) const {

        Vector e1 = m_p1 - m_p0;
        Vector e2 = m_p2 - m_p0;
        Vector h = cross(ray.direction(), e2);
        double a = dot(e1, h);

        if (a > -EPSILON && a < EPSILON) 
            return false;  // Ray is parallel to triangle

        double f = 1.0 / a;
        Vector s = ray.origin() - m_p0;
        double u = f * dot(s, h);

        if (u < 0.0 || u > 1.0) 
            return false;  // Outside triangle

        Vector q = cross(s, e1);
        double v = f * dot(ray.direction(), q);

        if (v < 0.0 || u + v > 1.0) 
            return false;  // Outside triangle

        double t_value = f * dot(e2, q);

        if (t_value < ray_interval.min || t_value > ray_interval.max) 
            return false;  // Outside ray interval

        rec.t = t_value;
        rec.p = ray.at(rec.t);
        Vector outward_normal = cross(e1, e2);
        rec.set_face_normal(ray, outward_normal);
        
        return true;
    }

    __host__ __device__
    Point intersection_point(const Ray& ray, double t) const {
        return ray.at(t);
    }

    __host__ __device__
    Vector normal() const {
        Vector e1 = m_p1 - m_p0;
        Vector e2 = m_p2 - m_p0;
        return cross(e1, e2).unit_vector();
    }

private:
    Point m_p0;
    Point m_p1;
    Point m_p2;

};

#endif