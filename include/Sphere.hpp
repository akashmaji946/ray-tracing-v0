#ifndef SPHERE_H
#define SPHERE_H

#include "commons.hpp"
#include "utility.hpp"
#include "Point.hpp"
#include "Ray.hpp"
#include "Vector.hpp"
#include "HitRecord.hpp"
#include "Hittable.hpp"
#include "Interval.hpp"


class Sphere: public Hittable {
public:

    __host__ __device__
    Sphere() = default;

    __host__ __device__
    Sphere(const Point& center, double radius)
        : m_center(center), m_radius(radius) {} 

    __host__ __device__    
    Sphere(const Sphere& other)
        : m_center(other.m_center), m_radius(other.m_radius) {} 

    __host__ __device__
    const Point& center() const { return m_center; }

    __host__ __device__
    double radius() const { return m_radius; }

    // a simple method to check for intersection
    __host__ __device__
    bool intersects_new(const Ray& ray, double& t) const{
        Vector oc = m_center - ray.origin();
        double a = ray.direction().squared_length();
        double h = dot(oc, ray.direction());
        double c = oc.squared_length() - m_radius * m_radius;
        double discriminant = h * h - a * c;
        if (discriminant < 0) {
            t = -1.0;
            return false; // No intersection    
        }
        double t1 = (h + std::sqrt(discriminant)) / a;
        double t2 = (h - std::sqrt(discriminant)) / a;
        t = (t1 < t2) ? t1 : t2;
        return true;

    }

    // Method to check if a ray intersects with the sphere
    __host__ __device__
    bool intersects(const Ray& ray, double& t) const {
        Vector oc = m_center - ray.origin();
        double a = dot(ray.direction(), ray.direction());
        double b = -2.0 * dot(oc, ray.direction());
        double c = dot(oc, oc) - m_radius * m_radius;
        double discriminant = b * b - 4 * a * c;
        if (discriminant < 0) {
            t = -1; // No intersection
            return false; // No intersection
        } 
        double t1 = (-b - std::sqrt(discriminant)) / (2.0 * a);
        double t2 = (-b + std::sqrt(discriminant)) / (2.0 * a);


        // t = (t1 < t2) ? t1 : t2; // Use the closest intersection point
        // if (t < 0) {
        //     t = -1; // Intersection is behind the ray origin
        //     return false; // No intersection
        // }
        // return true; // Intersection exists



        // default fallback case, to be seen later
        t = min(t1, t2); // Use the closest intersection point
        return true; // Intersection exists

    } 
      
    // Method to get the intersection point if it exists
    __host__ __device__
    Point intersection_point(const Ray& ray) const {
        double t = -1;
        if(!intersects(ray, t)) {
            // throw std::runtime_error("No intersection found");
            return Point(0, 0, 0);
        }
        return ray.at(t); // Return the intersection point
    }


    // if sphere is being hit by a ray, set the hit record
    __host__ __device__
    bool hitted(const Ray& ray, Interval ray_interval, HitRecord& rec) const {
        double t = -1;
        Vector oc = m_center - ray.origin();
        double a = ray.direction().squared_length();
        double h = dot(oc, ray.direction());
        double c = oc.squared_length() - m_radius * m_radius;
        double discriminant = h * h - a * c;
        if (discriminant < 0) {
            t = -1.0;
            return false; // No intersection    
        }

        double sqrtd = std::sqrt(discriminant);
        double root = (h - sqrtd) / a;
        if(root < ray_interval.min || root > ray_interval.max) {
            root = (h + sqrtd) / a; // Check the second root
            if(root < ray_interval.min || root > ray_interval.max) {
                return false; // No valid intersection in the interval
            }
        }
        t = root; // Use the valid root
        Point hit_point = ray.at(t);
        Vector outward_normal = (hit_point - m_center) / m_radius; // Normalized normal vector
        rec.set_face_normal(ray, outward_normal);
        rec.p = hit_point;
        rec.t = t;
        return true; // Intersection exists
    }

private:
    Point m_center; // Center of the sphere
    double m_radius; // Radius of the sphere
};

#endif // SPHERE_H