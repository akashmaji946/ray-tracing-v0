#ifndef SPHERE_H
#define SPHERE_H

#include "Point.hpp"
#include "Ray.hpp"
#include "Vector.hpp"


class Sphere {
public:
    Sphere() = default;
    Sphere(const Point& center, double radius)
        : m_center(center), m_radius(radius) {}     
    Sphere(const Sphere& other)
        : m_center(other.m_center), m_radius(other.m_radius) {} 
    const Point& center() const { return m_center; }
    double radius() const { return m_radius; }

    // Method to check if a ray intersects with the sphere
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
    Point intersection_point(const Ray& ray) const {

        Vector oc = m_center - ray.origin();
        double a = dot(ray.direction(), ray.direction());
        double b = -2.0 * dot(oc, ray.direction());
        double c = dot(oc, oc) - m_radius * m_radius;
        double discriminant = b * b - 4 * a * c;
        if (discriminant < 0) {
            throw std::runtime_error("No intersection point exists");
        }
        double t1 = (-b - std::sqrt(discriminant)) / (2.0 * a);
        double t2 = (-b + std::sqrt(discriminant)) / (2.0 * a);


        // if (t1 < 0 && t2 < 0) {
        //     throw std::runtime_error("Both intersection points are behind the ray origin");     
        // }
        // // Return the closest intersection point
        // if (t1 >= 0 && t2 >= 0) {
        //     return ray.at(std::min(t1, t2));
        // } else if (t1 >= 0) {
        //     return ray.at(t1);
        // } else {
        //     return ray.at(t2);      
        // }

        // default fallback case, to be seen later
        double t = min(t1, t2); // Use the closest intersection point
        return ray.at(t); // Return the intersection point
    }

private:
    Point m_center; // Center of the sphere
    double m_radius; // Radius of the sphere
};

#endif // SPHERE_H