#ifndef HIT_RECORD_H
#define HIT_RECORD_H

#include "commons.hpp"
#include "utility.hpp"
#include "Point.hpp"
#include "Vector.hpp"
#include "Ray.hpp"

class HitRecord {

public:
    Point p; // point of intersection
    Vector normal; // normal at the intersection point
    double t; // parameter along the ray
    bool front_face; // is the hit on the front face of the object

    HitRecord() = default;

    HitRecord(const Point& p, const Vector& normal, double t, bool front_face)
        : p(p), normal(normal), t(t), front_face(front_face) {}

    void set_face_normal(const Ray& ray, const Vector& outward_normal) {
        front_face = dot(ray.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : (-1 * outward_normal);
    }




};

#endif // HIT_RECORD_H