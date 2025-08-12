#ifndef HITTABLE_H
#define HITTABLE_H


#include "commons.hpp"
#include "utility.hpp"
#include "Ray.hpp"
#include "HitRecord.hpp"

class Hittable{
    public:
        //  if a ray intersects with the object
        virtual bool hitted(const Ray& ray, double t_min, double t_max, HitRecord& rec) const = 0;

        // Virtual destructor
        virtual ~Hittable() = default;

        // Optional: Function to get the bounding box of the hittable object
        // virtual bool bounding_box(double time0, double time1, AABB& output_box) const {
        //     return false; // Default implementation returns false
        // }

};


#endif // HITTABLE_H