#ifndef HITTABLE_H
#define HITTABLE_H


#include "commons.hpp"
#include "utility.hpp"
#include "Ray.hpp"
#include "HitRecord.hpp"
#include "Interval.hpp"

class Hittable{
    public:
        //  if a ray intersects with the object
        __host__ __device__
        bool hitted(const Ray& ray, Interval ray_interval, HitRecord& rec) const{
            return false;
        }

        // Virtual destructor
        __host__ __device__
        ~Hittable() = default;

        // Optional: Function to get the bounding box of the hittable object
        // virtual bool bounding_box(double time0, double time1, AABB& output_box) const {
        //     return false; // Default implementation returns false
        // }

};


#endif // HITTABLE_H