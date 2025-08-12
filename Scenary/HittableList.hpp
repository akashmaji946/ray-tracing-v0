#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H


#include "commons.hpp"
#include "utility.hpp"
#include "Hittable.hpp"
#include "Ray.hpp"
#include "HitRecord.hpp"


class HittableList : public Hittable {
public:
    std::vector<std::shared_ptr<Hittable>> objects; 

    HittableList() = default;
    HittableList(const std::vector<std::shared_ptr<Hittable>>& objects) : objects(objects) {}

    void add(const std::shared_ptr<Hittable>& object) {
        objects.push_back(object);
    }

    bool hitted(const Ray& ray, Interval ray_interval, HitRecord& rec) const override {
        HitRecord temp_rec;
        bool hit_anything = false;
        double closest_so_far = ray_interval.max; // Start with the maximum distance in the interval

        for (const auto& object : objects) {
            if (object->hitted(ray, Interval(ray_interval.min, closest_so_far), temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec; // Update the record with the closest hit
            }
        }
        return hit_anything;
    }

};


#endif // HITTABLE_LIST_H