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

    bool hit(const Ray& ray, double t_min, double t_max, HitRecord& rec) const override {
        HitRecord temp_rec;
        bool hit_anything = false;
        double closest_so_far = t_max;

        for (const auto& object : objects) {
            if (object->hit(ray, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec; // Update the record with the closest hit
            }
        }
        return hit_anything;
    }

};


#endif // HITTABLE_LIST_H