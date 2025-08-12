#include "utility.hpp"
#include "commons.hpp"
#include "Sphere.hpp"
#include "Ray.hpp"
#include "Point.hpp"
#include "Vector.hpp"
#include "Color.hpp"    
#include "HitRecord.hpp"
#include "Hittable.hpp"
#include "HittableList.hpp"

#include "Camera.hpp"

int main(){

    // setup world
    HittableList world;
    world.add(make_shared<Sphere>(Point(0, 0, -1), 0.5));
    world.add(make_shared<Sphere>(Point(0, -100.5, -1), 100));

    // camera setup
    Camera cam;
    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width = 400;

    cam.samples_per_pixel = 100; // Set samples per pixel for anti-aliasing

    cam.init();
    cam.render(world, true); // Render with anti-aliasing


}