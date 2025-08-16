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
#include "Triangle.hpp"
#include "Square.hpp"
#include "Camera.hpp"


int main(){

    // setup world
    HittableList world;
    world.add(make_shared<Sphere>(Point(0, 0, -1), 0.5));
    world.add(make_shared<Sphere>(Point(0, -100.5, -1), 100));

    world.add(make_shared<Triangle>(Point(0, 0, -1), Point(1, 0, -1), Point(0, 1, -1)));
    world.add(make_shared<Square>(Point(1, 1, -2), 1, 1));

    // camera setup
    Camera cam;
    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width = 1920;

    cam.samples_per_pixel = 1; // Set samples per pixel for anti-aliasing

    // lets measure the time it takes to render the image
    auto start = std::chrono::high_resolution_clock::now();

    cam.init();
    cam.render_cpu(world, true, false); 
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken to render the image [CPU]: " << duration.count() * 1000 << " milliseconds" << std::endl;


}