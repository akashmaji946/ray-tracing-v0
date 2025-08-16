#include <iostream>
#include "common/common.hpp"
#include "include/Ray.hpp"
#include "include/Vector.hpp"
#include "include/Point.hpp"
#include "include/Color.hpp"
#include "include/Triangle.hpp"
#include "include/Sphere.hpp"
#include "include/SurfaceInteraction.hpp"





int main() {
    
    printf("Hello, World !\n");

    Ray ray(Point(0, 0, 0), Vector(1, 0, 0));
    printf("Ray: %s\n", ray.toString().c_str());

    Vector v(1, 2, 3);
    printf("Vector: %s\n", v.toString().c_str());

    Color color(0.5, 0.5, 0.5);
    printf("Color: %s\n", color.toString().c_str());

    Point p(1, 2, 3);
    printf("Point: %s\n", p.toString().c_str());

    Triangle triangle(Vector(0, 0, 0), Vector(1, 0, 0), Vector(0, 1, 0));
    printf("Triangle: %s\n", triangle.toString().c_str());

    Sphere sphere(Point(0, 0, 0), 1);
    printf("Sphere: %s\n", sphere.toString().c_str());

    SurfaceInteraction interaction;


    

    
    

    return 0;


}
