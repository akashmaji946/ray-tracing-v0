
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
using namespace std;


Color rayGradColor(const Ray& ray){
    float a = (ray.direction().unit_vector().yy() + 1.0f) / 2.0f; // Normalize to [0, 1] along y-axis
    Color white(1.0, 1.0, 1.0); // start
    Color blue(0.5, 0.7, 1.0); // end
    return white * (1.0f - a) + blue * a; // Linear
}

Color rayColor(const Ray& ray, const HittableList& world) {
    HitRecord rec;
    double t = -1; // Initialize t to -1 to indicate no intersection

    if(world.hitted(ray, Interval(0, INF), rec)) {
        // If the ray hits an object, use the hit record to determine color
        Vector normal = rec.normal;
        normal.make_unit_vector(); // Normalize the normal vector
        // Use the normal to determine color
        Color c = 0.5 * (normal + Vector(1, 1, 1)); // Example color based on normal
        return Color(c);
    }


    return rayGradColor(ray); // If no intersection, use gradient color
}


void write_image_to_file(int IMG_HEIGHT, int IMG_WIDTH, std::ostream& out, \
    const Point& camera_origin, double focal_length, \
    const Vector& viewport_u, const Vector& viewport_v, \
    const Vector& d_u, const Vector& d_v, \
    const Point& upper_left_corner, const Point& pixel_origin, \
    HittableList& world) {

    // Sphere
    Sphere sphere(Point(0, 0, -1), 0.5); // Center at (0, 0, -1) with radius 1.0
    

    // write ppm headers
    out << "P3\n" << IMG_WIDTH << " " << IMG_HEIGHT << "\n255\n";

    for (int j = 0; j < IMG_HEIGHT; j++) {
        for (int i = 0; i < IMG_WIDTH; i++) {
            // Calculate the pixel position
            Point pixel_position = pixel_origin + d_u * i + d_v * j;

            // Create a ray from the camera origin to the pixel position
            Vector ray_direction = pixel_position - camera_origin;
            Ray ray(camera_origin, ray_direction);


            // Check for intersection with the sphere
            double t;
            Color color = rayColor(ray, world);
            
            cout << color << endl;

            // Write the color to the output stream
            write_color(out, color);
        }
    }
}

double ASPECT_RATIO = 16.0 / 9.0;


int main(){

    // world setup
    HittableList world;
    world.add(make_shared<Sphere>(Point(0, 0, -1), 0.5));
    world.add(make_shared<Sphere>(Point(0, -100.5, -1), 100));

    int IMG_WIDTH = 800;
    int IMG_HEIGHT = static_cast<int>(IMG_WIDTH / ASPECT_RATIO);
    IMG_HEIGHT = (IMG_HEIGHT < 1) ? 1 : IMG_HEIGHT; // Ensure height is at least 1

    // Camera setup
    Point camera_origin(0, 0, 0);
    double focal_length = 1.0;

    double viewport_height = 2.0;
    double viewport_width = viewport_height * ASPECT_RATIO;

    Vector viewport_u(viewport_width, 0, 0);
    Vector viewport_v(0, -viewport_height, 0);

    Vector d_u = viewport_u / static_cast<double>(IMG_WIDTH);
    Vector d_v = viewport_v / static_cast<double>(IMG_HEIGHT);

    // camera faces in the -ve z direction
    Vector upper_left_corner = camera_origin 
                               - viewport_u / 2.0 
                               - viewport_v / 2.0 
                               - Vector(0, 0, focal_length);

    Point pixel_origin = upper_left_corner
                         + d_u * 0.5 
                         + d_v * 0.5;

    std::cout << "Image dimensions: " << IMG_WIDTH << "x" << IMG_HEIGHT << std::endl;


    cout << "Image dimensions: " << IMG_WIDTH << "x" << IMG_HEIGHT << endl;
    // render the image
    ofstream out("simple_scene.ppm", std::ios::out | std::ios::trunc);
    if (!out) {
        cerr << "Error opening file for writing." << endl;
        return 1;
    }else{
        write_image_to_file(IMG_HEIGHT, IMG_WIDTH, out, camera_origin, focal_length, viewport_u, viewport_v, d_u, d_v, upper_left_corner, pixel_origin, world);
        out.close();
        cout << "Image written to simple_scene.ppm" << endl;
    }





    return 0;
}