

#ifndef CAMERA_H
#define CAMERA_H

#include "commons.hpp"
#include "utility.hpp"

class Camera {
public:
    double aspect_ratio;
    int image_width;

    int image_height;
    Point camera_origin;
    double focal_length;
    Vector viewport_u;
    Vector viewport_v;
    Vector d_u; 
    Vector d_v;
    Vector upper_left_corner;
    Point pixel_origin;
    bool INIT = false; // Flag to check if camera is initialized

    Camera() : aspect_ratio(16.0 / 9.0), image_width(800) {}

    void init() {
        // Initialize camera parameters if needed
        image_height = static_cast<int>(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height; // Ensure height is at least 1

        camera_origin = Point(0, 0, 0);
        focal_length = 1.0;
        double viewport_height = 2.0;
        double viewport_width = viewport_height * aspect_ratio;

        viewport_u = Vector(viewport_width, 0, 0);
        viewport_v = Vector(0, -viewport_height, 0);
        d_u = viewport_u / static_cast<double>(image_width);
        d_v = viewport_v / static_cast<double>(image_height);   

        // camera faces in the -ve z direction
        upper_left_corner = camera_origin 
                               - viewport_u / 2.0 
                               - viewport_v / 2.0 
                               - Vector(0, 0, focal_length);    

        pixel_origin = upper_left_corner
                         + d_u * 0.5 
                         + d_v * 0.5;       

        INIT = true; // Mark camera as initialized

    }

    void render(const HittableList& world) {

        if(!INIT) {
         init(); // Ensure camera is initialized before rendering
        }

        std::cout << "Image dimensions: " << image_width << "x" << image_height << std::endl;

        std::ofstream out("main.ppm", std::ios::out | std::ios::trunc);
        if (!out) {
            std::cerr << "Error opening file for writing." << std::endl;
            return;
        }
        write_image(world, out);

        std::cout << "Image written to main.ppm" << std::endl;

    }

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

    void write_image(const HittableList& world, std::ostream& out) {

        // Write PPM header
        out << "P3\n" << image_width << ' ' << image_height << "\n255\n";

        // Render the image
        for (int j = 0; j < image_height; j++) {
            for (int i = 0; i < image_width; i++) {
                // Calculate the pixel position
                Point pixel_position = pixel_origin + d_u * i + d_v * j;

                // Create a ray from the camera origin to the pixel position
                Vector ray_direction = pixel_position - camera_origin;
                Ray ray(camera_origin, ray_direction);

                // Check for intersection with the world
                Color color = rayColor(ray, world);

                // Write the color to the output stream
                write_color(out, color);
            }
        }

    }



};


#endif // CAMERA_H