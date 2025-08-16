
#ifndef CAMERA_H
#define CAMERA_H

#include "commons.hpp"
#include "utility.hpp"
#include "Point.hpp"
#include "Vector.hpp"
#include "Ray.hpp"
#include "HittableList.hpp"
#include "HitRecord.hpp"
#include "Interval.hpp"
#include "Color.hpp"



class Camera {

public:
    double aspect_ratio;
    int image_width;
    double samples_per_pixel; 

    int image_height;
    Point camera_origin;
    double focal_length;
    Vector viewport_u;
    Vector viewport_v;
    Vector d_u; 
    Vector d_v;
    Vector upper_left_corner;
    Point pixel_origin;
    double pixel_scale = 1.0; // Scale factor for pixel size
    bool INIT = false; // Flag to check if camera is initialized

    __host__ __device__
    Camera() : aspect_ratio(16.0 / 9.0), image_width(800), samples_per_pixel(10){}
    
    __host__ __device__
    Camera(double aspect_ratio, int image_width, int image_height, int samples_per_pixel) : aspect_ratio(aspect_ratio), image_width(image_width), image_height(image_height), samples_per_pixel(samples_per_pixel){}

    __host__ __device__
    void init() {
        // Initialize camera parameters if needed
        image_height = static_cast<int>(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height; // Ensure height is at least 1
        pixel_scale = 1.0 / static_cast<double>(samples_per_pixel); // Scale factor for pixel size

        camera_origin = Point(0, 0, 0);
        focal_length = 1.0;
        double viewport_height = 2.0;
        double viewport_width = viewport_height * ((double)(image_width) / (double)(image_height));

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

    __host__ __device__
    void render_cpu(const HittableList& world, bool antialias = true, bool toFile = true) {

        if(!INIT) {
         init(); // Ensure camera is initialized before rendering
        }

        printf("Image dimensions: %dx%d\n", image_width, image_height);

       if(toFile) {
            std::ofstream out("seam.ppm", std::ios::out | std::ios::trunc);
            if (!out) {
                printf("Error opening file for writing.\n");
                return;
            }
            if (antialias) {
                write_image_antialias_file(world, out, this);
            } else {
                write_image_file(world, out, this);
            }

            printf("Image written to seam.ppm\n");
       }else{
        // we will use stb_image.h to write the image to a buffer
        unsigned char* image_buffer = new unsigned char[image_width * image_height * 3];
        if (antialias) {
            write_image_antialias_stb(world, image_buffer, this);
        } else {
            write_image_stb(world, image_buffer, this);
        }
        // save the image to a file
        stbi_write_png("seam.png", image_width, image_height, 3, image_buffer, image_width * 3);
        delete[] image_buffer;
       }

    }

    __host__ __device__ 
    void render_gpu(const HittableList& world, bool antialias = true, bool toFile = true) {
        if(!INIT) {
            init(); // Ensure camera is initialized before rendering
        }

        // create a buffer to store the image
        unsigned char* image_buffer = new unsigned char[image_width * image_height * 3];

        // render the image
        render_image_gpu(world, image_buffer);

        // save the image to a file
        stbi_write_png("seam.png", image_width, image_height, 3, image_buffer, image_width * 3);
        delete[] image_buffer;

    }

    __host__ __device__
    void render_image_gpu(const HittableList& world, unsigned char* image_buffer) {
        // create a buffer to store the image
    }


    __host__ __device__
    Color rayGradColor(const Ray& ray) const {
        float a = (ray.direction().unit_vector().yy() + 1.0f) / 2.0f; // Normalize to [0, 1] along y-axis
        Color white(1.0, 1.0, 1.0); // start
        Color blue(0.5, 0.7, 1.0); // end
        return white * (1.0f - a) + blue * a; // Linear
    }

    __host__ __device__
    Color rayColor(const Ray& ray, const HittableList& world) const {
        HitRecord rec;
        double t = -1; // Initialize t to -1 to indicate no intersection

        if(world.hitted(ray, Interval(0, INF), rec)) {
            // If the ray hits an object, use the hit record to determine color
            Vector normal = rec.normal;
            normal.make_unit_vector(); // Normalize the normal vector
            // Use the normal to determine color
            Color c = 0.5 * (normal + Color(1, 1, 1)); // Example color based on normal
            return c;
        }


        return rayGradColor(ray); // If no intersection, use gradient color
    }

    __host__ __device__
    Vector sample_square() const {
        // Generate a random offset within the pixel square
        double offset_x = random_double(); // Random value between -0.5 and 0.5
        double offset_y = random_double(); // Random value between -0.5 and 0.5
        return Vector(offset_x - 0.5, offset_y - 0.5, 0.0); // Assuming z-component is not used for 2D pixel sampling
    }

    __host__ __device__
    Ray get_sampled_ray(int i, int j) const {

        Vector offset = sample_square();
        // Vector offset = Vector(0, 0, 0); // Get a random offset for anti-aliasing

        // Calculate the pixel position
        Point pixel_sample = pixel_origin + (d_u * (i + offset.xx())) + (d_v * (j + offset.yy()));
        // Create a ray from the camera origin to the pixel position
        Point ray_origin = camera_origin;
        Vector ray_direction = pixel_sample - ray_origin;
        Ray r =  Ray(ray_origin, ray_direction);
        return r;
    }


__host__ __device__
void write_image_file(const HittableList& world, std::ostream& out, Camera* cam) {

    // Write PPM header
    out << "P3\n" << cam->image_width << ' ' << cam->image_height << "\n255\n";

    // Render the image
    for (int j = 0; j < cam->image_height; j++) {
        for (int i = 0; i < cam->image_width; i++) {
            // Calculate the pixel position
            Point pixel_position = cam->pixel_origin + cam->d_u * i + cam->d_v * j;

            // Create a ray from the camera origin to the pixel position
            Vector ray_direction = pixel_position - cam->camera_origin;
            Ray ray(cam->camera_origin, ray_direction);

            // Check for intersection with the world
            Color color = cam->rayColor(ray, world);

            // Write the color to the output stream
            write_color(out, color);
        }
    }

}

__host__ __device__
void write_image_antialias_file(const HittableList& world, std::ostream& out, Camera* cam) {

    printf(">>> %f\n", cam->pixel_scale);
    // Write PPM header
    out << "P3\n" << cam->image_width << ' ' << cam->image_height << "\n255\n";

    // Render the image
    for (int j = 0; j < cam->image_height; j++) {
        for (int i = 0; i < cam->image_width; i++) {
            // // Calculate the pixel position
            // Point pixel_position = pixel_origin + d_u * i + d_v * j;

            Color accumulated_color(0, 0, 0); // Initialize accumulated color
            for (int s = 0; s < cam->samples_per_pixel; s++) {
                // Create a ray from the camera origin to the pixel position
                Ray ray = cam->get_sampled_ray(i, j);
                // Accumulate the color
                accumulated_color += cam->rayColor(ray, world);
            }
            accumulated_color /= cam->samples_per_pixel; // Average the color over samples
            // std::cout << "Pixel (" << i << ", " << j << ") - Accumulated Color: " 
            //           << accumulated_color.xx() << ", " 
            //           << accumulated_color.yy() << ", " 
            //           << accumulated_color.zz() << std::endl;
            // // Write the color to the output stream
            write_color(out, accumulated_color);
        }
    }

}

__host__ __device__
void write_image_antialias_stb(const HittableList& world, unsigned char* image_buffer,  Camera* cam) {

    for (int j = 0; j < cam->image_height; j++) {
        for (int i = 0; i < cam->image_width; i++) {
            // Calculate the pixel position
            Point pixel_position = cam->pixel_origin + cam->d_u * i + cam->d_v * j;


            // Create a ray from the camera origin to the pixel position
            Vector ray_direction = pixel_position - cam->camera_origin;
            Ray ray(cam->camera_origin, ray_direction);

            // Check for intersection with the world
            Color color = cam->rayColor(ray, world);

            // Write the color to the image buffer
            image_buffer[j * cam->image_width * 3 + i * 3 + 0] = static_cast<unsigned char>(255 * color.xx());
            image_buffer[j * cam->image_width * 3 + i * 3 + 1] = static_cast<unsigned char>(255 * color.yy());
            image_buffer[j * cam->image_width * 3 + i * 3 + 2] = static_cast<unsigned char>(255 * color.zz());
        }
    }
}

__host__ __device__
void write_image_stb(const HittableList& world, unsigned char* image_buffer, Camera* cam) {
    for (int j = 0; j < cam->image_height; j++) {
        for (int i = 0; i < cam->image_width; i++) {
            // Calculate the pixel position
            Point pixel_position = cam->pixel_origin + cam->d_u * i + cam->d_v * j;

            // Create a ray from the camera origin to the pixel position
            Vector ray_direction = pixel_position - cam->camera_origin;
            Ray ray(cam->camera_origin, ray_direction);

            // Check for intersection with the world
            Color color = cam->rayColor(ray, world);

            // Write the color to the image buffer
            image_buffer[j * cam->image_width * 3 + i * 3 + 0] = static_cast<unsigned char>(255 * color.xx());
            image_buffer[j * cam->image_width * 3 + i * 3 + 1] = static_cast<unsigned char>(255 * color.yy());
            image_buffer[j * cam->image_width * 3 + i * 3 + 2] = static_cast<unsigned char>(255 * color.zz());
        }
    }
}


};


#endif // CAMERA_H