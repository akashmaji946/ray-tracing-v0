#ifndef WRITERS_CU
#define WRITERS_CU


#include "commons.hpp"
#include "Camera.hpp"
#include "HittableList.hpp"
#include "Hittable.hpp"
#include "HitRecord.hpp"
#include "Interval.hpp"
#include "Ray.hpp"
#include "Vector.hpp"
#include "utility.hpp"
#include "Vector.hpp"
#include "Vector.hpp"

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

#endif // WRITERS_CU