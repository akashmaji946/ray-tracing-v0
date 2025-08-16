#ifndef CAMERA_H
#define CAMERA_H

#include "../common/common.hpp"
#include "../include/Vector.hpp"
#include "../include/Point.hpp"
#include "../include/Ray.hpp"

class Camera {
public:
    // Constructors
    __host__ __device__
    Camera()
        : aspect_ratio(16.0 / 9.0), image_width(800), image_height(450), samples_per_pixel(10),
          pixel_scale(1.0 / 10), INIT(false), camera_origin(Point(0, 0, 0)), focal_length(1.0),
          viewport_u(Vector(0, 0, 0)), viewport_v(Vector(0, 0, 0)), d_u(Vector(0, 0, 0)), d_v(Vector(0, 0, 0)),
          upper_left_corner(Point(0, 0, 0)), pixel_origin(Point(0, 0, 0)) {}

    __host__ __device__
    Camera(double aspect_ratio, int image_width, int image_height, int samples_per_pixel)
        : aspect_ratio(aspect_ratio), image_width(image_width), image_height(image_height), samples_per_pixel(samples_per_pixel),
          pixel_scale(1.0 / samples_per_pixel), INIT(false), camera_origin(Point(0, 0, 0)), focal_length(1.0),
          viewport_u(Vector(0, 0, 0)), viewport_v(Vector(0, 0, 0)), d_u(Vector(0, 0, 0)), d_v(Vector(0, 0, 0)),
          upper_left_corner(Point(0, 0, 0)), pixel_origin(Point(0, 0, 0)) {}

    // Methods
    __host__ __device__ void init() {
        if (INIT) return;
        image_height = static_cast<int>(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;
        pixel_scale = 1.0 / static_cast<double>(samples_per_pixel);
        camera_origin = Point(0, 0, 0);
        focal_length = 1.0;
        double viewport_height = 2.0;
        double viewport_width = viewport_height * ((double)(image_width) / (double)(image_height));
        viewport_u = Vector(viewport_width, 0, 0);
        viewport_v = Vector(0, -viewport_height, 0);
        d_u = viewport_u / static_cast<double>(image_width);
        d_v = viewport_v / static_cast<double>(image_height);
        Vector v = camera_origin - viewport_u / 2.0 - viewport_v / 2.0 - Vector(0, 0, focal_length);
        upper_left_corner = Point(v.m_x, v.m_y, v.m_z);
        Point pixel_origin_vec = upper_left_corner + d_u * 0.5 + d_v * 0.5;
        pixel_origin = Point(pixel_origin_vec.m_x, pixel_origin_vec.m_y, pixel_origin_vec.m_z);
        INIT = true;
    }

    __host__ __device__ Ray get_ray(int i, int j) const {
        Point pixel_sample = pixel_origin + (d_u * i) + (d_v * j);
        Point ray_origin = camera_origin;
        Vector ray_direction = pixel_sample - ray_origin;
        return Ray(ray_origin, ray_direction);
    }

public:
    double aspect_ratio;
    int image_width;
    int image_height;
    int samples_per_pixel;
    double pixel_scale;
    bool INIT;

    Point camera_origin;
    double focal_length;
    Vector viewport_u;
    Vector viewport_v;
    Vector d_u;
    Vector d_v;
    Point upper_left_corner;
    Point pixel_origin;
};

#endif // CAMERA_H