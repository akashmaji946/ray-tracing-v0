#include <cstdio>
#include <iostream>
#include "include/Camera.hpp"
#include "include/Color.hpp"
#include "include/Ray.hpp"
#include "include/Triangle.hpp"
#include "include/Sphere.hpp"
#include "include/SurfaceInteraction.hpp"
#include "include/Scene.hpp"
#include <curand_kernel.h>
#include <iostream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "common/stb_image_write.h"

// Error checking macro
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

// CUDA kernel
__global__ void render(unsigned char* image, const Camera* cam, \
    const Scene* scene, \
    int image_width, int image_height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= image_width || y >= image_height) return;

    Ray r = cam->get_ray(x, y);

    float closest_t = 1e30f;
    int hit_type = -1; // 0: triangle, 1: sphere
    Color pixel_color(0, 0, 0);
    SurfaceInteraction closest_si;

    for (int i = 0; i < scene->num_objects; ++i) {
        SurfaceInteraction si;
        if (scene->objects[i].intersect_si(r, si) && si.t < closest_t) {
            closest_t = si.t;
            hit_type = (scene->objects[i].type == OBJ_TRIANGLE) ? 0 : 1;
            closest_si = si;
        }
    }

    if (hit_type == 0) {
        pixel_color = Color(1, 0, 0); // Red for triangle
    } else if (hit_type == 1) {
        pixel_color = Color(0, 0, 1); // Blue for sphere
    } else {
        pixel_color = Color(0, 0, 0); // Black background
    }

    int idx = (y * image_width + x) * 3;
    image[idx + 0] = static_cast<unsigned char>(255.99f * pixel_color.m_r);
    image[idx + 1] = static_cast<unsigned char>(255.99f * pixel_color.m_g);
    image[idx + 2] = static_cast<unsigned char>(255.99f * pixel_color.m_b);
}


__global__ void render_antialias(unsigned char* image, const Camera* cam, \
    const Scene* scene, \
    int image_width, int image_height) {
        
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= image_width || y >= image_height) return;

    // Initialize random number generator for this thread
    curandState_t state;
    int idx = (y * image_width + x) * 3;
    curand_init(idx, 0, 0, &state);

    Color pixel_color(0, 0, 0);
    const int num_samples = cam->samples_per_pixel;

    for (int s = 0; s < num_samples; ++s) {
        float random_x_offset = curand_uniform(&state) - 0.5f;
        float random_y_offset = curand_uniform(&state) - 0.5f;

        // Add the random offset to the pixel coordinates.
        float u = (x + random_x_offset);
        float v = (y + random_y_offset);

        Ray r = cam->get_ray(u, v);
        
        float closest_t = 1e30f;
        int hit_type = -1; // 0: triangle, 1: sphere
        SurfaceInteraction closest_si;

        for (int i = 0; i < scene->num_objects; ++i) {
            SurfaceInteraction si;
            if (scene->objects[i].intersect_si(r, si) && si.t < closest_t) {
                closest_t = si.t;
                hit_type = (scene->objects[i].type == OBJ_TRIANGLE) ? 0 : 1;
                closest_si = si;
            }
        }

        if (hit_type == 0) {
            pixel_color += Color(1, 0, 0); // Red for triangle
        } else if (hit_type == 1) {
            pixel_color += Color(0, 0, 1); // Blue for sphere
        } else {
            pixel_color += Color(0, 0, 0); // Black background
        }
    }

    pixel_color /= static_cast<float>(num_samples);

    
    image[idx + 0] = static_cast<unsigned char>(255.99f * pixel_color.m_r);
    image[idx + 1] = static_cast<unsigned char>(255.99f * pixel_color.m_g);
    image[idx + 2] = static_cast<unsigned char>(255.99f * pixel_color.m_b);
}

__global__ void render_cool(unsigned char* image, const Camera* cam, \
    const Scene* scene, \
    int image_width, int image_height) {
        
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= image_width || y >= image_height) return;

    curandState_t state;
    curand_init(x + y * image_width, 0, 0, &state);

    Color pixel_color(0, 0, 0);
    const int num_samples = 100;

    for (int s = 0; s < num_samples; ++s) {
        float random_x_offset = curand_uniform(&state) - 0.5f;
        float random_y_offset = curand_uniform(&state) - 0.5f;

        float u = (x + random_x_offset);
        float v = (y + random_y_offset);

        Ray r = cam->get_ray(u, v);
        
        float closest_t = 1e30f;
        int hit_type = -1; // 0: triangle, 1: sphere
        SurfaceInteraction closest_si;

        for (int i = 0; i < scene->num_objects; ++i) {
            SurfaceInteraction si;
            if (scene->objects[i].intersect_si(r, si) && si.t < closest_t) {
                closest_t = si.t;
                hit_type = (scene->objects[i].type == OBJ_TRIANGLE) ? 0 : 1;
                closest_si = si;
            }
        }

        if (hit_type != -1) {
            // Normalize the normal vector to ensure its components are within [-1, 1].
            Vector N = (closest_si.normal).unit_vector();
            // Map the vector components from [-1, 1] to [0, 1] for color.
            pixel_color +=  Color(N.m_x + 1, N.m_y + 1, N.m_z + 1) * 0.5f; 
        } else {
            // Background color
            pixel_color += Color(0.1f, 0.1f, 0.1f);
        }
    }

    pixel_color /= static_cast<float>(num_samples);
    
    // Clamp the color values to a max of 1.0
    pixel_color.m_r = fminf(pixel_color.m_r, 1.0f);
    pixel_color.m_g = fminf(pixel_color.m_g, 1.0f);
    pixel_color.m_b = fminf(pixel_color.m_b, 1.0f);

    int idx = (y * image_width + x) * 3;
    image[idx + 0] = static_cast<unsigned char>(255.99f * pixel_color.m_r);
    image[idx + 1] = static_cast<unsigned char>(255.99f * pixel_color.m_g);
    image[idx + 2] = static_cast<unsigned char>(255.99f * pixel_color.m_b);
}

int main() {

    const int image_width = 1920, 
             image_height = 1080;
    const int samples_per_pixel = 100;

    size_t img_size = image_width * image_height * 3;

    unsigned char *d_image;
    unsigned char *h_image = new unsigned char[img_size];

    checkCudaErrors(cudaMalloc((void**)&d_image, img_size * sizeof(unsigned char)));

    dim3 block(16, 16);
    dim3 grid((image_width + block.x - 1) / block.x,
              (image_height + block.y - 1) / block.y);

    // Camera setup
    Camera cam(16.0/9.0, image_width, image_height, samples_per_pixel);
    cam.init();

    Camera *d_cam;
    checkCudaErrors(cudaMalloc((void**)&d_cam, sizeof(Camera)));
    checkCudaErrors(cudaMemcpy(d_cam, &cam, sizeof(Camera), cudaMemcpyHostToDevice));

    // Use:
    const int num_objects = 2;
    Object objects[num_objects];

    // Add a triangle
    objects[0].type = OBJ_TRIANGLE;
    objects[0].triangle = Triangle(
        Vector(-0.5f, -0.5f, -1.0f),
        Vector(0.5f, -0.5f, -1.0f),
        Vector(0.0f, 0.5f, -1.0f)
    );

    // Add a sphere
    objects[1].type = OBJ_SPHERE;
    objects[1].sphere = Sphere(Point(0, 0, -2), 1.0f);

    // Copy objects to device
    Object* d_objects;
    size_t objects_size = num_objects * sizeof(Object);
    checkCudaErrors(cudaMalloc((void**)&d_objects, objects_size));
    checkCudaErrors(cudaMemcpy(d_objects, objects, objects_size, cudaMemcpyHostToDevice));

    // Setup scene struct
    Scene h_scene;
    h_scene.objects = d_objects;
    h_scene.num_objects = num_objects;

    Scene* d_scene;
    checkCudaErrors(cudaMalloc((void**)&d_scene, sizeof(Scene)));
    checkCudaErrors(cudaMemcpy(d_scene, &h_scene, sizeof(Scene), cudaMemcpyHostToDevice));

    // Launch kernel
    printf(">Rendering...\n");
    render_cool<<<grid, block>>>(d_image, d_cam, d_scene, image_width, image_height);
    
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(h_image, d_image, img_size, cudaMemcpyDeviceToHost));

    stbi_write_png("image.png", image_width, image_height, 3, h_image, image_width * 3);
    std::cout << "Image saved as image.png\n";

    cudaFree(d_image);
    cudaFree(d_cam);
    cudaFree(d_objects);
    cudaFree(d_scene);
    delete[] h_image;
    cudaDeviceReset();

    return 0;
}