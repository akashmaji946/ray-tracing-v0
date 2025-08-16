#include <iostream>
#include "common/common.hpp"
#include "include/Ray.hpp"
#include "include/Vector.hpp"
#include "include/Point.hpp"
#include "include/Color.hpp"
#include "include/Triangle.hpp"
#include "include/Camera.hpp"
#include "include/SurfaceInteraction.hpp"

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

// Dummy integrator: always returns blue
__device__ void integrator(const Ray* ray_, SurfaceInteraction* si, Color* L, int max_depth) {
    for (int i = 0; i < max_depth; i++)
        *L = Color(0, 0, 1);
}

// CUDA kernel
__global__ void render(unsigned char* image,  const Camera* cam, const Triangle* tri, int image_width, int image_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= image_width || y >= image_height) return;

    // Get ray for pixel (assume Camera::get_ray is __device__)
    Ray r = cam->get_ray(x, y);

    SurfaceInteraction si;
    Color col;
    integrator(&r, &si, &col, 5);

    int idx = (y * image_width + x) * 3;
    image[idx + 0] = static_cast<unsigned char>(255.99f * col.m_r);
    image[idx + 1] = static_cast<unsigned char>(255.99f * col.m_g);
    image[idx + 2] = static_cast<unsigned char>(255.99f * col.m_b);
}

int main() {

    const int image_width = 2000, image_height = 2000;
    size_t img_size = image_width * image_height * 3;

    unsigned char *d_image;
    unsigned char *h_image = new unsigned char[img_size];

    checkCudaErrors(cudaMalloc((void**)&d_image, img_size * sizeof(unsigned char)));

    dim3 block(16, 16);
    dim3 grid((image_width + block.x - 1) / block.x,
              (image_height + block.y - 1) / block.y);

    // Camera setup (adapt to your Camera constructor)
    Point camera_origin(0, 0, 0);
    double focal_length = 1.0;
    double viewport_height = 2.0;
    double viewport_width = viewport_height * (double(image_width) / image_height);

    Camera cam( 16.0/9.0, image_width, image_height, 100);
    cam.init();

    Camera *d_cam;
    checkCudaErrors(cudaMalloc((void**)&d_cam, sizeof(Camera)));
    checkCudaErrors(cudaMemcpy(d_cam, &cam, sizeof(Camera), cudaMemcpyHostToDevice));

    // Triangle setup (adapt to your Triangle constructor)
    Triangle tri(
        Vector(-0.5f, -0.5f, -1.0f),
        Vector(0.5f, -0.5f, -1.0f),
        Vector(0.0f, 0.5f, -1.0f)
    );

    Triangle* d_tri;
    checkCudaErrors(cudaMalloc((void**)&d_tri, sizeof(Triangle)));
    checkCudaErrors(cudaMemcpy(d_tri, &tri, sizeof(Triangle), cudaMemcpyHostToDevice));

    // Launch kernel
    render<<<grid, block>>>(d_image, d_cam, d_tri, image_width, image_height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(h_image, d_image, img_size, cudaMemcpyDeviceToHost));

    stbi_write_png("image.png", image_width, image_height, 3, h_image, image_width * 3);
    std::cout << "Image saved as image.png\n";

    cudaFree(d_image);
    cudaFree(d_cam);
    cudaFree(d_tri);
    delete[] h_image;
    cudaDeviceReset();

    return 0;
}


/*

I am now only sending triangles, I want to send a scene object, which contains many triangles, spheres, squares etc. we can create a struct for Object, specifying object type as triangle, square, ... and use a Scene struct to hold these objects, and pass the Scene object to the render, and check inside render for nearest interaction for each ray
*/