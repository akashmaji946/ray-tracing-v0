// Minimal CUDA path tracer scaffold rendering a sphere and ground into a PNG

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <chrono>
using namespace std::chrono;

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

static inline void check_cuda(cudaError_t result, char const* const func, char const* const file, int const line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result)
                  << " at " << file << ":" << line << " '" << func << "'\n";
        cudaDeviceReset();
        std::exit(99);
    }
}

// ------------------ math helpers ------------------
__device__ __host__ inline float3 make_float3_add(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ __host__ inline float3 make_float3_sub(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ __host__ inline float3 make_float3_mul(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__device__ __host__ inline float dot3(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ __host__ inline float3 cross3(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}
__device__ inline float3 normalize3(const float3& v) {
    float len = sqrtf(dot3(v, v));
    return make_float3(v.x / len, v.y / len, v.z / len);
}

// ------------------ tiny RNG ------------------
__device__ inline uint32_t lcg_step(uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}
__device__ inline float rand_uniform(uint32_t& state) {
    // Take top 24 bits to get a float in [0,1)
    return (lcg_step(state) >> 8) * (1.0f / 16777216.0f);
}

struct RayGPU {
    float3 origin;
    float3 direction;
};

__device__ inline float3 ray_at(const RayGPU& r, float t) {
    return make_float3_add(r.origin, make_float3_mul(r.direction, t));
}

// ------------------ scene ------------------
struct DeviceSphere {
    float3 center;
    float  radius;
};

struct DeviceCamera {
    int image_width;
    int image_height;
    float3 origin;
    float3 pixel_origin;
    float3 d_u;
    float3 d_v;
};

__device__ inline RayGPU get_ray(const DeviceCamera& cam, float i, float j) {
    float3 pixel = make_float3_add(cam.pixel_origin,
                                   make_float3_add(make_float3_mul(cam.d_u, i),
                                                   make_float3_mul(cam.d_v, j)));
    return {cam.origin, make_float3_sub(pixel, cam.origin)};
}

__device__ bool hit_sphere(const DeviceSphere& s, const RayGPU& r, float t_min, float t_max, float& t_hit, float3& normal) {
    float3 oc = make_float3_sub(r.origin, s.center);
    float a = dot3(r.direction, r.direction);
    float half_b = dot3(oc, r.direction);
    float c = dot3(oc, oc) - s.radius * s.radius;
    float discriminant = half_b * half_b - a * c;
    if (discriminant < 0.0f) return false;
    float sqrt_d = sqrtf(discriminant);

    // find nearest root within [t_min, t_max]
    float root = (-half_b - sqrt_d) / a;
    if (root < t_min || root > t_max) {
        root = (-half_b + sqrt_d) / a;
        if (root < t_min || root > t_max) return false;
    }

    t_hit = root;
    float3 p = ray_at(r, t_hit);
    normal = make_float3_mul(make_float3_sub(p, s.center), 1.0f / s.radius);
    return true;
}

// ------------- triangles and quads -------------
struct DeviceTriangle {
    float3 v0, v1, v2;
};

__device__ bool hit_triangle(const DeviceTriangle& tri, const RayGPU& r, float t_min, float t_max, float& t_hit, float3& normal) {
    const float EPS = 1e-7f;
    float3 e1 = make_float3_sub(tri.v1, tri.v0);
    float3 e2 = make_float3_sub(tri.v2, tri.v0);
    float3 h = cross3(r.direction, e2);
    float a = dot3(e1, h);
    if (fabsf(a) < EPS) return false; // parallel
    float f = 1.0f / a;
    float3 s = make_float3_sub(r.origin, tri.v0);
    float u = f * dot3(s, h);
    if (u < 0.0f || u > 1.0f) return false;
    float3 q = cross3(s, e1);
    float v = f * dot3(r.direction, q);
    if (v < 0.0f || u + v > 1.0f) return false;
    float t = f * dot3(e2, q);
    if (t < t_min || t > t_max) return false;
    t_hit = t;
    normal = normalize3(cross3(e1, e2));
    return true;
}

struct DeviceQuad {
    float3 p0, p1, p2, p3; // two triangles: (p0,p1,p2) and (p0,p2,p3)
};

__device__ bool hit_quad(const DeviceQuad& qd, const RayGPU& r, float t_min, float t_max, float& t_hit, float3& normal) {
    DeviceTriangle t0{qd.p0, qd.p1, qd.p2};
    DeviceTriangle t1{qd.p0, qd.p2, qd.p3};
    float t_best = t_max;
    float3 n_best = make_float3(0,0,0);
    bool hit = false;
    float t_hit_local; float3 n_local;
    if (hit_triangle(t0, r, t_min, t_best, t_hit_local, n_local)) { hit = true; t_best = t_hit_local; n_best = n_local; }
    if (hit_triangle(t1, r, t_min, t_best, t_hit_local, n_local)) { hit = true; t_best = t_hit_local; n_best = n_local; }
    if (hit) { t_hit = t_best; normal = n_best; }
    return hit;
}

// ------------------ kernel ------------------
__global__ void render_kernel(unsigned char* image,
                              DeviceCamera cam,
                              const DeviceSphere* spheres, int num_spheres,
                              const DeviceTriangle* tris, int num_tris,
                              const DeviceQuad* quads, int num_quads,
                              int samples_per_pixel) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cam.image_width || y >= cam.image_height) return;

    int idx = (y * cam.image_width + x) * 3;

    // Accumulate samples with subpixel jitter
    uint32_t seed = static_cast<uint32_t>(y * cam.image_width + x) ^ 0x9E3779B9u;
    float3 accum = make_float3(0.0f, 0.0f, 0.0f);
    for (int s = 0; s < samples_per_pixel; ++s) {
        float dx = rand_uniform(seed) - 0.5f; // [-0.5, 0.5)
        float dy = rand_uniform(seed) - 0.5f;
        RayGPU ray = get_ray(cam, static_cast<float>(x) + dx, static_cast<float>(y) + dy);

        // shade: find closest hit
        float t_closest = 1e30f;
        bool  any_hit = false;
        float3 nrm = make_float3(0, 0, 0);
        for (int i = 0; i < num_spheres; ++i) {
            float t_hit; float3 normal;
            if (hit_sphere(spheres[i], ray, 0.001f, t_closest, t_hit, normal)) {
                any_hit = true; t_closest = t_hit; nrm = normalize3(normal);
            }
        }
        for (int i = 0; i < num_tris; ++i) {
            float t_hit; float3 normal;
            if (hit_triangle(tris[i], ray, 0.001f, t_closest, t_hit, normal)) {
                any_hit = true; t_closest = t_hit; nrm = normalize3(normal);
            }
        }
        for (int i = 0; i < num_quads; ++i) {
            float t_hit; float3 normal;
            if (hit_quad(quads[i], ray, 0.001f, t_closest, t_hit, normal)) {
                any_hit = true; t_closest = t_hit; nrm = normalize3(normal);
            }
        }

        float3 color;
        if (any_hit) {
            color = make_float3(0.5f * (nrm.x + 1.0f), 0.5f * (nrm.y + 1.0f), 0.5f * (nrm.z + 1.0f));
        } else {
            float3 unit_dir = normalize3(ray.direction);
            float t = 0.5f * (unit_dir.y + 1.0f);
            color = make_float3((1.0f - t) + t * 0.5f, (1.0f - t) + t * 0.7f, 1.0f);
        }
        accum = make_float3_add(accum, color);
    }
    float inv = 1.0f / static_cast<float>(samples_per_pixel);
    float3 color = make_float3_mul(accum, inv);

    image[idx + 0] = static_cast<unsigned char>(fminf(255.0f, 255.99f * color.x));
    image[idx + 1] = static_cast<unsigned char>(fminf(255.0f, 255.99f * color.y));
    image[idx + 2] = static_cast<unsigned char>(fminf(255.0f, 255.99f * color.z));
}

// ------------------ host ------------------
int main() {

    // measure the time it takes to render the image
   

    // camera setup (host computes parameters and passes a POD struct)
    const float aspect_ratio = 16.0f / 9.0f;
    const int   image_width = 1920;
    const int   image_height = static_cast<int>(image_width / aspect_ratio);

    DeviceCamera hcam{};
    hcam.image_width = image_width;
    hcam.image_height = image_height;
    hcam.origin = make_float3(0.0f, 0.0f, 0.0f);
    const float focal_length = 1.0f;
    const float viewport_height = 2.0f;
    const float viewport_width = viewport_height * (static_cast<float>(image_width) / static_cast<float>(image_height));
    float3 viewport_u = make_float3(viewport_width, 0.0f, 0.0f);
    float3 viewport_v = make_float3(0.0f, -viewport_height, 0.0f);
    hcam.d_u = make_float3_mul(viewport_u, 1.0f / static_cast<float>(image_width));
    hcam.d_v = make_float3_mul(viewport_v, 1.0f / static_cast<float>(image_height));
    float3 upper_left = make_float3_sub(
        make_float3_sub(make_float3_sub(hcam.origin, make_float3_mul(viewport_u, 0.5f)),
                         make_float3_mul(viewport_v, 0.5f)),
        make_float3(0.0f, 0.0f, focal_length));
    hcam.pixel_origin = make_float3_add(upper_left, make_float3_add(make_float3_mul(hcam.d_u, 0.5f), make_float3_mul(hcam.d_v, 0.5f)));

    // scene: two spheres (a small one and a ground), one triangle, one quad
    DeviceSphere hspheres[2];
    hspheres[0] = {make_float3(0.0f, 0.0f, -1.0f), 0.5f};
    hspheres[1] = {make_float3(0.0f, -100.5f, -1.0f), 100.0f};
    DeviceTriangle htris[1];
    htris[0] = {make_float3(-0.5f, -0.25f, -0.8f), make_float3(0.5f, -0.25f, -0.8f), make_float3(0.0f, 0.5f, -0.8f)};
    DeviceQuad hquads[1];
    hquads[0] = {make_float3(-1.0f, -0.8f, -1.2f), make_float3(1.0f, -0.8f, -1.2f), make_float3(1.0f, 0.0f, -1.2f), make_float3(-1.0f, 0.0f, -1.2f)};

    DeviceSphere* dspheres = nullptr;
    DeviceTriangle* dtris = nullptr;
    DeviceQuad* dquads = nullptr;
    checkCudaErrors(cudaMalloc(&dspheres, sizeof(hspheres)));
    checkCudaErrors(cudaMemcpy(dspheres, hspheres, sizeof(hspheres), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc(&dtris, sizeof(htris)));
    checkCudaErrors(cudaMemcpy(dtris, htris, sizeof(htris), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc(&dquads, sizeof(hquads)));
    checkCudaErrors(cudaMemcpy(dquads, hquads, sizeof(hquads), cudaMemcpyHostToDevice));

    // image buffers
    const size_t num_pixels = static_cast<size_t>(image_width) * static_cast<size_t>(image_height);
    unsigned char* h_image = new unsigned char[num_pixels * 3];
    unsigned char* d_image = nullptr;
    checkCudaErrors(cudaMalloc(&d_image, num_pixels * 3 * sizeof(unsigned char)));


    auto start = std::chrono::high_resolution_clock::now();
    // launch
    dim3 block(16, 16);
    dim3 grid((image_width + block.x - 1) / block.x, (image_height + block.y - 1) / block.y);
    const int samples_per_pixel = 100;
    render_kernel<<<grid, block>>>(d_image, hcam, dspheres, 2, dtris, 1, dquads, 1, samples_per_pixel);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // copy back and save
    checkCudaErrors(cudaMemcpy(h_image, d_image, num_pixels * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    stbi_write_png("works.png", image_width, image_height, 3, h_image, image_width * 3);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Time taken to render the image [GPU]: " << duration.count() << " milliseconds" << std::endl;

    // cleanup
    checkCudaErrors(cudaFree(d_image));
    checkCudaErrors(cudaFree(dspheres));
    checkCudaErrors(cudaFree(dtris));
    checkCudaErrors(cudaFree(dquads));
    delete[] h_image;
    return 0;
}