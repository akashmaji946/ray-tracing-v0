// Minimal CUDA path tracer scaffold rendering a sphere and ground into a PNG

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <chrono>
#include <fstream>
using namespace std;


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

// ------------------ Vector3 class ------------------
class Vector3 {
public:
    float x, y, z;

    __host__ __device__ Vector3() : x(0), y(0), z(0) {}
    __host__ __device__ Vector3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    __host__ __device__ Vector3 operator+(const Vector3& o) const { return Vector3(x + o.x, y + o.y, z + o.z); }
    __host__ __device__ Vector3 operator-(const Vector3& o) const { return Vector3(x - o.x, y - o.y, z - o.z); }
    __host__ __device__ Vector3 operator*(float s) const { return Vector3(x * s, y * s, z * s); }
    __host__ __device__ Vector3& operator+=(const Vector3& o) { x += o.x; y += o.y; z += o.z; return *this; }

    __host__ __device__ float dot(const Vector3& o) const { return x*o.x + y*o.y + z*o.z; }
    __host__ __device__ Vector3 cross(const Vector3& o) const { return Vector3(y*o.z - z*o.y, z*o.x - x*o.z, x*o.y - y*o.x); }
    __host__ __device__ Vector3 normalized() const { float len = sqrtf(this->dot(*this)); return Vector3(x/len, y/len, z/len); }
};

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
    Vector3 origin;
    Vector3 direction;
};

__device__ inline Vector3 ray_at(const RayGPU& r, float t) {
    return r.origin + r.direction * t;
}

// ------------------ Surface Interaction Record ------------------
class SurfaceInteractionRecord {
public:
    __host__ __device__ SurfaceInteractionRecord() : hit(false), t(0.0f) {}

    bool hit;
    Vector3 normal;
    Vector3 position;
    Vector3 bsdf;  // For future material support
    float t;
};

// ------------------ scene classes ------------------
class DeviceSphere {
public:
    Vector3 center;
    float radius;

    __host__ __device__ DeviceSphere() : center(Vector3(0,0,0)), radius(0) {}
    __host__ __device__ DeviceSphere(const Vector3& c, float r) : center(c), radius(r) {}

    __device__ bool hit(const RayGPU& r, float t_min, float t_max, SurfaceInteractionRecord& rec) const {
        Vector3 oc = r.origin - center;
        float a = r.direction.dot(r.direction);
        float half_b = oc.dot(r.direction);
        float c = oc.dot(oc) - radius * radius;
        float discriminant = half_b * half_b - a * c;
        if (discriminant < 0.0f) return false;
        float sqrt_d = sqrtf(discriminant);

        // find nearest root within [t_min, t_max]
        float root = (-half_b - sqrt_d) / a;
        if (root < t_min || root > t_max) {
            root = (-half_b + sqrt_d) / a;
            if (root < t_min || root > t_max) return false;
        }

        rec.t = root;
        rec.position = ray_at(r, rec.t);
        rec.normal = (rec.position - center) * (1.0f / radius);
        rec.hit = true;
        return true;
    }
};

struct DeviceCamera {
    int image_width;
    int image_height;
    Vector3 origin;
    Vector3 pixel_origin;
    Vector3 d_u;
    Vector3 d_v;
};

__device__ inline RayGPU get_ray(const DeviceCamera& cam, float i, float j) {
    Vector3 pixel = cam.pixel_origin + cam.d_u * i + cam.d_v * j;
    return {cam.origin, pixel - cam.origin};
}

__device__ bool hit_sphere(const DeviceSphere& s, const RayGPU& r, float t_min, float t_max, float& t_hit, Vector3& normal) {
    Vector3 oc = r.origin - s.center;
    float a = r.direction.dot(r.direction);
    float half_b = oc.dot(r.direction);
    float c = oc.dot(oc) - s.radius * s.radius;
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
    Vector3 p = ray_at(r, t_hit);
    normal = (p - s.center) * (1.0f / s.radius);
    return true;
}

class DeviceTriangle {
public:
    Vector3 v0, v1, v2;

    __host__ __device__ DeviceTriangle() : v0(Vector3(0,0,0)), v1(Vector3(0,0,0)), v2(Vector3(0,0,0)) {}
    __host__ __device__ DeviceTriangle(const Vector3& v0_, const Vector3& v1_, const Vector3& v2_) : v0(v0_), v1(v1_), v2(v2_) {}

    __device__ bool hit(const RayGPU& r, float t_min, float t_max, SurfaceInteractionRecord& rec) const {
        const float EPS = 1e-7f;
        Vector3 e1 = v1 - v0;
        Vector3 e2 = v2 - v0;
        Vector3 h = r.direction.cross(e2);
        float a = e1.dot(h);
        if (fabsf(a) < EPS) return false; // parallel
        float f = 1.0f / a;
        Vector3 s = r.origin - v0;
        float u = f * s.dot(h);
        if (u < 0.0f || u > 1.0f) return false;
        Vector3 q = s.cross(e1);
        float v = f * r.direction.dot(q);
        if (v < 0.0f || u + v > 1.0f) return false;
        float t = f * e2.dot(q);
        if (t < t_min || t > t_max) return false;
        rec.t = t;
        rec.position = ray_at(r, rec.t);
        rec.normal = e1.cross(e2).normalized();
        rec.hit = true;
        return true;
    }
};

class DeviceQuad {
public:
    Vector3 p0, p1, p2, p3; // two triangles: (p0,p1,p2) and (p0,p2,p3)

    __host__ __device__ DeviceQuad() : p0(Vector3(0,0,0)), p1(Vector3(0,0,0)), p2(Vector3(0,0,0)), p3(Vector3(0,0,0)) {}
    __host__ __device__ DeviceQuad(const Vector3& p0_, const Vector3& p1_, const Vector3& p2_, const Vector3& p3_) : p0(p0_), p1(p1_), p2(p2_), p3(p3_) {}

    __device__ bool hit(const RayGPU& r, float t_min, float t_max, SurfaceInteractionRecord& rec) const {
        DeviceTriangle t0(p0, p1, p2);
        DeviceTriangle t1(p0, p2, p3);
        float t_best = t_max;
        Vector3 n_best(0,0,0);
        bool hit = false;
        SurfaceInteractionRecord rec_local;
        if (t0.hit(r, t_min, t_best, rec_local)) { hit = true; t_best = rec_local.t; n_best = rec_local.normal; }
        if (t1.hit(r, t_min, t_best, rec_local)) { hit = true; t_best = rec_local.t; n_best = rec_local.normal; }
        if (hit) { rec.t = t_best; rec.normal = n_best; rec.position = ray_at(r, rec.t); rec.hit = true; }
        return hit;
    }
};

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
    Vector3 accum(0.0f, 0.0f, 0.0f);
    for (int s = 0; s < samples_per_pixel; ++s) {
        float dx = rand_uniform(seed) - 0.5f; // [-0.5, 0.5)
        float dy = rand_uniform(seed) - 0.5f;
        RayGPU ray = get_ray(cam, static_cast<float>(x) + dx, static_cast<float>(y) + dy);

        // shade: find closest hit
        float t_closest = 1e30f;
        bool  any_hit = false;
        Vector3 nrm(0, 0, 0);
        for (int i = 0; i < num_spheres; ++i) {
            SurfaceInteractionRecord rec;
            if (spheres[i].hit(ray, 0.001f, t_closest, rec)) {
                any_hit = true; t_closest = rec.t; nrm = rec.normal.normalized();
            }
        }
        for (int i = 0; i < num_tris; ++i) {
            SurfaceInteractionRecord rec;
            if (tris[i].hit(ray, 0.001f, t_closest, rec)) {
                any_hit = true; t_closest = rec.t; nrm = rec.normal.normalized();
            }
        }
        for (int i = 0; i < num_quads; ++i) {
            SurfaceInteractionRecord rec;
            if (quads[i].hit(ray, 0.001f, t_closest, rec)) {
                any_hit = true; t_closest = rec.t; nrm = rec.normal.normalized();
            }
        }

        Vector3 color;
        if (any_hit) {
            color = Vector3(0.5f * (nrm.x + 1.0f), 0.5f * (nrm.y + 1.0f), 0.5f * (nrm.z + 1.0f));
        } else {
            Vector3 unit_dir = ray.direction.normalized();
            float t = 0.5f * (unit_dir.y + 1.0f);
            color = Vector3((1.0f - t) + t * 0.5f, (1.0f - t) + t * 0.7f, 1.0f);
        }
        accum += color;
    }
    float inv = 1.0f / static_cast<float>(samples_per_pixel);
    Vector3 color = accum * inv;

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
    hcam.origin = Vector3(0.0f, 0.0f, 0.0f);
    const float focal_length = 1.0f;
    const float viewport_height = 2.0f;
    const float viewport_width = viewport_height * (static_cast<float>(image_width) / static_cast<float>(image_height));
    Vector3 viewport_u(viewport_width, 0.0f, 0.0f);
    Vector3 viewport_v(0.0f, -viewport_height, 0.0f);
    hcam.d_u = viewport_u * (1.0f / static_cast<float>(image_width));
    hcam.d_v = viewport_v * (1.0f / static_cast<float>(image_height));
    Vector3 upper_left = hcam.origin - viewport_u * 0.5f - viewport_v * 0.5f - Vector3(0.0f, 0.0f, focal_length);
    hcam.pixel_origin = upper_left + hcam.d_u * 0.5f + hcam.d_v * 0.5f;

    // scene: two spheres (a small one and a ground), one triangle, one quad
    DeviceSphere hspheres[2];
    hspheres[0] = {Vector3(0.0f, 0.0f, -1.0f), 0.5f};
    hspheres[1] = {Vector3(0.0f, -100.5f, -1.0f), 100.0f};
    DeviceTriangle htris[1];
    htris[0] = {Vector3(-0.5f, -0.25f, -0.8f), Vector3(0.5f, -0.25f, -0.8f), Vector3(0.0f, 0.5f, -0.8f)};
    DeviceQuad hquads[1];
    hquads[0] = {Vector3(-1.0f, -0.8f, -1.2f), Vector3(1.0f, -0.8f, -1.2f), Vector3(1.0f, 0.0f, -1.2f), Vector3(-1.0f, 0.0f, -1.2f)};

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
    const int samples_per_pixel = 10000;
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