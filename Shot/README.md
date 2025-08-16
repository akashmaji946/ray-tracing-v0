# Shot

A CUDA-based ray tracing project.

## Project Structure

```
Shot/
  ├── include/    # Header files (e.g., Ray.hpp, Sphere.hpp, ...)
  ├── src/        # Source files implementing the headers
  ├── common/     # Common utilities and standard headers (common.hpp, common.cpp)
  ├── main.cu     # Main CUDA entry point
  ├── CMakeLists.txt
  └── README.md
```

## Build Instructions

1. **Install CMake (>= 3.18) and a CUDA-capable compiler.**
2. From the project root, run:
   ```sh
   mkdir build && cd build
   cmake ..
   make
   ```
3. The executable `Shot` will be created in the `build/` directory.

## Notes
- Place your header files in `include/` (e.g., `Ray.hpp`, `Sphere.hpp`, ...).
- Place their implementations in `src/` (e.g., `Ray.cpp`, `Sphere.cpp`, ...).
- Use `common/` for shared utilities and headers.
- The main entry point is `main.cu`.
