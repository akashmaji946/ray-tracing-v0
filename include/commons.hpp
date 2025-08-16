#ifndef COMMONS_H
#define COMMONS_H

#include <stdexcept>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <sstream>
#include <algorithm>
#include <cassert>
#include <iomanip>
#include <limits>
#include <utility>
#include <time.h>
#include <random>
#include <chrono>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
#define __host__
#define __device__
#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

#endif // COMMONS_H