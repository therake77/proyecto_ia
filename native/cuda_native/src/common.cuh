#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace cuda_native {

// Thrown by kernels that are intentionally not implemented yet (matmul
// tiling, reductions, fused softmax-CE -- see each .cu file's TODO). Mapped
// to Python's builtin NotImplementedError in bindings.cpp, matching the
// `raise NotImplementedError` convention used everywhere else in this
// project's Python skeleton.
struct NotImplemented : std::runtime_error {
    using std::runtime_error::runtime_error;
};

inline void throw_on_cuda_error(cudaError_t err, const char* expr, const char* file, int line) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string(expr) + " failed: " + cudaGetErrorString(err) +
            " (" + file + ":" + std::to_string(line) + ")");
    }
}

constexpr int kDefaultBlockSize = 256;

inline int grid_size_for(int64_t n, int block_size = kDefaultBlockSize) {
    return static_cast<int>((n + block_size - 1) / block_size);
}

} // namespace cuda_native

#define CUDA_CHECK(expr) \
    ::cuda_native::throw_on_cuda_error((expr), #expr, __FILE__, __LINE__)
