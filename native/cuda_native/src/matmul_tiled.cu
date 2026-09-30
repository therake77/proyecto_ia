#include "matmul_tiled.cuh"
#include "common.cuh"

// Ported from src/backend/cuda_kernels/matmul_tiled.py: MATMUL_NAIVE_SRC is
// fully written there and ported verbatim below. MATMUL_TILED_SRC was
// already a TODO stub ("// TODO: shared-memory tiling") -- kept as a TODO
// here too, now throwing NotImplemented instead of compiling to a kernel
// that silently does nothing.

namespace cuda_native {

namespace {

constexpr int kTileDim = 16;

__global__ void matmul_naive_kernel(const float* a, const float* b, float* out, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float acc = 0.0f;
        for (int i = 0; i < k; ++i) acc += a[row * k + i] * b[i * n + col];
        out[row * n + col] = acc;
    }
}

} // namespace

void launch_matmul_naive(const DeviceArray& a, const DeviceArray& b, DeviceArray& out) {
    const auto& a_shape = a.shape();
    const auto& b_shape = b.shape();
    if (a_shape.size() != 2 || b_shape.size() != 2 || a_shape[1] != b_shape[0]) {
        throw std::invalid_argument("launch_matmul_naive: expected compatible 2D shapes");
    }
    int m = static_cast<int>(a_shape[0]);
    int k = static_cast<int>(a_shape[1]);
    int n = static_cast<int>(b_shape[1]);

    dim3 block(kTileDim, kTileDim);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    matmul_naive_kernel<<<grid, block, 0, out.stream()>>>(
        a.data_as<float>(), b.data_as<float>(), out.data_as<float>(), m, k, n);
    CUDA_CHECK(cudaGetLastError());
}

void launch_matmul_tiled(const DeviceArray&, const DeviceArray&, DeviceArray&) {
    throw NotImplemented("launch_matmul_tiled: shared-memory tiling not implemented yet");
}

} // namespace cuda_native
