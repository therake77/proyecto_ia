#include "elementwise.cuh"
#include "common.cuh"

// Ported from the original RawKernel source strings in
// src/backend/cuda_kernels/elementwise.py (ELEMENTWISE_ADD_SRC,
// RELU_FORWARD_SRC, RELU_BACKWARD_SRC) -- same math, now compiled instead of
// JIT'd through cupy.RawKernel.
//
// TODO(E4): broadcasting (M4's "elementwise con broadcasting" requirement)
// and a `mul` kernel are not implemented yet, matching the original source
// files, which only had add/relu. Add them together with
// CudaBackend.mul / CudaBackend.add's broadcast path.

namespace cuda_native {

namespace {

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}

__global__ void relu_forward_kernel(const float* x, float* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = x[idx] > 0.0f ? x[idx] : 0.0f;
}

__global__ void relu_backward_kernel(const float* x, const float* grad_out, float* grad_in, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) grad_in[idx] = x[idx] > 0.0f ? grad_out[idx] : 0.0f;
}

} // namespace

void launch_add(const DeviceArray& a, const DeviceArray& b, DeviceArray& out) {
    if (a.shape() != b.shape() || a.shape() != out.shape()) {
        throw std::invalid_argument("launch_add: shapes must match (broadcasting not implemented yet)");
    }
    int64_t n = out.numel();
    int grid = grid_size_for(n);
    elementwise_add_kernel<<<grid, kDefaultBlockSize, 0, out.stream()>>>(
        a.data_as<float>(), b.data_as<float>(), out.data_as<float>(), n);
    CUDA_CHECK(cudaGetLastError());
}

void launch_relu_forward(const DeviceArray& x, DeviceArray& out) {
    if (x.shape() != out.shape()) {
        throw std::invalid_argument("launch_relu_forward: shapes must match");
    }
    int64_t n = out.numel();
    int grid = grid_size_for(n);
    relu_forward_kernel<<<grid, kDefaultBlockSize, 0, out.stream()>>>(
        x.data_as<float>(), out.data_as<float>(), n);
    CUDA_CHECK(cudaGetLastError());
}

void launch_relu_backward(const DeviceArray& x, const DeviceArray& grad_out, DeviceArray& grad_in) {
    if (x.shape() != grad_out.shape() || x.shape() != grad_in.shape()) {
        throw std::invalid_argument("launch_relu_backward: shapes must match");
    }
    int64_t n = grad_in.numel();
    int grid = grid_size_for(n);
    relu_backward_kernel<<<grid, kDefaultBlockSize, 0, grad_in.stream()>>>(
        x.data_as<float>(), grad_out.data_as<float>(), grad_in.data_as<float>(), n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda_native
