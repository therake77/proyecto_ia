#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "device_context.hpp"

namespace cuda_native {

enum class DType {
    Float32,
};

std::size_t dtype_size(DType dtype);

// A pool-allocated GPU buffer plus the metadata CudaBackend needs to treat
// it as a Tensor's GPU storage (shape, dtype, the stream it was allocated
// and last used on). Allocation and free are both stream-ordered
// (cudaMallocAsync / cudaFreeAsync against the DeviceContext's memory
// pool) -- CudaBackend never touches cupy.ndarray; this is the actual
// storage type a GPU Tensor.data holds.
class DeviceArray {
public:
    DeviceArray(std::vector<int64_t> shape, DType dtype, cudaStream_t stream = nullptr);
    ~DeviceArray();

    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    DeviceArray(DeviceArray&& other) noexcept;
    DeviceArray& operator=(DeviceArray&& other) noexcept;

    void* data() const { return ptr_; }
    template <typename T>
    T* data_as() const { return static_cast<T*>(ptr_); }

    const std::vector<int64_t>& shape() const { return shape_; }
    DType dtype() const { return dtype_; }
    int64_t numel() const;
    std::size_t nbytes() const { return static_cast<std::size_t>(numel()) * dtype_size(dtype_); }
    cudaStream_t stream() const { return stream_; }
    const std::shared_ptr<DeviceContext>& context() const { return ctx_; }

private:
    void release() noexcept;

    std::shared_ptr<DeviceContext> ctx_;
    std::vector<int64_t> shape_;
    DType dtype_ = DType::Float32;
    cudaStream_t stream_ = nullptr;
    void* ptr_ = nullptr;
};

// Host <-> device transfer helpers behind the to_device/to_host bindings.
// `host_ptr` must point to at least the array's nbytes() of contiguous,
// row-major memory.
DeviceArray copy_to_device(const void* host_ptr, std::vector<int64_t> shape, DType dtype);
void copy_to_host(const DeviceArray& array, void* host_ptr);

} // namespace cuda_native
