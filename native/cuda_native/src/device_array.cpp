#include "device_array.hpp"
#include "common.cuh"

#include <functional>
#include <numeric>
#include <utility>

namespace cuda_native {

std::size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32:
            return sizeof(float);
    }
    throw std::runtime_error("unknown dtype");
}

DeviceArray::DeviceArray(std::vector<int64_t> shape, DType dtype, cudaStream_t stream)
    : ctx_(DeviceContext::instance()),
      shape_(std::move(shape)),
      dtype_(dtype),
      stream_(stream != nullptr ? stream : ctx_->default_stream()) {
    std::size_t bytes = nbytes();
    if (bytes > 0) {
        CUDA_CHECK(cudaMallocAsync(&ptr_, bytes, ctx_->pool(), stream_));
    }
}

DeviceArray::~DeviceArray() { release(); }

void DeviceArray::release() noexcept {
    if (ptr_ != nullptr) {
        // Best-effort: a destructor must not throw. Freeing against a
        // stream that is still mid-launch is fine -- cudaFreeAsync is
        // itself stream-ordered, the free only takes effect once prior
        // work on `stream_` completes.
        cudaFreeAsync(ptr_, stream_);
        ptr_ = nullptr;
    }
}

DeviceArray::DeviceArray(DeviceArray&& other) noexcept
    : ctx_(std::move(other.ctx_)),
      shape_(std::move(other.shape_)),
      dtype_(other.dtype_),
      stream_(other.stream_),
      ptr_(other.ptr_) {
    other.ptr_ = nullptr;
}

DeviceArray& DeviceArray::operator=(DeviceArray&& other) noexcept {
    if (this != &other) {
        release();
        ctx_ = std::move(other.ctx_);
        shape_ = std::move(other.shape_);
        dtype_ = other.dtype_;
        stream_ = other.stream_;
        ptr_ = other.ptr_;
        other.ptr_ = nullptr;
    }
    return *this;
}

int64_t DeviceArray::numel() const {
    return std::accumulate(shape_.begin(), shape_.end(), int64_t{1}, std::multiplies<int64_t>());
}

DeviceArray copy_to_device(const void* host_ptr, std::vector<int64_t> shape, DType dtype) {
    DeviceArray array(std::move(shape), dtype);
    std::size_t bytes = array.nbytes();
    if (bytes > 0) {
        CUDA_CHECK(cudaMemcpyAsync(array.data(), host_ptr, bytes, cudaMemcpyHostToDevice, array.stream()));
        array.context()->synchronize(array.stream());
    }
    return array;
}

void copy_to_host(const DeviceArray& array, void* host_ptr) {
    std::size_t bytes = array.nbytes();
    if (bytes > 0) {
        CUDA_CHECK(cudaMemcpyAsync(host_ptr, array.data(), bytes, cudaMemcpyDeviceToHost, array.stream()));
        array.context()->synchronize(array.stream());
    }
}

} // namespace cuda_native
