#include "device_context.hpp"
#include "common.cuh"

#include <mutex>

namespace cuda_native {

namespace {
std::weak_ptr<DeviceContext> g_instance;
std::mutex g_instance_mutex;
} // namespace

std::shared_ptr<DeviceContext> DeviceContext::instance() {
    std::lock_guard<std::mutex> lock(g_instance_mutex);
    if (auto existing = g_instance.lock()) {
        return existing;
    }
    // std::shared_ptr<DeviceContext>(new DeviceContext()) instead of
    // make_shared: constructor is private, only this factory may call it.
    auto created = std::shared_ptr<DeviceContext>(new DeviceContext());
    g_instance = created;
    return created;
}

DeviceContext::DeviceContext() {
    CUDA_CHECK(cudaGetDevice(&device_));
    CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool_, device_));

    // Keep freed blocks in the pool instead of returning them to the OS
    // immediately: training steps repeatedly alloc/free similarly-sized
    // activation buffers, so reuse matters more than returning memory early.
    uint64_t release_threshold = UINT64_MAX;
    CUDA_CHECK(cudaMemPoolSetAttribute(
        pool_, cudaMemPoolAttrReleaseThreshold, &release_threshold));

    CUDA_CHECK(cudaStreamCreateWithFlags(&default_stream_, cudaStreamNonBlocking));
}

DeviceContext::~DeviceContext() {
    if (default_stream_ != nullptr) {
        cudaStreamSynchronize(default_stream_);
        cudaStreamDestroy(default_stream_);
    }
}

cudaStream_t DeviceContext::create_stream() {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    return stream;
}

void DeviceContext::destroy_stream(cudaStream_t stream) {
    if (stream == nullptr) {
        return; // the default stream is owned by this DeviceContext, not the caller
    }
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

void DeviceContext::synchronize(cudaStream_t stream) const {
    CUDA_CHECK(cudaStreamSynchronize(stream != nullptr ? stream : default_stream_));
}

} // namespace cuda_native
