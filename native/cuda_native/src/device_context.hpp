#pragma once

#include <cuda_runtime.h>

#include <memory>

namespace cuda_native {

// Process-wide owner of the CUDA device, its stream-ordered memory pool, and
// a default stream. Reference-counted via shared_ptr so every DeviceArray
// that references it keeps it alive -- deliberately avoids a raw global
// singleton, which would be destroyed at an unpredictable point relative to
// CUDA runtime teardown when the interpreter exits.
//
// CUDA Graphs are not implemented here. Every kernel launch (see
// elementwise.cu, matmul_tiled.cu, reductions.cu, softmax_ce.cu) is issued
// against a DeviceContext-owned stream, which is what would let
// cudaStreamBeginCapture/cudaGraphLaunch be added later without touching
// per-kernel code -- not built in this pass.
class DeviceContext : public std::enable_shared_from_this<DeviceContext> {
public:
    static std::shared_ptr<DeviceContext> instance();

    ~DeviceContext();

    DeviceContext(const DeviceContext&) = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;

    int device() const { return device_; }
    cudaMemPool_t pool() const { return pool_; }
    cudaStream_t default_stream() const { return default_stream_; }

    // Convenience for callers that want ops issued on a stream other than
    // the default one. Not a stream pool -- just create/destroy pairs.
    cudaStream_t create_stream();
    void destroy_stream(cudaStream_t stream);

    void synchronize(cudaStream_t stream = nullptr) const;

private:
    DeviceContext();

    int device_ = 0;
    cudaMemPool_t pool_ = nullptr;
    cudaStream_t default_stream_ = nullptr;
};

} // namespace cuda_native
