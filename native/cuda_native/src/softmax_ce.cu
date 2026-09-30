#include "softmax_ce.cuh"
#include "common.cuh"

namespace cuda_native {

void launch_softmax_cross_entropy(const DeviceArray&, const DeviceArray&, DeviceArray&, DeviceArray&) {
    throw NotImplemented(
        "launch_softmax_cross_entropy: fused softmax+cross-entropy not implemented yet");
}

} // namespace cuda_native
