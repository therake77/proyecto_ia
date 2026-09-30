#include "reductions.cuh"
#include "common.cuh"

namespace cuda_native {

void launch_sum_axis(const DeviceArray&, int, bool, DeviceArray&) {
    throw NotImplemented("launch_sum_axis: in-block tree reduction not implemented yet");
}

void launch_max_axis(const DeviceArray&, int, bool, DeviceArray&) {
    throw NotImplemented("launch_max_axis: in-block tree reduction not implemented yet");
}

} // namespace cuda_native
