#pragma once

#include "device_array.hpp"

namespace cuda_native {

// Both TODO(E4, week 4): ported from src/backend/cuda_kernels/reductions.py
// (SUM_AXIS_TREE_SRC / MAX_AXIS_TREE_SRC), which were already TODO stubs
// there ("// TODO: in-block tree reduction"). Throw NotImplemented until
// the tree reduction is written.
void launch_sum_axis(const DeviceArray& a, int axis, bool keepdims, DeviceArray& out);
void launch_max_axis(const DeviceArray& a, int axis, bool keepdims, DeviceArray& out);

} // namespace cuda_native
