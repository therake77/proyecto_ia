#pragma once

#include "device_array.hpp"

namespace cuda_native {

// out[m,n] = a[m,k] @ b[k,n]. Row-major, one thread per output element.
void launch_matmul_naive(const DeviceArray& a, const DeviceArray& b, DeviceArray& out);

// Shared-memory tiled matmul. TODO(E4, week 4 per the course schedule):
// throws NotImplemented until the tiling is written.
void launch_matmul_tiled(const DeviceArray& a, const DeviceArray& b, DeviceArray& out);

} // namespace cuda_native
