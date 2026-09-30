#pragma once

#include "device_array.hpp"

namespace cuda_native {

// out = a + b. a, b, out must have identical shape (no broadcasting yet --
// see the TODO in elementwise.cu; CudaBackend.add currently only needs the
// same-shape case for gradcheck to pass on M1's elementwise tests).
void launch_add(const DeviceArray& a, const DeviceArray& b, DeviceArray& out);

// out = relu(x) = max(x, 0).
void launch_relu_forward(const DeviceArray& x, DeviceArray& out);

// grad_in = grad_out where x > 0, else 0.
void launch_relu_backward(const DeviceArray& x, const DeviceArray& grad_out, DeviceArray& grad_in);

} // namespace cuda_native
