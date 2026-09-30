#pragma once

#include "device_array.hpp"

namespace cuda_native {

// Fused softmax + cross-entropy, forward and grad (p - y) in one kernel,
// numerically stable via row-wise log-sum-exp. TODO(E4, week 6 per the
// course schedule): ported from src/backend/cuda_kernels/softmax_ce.py
// (SOFTMAX_CROSS_ENTROPY_FUSED_SRC), already a TODO stub there. Throws
// NotImplemented until written.
//
// logits: [batch, num_classes]; targets: [batch] (class indices).
// Returns per-example loss in `loss_out` [batch] and writes dL/dlogits
// (= p - y) into `grad_out` [batch, num_classes].
void launch_softmax_cross_entropy(
    const DeviceArray& logits,
    const DeviceArray& targets,
    DeviceArray& loss_out,
    DeviceArray& grad_out);

} // namespace cuda_native
