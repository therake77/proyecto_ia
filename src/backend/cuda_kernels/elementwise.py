ELEMENTWISE_ADD_SRC = r"""
extern "C" __global__
void elementwise_add(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}
"""

RELU_FORWARD_SRC = r"""
extern "C" __global__
void relu_forward(const float* x, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = x[idx] > 0.0f ? x[idx] : 0.0f;
}
"""

RELU_BACKWARD_SRC = r"""
extern "C" __global__
void relu_backward(const float* x, const float* grad_out, float* grad_in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) grad_in[idx] = x[idx] > 0.0f ? grad_out[idx] : 0.0f;
}
"""
