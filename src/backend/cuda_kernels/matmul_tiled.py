MATMUL_NAIVE_SRC = r"""
extern "C" __global__
void matmul_naive(const float* a, const float* b, float* out, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float acc = 0.0f;
        for (int i = 0; i < k; ++i) acc += a[row * k + i] * b[i * n + col];
        out[row * n + col] = acc;
    }
}
"""

MATMUL_TILED_SRC = r"""
extern "C" __global__
void matmul_tiled(const float* a, const float* b, float* out, int m, int k, int n) {
    // TODO: shared-memory tiling
}
"""
