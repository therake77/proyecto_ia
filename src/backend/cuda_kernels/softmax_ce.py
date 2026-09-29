SOFTMAX_CROSS_ENTROPY_FUSED_SRC = r"""
extern "C" __global__
void softmax_ce_fused(const float* logits, const int* labels, float* loss,
                       float* grad, int batch, int classes) {
    // TODO: row-wise log-sum-exp, fused forward + grad (p - y)
}
"""
