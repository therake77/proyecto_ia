SUM_AXIS_TREE_SRC = r"""
extern "C" __global__
void sum_axis_tree(const float* x, float* out, int outer, int axis_len) {
    // TODO: in-block tree reduction
}
"""

MAX_AXIS_TREE_SRC = r"""
extern "C" __global__
void max_axis_tree(const float* x, float* out, int outer, int axis_len) {
    // TODO: in-block tree reduction
}
"""
