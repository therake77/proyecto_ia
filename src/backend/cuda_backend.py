import numpy as np

from backend.base import Backend

try:
    import cuda_native as _native
except ImportError:
    _native = None


class CudaBackend(Backend):
    """GPU backend. Storage is `cuda_native.DeviceArray` -- a pool-allocated
    buffer owned by the native extension's own CUDA streams/memory pool, not
    `cupy.ndarray`. See native/cuda_native/README.md for build instructions
    and which kernels are implemented vs. still TODO.
    """

    name = "cuda"

    def __init__(self) -> None:
        if _native is None:
            raise RuntimeError(
                "cuda_native extension is not built. Run "
                "`uv sync --group cuda-native` on a CUDA-capable machine "
                "with a C++/CUDA toolchain installed (see "
                "native/cuda_native/README.md)."
            )

    def add(self, a, b):
        return _native.add(a, b)

    def mul(self, a, b):
        raise NotImplementedError

    def matmul(self, a, b):
        return _native.matmul_naive(a, b)

    def exp(self, a):
        raise NotImplementedError

    def log(self, a):
        raise NotImplementedError

    def sum(self, a, axis=None, keepdims=False):
        raise NotImplementedError

    def max(self, a, axis=None, keepdims=False):
        raise NotImplementedError

    def relu(self, a):
        return _native.relu_forward(a)

    def reshape(self, a, shape):
        raise NotImplementedError

    def transpose(self, a, axes=None):
        raise NotImplementedError

    def broadcast_to(self, a, shape):
        raise NotImplementedError

    def unbroadcast(self, grad, shape):
        raise NotImplementedError

    def zeros_like(self, a):
        return _native.to_device(np.zeros(a.shape, dtype=np.float32))

    def to_numpy(self, a):
        return _native.to_host(a)
