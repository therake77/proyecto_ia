from backend.base import Backend

try:
    import cupy as cp
except ImportError:
    cp = None


class CudaBackend(Backend):
    name = "cuda"

    def __init__(self) -> None:
        if cp is None:
            raise RuntimeError("CuPy is required for CudaBackend")

    def add(self, a, b):
        raise NotImplementedError

    def mul(self, a, b):
        raise NotImplementedError

    def matmul(self, a, b):
        raise NotImplementedError

    def exp(self, a):
        raise NotImplementedError

    def log(self, a):
        raise NotImplementedError

    def sum(self, a, axis=None, keepdims=False):
        raise NotImplementedError

    def max(self, a, axis=None, keepdims=False):
        raise NotImplementedError

    def relu(self, a):
        raise NotImplementedError

    def reshape(self, a, shape):
        raise NotImplementedError

    def transpose(self, a, axes=None):
        raise NotImplementedError

    def broadcast_to(self, a, shape):
        raise NotImplementedError

    def unbroadcast(self, grad, shape):
        raise NotImplementedError

    def zeros_like(self, a):
        raise NotImplementedError

    def to_numpy(self, a):
        raise NotImplementedError
