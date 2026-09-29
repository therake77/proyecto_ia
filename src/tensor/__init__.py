from __future__ import annotations

from typing import Optional

from backend.base import Backend


class Tensor:
    def __init__(
        self,
        data,
        backend: Backend,
        requires_grad: bool = False,
        grad_fn=None,
    ) -> None:
        self.data = data
        self.backend = backend
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self.grad: Optional["Tensor"] = None

    @property
    def shape(self):
        raise NotImplementedError

    @property
    def dtype(self):
        raise NotImplementedError

    @property
    def device(self) -> str:
        return self.backend.name

    def to(self, device: str) -> "Tensor":
        raise NotImplementedError

    def backward(self, grad_output: Optional["Tensor"] = None) -> None:
        raise NotImplementedError

    def zero_grad(self) -> None:
        self.grad = None

    def __add__(self, other: "Tensor") -> "Tensor":
        from ops.elementwise import Add
        return Add.apply(self, other)

    def __mul__(self, other: "Tensor") -> "Tensor":
        from ops.elementwise import Mul
        return Mul.apply(self, other)

    def matmul(self, other: "Tensor") -> "Tensor":
        from ops.matmul import MatMul
        return MatMul.apply(self, other)

    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        from ops.reductions import Sum
        return Sum.apply(self, axis=axis, keepdims=keepdims)

    def reshape(self, *shape: int) -> "Tensor":
        from ops.shape import Reshape
        return Reshape.apply(self, shape)
