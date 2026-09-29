from __future__ import annotations


class Context:
    def __init__(self) -> None:
        self.saved_tensors: tuple = ()

    def save_for_backward(self, *tensors) -> None:
        self.saved_tensors = tensors


class Function:
    @staticmethod
    def forward(ctx: Context, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, grad_output):
        raise NotImplementedError

    @classmethod
    def apply(cls, *args, **kwargs):
        raise NotImplementedError
