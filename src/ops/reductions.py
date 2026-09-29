from autograd.function import Function


class Sum(Function):
    @staticmethod
    def forward(ctx, a, axis=None, keepdims=False):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


class Max(Function):
    @staticmethod
    def forward(ctx, a, axis=None, keepdims=False):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError
