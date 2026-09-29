from autograd.function import Function


class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError
