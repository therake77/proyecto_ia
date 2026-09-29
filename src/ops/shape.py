from autograd.function import Function


class Reshape(Function):
    @staticmethod
    def forward(ctx, a, shape):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


class Transpose(Function):
    @staticmethod
    def forward(ctx, a, axes=None):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


class BroadcastTo(Function):
    @staticmethod
    def forward(ctx, a, shape):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError
