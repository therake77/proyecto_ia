from autograd.function import Function


class ReLU(Function):
    @staticmethod
    def forward(ctx, a):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


class Sigmoid(Function):
    @staticmethod
    def forward(ctx, a):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


class Tanh(Function):
    @staticmethod
    def forward(ctx, a):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError
