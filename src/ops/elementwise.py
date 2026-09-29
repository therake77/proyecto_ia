from autograd.function import Function


class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


class Exp(Function):
    @staticmethod
    def forward(ctx, a):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


class Log(Function):
    @staticmethod
    def forward(ctx, a):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError
