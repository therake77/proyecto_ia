from autograd.function import Function


class SoftmaxCrossEntropy(Function):
    @staticmethod
    def forward(ctx, logits, labels):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError
