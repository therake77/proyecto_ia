from nn.module import Module


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class Sequential(Module):
    def __init__(self, *layers: Module) -> None:
        super().__init__()
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p
        self.training = True

    def forward(self, x):
        raise NotImplementedError


class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
