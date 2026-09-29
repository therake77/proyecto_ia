from optim.optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, parameters, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8) -> None:
        super().__init__(parameters, lr)
        self.betas = betas
        self.eps = eps

    def step(self) -> None:
        raise NotImplementedError


class AdamW(Adam):
    def __init__(self, parameters, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 1e-2) -> None:
        super().__init__(parameters, lr, betas, eps)
        self.weight_decay = weight_decay

    def step(self) -> None:
        raise NotImplementedError
