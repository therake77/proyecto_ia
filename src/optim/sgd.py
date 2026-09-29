from optim.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, parameters, lr: float, momentum: float = 0.0, nesterov: bool = False) -> None:
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.nesterov = nesterov

    def step(self) -> None:
        raise NotImplementedError
