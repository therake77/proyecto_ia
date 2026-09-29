from optim.optimizer import Optimizer


class Scheduler:
    def __init__(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer
        self.base_lr = optimizer.lr

    def step(self, epoch: int) -> None:
        raise NotImplementedError


class StepLR(Scheduler):
    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1) -> None:
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self, epoch: int) -> None:
        raise NotImplementedError


class CosineLR(Scheduler):
    def __init__(self, optimizer: Optimizer, total_epochs: int) -> None:
        super().__init__(optimizer)
        self.total_epochs = total_epochs

    def step(self, epoch: int) -> None:
        raise NotImplementedError


class WarmupLR(Scheduler):
    def __init__(self, optimizer: Optimizer, warmup_epochs: int, base_scheduler: Scheduler) -> None:
        super().__init__(optimizer)
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler

    def step(self, epoch: int) -> None:
        raise NotImplementedError
