from __future__ import annotations

from typing import Iterable

from tensor import Tensor


class Optimizer:
    def __init__(self, parameters: Iterable[Tensor], lr: float) -> None:
        self.parameters = list(parameters)
        self.lr = lr

    def step(self) -> None:
        raise NotImplementedError

    def zero_grad(self) -> None:
        for p in self.parameters:
            p.zero_grad()
