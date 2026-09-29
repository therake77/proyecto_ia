from __future__ import annotations

from typing import Iterator

from tensor import Tensor


class Module:
    def __init__(self) -> None:
        self._parameters: dict[str, Tensor] = {}
        self._modules: dict[str, "Module"] = {}

    def register_parameter(self, name: str, param: Tensor) -> None:
        self._parameters[name] = param

    def register_module(self, name: str, module: "Module") -> None:
        self._modules[name] = module

    def parameters(self) -> Iterator[Tensor]:
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
