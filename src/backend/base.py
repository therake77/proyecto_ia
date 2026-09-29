from abc import ABC, abstractmethod


class Backend(ABC):
    name: str

    @abstractmethod
    def add(self, a, b): ...

    @abstractmethod
    def mul(self, a, b): ...

    @abstractmethod
    def matmul(self, a, b): ...

    @abstractmethod
    def exp(self, a): ...

    @abstractmethod
    def log(self, a): ...

    @abstractmethod
    def sum(self, a, axis=None, keepdims=False): ...

    @abstractmethod
    def max(self, a, axis=None, keepdims=False): ...

    @abstractmethod
    def relu(self, a): ...

    @abstractmethod
    def reshape(self, a, shape): ...

    @abstractmethod
    def transpose(self, a, axes=None): ...

    @abstractmethod
    def broadcast_to(self, a, shape): ...

    @abstractmethod
    def unbroadcast(self, grad, shape): ...

    @abstractmethod
    def zeros_like(self, a): ...

    @abstractmethod
    def to_numpy(self, a): ...
