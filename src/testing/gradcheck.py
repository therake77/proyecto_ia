from typing import Callable, Sequence

from tensor import Tensor


def gradcheck(
    func: Callable[..., Tensor],
    inputs: Sequence[Tensor],
    eps: float = 1e-5,
    tol: float = 1e-5,
) -> bool:
    raise NotImplementedError
