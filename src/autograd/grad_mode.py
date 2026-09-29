import contextlib
import threading

_state = threading.local()


def is_grad_enabled() -> bool:
    return getattr(_state, "enabled", True)


@contextlib.contextmanager
def no_grad():
    previous = is_grad_enabled()
    _state.enabled = False
    try:
        yield
    finally:
        _state.enabled = previous
