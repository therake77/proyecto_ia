import pytest

try:
    import cuda_native
except ImportError:
    cuda_native = None

pytestmark = pytest.mark.skipif(
    cuda_native is None,
    reason="cuda_native extension not built (see native/cuda_native/README.md)",
)


def test_placeholder():
    pytest.skip("TODO: E4")
