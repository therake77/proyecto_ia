"""Smoke tests for the cuda_native extension: DeviceContext/DeviceArray
plumbing (allocation, host<->device round trip, streams) and the kernels
that are actually implemented (add, relu, matmul_naive). Skips cleanly if
the extension hasn't been built -- see README.md.
"""

import numpy as np
import pytest

cuda_native = pytest.importorskip("cuda_native")


def test_to_device_to_host_roundtrip():
    host = np.arange(12, dtype=np.float32).reshape(3, 4)
    device = cuda_native.to_device(host)
    assert device.shape == [3, 4]
    assert device.dtype == "float32"
    np.testing.assert_array_equal(cuda_native.to_host(device), host)


def test_add():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    out = cuda_native.add(cuda_native.to_device(a), cuda_native.to_device(b))
    np.testing.assert_allclose(cuda_native.to_host(out), a + b)


def test_relu_forward_and_backward():
    x = np.array([-1.0, 0.5, 2.0, -3.0], dtype=np.float32)
    grad_out = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    x_dev = cuda_native.to_device(x)
    out = cuda_native.relu_forward(x_dev)
    np.testing.assert_allclose(cuda_native.to_host(out), np.maximum(x, 0.0))

    grad_in = cuda_native.relu_backward(x_dev, cuda_native.to_device(grad_out))
    np.testing.assert_allclose(cuda_native.to_host(grad_in), (x > 0).astype(np.float32))


def test_matmul_naive():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((8, 5), dtype=np.float32)
    b = rng.standard_normal((5, 3), dtype=np.float32)
    out = cuda_native.matmul_naive(cuda_native.to_device(a), cuda_native.to_device(b))
    np.testing.assert_allclose(cuda_native.to_host(out), a @ b, rtol=1e-5, atol=1e-5)


def test_stream_create_destroy_and_synchronize():
    stream = cuda_native.create_stream()
    cuda_native.synchronize(stream)
    cuda_native.destroy_stream(stream)
    cuda_native.synchronize()  # default stream, handle 0


@pytest.mark.parametrize(
    "call",
    [
        lambda: cuda_native.matmul_tiled(
            cuda_native.to_device(np.zeros((2, 2), dtype=np.float32)),
            cuda_native.to_device(np.zeros((2, 2), dtype=np.float32)),
        ),
        lambda: cuda_native.sum_axis(cuda_native.to_device(np.zeros((2, 2), dtype=np.float32)), axis=0),
        lambda: cuda_native.max_axis(cuda_native.to_device(np.zeros((2, 2), dtype=np.float32)), axis=0),
        lambda: cuda_native.softmax_cross_entropy(
            cuda_native.to_device(np.zeros((2, 3), dtype=np.float32)),
            cuda_native.to_device(np.zeros((2,), dtype=np.float32)),
        ),
    ],
)
def test_not_yet_implemented_kernels_raise_not_implemented_error(call):
    with pytest.raises(NotImplementedError):
        call()
