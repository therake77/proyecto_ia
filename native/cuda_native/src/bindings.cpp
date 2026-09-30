#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <cstdint>
#include <vector>

#include "common.cuh"
#include "device_context.hpp"
#include "device_array.hpp"
#include "elementwise.cuh"
#include "matmul_tiled.cuh"
#include "reductions.cuh"
#include "softmax_ce.cuh"

namespace py = pybind11;
using namespace cuda_native;

namespace {

std::vector<int64_t> reduced_shape(const std::vector<int64_t>& shape, int axis, bool keepdims) {
    std::vector<int64_t> out;
    out.reserve(shape.size());
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (static_cast<int>(i) == axis) {
            if (keepdims) out.push_back(1);
        } else {
            out.push_back(shape[i]);
        }
    }
    return out;
}

DeviceArray py_to_device(py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
    std::vector<int64_t> shape(arr.shape(), arr.shape() + arr.ndim());
    return copy_to_device(arr.data(), std::move(shape), DType::Float32);
}

py::array_t<float> py_to_host(const DeviceArray& arr) {
    std::vector<py::ssize_t> shape(arr.shape().begin(), arr.shape().end());
    py::array_t<float> result(shape);
    copy_to_host(arr, result.mutable_data());
    return result;
}

std::uintptr_t py_create_stream() {
    return reinterpret_cast<std::uintptr_t>(DeviceContext::instance()->create_stream());
}

void py_destroy_stream(std::uintptr_t stream) {
    DeviceContext::instance()->destroy_stream(reinterpret_cast<cudaStream_t>(stream));
}

void py_synchronize(std::uintptr_t stream) {
    DeviceContext::instance()->synchronize(reinterpret_cast<cudaStream_t>(stream));
}

DeviceArray py_add(const DeviceArray& a, const DeviceArray& b) {
    DeviceArray out(a.shape(), a.dtype(), a.stream());
    launch_add(a, b, out);
    return out;
}

DeviceArray py_relu_forward(const DeviceArray& x) {
    DeviceArray out(x.shape(), x.dtype(), x.stream());
    launch_relu_forward(x, out);
    return out;
}

DeviceArray py_relu_backward(const DeviceArray& x, const DeviceArray& grad_out) {
    DeviceArray grad_in(x.shape(), x.dtype(), x.stream());
    launch_relu_backward(x, grad_out, grad_in);
    return grad_in;
}

DeviceArray py_matmul_naive(const DeviceArray& a, const DeviceArray& b) {
    if (a.shape().size() != 2 || b.shape().size() != 2) {
        throw std::invalid_argument("matmul_naive: expected 2D arrays");
    }
    DeviceArray out({a.shape()[0], b.shape()[1]}, a.dtype(), a.stream());
    launch_matmul_naive(a, b, out);
    return out;
}

DeviceArray py_matmul_tiled(const DeviceArray& a, const DeviceArray& b) {
    if (a.shape().size() != 2 || b.shape().size() != 2) {
        throw std::invalid_argument("matmul_tiled: expected 2D arrays");
    }
    DeviceArray out({a.shape()[0], b.shape()[1]}, a.dtype(), a.stream());
    launch_matmul_tiled(a, b, out);
    return out;
}

DeviceArray py_sum_axis(const DeviceArray& a, int axis, bool keepdims) {
    DeviceArray out(reduced_shape(a.shape(), axis, keepdims), a.dtype(), a.stream());
    launch_sum_axis(a, axis, keepdims, out);
    return out;
}

DeviceArray py_max_axis(const DeviceArray& a, int axis, bool keepdims) {
    DeviceArray out(reduced_shape(a.shape(), axis, keepdims), a.dtype(), a.stream());
    launch_max_axis(a, axis, keepdims, out);
    return out;
}

std::pair<DeviceArray, DeviceArray> py_softmax_cross_entropy(
    const DeviceArray& logits, const DeviceArray& targets) {
    if (logits.shape().size() != 2) {
        throw std::invalid_argument("softmax_cross_entropy: logits must be 2D [batch, classes]");
    }
    int64_t batch = logits.shape()[0];
    DeviceArray loss({batch}, logits.dtype(), logits.stream());
    DeviceArray grad(logits.shape(), logits.dtype(), logits.stream());
    launch_softmax_cross_entropy(logits, targets, loss, grad);
    return {std::move(loss), std::move(grad)};
}

} // namespace

PYBIND11_MODULE(cuda_native, m) {
    m.doc() =
        "Pybind11/CUDA extension for proyecto-ia's CudaBackend. Owns its own "
        "CUDA streams and a stream-ordered memory pool (see DeviceContext) -- "
        "CudaBackend does not use cupy.ndarray for GPU Tensor storage.";

    py::register_exception<NotImplemented>(m, "NotImplementedCppError", PyExc_NotImplementedError);

    py::class_<DeviceArray>(m, "DeviceArray")
        .def_property_readonly("shape", &DeviceArray::shape)
        .def_property_readonly("dtype", [](const DeviceArray&) { return "float32"; })
        .def_property_readonly("nbytes", &DeviceArray::nbytes)
        .def_property_readonly("stream", [](const DeviceArray& a) {
            return reinterpret_cast<std::uintptr_t>(a.stream());
        })
        .def("__repr__", [](const DeviceArray& a) {
            std::string s = "DeviceArray(shape=(";
            for (std::size_t i = 0; i < a.shape().size(); ++i) {
                s += std::to_string(a.shape()[i]);
                if (i + 1 < a.shape().size()) s += ", ";
            }
            s += "), dtype=float32)";
            return s;
        });

    m.def("to_device", &py_to_device, py::arg("array"),
          "Copy a contiguous float32 numpy array to a pool-allocated DeviceArray.");
    m.def("to_host", &py_to_host, py::arg("array"),
          "Copy a DeviceArray back to a numpy array.");

    m.def("create_stream", &py_create_stream,
          "Create a new CUDA stream, returned as an opaque integer handle.");
    m.def("destroy_stream", &py_destroy_stream, py::arg("stream"));
    m.def("synchronize", &py_synchronize, py::arg("stream") = 0,
          "Synchronize the given stream handle, or the default stream if 0.");

    m.def("add", &py_add, py::arg("a"), py::arg("b"));
    m.def("relu_forward", &py_relu_forward, py::arg("x"));
    m.def("relu_backward", &py_relu_backward, py::arg("x"), py::arg("grad_out"));

    m.def("matmul_naive", &py_matmul_naive, py::arg("a"), py::arg("b"));
    m.def("matmul_tiled", &py_matmul_tiled, py::arg("a"), py::arg("b"));

    m.def("sum_axis", &py_sum_axis, py::arg("a"), py::arg("axis"), py::arg("keepdims") = false);
    m.def("max_axis", &py_max_axis, py::arg("a"), py::arg("axis"), py::arg("keepdims") = false);

    m.def("softmax_cross_entropy", &py_softmax_cross_entropy, py::arg("logits"), py::arg("targets"));
}
