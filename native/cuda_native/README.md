# cuda_native

Pybind11/CUDA extension backing `CudaBackend` (`src/backend/cuda_backend.py`). Unlike
the course spec's suggested CuPy-based approach, this extension owns its own GPU memory:
a `DeviceContext` (one CUDA device, one stream-ordered `cudaMemPool_t`, a default
`cudaStream_t`) and a `DeviceArray` type (pool-allocated via `cudaMallocAsync`, freed via
`cudaFreeAsync`) are what a GPU `Tensor.data` actually holds — not `cupy.ndarray`. CuPy
stays available elsewhere in the project only as an optional dev/test cross-check tool.

This is a **separate uv workspace member**, not part of the root package, precisely
because it needs a native toolchain that the rest of the team (E1/E2/E3/E5) does not.
Nothing outside this directory requires CUDA, CMake, or a C++ compiler to be installed.

## Prerequisites (only needed to build/run this package)

- An NVIDIA GPU + driver, and the **CUDA Toolkit** (`nvcc` on `PATH`).
- **CMake ≥ 3.24**.
- A C++17 host compiler compatible with your CUDA Toolkit version:
  - Windows: MSVC Build Tools (Visual Studio Build Tools), version matched to the CUDA
    Toolkit release notes for supported host compilers.
  - Linux: a supported `gcc`/`g++` version per the same compatibility table.
- `pybind11` and `scikit-build-core` are pulled automatically as PEP 517 build
  dependencies — no manual install needed beyond the three items above.

## Building

From the repo root (once the root `pyproject.toml` workspace wiring is in place):

```sh
uv sync --group cuda-native
```

Or directly, from this directory:

```sh
uv pip install -e .
# or, to just produce a wheel:
uv build
```

Either invokes `scikit-build-core`, which runs CMake → `nvcc`/host compiler → links the
`cuda_native` extension module (`CMakeLists.txt` in this directory).

## Verifying the build

```sh
uv run python -c "import numpy as np, cuda_native as c; a = c.to_device(np.ones(4, dtype=np.float32)); print(c.to_host(a))"
uv run pytest native/cuda_native/tests
```

Verified working end-to-end (build + all kernel/round-trip/stream tests) on Windows 11 +
CUDA 13.1 + VS 2022 Build Tools (MSVC 19.44) + an RTX 3060. That's the default and only
path currently confirmed to work on Windows — see "Using clang instead of MSVC" below
for why.

## Using clang instead of MSVC / gcc

**Linux (e.g. Fedora): should work, not yet verified in this repo.** nvcc officially
supports clang as a host compiler on Linux, and CMake's Clang-as-CUDA-compiler support
is mature there. Two ways to try it, both standard CMake mechanisms (no project-specific
config needed) since the default "Unix Makefiles"/Ninja generators on Linux already
honor these:

```sh
# nvcc still drives device codegen, just with clang as its host compiler:
CUDAHOSTCXX=clang++ CXX=clang++ uv sync --group cuda-native

# or let clang itself compile the .cu files (bypasses nvcc entirely):
CXX=clang++ CUDACXX=clang++ uv sync --group cuda-native
```

If you hit anything, it's most likely a CUDA-Toolkit-version-vs-clang-version support
gap (Clang maintains a "max tested CUDA version" per release, similar to the MSVC
compatibility table) — check `clang --version` output for a `-Wunknown-cuda-version`
warning as the first thing to look at.

**Windows: does not currently work, confirmed.** We tried getting clang to replace MSVC
here (both as the CUDA-language compiler via CMake, and via `nvcc -ccbin clang++`) and
hit two separate, reproducible dead ends:

1. `CMAKE_CXX_COMPILER=clang++` / `CMAKE_CUDA_COMPILER=clang++` with a Ninja generator
   (required — the default "Visual Studio" generator silently ignores both of those and
   always drives MSBuild+cl.exe) gets past CXX compiler identification (`Clang 22.1.8
   with GNU-like command-line`) but then fails CUDA compiler identification:
   ```
   CMake Error: Error required internal CMake variable not set, cmake may not be built correctly.
   Missing variable is: _CMAKE_CUDA_WHOLE_FLAG
   ```
   at `CMakeDetermineCompilerABI.cmake` — this looks like a real gap in CMake 3.31's
   Windows+Clang+CUDA(NVPTX) platform module, not a config mistake on our end.
2. `nvcc -ccbin clang++ -allow-unsupported-compiler` fails immediately with
   `nvcc fatal : Host compiler targets unsupported OS` — nvcc's Windows host-compiler
   detection doesn't recognize `clang++` (even one targeting `x86_64-pc-windows-msvc`)
   at all, `--allow-unsupported-compiler` doesn't help.

Also worth knowing even if a way around the above is found: on Windows, clang here still
depends on VS Build Tools regardless — it auto-detects and pulls the MSVC STL headers
and Windows SDK from the same Build Tools install (`...VC\Tools\MSVC\...\include`,
`...Windows Kits\10\Include\...`) rather than being MSVC-free. There's no way around
that short of a MinGW-w64 target, which is a materially different, less-tested
combination with CUDA. So on Windows, MSVC stays the supported path for now; if this
matters to you, prefer building on Linux instead.

## What's implemented vs. TODO

Ported from the original `src/backend/cuda_kernels/*.py` RawKernel source strings —
same math, now compiled instead of CuPy-JIT'd:

| Kernel | Status |
| --- | --- |
| `add` (elementwise, same-shape) | implemented |
| `relu_forward` / `relu_backward` | implemented |
| `matmul_naive` | implemented |
| elementwise `mul`, broadcasting | **not implemented** (the original source didn't have these either) |
| `matmul_tiled` (shared-memory tiling) | **not implemented** — raises `NotImplementedError` |
| `sum_axis` / `max_axis` (tree reduction) | **not implemented** — raises `NotImplementedError` |
| `softmax_cross_entropy` (fused, log-sum-exp) | **not implemented** — raises `NotImplementedError` |

These match the course schedule (weeks 4–6) and the M4 rubric criteria — they're E4's
next implementation steps, not something this scaffolding pass fills in.

## Free-threaded CPython note

The project's `.venv` is built on **free-threaded CPython 3.14** (no-GIL). This module
does not declare itself thread-safe (`Py_MOD_GIL_NOT_USED`), so importing it makes the
interpreter fall back to running with the GIL re-enabled — you'll see a `RuntimeWarning`
at import time. That's expected and harmless for how this extension is used today
(nothing elsewhere in the project relies on free-threading); revisit only if that
changes.

## CUDA Graphs

Not implemented. Every kernel launch goes through a `DeviceContext`-owned stream so
`cudaStreamBeginCapture`/`cudaGraphLaunch` could be added later without restructuring
the kernel code, but no capture/replay API exists yet.
