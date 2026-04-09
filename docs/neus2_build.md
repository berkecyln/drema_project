# NeuS2 Setup Guide

## 1. Clone

```bash
cd ~/Projects/drema/drema_project
git clone --recursive https://github.com/19reborn/NeuS2.git
cd NeuS2
```

If got submodule errors, run:
```bash
git submodule update --init --recursive
```

## 2. Conda dependencies

```bash
conda activate drema_env

conda install -c conda-forge xorg-libxcursor xorg-libxrandr xorg-libxi xorg-libxinerama -y
conda install -c conda-forge glew mesalib -y
conda install -c conda-forge libgl libglvnd libglx -y
```

## 3. Configure

```bash
export TCNN_CUDA_ARCHITECTURES=120   # adjust to your GPU arch
export CUDA_HOME=$CONDA_PREFIX

cmake . -B build \
  -DPYTHON_EXECUTABLE=$(which python) \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DNGP_BUILD_WITH_GUI=OFF
```

`NGP_BUILD_WITH_GUI=OFF` removes the libGL linker dependency.

## 4. Build

```bash
export CPATH=$CONDA_PREFIX/include:$CPATH   # lets nvcc find GL/gl.h from mesalib

cmake --build build --config RelWithDebInfo -j$(nproc)
```

Output: `build/pyngp.cpython-310-x86_64-linux-gnu.so`

## 5. Verify

```bash
cd build
python -c "import pyngp; print('OK:', pyngp.__file__)"
```

## Notes

- `CPATH` must be set every time you rebuild (nvcc doesn't inherit conda's include path automatically).
- `TCNN_CUDA_ARCHITECTURES=120` is for RTX 50-series. For RTX 30xx use `86`, RTX 40xx use `89`.
- The `testbed` binary also builds but is unused in the pipeline.
- `pyngp.so` lives in `NeuS2/build/`, import it by adding that path to `sys.path`.
