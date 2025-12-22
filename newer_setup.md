# DreMa Setup for Newer Systems

**Tested System:** Linux 6.17.9-arch1-1
**Tested GPU:** NVIDIA RTX 5070 Ti

This project uses Gaussian Splatting submodules that are highly sensitive to compiler versions. The original code was designed for CUDA 11.8. Setting this up on a modern Linux distribution (like Arch) is tricky because the system's default C++ compilers (GCC 13+) are too new for CUDA 11.8, and the system libraries (glibc) are often incompatible with older build tools.

## Installation Steps

Use 'environment.yml' to create the environment

```bash
conda env create -f comprehensive_environment.yml
conda activate drema_env
```

During environmnet setup if pytorch libraries fails please download them manually and ensure you use the specific CUDA 11.8 index:

```bash
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
```
After everything finished please download submodules via

```bash
pip install submodules/simple-knn
pip install submodules/diff-gaussian-rasterization
pip install submodules/diff-gaussian-rasterization-depth
pip install submodules/diff-surfel-rasterization
```

**Note1:**
The submodule `diff-surfel-rasterization` has a bug in its `setup.py`. It hardcodes a path to a local `third_party` folder that doesn't exist. We need to point it to the Conda-installed `glm` library. 

**The Fix:**
Open `submodules/diff-surfel-rasterization/setup.py` and remove the `extra_compile_args` line that references `third_party/glm`. 

However please do this if you take an error during 'pip install submodules/diff-surfel-rasterization' since I dididnt get this error durong ubuntu setup but only in linux setup.

**Note2:**
You may see a warning: `NVIDIA GeForce RTX 5070 Ti ... with CUDA capability sm_120 is not compatible...`. 
This is expected. You are running older CUDA 11.8 code on a brand-new GPU architecture. It should work in compatibility mode for this project.
