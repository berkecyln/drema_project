# DreMa Setup for RTX 50-series Cards

**Tested System:** Linux 6.17.9-arch1-1

**Tested GPU:** NVIDIA RTX 5070 Ti (Architecture: sm_120)

Newer NVIDIA GPUs (RTX 50-series) require `PyTorch >= 2.7.0`, `CUDA >= 12.8` and `CUDA Toolkit >= 12.8.61` , however project mainly developed on `CUDA 11.8` so to make it work project on newer versions please follow below instructions.

## Installation Steps

1. **Create Environment**

   Please check `environmnet.yml` for dependicies.

   ```bash
   conda env create -f environment.yml
   conda activate drema_env
   ```

2. **Manual Dependency Fixes**
   If automated install has issues, ensure you are using the Nightly build.
   ```bash
   pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
   pip install "numpy<2.0"
   ```

3. **Compile Submodules**
   The submodules must be compiled with the compatible compiler and the correct architecture flag:
   ```bash
   export CUDA_HOME=$CONDA_PREFIX
   export TORCH_CUDA_ARCH_LIST="12.0"
   
   # Install
   pip install submodules/simple-knn --no-cache-dir
   pip install submodules/diff-gaussian-rasterization --no-cache-dir
   pip install submodules/diff-gaussian-rasterization-depth --no-cache-dir
   pip install submodules/diff-surfel-rasterization --no-cache-dir
   ```
4. **SAM3 Setup**
   
   SAM3 is not available via PyPI currently, must be installed from GitHub and kept in project folder.

   First you need to request access for SAM3 repository from https://huggingface.co/facebook/sam3 then after you have to create a login token for sam in Hugging Face.

   To authenticate yourself, run and paste your token
   ```bash
   huggingface-cli login
   ```

   Then clone and install SAM3:
   ```bash
   cd /home/your_username/Projects/drema_project
   git clone https://github.com/facebookresearch/sam3.git
   cd sam3
   pip install -e ".[notebooks]"
   ```

