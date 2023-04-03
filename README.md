# TRACED: Execution-aware Pre-training for Source Code

## Install Dependencies

```
# Create conda env

conda create -n traced python=3.8.13;
conda activate traced;

# Install Python Packages

pip install -r requirements.txt;
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
https://github.com/NVIDIA/apex.git
cd apex;
pip install -v --disable-pip-version-check --no-cache-dir ./
```

## Data and pre-trained checkpoint

- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7790005.svg)](https://doi.org/10.5281/zenodo.7790005)

## Tasks

- Execution Coverage Prediction: Check `run_finetune_exec_cov.py`
- Runtime Value Prediction: Check `run_finetune_runtime_value.py`
- Clone Retrieval: Check `run_finetune_clone.py`
- Vulnerablity Detection: Check `run_finetune_vul_detect.py`