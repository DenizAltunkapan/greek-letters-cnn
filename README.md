# Greek Letters ML

This project trains and evaluates a convolutional neural network to classify handwritten Greek letters from image datasets.  
It focuses on repeatable training and architecture comparison so model changes can be measured consistently over time.

For a quick repository map, see [DIRECTORY.md](DIRECTORY.md).

## Training Setup

Training is config-driven and uses [ml/config/config.yml](ml/config/config.yml) by default.

Run training from [ml](ml) with:

```bash
python train.py
```

To use a custom config:

```bash
python train.py --config /path/to/config.yml
```

The script automatically uses GPU if available (`torch.cuda.is_available()`), otherwise it runs on CPU.
Training outputs are saved directly in [ml/models](ml/models) and [ml/plots](ml/plots) with a timestamp suffix.
Additionally, the latest model/plot are updated at the configured base paths for convenient default loading.
The same config is also used by [test.py](ml/test.py) and [predict.py](ml/predict.py).

## Linux / macOS

### CPU setup

```bash
cd /path/to/greek-letters-ml
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cd ml
python train.py
```

### GPU setup (Linux with NVIDIA CUDA)

```bash
cd /path/to/greek-letters-ml
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# Install CUDA-enabled PyTorch matching your CUDA driver:
# https://pytorch.org/get-started/locally/
cd ml
python train.py
```

Notes:
- macOS generally runs on CPU (or Apple Silicon acceleration depending on your local PyTorch build).
- Verify device detection:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## Windows (PowerShell)

### CPU setup

```powershell
cd C:\path\to\greek-letters-ml
py -m venv .venv
# Preferred for regular use:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
# Fallback only if activation is still blocked in this session:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
cd ml
python train.py
```

### GPU setup (NVIDIA CUDA)

```powershell
cd C:\path\to\greek-letters-ml
py -m venv .venv
# Preferred for regular use:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
# Fallback only if activation is still blocked in this session:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
# Install CUDA-enabled PyTorch matching your CUDA driver:
# https://pytorch.org/get-started/locally/
cd ml
python train.py
```

Verify GPU in Windows:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

## Training Config

Use [ml/config/config.yml](ml/config/config.yml) to control:
- model/plot storage paths (`paths`)
- dataset and batch size (`data`)
- seeds and epochs (`experiment`)
- early stopping and scheduler behavior (`trainer`)
- model and optimizer candidates for comparison (`search_space`)

Path note:
- Relative paths in [config.yml](ml/config/config.yml) are resolved against [ml](ml), not your current terminal folder.

Run evaluation and prediction with the same config:

```bash
python test.py --config /path/to/config.yml
python predict.py /path/to/image.png --config /path/to/config.yml
```
