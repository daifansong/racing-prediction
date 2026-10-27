# racing-prediction

## Setup

Use Python 3.12. Create a virtualenv at the repo root, then install the shared dependencies:

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`torch` is not in `requirements.txt` because Mac (MPS) and NVIDIA (CUDA) need different wheels. Install it separately for your machine.

### Mac (Apple Silicon, MPS)

```bash
pip install torch==2.14.1
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

### NVIDIA (Linux, CUDA)

Check the driver first:

```bash
nvidia-smi
```

Install a CUDA build that matches the machine. Example for CUDA 12.6:

```bash
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Other common indexes: `cu124`, `cu128`. Pick the one that matches the GPU driver / CUDA toolkit on that host: https://pytorch.org/get-started/locally/

Do not copy a Mac `venv` onto the NVIDIA machine. Create a fresh venv there and install again.
