# fireants-fused-ops-linux-install

Two scripts and a starter conda environment for installing
[FireANTs](https://github.com/rohitrango/FireANTs) **plus the optional
`fireants_fused_ops` CUDA extension** on Linux without the usual headaches.

The fused-ops extension gives ~80% faster registration and ~20% lower GPU
memory use, but it must be **compiled from source for your specific GPU**,
which is where most installs fail.  These scripts handle the version pinning
and toolchain setup automatically.

---

## What's in this repo

| File | Purpose |
|---|---|
| [`fireants_check.py`](fireants_check.py) | **Diagnostic only.** Reports PASS / WARN / FAIL for every component (driver, GPU, nvcc, gcc, PyTorch, fireants, fused_ops) and prints copy-paste fix commands. Does **not** install anything — safe to share with anyone. |
| [`fireants_install.py`](fireants_install.py) | **Auto-installer.** Creates a conda env, installs PyTorch (cu121), fireants 1.4.0, the CUDA toolkit 12.1, gcc 12, then auto-detects your GPU's compute capability and builds `fireants_fused_ops` for it. |
| [`pre_env_from_history.yml`](pre_env_from_history.yml) | A `conda env export --from-history` from a known-good install, used as the starting point by the installer (optional). Lists explicitly-installed packages only — portable across machines. |

---

## Quick start

> Requires: Linux, an NVIDIA GPU, the NVIDIA driver, and `conda` already
> installed. If any of those are missing, see [Prerequisites](#prerequisites)
> at the bottom of this README.

### 1. Diagnose first (no install)
```bash
git clone https://github.com/<your-username>/fireants-fused-ops-linux-install.git
cd fireants-fused-ops-linux-install
python fireants_check.py
```
You'll get a row for each component and a list of suggested fixes if anything
is wrong. **Run this even if you plan to use the installer** — it tells you
whether your driver/GPU are compatible *before* you spend 15 minutes building.

### 2. Install
```bash
# Option A: bootstrap from the included yml (recommended)
python fireants_install.py \
    --env-yml pre_env_from_history.yml \
    --env-name fireants

# Option B: skip the yml, build the env from scratch
python fireants_install.py --env-name fireants

# Option C: install into an env you already have
conda activate myenv
python fireants_install.py --env-name myenv --use-existing
```
The installer:

1. Verifies you have an NVIDIA driver ≥ 525 (required for CUDA 12.1).
2. Creates / updates the conda env.
3. Installs PyTorch 2.5.1 + cu121 wheel.
4. Installs `fireants==1.4.0`.
5. Installs the CUDA 12.1 **toolkit** (`nvcc`) into the env — does **not**
   touch your system CUDA install.
6. Installs `gcc=12` / `gxx=12` into the env (CUDA 12.1 only supports
   gcc ≤ 13; modern distros ship gcc 14/15 which causes the
   `pybind11::cpp_function` errors most failed installs hit).
7. Reads your GPU's compute capability from `nvidia-smi` and builds
   `fireants_fused_ops` for it.
8. Verifies the extension imports and runs.

When it finishes:
```bash
conda activate fireants
python -c "import torch, fireants_fused_ops; print('ready')"
```

---

## Why is this hard? What does each piece do?

A surprising amount of confusion comes from CUDA having three independent
things all called "CUDA version":

| Layer | What it is | Where it comes from |
|---|---|---|
| **GPU driver** (e.g. 580) | Lets the kernel talk to the GPU. The `CUDA Version: 13.0` line in `nvidia-smi` is the *highest* CUDA runtime this driver can run — a **ceiling**, not what's installed. | NVIDIA proprietary driver install |
| **CUDA toolkit** (e.g. 12.1) | Contains `nvcc`, the CUDA compiler. Used to **build** CUDA code. PyTorch's cu121 wheel needs a 12.1 toolkit to produce binary-compatible extensions. | `conda install -c nvidia/label/cuda-12.1.1 cuda-toolkit` |
| **PyTorch CUDA libs** (e.g. 12.1) | The CUDA runtime libraries bundled inside the PyTorch wheel. PyTorch ships its own. | `pip install torch==2.5.1 --index-url ...cu121` |

Plus two more pieces:

- **`gcc` / `g++`** — the standard C/C++ compiler. `nvcc` calls it under the
  hood for the CPU-side code. **CUDA 12.1 only supports gcc 6–13.** If your
  system gcc is 14+, the build fails with cryptic pybind11 errors.
- **GPU compute capability** — every NVIDIA GPU has one (e.g. 6.1 for
  GTX 1080, 8.6 for RTX 3090, 9.0 for H100). The `.so` file we build only
  runs on the architectures we list at compile time. The installer detects
  yours automatically.

The installer pins everything to versions that work together and uses
**conda-env-local** `nvcc` and `gcc` so they don't conflict with whatever
you have system-wide.

---

## Troubleshooting

**`pybind11::cpp_function` errors during build**
You're using a system `gcc` that's too new (≥14) instead of the conda one.
Re-activate the env (`conda deactivate && conda activate <env>`) and confirm
`which gcc` shows a path inside the conda env. If it doesn't, run
`conda install -c conda-forge gcc=12 gxx=12 -y`.

**`error: [Errno 2] No such file or directory: '.../bin/nvcc'`**
The CUDA toolkit didn't get installed into the env. Re-run:
```bash
conda install -c "nvidia/label/cuda-12.1.1" cuda-toolkit -y
export CUDA_HOME=$CONDA_PREFIX
```

**`No kernel image is available for execution on the device`**
The `.so` was built for a different GPU arch. Rebuild after setting
`TORCH_CUDA_ARCH_LIST` to *your* compute capability (the installer does this
automatically — only an issue if you copied a `.so` from another machine).

**`libc10.so: cannot open shared object file`**
You imported `fireants_fused_ops` before `torch`. Always
`import torch` first.

**`No module named 'simpleitk'` while installing fireants on Python 3.13**
SimpleITK has no Python 3.13 wheel as of early 2026. Use `python=3.12`.

---

## Prerequisites

> The installer/diagnostic scripts check for all of these and tell you which
> are missing.

### NVIDIA GPU + driver
You need an NVIDIA GPU with a driver version **≥ 525** (required for CUDA
12.1). Check with:
```bash
nvidia-smi
```
If `nvidia-smi` is missing or shows an older driver, install the proprietary
NVIDIA driver from your distro's package manager. Example for Ubuntu:
```bash
sudo ubuntu-drivers autoinstall
sudo reboot
```
For other distros: see https://www.nvidia.com/Download/index.aspx.

The compute-capability requirement depends on your GPU; anything ≥ Pascal
(GTX 10-series, 2016+) works. Look yours up at
https://developer.nvidia.com/cuda-gpus.

### Miniconda / Anaconda
You need `conda` available on your PATH. If `which conda` returns nothing:
```bash
# Linux x86_64 quick install
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash    # or `init zsh` if you use zsh
exec $SHELL                              # restart shell so PATH updates
conda --version                          # should now print the version
```
Full instructions: https://docs.conda.io/projects/miniconda/en/latest/

### Optional: a starting environment
You don't have to create one yourself — the installer will. But if you'd
rather start from the included yml manually:
```bash
conda env create -n fireants -f pre_env_from_history.yml
conda activate fireants
python fireants_install.py --env-name fireants --use-existing
```

---

## License

MIT — see [LICENSE](LICENSE).

The FireANTs library itself is distributed under its own license — see the
upstream repo: https://github.com/rohitrango/FireANTs.
