# fireants-fused-ops-linux-install

Two scripts and a minimal conda environment for installing
[FireANTs](https://github.com/rohitrango/FireANTs) **plus the optional
`fireants_fused_ops` CUDA extension** on Linux without the usual headaches.

The fused-ops extension gives ~80% faster registration and ~20% lower GPU
memory use, but it must be **compiled from source for your specific GPU**,
which is where most installs fail.  These scripts detect your GPU, pick a
CUDA/PyTorch combination that actually supports it, and handle the toolchain
setup automatically.

---

## What's in this repo

| File | Purpose |
|---|---|
| [`fireants_check.py`](fireants_check.py) | **Diagnostic only.** Reports PASS / WARN / FAIL for every component (driver, GPU, nvcc, gcc, PyTorch, fireants, fused_ops, FSL, Jupyter kernel) and prints copy-paste fix commands. Does **not** install anything — safe to share with anyone. |
| [`fireants_install.py`](fireants_install.py) | **Auto-installer.** Creates a conda env, picks the right CUDA toolkit + PyTorch build for your GPU, installs fireants, builds `fireants_fused_ops` for your compute capability, and registers a Jupyter kernel. |
| [`environment.yml`](environment.yml) | Minimal conda env used by default: python 3.12, pip, tk, git, gcc/gxx 12. Nothing else — everything version-sensitive is decided at install time. |
| [`constraints.txt`](constraints.txt) | Global pip constraints (`numpy<2`) applied to every pip install. |
| [`requirements-fireants.txt`](requirements-fireants.txt) | Runtime deps for fireants itself, with the broken `simpleitk==2.2.1` pin worked around. |
| [`requirements-analysis.txt`](requirements-analysis.txt) | Optional neuroimaging/stats stack (pingouin, nipype, nilearn, statsmodels, scikit-learn, plotly, jupyter…). Installed only with `--with-analysis-stack`. |
| [`pre_env_from_history.yml`](pre_env_from_history.yml) | **Not needed.** A 350-package `conda env export` from one developer's everyday Anaconda env, kept only for reproducing that machine exactly. See [why you probably don't want it](#about-pre_env_from_historyyml). |

---

## Quick start

> Requires: Linux, an NVIDIA GPU, the NVIDIA driver, and `conda` already
> installed. If any of those are missing, see [Prerequisites](#prerequisites)
> at the bottom of this README.

### 1. Diagnose first (no install)
```bash
git clone https://github.com/AzadAzargushasb/fireants-fused-ops-linux-install.git
cd fireants-fused-ops-linux-install
python fireants_check.py
```
You'll get a row for each component and a list of suggested fixes if anything
is wrong. **Run this even if you plan to use the installer** — it tells you
whether your driver/GPU are usable *before* you spend 15 minutes building.

### 2. Install
```bash
# Option A: minimal env (recommended)
python fireants_install.py --env-name fireants

# Option B: ...plus the neuroimaging analysis stack
python fireants_install.py --env-name fireants --with-analysis-stack

# Option C: install into an env you already have
python fireants_install.py --env-name myenv --use-existing
```
The installer:

1. Verifies conda, the NVIDIA driver, and `nvidia-smi`.
2. Reads your GPU's **compute capability** and picks a [CUDA lane](#gpu-support-matrix).
3. Creates / updates the conda env from `environment.yml`.
4. Installs the lane's PyTorch build and `fireants`.
5. Installs the lane's CUDA **toolkit** (`nvcc`) into the env — does **not**
   touch your system CUDA install.
6. Installs `gcc=12` / `gxx=12` into the env (modern distros ship gcc 14/15,
   which causes the `pybind11::cpp_function` errors most failed installs hit).
7. Builds `fireants_fused_ops` for your GPU's architecture.
8. Registers a Jupyter kernel and verifies everything imports and runs.

When it finishes:
```bash
conda activate fireants
python -c "import torch, fireants_fused_ops; print('ready')"
```

### Useful flags

| Flag | Effect |
|---|---|
| `--with-analysis-stack` | Also install `requirements-analysis.txt`. |
| `--no-kernel` | Skip Jupyter kernel registration. |
| `--fireants-version 1.5.0` | Install a different fireants release (default `1.4.0`). |
| `--compute-cap 8.6` | Force the GPU arch instead of asking `nvidia-smi` (for old drivers). |
| `--env-yml <file>` | Bootstrap from some other conda yml instead of `environment.yml`. |
| `--use-existing` | Install into an env that already exists. |

---

## GPU support matrix

There is **no single CUDA + PyTorch stack that works on every NVIDIA GPU**, so
the installer picks one based on your GPU's compute capability:

| Compute capability | Example GPUs | CUDA toolkit | PyTorch |
|---|---|---|---|
| 5.0 – 9.0 (`sm_50`–`sm_90`) | GTX 10-series, RTX 20/30/40-series, A100, H100 | 12.1 | 2.5.1 + cu121 |
| 10.0 / 12.0 (`sm_100`, `sm_120`) | **RTX 50-series (5060/5070/5080/5090)**, B100, B200 | 12.8 | 2.11.0 + cu128 (falls back to 2.7.0) |
| below 5.0 | Kepler and older | *unsupported* | — |

Why it has to work this way:

* The **cu121 wheel index ends at torch 2.5.1** — that is the newest cu121
  build that will ever exist, and those wheels contain kernels only for
  `sm_50`–`sm_90`.
* **CUDA 12.8 was the first toolkit that can compile for Blackwell** (`sm_120`)
  at all, and PyTorch only gained Blackwell kernels in **2.7.0**.
* The cu128 builds in turn dropped Maxwell/Pascal, so a GTX 1080 and an
  RTX 5060 genuinely cannot share one install.

FireANTs' `fused_ops` sources are known-good against torch 2.5.1. On the
Blackwell lane the installer tries the newest cu128 torch first and
automatically falls back to 2.7.0 if the CUDA extension won't compile.

Adding a lane later (e.g. cu130 for a future architecture) is a one-line edit
to `CUDA_LANES` at the top of both scripts.

---

## Using the env in JupyterLab

The installer does this for you unless you pass `--no-kernel`. To do it by
hand:

```bash
conda activate fireants
pip install ipykernel
python -m ipykernel install --user --name fireants --display-name "Python (fireants)"
```

`--user` writes `~/.local/share/jupyter/kernels/fireants/`, which is on the
default `JUPYTER_PATH`. That means **whichever** JupyterLab you launch will see
it — including one shipped by something else entirely, such as FSL's
`/opt/fsl/bin/jupyter-lab`. Restart JupyterLab and pick *Python (fireants)*
from the kernel list.

Inside a notebook, always `import torch` **before** `import fireants_fused_ops`
(see troubleshooting below).

---

## Why is this hard? What does each piece do?

A surprising amount of confusion comes from CUDA having three independent
things all called "CUDA version":

| Layer | What it is | Where it comes from |
|---|---|---|
| **GPU driver** (e.g. 580) | Lets the kernel talk to the GPU. The `CUDA Version: 13.0` line in `nvidia-smi` is the *highest* CUDA runtime this driver can run — a **ceiling**, not what's installed. | NVIDIA proprietary driver install |
| **CUDA toolkit** (e.g. 12.1) | Contains `nvcc`, the CUDA compiler. Used to **build** CUDA code. Must match the CUDA version PyTorch was built against. | `conda install -c nvidia/label/cuda-12.1.1 cuda-toolkit` |
| **PyTorch CUDA libs** (e.g. 12.1) | The CUDA runtime libraries bundled inside the PyTorch wheel. PyTorch ships its own. | `pip install torch --index-url ...` |

Plus two more pieces:

- **`gcc` / `g++`** — the standard C/C++ compiler. `nvcc` calls it under the
  hood for the CPU-side code. Each CUDA release accepts only a window of gcc
  versions; gcc 12 is safe for both 12.1 and 12.8. If your system gcc is 14+,
  the build fails with cryptic pybind11 errors.
- **GPU compute capability** — the `.so` file we build only runs on the
  architectures listed at compile time, and PyTorch must independently ship
  kernels for that same architecture. `torch.cuda.get_arch_list()` is the
  authoritative check.

---

## Troubleshooting

**`ValueError: Unknown CUDA arch (12.0) or GPU not supported`**
Your PyTorch is too old for your GPU. This is the RTX 50-series (Blackwell,
`sm_120`) case: torch 2.5.1+cu121 knows nothing about compute capability 12.0,
so it rejects it before compiling anything. No `TORCH_CUDA_ARCH_LIST` setting
can fix this — you need a cu128 build. Re-run `fireants_install.py`, which now
picks the right lane automatically, or install manually:
```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
conda install -c "nvidia/label/cuda-12.8.2" cuda-toolkit -y
```

**`NVIDIA GeForce RTX 50xx with CUDA capability sm_120 is not compatible with the current PyTorch installation`**
Same root cause, and it is more serious than it looks: this is a *warning*, not
an error, so torch keeps running and everything appears to work — but the GPU
is unusable, with or without fused_ops. `fireants_check.py` reports this as a
FAIL. Fix it as above.

**`sqlite3.OperationalError: database is locked` during `conda env create`**
A bug in `conda-libmamba-solver` **26.4.0** (shipped with conda 26.3.x), which
enabled sharded repodata by default but opens its SQLite shard cache without
WAL mode or a busy timeout — its own reader and writer threads then race and
conda aborts mid-solve
([upstream #925](https://github.com/conda/conda-libmamba-solver/issues/925)).
Fixed in 26.4.1 and 26.4.2. `fireants_install.py` detects the affected version
and works around it automatically. To fix it by hand:
```bash
# if you can write to base conda (the real fix):
conda update -n base -c conda-forge conda-libmamba-solver

# otherwise (shared machine, read-only base):
rm -f ~/.conda/pkgs/cache/repodata_shards*.db*
export CONDA_PLUGINS_USE_SHARDED_REPODATA=false
```
Large environments make it far more likely, which is one reason
`environment.yml` is kept small.

**`pybind11::cpp_function` errors during build**
You're using a system `gcc` that's too new (≥14) instead of the conda one.
Re-activate the env (`conda deactivate && conda activate <env>`) and confirm
`which gcc` shows a path inside the conda env. If it doesn't, run
`conda install -c conda-forge gcc=12 gxx=12 -y`.

**`error: [Errno 2] No such file or directory: '.../bin/nvcc'`**
The CUDA toolkit didn't get installed into the env. Re-run the installer, or
install the toolkit for your lane manually (see the matrix above) and
`export CUDA_HOME=$CONDA_PREFIX`.

**`No kernel image is available for execution on the device`**
The `.so` was built for a different GPU arch. Rebuild after setting
`TORCH_CUDA_ARCH_LIST` to *your* compute capability (the installer does this
automatically — only an issue if you copied a `.so` from another machine).

**`libc10.so: cannot open shared object file`**
You imported `fireants_fused_ops` before `torch`. Always `import torch` first.

**`No module named 'simpleitk'` while installing fireants on Python 3.13**
SimpleITK has no Python 3.13 wheel as of 2026. Use `python=3.12`.

**`nipype.interfaces.fsl` can't find FSL**
FSL is a system install, not a pip package. Set `FSLDIR` and put
`$FSLDIR/bin` on your `PATH`.

---

## About `pre_env_from_history.yml`

This file is a `conda env export --from-history` of one developer's everyday
Anaconda environment. It lists ~350 explicitly-installed packages — spyder,
scrapy, qt, mkl, astropy, and so on — almost none of which FireANTs needs.

Using it is slower and more fragile than `environment.yml`:

* it takes minutes to solve rather than seconds,
* it pins every package to that one machine's versions,
* it pulls in the Anaconda `defaults` channel, which is gated behind a Terms
  of Service acceptance,
* and the sheer size of the solve is what makes the conda SQLite bug above
  much more likely to fire.

It is kept only for reproducing that specific environment. If you want it:
```bash
python fireants_install.py --env-yml pre_env_from_history.yml --env-name fireants
```

---

## Prerequisites

> The installer/diagnostic scripts check for all of these and tell you which
> are missing.

### NVIDIA GPU + driver
You need an NVIDIA GPU with a driver version **≥ 525**. Check with:
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

Anything from Maxwell (2014) onward is supported; look your GPU's compute
capability up at https://developer.nvidia.com/cuda-gpus and check it against
the [matrix above](#gpu-support-matrix).

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

---

## License

MIT — see [LICENSE](LICENSE).

The FireANTs library itself is distributed under its own license — see the
upstream repo: https://github.com/rohitrango/FireANTs.
