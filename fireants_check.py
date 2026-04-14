#!/usr/bin/env python3
"""
fireants_check.py  --  DIAGNOSTIC ONLY (does NOT install anything)

Run this on any machine to find out whether it can run FireANTs + the optional
fused_ops CUDA extension, and what's missing if it can't.

  Usage:   python fireants_check.py
  Output:  one row per component (PASS / WARN / FAIL) and copy-paste fix
           commands at the bottom.

------------------------------------------------------------------------------
BACKGROUND  (read once, then skip)
------------------------------------------------------------------------------

FireANTs has TWO parts:

  1. The pure-Python package `fireants` (pip-installable, works anywhere with
     PyTorch + CUDA).  This alone is enough to run registrations.

  2. An OPTIONAL compiled CUDA extension `fireants_fused_ops` that gives ~80%
     speedup and ~20% less GPU memory.  This must be COMPILED FROM SOURCE on
     the target machine because the .so file is GPU-architecture-specific.

To compile (2) you need three things that often confuse people:

  * GPU DRIVER  (e.g. 580.142)
        Ships with the OS / NVIDIA driver install.  Shows up in `nvidia-smi`.
        The "CUDA Version: 13.0" line in nvidia-smi is the *highest* CUDA
        runtime this driver can run -- a CEILING, not what's installed.

  * CUDA TOOLKIT  (e.g. 12.1)
        Contains `nvcc`, the NVIDIA CUDA compiler.  This is what turns
        `.cu` source files into GPU machine code.  It is SEPARATE from the
        driver and must be installed on its own.  PyTorch's `cu121` wheels
        were built against CUDA 12.1, so the toolkit version we install must
        match (12.1) so the produced .so is binary-compatible with PyTorch.

  * gcc  (the GNU C/C++ compiler)
        Standard Linux compiler.  `nvcc` calls `gcc` under the hood for the
        host (CPU) parts of CUDA code.  Each CUDA toolkit version supports
        only a window of gcc versions:
            CUDA 12.1  ->  gcc 6 .. 12  (NOT 13, NOT 15)
        Modern Linux distros ship gcc 14 or 15, which is too new -> we must
        install an older gcc into the conda env.

  * GPU COMPUTE CAPABILITY  (e.g. "6.1" for GTX 1080, "8.6" for RTX 3090)
        Each NVIDIA GPU has a "compute capability" -- the CUDA feature
        version it supports.  When we compile a .so we tell nvcc which
        capabilities to build for via TORCH_CUDA_ARCH_LIST.  If we build
        for "6.1" only and try to load the .so on a "9.0" GPU, it will
        fail with "no kernel image is available for execution".

The most common build failure (the `pybind11::cpp_function` errors your
professor saw) is an ABI mismatch caused by a too-new SYSTEM gcc being
picked up instead of a CUDA-compatible gcc inside the conda env.

This script checks all of the above and tells you exactly what to fix.
"""

from __future__ import annotations
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

# ---------- tiny helpers ---------------------------------------------------- #

GREEN = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"; DIM = "\033[2m"; END = "\033[0m"
TICK  = f"{GREEN}[ PASS ]{END}"
WARN  = f"{YELLOW}[ WARN ]{END}"
FAIL  = f"{RED}[ FAIL ]{END}"

@dataclass
class Result:
    name: str
    status: str           # "pass" | "warn" | "fail"
    detail: str
    fix: str = ""

results: list[Result] = []

def add(name, status, detail, fix=""):
    results.append(Result(name, status, detail, fix))
    sym = {"pass": TICK, "warn": WARN, "fail": FAIL}[status]
    print(f"  {sym}  {name:<28}  {detail}")

def run(cmd: str) -> tuple[int, str]:
    """Run a shell command, return (returncode, combined stdout+stderr)."""
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()

# ---------- individual checks ----------------------------------------------- #

def check_os():
    if sys.platform != "linux":
        add("OS", "fail", f"{sys.platform}",
            fix="FireANTs fused_ops only supported on Linux. Use WSL2 on Windows.")
        return False
    add("OS", "pass", "linux")
    return True

def check_conda():
    """conda is required because we install gcc + cuda-toolkit into the env
    rather than touching the system."""
    if shutil.which("conda") is None:
        add("conda", "fail", "not found",
            fix="Install Miniconda: https://docs.conda.io/en/latest/miniconda.html")
        return False
    rc, out = run("conda --version")
    add("conda", "pass", out)
    # Which env are we in?
    env = os.environ.get("CONDA_DEFAULT_ENV", "(none -- not inside a conda env)")
    prefix = os.environ.get("CONDA_PREFIX", "")
    add("active conda env", "pass" if prefix else "warn",
        f"{env}  ({prefix})" if prefix else env,
        fix="Activate your env first:  conda activate <envname>")
    return True

def check_python():
    v = sys.version_info
    s = f"{v.major}.{v.minor}.{v.micro}"
    # FireANTs 1.4.0 wheels exist for 3.9-3.12.  3.13 has no SimpleITK wheel yet.
    if (v.major, v.minor) < (3, 9):
        add("python", "fail", s, fix="Use python 3.9-3.12.")
    elif (v.major, v.minor) > (3, 12):
        add("python", "warn", s,
            fix="python 3.13+ has no SimpleITK wheel as of early 2026. "
                "Recreate env with python=3.12.")
    else:
        add("python", "pass", s)

def check_driver():
    """nvidia-smi reports driver version + CUDA CEILING (not what's installed).
    The driver must be new enough to run CUDA 12.1 -> driver >= 525."""
    if shutil.which("nvidia-smi") is None:
        add("nvidia driver", "fail", "nvidia-smi not found",
            fix="Install NVIDIA proprietary driver (>=525 for CUDA 12.1).")
        return None
    rc, out = run("nvidia-smi --query-gpu=driver_version --format=csv,noheader")
    if rc != 0:
        add("nvidia driver", "fail", "nvidia-smi failed",
            fix="Driver may be broken; reboot or reinstall NVIDIA driver.")
        return None
    drv = out.splitlines()[0].strip()
    major = int(drv.split(".")[0])
    if major < 525:
        add("nvidia driver", "fail", drv,
            fix=f"Driver {drv} too old for CUDA 12.1. Need >=525.")
    else:
        add("nvidia driver", "pass", drv)
    return drv

def check_gpu_arch():
    """Compute capability decides what arch nvcc must compile for."""
    rc, out = run("nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader")
    if rc != 0 or not out:
        add("GPU + compute_cap", "fail", "could not query",
            fix="Run `nvidia-smi` manually and look up your GPU's compute capability.")
        return None
    line = out.splitlines()[0]
    name, cap = [x.strip() for x in line.split(",")]
    add("GPU + compute_cap", "pass", f"{name} (sm_{cap.replace('.', '')})")
    return cap

def check_nvcc():
    """nvcc inside the active conda env is what we need.  System nvcc (e.g.
    /opt/cuda/bin/nvcc) is often an old version (CUDA 9 from 2017) that
    won't link against modern PyTorch -- avoid it."""
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        add("nvcc (CUDA toolkit)", "fail", "not found",
            fix="conda install -c 'nvidia/label/cuda-12.1.1' cuda-toolkit -y")
        return None
    rc, out = run("nvcc --version")
    # Parse "release 12.1, V12.1.105"
    ver = "?"
    for tok in out.split():
        if tok.startswith("V") and "." in tok:
            ver = tok.lstrip("V").rstrip(",")
            break
    prefix = os.environ.get("CONDA_PREFIX", "")
    in_env = prefix and nvcc.startswith(prefix)
    if not in_env:
        add("nvcc (CUDA toolkit)", "warn", f"{ver} at {nvcc}  (NOT in conda env)",
            fix="System nvcc may be wrong version. Install env-local toolkit:\n"
                "  conda install -c 'nvidia/label/cuda-12.1.1' cuda-toolkit -y")
    elif not ver.startswith("12.1"):
        add("nvcc (CUDA toolkit)", "warn", f"{ver}  (PyTorch needs 12.1)",
            fix="conda install -c 'nvidia/label/cuda-12.1.1' cuda-toolkit -y")
    else:
        add("nvcc (CUDA toolkit)", "pass", f"{ver}  ({nvcc})")
    return ver

def check_gcc():
    """CUDA 12.1 supports gcc 6..12.  gcc 13+ will fail with cryptic errors."""
    gcc = shutil.which("gcc")
    if gcc is None:
        add("gcc (C++ compiler)", "fail", "not found",
            fix="conda install -c conda-forge gcc=12 gxx=12 -y")
        return None
    rc, out = run("gcc -dumpversion")
    ver = out.strip()
    major = int(ver.split(".")[0])
    prefix = os.environ.get("CONDA_PREFIX", "")
    in_env = prefix and gcc.startswith(prefix)
    if major > 13:
        add("gcc (C++ compiler)", "fail",
            f"{ver} at {gcc}  (CUDA 12.1 supports gcc <=13)",
            fix="conda install -c conda-forge gcc=12 gxx=12 -y")
    elif not in_env:
        add("gcc (C++ compiler)", "warn", f"{ver} at {gcc}  (system gcc, prefer env-local)",
            fix="conda install -c conda-forge gcc=12 gxx=12 -y")
    else:
        add("gcc (C++ compiler)", "pass", f"{ver}  ({gcc})")
    return ver

def check_torch():
    try:
        import torch  # noqa
    except Exception as e:
        add("PyTorch", "fail", f"import failed: {e}",
            fix="pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121")
        return None
    detail = f"{torch.__version__} (CUDA build {torch.version.cuda})"
    if torch.version.cuda != "12.1":
        add("PyTorch", "warn", detail,
            fix="For fused_ops, install torch==2.5.1+cu121:\n"
                "  pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121")
    elif not torch.cuda.is_available():
        add("PyTorch", "fail", detail + " -- CUDA NOT available at runtime",
            fix="Driver may be missing or too old. Run `nvidia-smi`.")
    else:
        add("PyTorch", "pass", f"{detail}, GPU: {torch.cuda.get_device_name(0)}")
    return torch

def check_fireants():
    try:
        import fireants  # noqa
        from importlib.metadata import version
        v = version("fireants")
    except Exception as e:
        add("fireants (python)", "fail", f"import failed: {e}",
            fix="pip install fireants==1.4.0")
        return None
    add("fireants (python)", "pass", v)
    return v

def check_fused_ops():
    """The whole reason we ran this script."""
    try:
        import torch  # must import torch first so libc10.so is loaded
        import fireants_fused_ops as ffo  # noqa
    except ImportError as e:
        add("fireants_fused_ops",  "warn", f"NOT installed ({e})",
            fix="Build from source -- see install commands at bottom of report.")
        return False
    add("fireants_fused_ops", "pass", f"installed at {ffo.__file__}")
    return True

# ---------- main ------------------------------------------------------------ #

def main():
    print(f"\n{DIM}== FireANTs / fused_ops environment check =={END}\n")
    if not check_os():       print_summary(); return
    check_conda()
    check_python()
    check_driver()
    cap = check_gpu_arch()
    check_nvcc()
    check_gcc()
    check_torch()
    check_fireants()
    check_fused_ops()
    print_summary(cap)

def print_summary(cap: str | None = None):
    n_fail = sum(r.status == "fail" for r in results)
    n_warn = sum(r.status == "warn" for r in results)
    print()
    if n_fail == 0 and n_warn == 0:
        print(f"{GREEN}All checks passed -- you can run FireANTs with fused_ops.{END}")
        return
    print(f"{YELLOW}Issues found:{END}  {n_fail} fail, {n_warn} warn\n")
    print("Suggested fixes (run inside your conda env):\n")
    seen = set()
    for r in results:
        if r.fix and r.fix not in seen:
            seen.add(r.fix)
            print(f"  # {r.name}")
            for ln in r.fix.splitlines():
                print(f"  {ln}")
            print()
    # If everything else looks good but fused_ops missing, print the build recipe.
    if any(r.name == "fireants_fused_ops" and r.status != "pass" for r in results):
        arch = cap or "<your_compute_cap>"  # e.g. 6.1, 8.6, 9.0
        env_prefix = os.environ.get("CONDA_PREFIX", "$CONDA_PREFIX")
        print(f"  # Build fused_ops for THIS machine's GPU (compute capability {arch})")
        print(f"  export CUDA_HOME={env_prefix}")
        print(f'  export TORCH_CUDA_ARCH_LIST="{arch}"')
        print( "  git clone https://github.com/rohitrango/FireANTs.git  # if not already cloned")
        print( "  cd FireANTs/fused_ops")
        print( "  python setup.py build_ext && python setup.py install")
        print()
    print(f"{DIM}Or run fireants_install.py to do all of the above automatically.{END}\n")

if __name__ == "__main__":
    main()
