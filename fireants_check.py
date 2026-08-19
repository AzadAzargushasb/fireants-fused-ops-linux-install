#!/usr/bin/env python3
"""
fireants_check.py  --  DIAGNOSTIC ONLY (does NOT install anything)

Run this on any machine to find out whether it can run FireANTs + the optional
fused_ops CUDA extension, and what's missing if it can't.

  Usage:   conda activate <your-env> && python fireants_check.py
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

The pieces that have to agree with each other:

  * GPU COMPUTE CAPABILITY  ("6.1" for a GTX 1080, "8.6" for an RTX 3090,
        "12.0" for an RTX 5060).  This is the root of everything: it decides
        which CUDA toolkit and which PyTorch build can be used at all.

  * GPU DRIVER  (e.g. 580.142)
        Ships with the OS / NVIDIA driver install.  Shows up in `nvidia-smi`.
        The "CUDA Version: 13.0" line in nvidia-smi is the *highest* CUDA
        runtime this driver can run -- a CEILING, not what's installed.

  * CUDA TOOLKIT  (e.g. 12.1, or 12.8 for Blackwell)
        Contains `nvcc`, which turns `.cu` files into GPU machine code.
        Separate from the driver.  CUDA 12.8 was the first release that can
        emit code for Blackwell (sm_120) at all.

  * PYTORCH BUILD  (e.g. 2.5.1+cu121, or 2.7.0+cu128)
        Must be built against the same CUDA version AND actually contain
        kernels for your GPU's architecture.  `torch.cuda.get_arch_list()`
        is the authoritative answer -- cu121 builds stop at sm_90.

  * gcc  (the GNU C/C++ compiler)
        `nvcc` calls `gcc` under the hood for the host (CPU) parts.  Modern
        distros ship gcc 14/15, too new for CUDA 12.x -> we install gcc 12
        into the conda env.  A too-new system gcc is the classic cause of
        the `pybind11::cpp_function` ABI errors.

This script checks all of the above and tells you exactly what to fix.
"""

from __future__ import annotations
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass

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

# ---------- GPU / CUDA lane matrix ------------------------------------------ #
# Keep in sync with the identical table in fireants_install.py.  Duplicated on
# purpose so either script can be copied to a machine on its own.

@dataclass
class Lane:
    cap_lo: float
    cap_hi: float
    cuda_label: str
    cuda_version: str
    torch_versions: list
    torch_index: str
    blurb: str

CUDA_LANES = [
    Lane(5.0, 10.0, "nvidia/label/cuda-12.1.1", "12.1",
         ["2.5.1"], "https://download.pytorch.org/whl/cu121",
         "Maxwell..Hopper (sm_50..sm_90): GTX 10xx, RTX 20/30/40xx, A100, H100"),
    Lane(10.0, 99.0, "nvidia/label/cuda-12.8.2", "12.8",
         ["2.11.0", "2.7.0"], "https://download.pytorch.org/whl/cu128",
         "Blackwell (sm_100/sm_120): RTX 50-series, B100/B200"),
]

def cap_to_sm(cap: str) -> str:
    """'8.6' -> 'sm_86';  '12.0' -> 'sm_120'."""
    return "sm_" + cap.replace(".", "")

def arch_supported(cap: str, arches: list) -> bool:
    """Can a torch build advertising `arches` actually run on a GPU of `cap`?

    NOT an exact-match test.  CUDA binary compatibility runs UPWARD within a
    major version: a cubin built for sm_Xy executes on any device sm_Xz with
    z >= y.  Every cu121 wheel ships sm_60 and no sm_61, yet runs perfectly on
    a GTX 1080 (6.1) -- an exact-match check would condemn a working machine.
    PTX (`compute_Xy`) JITs forward onto any device >= X.y.

    What it still catches is the case that matters: an RTX 50-series card
    (12.0) against a build whose newest arch is sm_90, where no cubin and no
    PTX can bridge the gap.
    """
    try:
        major, minor = (int(x) for x in cap.split("."))
    except ValueError:
        return True                      # unknown capability -> don't cry wolf
    for a in arches:
        m = re.fullmatch(r"(sm|compute)_(\d+)[a-z]*", a)
        if not m:
            continue
        kind, num = m.group(1), m.group(2)
        a_major, a_minor = int(num[:-1]), int(num[-1])
        if kind == "sm" and a_major == major and a_minor <= minor:
            return True                  # binary-compatible cubin
        if kind == "compute" and (a_major, a_minor) <= (major, minor):
            return True                  # PTX JITs forward
    return False

def select_lane(cap: str | None) -> Lane | None:
    if not cap:
        return None
    try:
        c = float(cap)
    except ValueError:
        return None
    for lane in CUDA_LANES:
        if lane.cap_lo <= c < lane.cap_hi:
            return lane
    return None

def torch_install_cmd(lane: Lane) -> str:
    return (f"pip install torch=={lane.torch_versions[0]} "
            f"--index-url {lane.torch_index}")

# ---------- individual checks ----------------------------------------------- #

def check_os():
    if sys.platform != "linux":
        add("OS", "fail", f"{sys.platform}",
            fix="FireANTs fused_ops only supported on Linux. Use WSL2 on Windows.")
        return False
    add("OS", "pass", "linux")
    return True

def conda_info() -> dict:
    p = subprocess.run(["conda", "info", "--json"], capture_output=True, text=True)
    try:
        return json.loads(p.stdout)
    except (ValueError, TypeError):
        return {}

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
    check_solver()
    return True

def check_solver():
    """conda-libmamba-solver 26.4.0 turned sharded repodata on by default but
    opens its SQLite shard cache without WAL mode or a busy timeout.  Its own
    reader and writer threads then race, and conda dies mid-solve with
    `sqlite3.OperationalError: database is locked`.  Fixed in 26.4.1/26.4.2.
    See https://github.com/conda/conda-libmamba-solver/issues/925"""
    info = conda_info()
    ua = (info.get("solver") or {}).get("user_agent", "")
    m = re.search(r"conda-libmamba-solver/(\d+)\.(\d+)\.(\d+)", ua)
    if not m:
        return
    ver = tuple(int(g) for g in m.groups())
    vs = ".".join(str(x) for x in ver)
    if not ((26, 4, 0) <= ver < (26, 4, 1)):
        add("conda solver", "pass", f"conda-libmamba-solver {vs}")
        return
    if info.get("root_writable"):
        fix = ("This version races on its own SQLite shard cache and can abort\n"
               "`conda env create` with 'database is locked'. Upgrade:\n"
               "  conda update -n base -c conda-forge conda-libmamba-solver")
    else:
        fix = ("This version races on its own SQLite shard cache and can abort\n"
               "`conda env create` with 'database is locked'. Base conda is\n"
               "read-only here, so use the documented opt-out instead:\n"
               "  export CONDA_PLUGINS_USE_SHARDED_REPODATA=false")
    add("conda solver", "warn", f"conda-libmamba-solver {vs}  (known sqlite race)",
        fix=fix)

    # Surface the cache file itself -- deleting it clears the corrupt-DB variant.
    for d in info.get("pkgs_dirs", []):
        for path in glob.glob(os.path.join(d, "cache", "repodata_shards*.db")):
            mb = os.path.getsize(path) / 1e6
            add("repodata shard cache", "warn", f"{path}  ({mb:.1f} MB)",
                fix=f"If conda dies with 'database is locked', remove it:\n"
                    f"  rm -f {path}*")

def check_python():
    v = sys.version_info
    s = f"{v.major}.{v.minor}.{v.micro}"
    # FireANTs wheels exist for 3.9-3.12.  3.13 has no SimpleITK wheel yet.
    if (v.major, v.minor) < (3, 9):
        add("python", "fail", s, fix="Use python 3.9-3.12.")
    elif (v.major, v.minor) > (3, 12):
        add("python", "warn", s,
            fix="python 3.13+ has no SimpleITK wheel as of 2026. "
                "Recreate env with python=3.12.")
    else:
        add("python", "pass", s)

def check_driver(lane: Lane | None):
    """nvidia-smi reports driver version + CUDA CEILING (not what's installed)."""
    if shutil.which("nvidia-smi") is None:
        add("nvidia driver", "fail", "nvidia-smi not found",
            fix="Install the NVIDIA proprietary driver (>=525).")
        return None
    rc, out = run("nvidia-smi --query-gpu=driver_version --format=csv,noheader")
    if rc != 0 or not out:
        add("nvidia driver", "fail", "nvidia-smi failed",
            fix="Driver may be broken; reboot or reinstall the NVIDIA driver.")
        return None
    drv = out.splitlines()[0].strip()
    major = int(drv.split(".")[0])
    cuda_ver = lane.cuda_version if lane else "12.x"
    if major < 525:
        add("nvidia driver", "fail", drv,
            fix=f"Driver {drv} too old for CUDA {cuda_ver}. Need >=525.")
    else:
        add("nvidia driver", "pass", drv)
    return drv

def check_gpu_arch():
    """Compute capability decides what arch nvcc must compile for."""
    rc, out = run("nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader")
    if rc != 0 or not out:
        add("GPU + compute_cap", "fail", "could not query",
            fix="Run `nvidia-smi` manually and look your GPU up at "
                "https://developer.nvidia.com/cuda-gpus")
        return None
    line = out.splitlines()[0]
    parts = [x.strip() for x in line.split(",")]
    if len(parts) < 2 or not re.fullmatch(r"\d+\.\d+", parts[1]):
        add("GPU + compute_cap", "warn", line,
            fix="nvidia-smi is too old to report compute_cap. Look your GPU up "
                "at https://developer.nvidia.com/cuda-gpus")
        return None
    name, cap = parts[0], parts[1]
    add("GPU + compute_cap", "pass", f"{name} ({cap_to_sm(cap)})")
    return cap

def report_lane(cap: str | None, lane: Lane | None):
    if cap is None:
        return
    if lane is None:
        add("supported CUDA stack", "fail",
            f"no known CUDA/PyTorch build supports {cap_to_sm(cap)}",
            fix="The oldest architecture PyTorch still ships kernels for is "
                "sm_50 (Maxwell, 2014).")
        return
    add("supported CUDA stack", "pass",
        f"CUDA {lane.cuda_version} + torch {lane.torch_versions[0]} "
        f"({lane.torch_index.rsplit('/', 1)[-1]})")

def check_nvcc(lane: Lane | None):
    """nvcc inside the active conda env is what we need.  System nvcc (e.g.
    /opt/cuda/bin/nvcc) is often a different version that won't produce a .so
    binary-compatible with the installed PyTorch -- avoid it."""
    want = lane.cuda_version if lane else None
    label = lane.cuda_label if lane else "nvidia/label/cuda-12.1.1"
    install = f"conda install -c '{label}' cuda-toolkit -y"
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        add("nvcc (CUDA toolkit)", "fail", "not found", fix=install)
        return None
    rc, out = run("nvcc --version")
    m = re.search(r"release (\d+\.\d+)", out)
    ver = m.group(1) if m else "?"
    prefix = os.environ.get("CONDA_PREFIX", "")
    in_env = bool(prefix) and nvcc.startswith(prefix)
    if want and not ver.startswith(want):
        add("nvcc (CUDA toolkit)", "fail",
            f"{ver} at {nvcc}  (this GPU needs CUDA {want})", fix=install)
    elif not in_env:
        add("nvcc (CUDA toolkit)", "warn", f"{ver} at {nvcc}  (NOT in conda env)",
            fix="System nvcc may be the wrong version. Install an env-local "
                f"toolkit:\n  {install}")
    else:
        add("nvcc (CUDA toolkit)", "pass", f"{ver}  ({nvcc})")
    return ver

def check_gcc():
    """CUDA 12.x supports gcc up to 13/14 depending on minor version; gcc 15+
    will fail with cryptic errors.  gcc 12 is safe across the board."""
    gcc = shutil.which("gcc")
    if gcc is None:
        add("gcc (C++ compiler)", "fail", "not found",
            fix="conda install -c conda-forge gcc=12 gxx=12 -y")
        return None
    rc, out = run("gcc -dumpversion")
    ver = out.strip()
    major = int(ver.split(".")[0]) if ver.split(".")[0].isdigit() else 99
    prefix = os.environ.get("CONDA_PREFIX", "")
    in_env = bool(prefix) and gcc.startswith(prefix)
    if major > 13:
        add("gcc (C++ compiler)", "fail",
            f"{ver} at {gcc}  (CUDA 12.x wants gcc <=13)",
            fix="conda install -c conda-forge gcc=12 gxx=12 -y")
    elif not in_env:
        add("gcc (C++ compiler)", "warn", f"{ver} at {gcc}  (system gcc, prefer env-local)",
            fix="conda install -c conda-forge gcc=12 gxx=12 -y")
    else:
        add("gcc (C++ compiler)", "pass", f"{ver}  ({gcc})")
    return ver

def check_torch(cap: str | None, lane: Lane | None):
    """The check that matters most.

    It is not enough for torch to import, report a CUDA build, and even see the
    GPU: the wheel must actually CONTAIN compiled kernels for this GPU's
    architecture.  A cu121 wheel on an RTX 5060 satisfies every naive check and
    is still completely unusable -- torch itself only emits a UserWarning.
    `torch.cuda.get_arch_list()` is the authoritative answer."""
    fix_install = torch_install_cmd(lane) if lane else \
        "pip install torch --index-url https://download.pytorch.org/whl/cu121"
    try:
        import torch  # noqa
    except Exception as e:
        add("PyTorch", "fail", f"import failed: {e}", fix=fix_install)
        return None

    detail = f"{torch.__version__} (CUDA build {torch.version.cuda})"
    if not torch.cuda.is_available():
        add("PyTorch", "fail", detail + " -- CUDA NOT available at runtime",
            fix="Driver may be missing or too old. Run `nvidia-smi`.")
        return torch

    arches = torch.cuda.get_arch_list()
    if cap and not arch_supported(cap, arches):
        add("PyTorch", "fail",
            f"{detail} has NO {cap_to_sm(cap)} kernels for this GPU",
            fix=f"This torch build supports: {' '.join(arches)}\n"
                f"Your GPU is {cap_to_sm(cap)}, so this install cannot use it "
                f"at all\n(fused_ops or not). Install the right build:\n"
                f"  {fix_install}")
        return torch

    if lane and torch.version.cuda != lane.cuda_version:
        add("PyTorch", "warn",
            f"{detail} -- fused_ops wants a CUDA {lane.cuda_version} build",
            fix=f"For fused_ops, match the toolkit:\n  {fix_install}")
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
            fix="pip install --no-deps fireants==1.4.0")
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

def check_fsl():
    """Only relevant if you use nipype.interfaces.fsl. FSL is a system install,
    not a pip package, so it can't be fixed from inside the env."""
    try:
        import nipype  # noqa
    except Exception:
        return          # analysis stack not installed -- nothing to say
    fsldir = os.environ.get("FSLDIR", "")
    flirt = shutil.which("flirt")
    if fsldir and flirt:
        add("FSL (for nipype)", "pass", f"{fsldir}  (flirt: {flirt})")
    else:
        add("FSL (for nipype)", "warn",
            f"FSLDIR={fsldir or 'unset'}, flirt {'found' if flirt else 'not on PATH'}",
            fix="nipype.interfaces.fsl needs FSL installed system-wide:\n"
                "  export FSLDIR=/path/to/fsl && export PATH=$FSLDIR/bin:$PATH")

def check_jupyter_kernel():
    """Is this env registered so JupyterLab can select it?"""
    env = os.environ.get("CONDA_DEFAULT_ENV", "")
    prefix = os.environ.get("CONDA_PREFIX", "")
    if not env or not prefix:
        return
    kernel_dirs = [
        os.path.expanduser(f"~/.local/share/jupyter/kernels/{env}"),
        os.path.join(prefix, "share", "jupyter", "kernels", env),
    ]
    found = next((d for d in kernel_dirs if os.path.isdir(d)), None)
    if found:
        add("jupyter kernel", "pass", found)
    else:
        add("jupyter kernel", "warn", f"no kernelspec named '{env}'",
            fix="Make JupyterLab see this env:\n"
                "  pip install ipykernel\n"
                f"  python -m ipykernel install --user --name {env} "
                f"--display-name 'Python ({env})'")

# ---------- main ------------------------------------------------------------ #

def main():
    print(f"\n{DIM}== FireANTs / fused_ops environment check =={END}\n")
    if not check_os():
        print_summary(); return
    check_conda()
    check_python()
    cap = check_gpu_arch()
    lane = select_lane(cap)
    check_driver(lane)
    report_lane(cap, lane)
    check_nvcc(lane)
    check_gcc()
    check_torch(cap, lane)
    check_fireants()
    check_fused_ops()
    check_fsl()
    check_jupyter_kernel()
    print_summary(cap, lane)

def print_summary(cap: str | None = None, lane: Lane | None = None):
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
        if r.status != "pass" and r.fix and r.fix not in seen:
            seen.add(r.fix)
            print(f"  # {r.name}")
            for ln in r.fix.splitlines():
                print(f"  {ln}")
            print()

    # Only offer the build recipe if a build could actually succeed.  Printing
    # `TORCH_CUDA_ARCH_LIST=12.0` at someone whose torch has no sm_120 kernels
    # just sends them into a build that cannot work.
    torch_broken = any(r.name == "PyTorch" and r.status == "fail" for r in results)
    fused_missing = any(r.name == "fireants_fused_ops" and r.status != "pass"
                        for r in results)
    if fused_missing and not torch_broken and cap:
        env_prefix = os.environ.get("CONDA_PREFIX", "$CONDA_PREFIX")
        print(f"  # Build fused_ops for THIS machine's GPU (compute capability {cap})")
        print(f"  export CUDA_HOME={env_prefix}")
        print(f'  export TORCH_CUDA_ARCH_LIST="{cap}"')
        print( "  git clone https://github.com/rohitrango/FireANTs.git  # if not already cloned")
        print( "  cd FireANTs/fused_ops")
        print( "  python setup.py build_ext && python setup.py install")
        print()
    elif fused_missing and torch_broken:
        print("  # fused_ops")
        print("  Fix PyTorch first -- building fused_ops against a torch that has")
        print("  no kernels for your GPU cannot work.")
        print()
    print(f"{DIM}Or run fireants_install.py to do all of the above automatically.{END}\n")

if __name__ == "__main__":
    main()
