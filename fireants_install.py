#!/usr/bin/env python3
"""
fireants_install.py  --  AUTO-INSTALLER  (will install / build things!)

Sets up a working FireANTs + fireants_fused_ops environment from scratch.

  Usage:
    # (recommended) bootstrap from a yml exported on a known-good machine
    python fireants_install.py --env-yml /path/to/pre_env_from_history.yml

    # or create a fresh env with the right pieces
    python fireants_install.py --env-name fireants

    # or update an env that already exists / is already activated
    python fireants_install.py --env-name pre_env --use-existing

  What it does (in order):
    1. Verifies you have conda + an NVIDIA GPU + driver new enough for CUDA 12.1.
    2. Creates / updates a conda env (from your yml if given, else fresh).
    3. Inside that env, installs:
         - python 3.12 (if creating fresh)
         - PyTorch 2.5.1 + CUDA 12.1
         - fireants 1.4.0
         - cuda-toolkit 12.1   (provides nvcc -- the CUDA compiler)
         - gcc 12 + gxx 12     (CUDA 12.1 supports gcc <= 13 only)
    4. Detects this machine's GPU compute capability via nvidia-smi.
    5. Clones https://github.com/rohitrango/FireANTs and builds fused_ops
       for that compute capability (the .so is GPU-arch-specific).
    6. Verifies `import fireants_fused_ops` succeeds.

------------------------------------------------------------------------------
WHY THIS IS NOT A ONE-LINER
------------------------------------------------------------------------------
fused_ops is a CUDA C++ extension.  Building it requires three things that
must all match each other:

  * nvcc           -- NVIDIA's CUDA compiler.  Lives in the "CUDA toolkit".
                      Must be CUDA 12.1 to match the PyTorch wheel we use.

  * gcc / g++      -- The standard Linux C/C++ compiler.  nvcc invokes it
                      for the CPU-side parts of the code.  CUDA 12.1 only
                      supports gcc 6..13.  Modern distros ship gcc 14/15
                      which produces obscure pybind11::cpp_function errors --
                      this is the #1 cause of "fused_ops won't build".

  * GPU compute    -- Each NVIDIA GPU has a "compute capability" (e.g. 6.1
    capability       for GTX 1080, 8.6 for RTX 3090, 9.0 for H100).  The .so
                     we build only runs on the architectures we list in
                     TORCH_CUDA_ARCH_LIST.  This script auto-detects yours.

The driver version (the "CUDA Version: 13.0" you see in `nvidia-smi`) is
just the *ceiling* the driver can run -- it does NOT mean CUDA 13 toolkit
is installed.  This script ignores it except to verify it's >= 525.
"""

from __future__ import annotations
import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

# ---------- pretty output --------------------------------------------------- #

GREEN = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"; BOLD = "\033[1m"; END = "\033[0m"

def step(msg):  print(f"\n{BOLD}==>{END} {msg}")
def info(msg):  print(f"    {msg}")
def ok(msg):    print(f"    {GREEN}[OK]{END}  {msg}")
def warn(msg):  print(f"    {YELLOW}[WARN]{END} {msg}")
def die(msg, code=1):
    print(f"\n{RED}[FATAL]{END} {msg}\n"); sys.exit(code)

# ---------- shell helpers --------------------------------------------------- #

def sh(cmd: str, check: bool = True, capture: bool = False) -> str:
    """Run a shell command. If capture, return stdout. Echo otherwise."""
    if capture:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if check and p.returncode != 0:
            die(f"command failed:\n  {cmd}\n{p.stderr}")
        return (p.stdout + p.stderr).strip()
    else:
        info(f"$ {cmd}")
        rc = subprocess.call(cmd, shell=True)
        if check and rc != 0:
            die(f"command failed (exit {rc}):\n  {cmd}")
        return ""

def in_env(env_name: str, cmd: str, check: bool = True, capture: bool = False) -> str:
    """Run `cmd` inside the given conda env (whether or not we're in one now)."""
    # `conda run -n <env>` runs without needing to source conda.sh.
    # NOTE: must use shlex.quote() (NOT Python's repr()) -- repr() produces
    # \-escaped single quotes which are NOT valid bash syntax, leading to
    # silently-malformed commands and false positives in existence checks.
    full = f"conda run -n {env_name} --no-capture-output bash -c {shlex.quote(cmd)}"
    return sh(full, check=check, capture=capture)

def env_has_module(env_name: str, module: str) -> bool:
    """Return True iff `import <module>` succeeds inside the env.
    Uses the EXIT CODE, not stdout matching, so it's robust to weird
    error messages that happen to contain literal substrings."""
    rc = subprocess.call(
        ["conda", "run", "-n", env_name, "python", "-c", f"import {module}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return rc == 0

# ---------- prechecks ------------------------------------------------------- #

def check_platform():
    step("Checking platform")
    if sys.platform != "linux":
        die("fused_ops only supported on Linux. Use WSL2 on Windows.")
    ok("linux")
    if shutil.which("conda") is None:
        die("conda not found. Install Miniconda first: "
            "https://docs.conda.io/en/latest/miniconda.html")
    ok(f"conda: {sh('conda --version', capture=True)}")
    if shutil.which("nvidia-smi") is None:
        die("nvidia-smi not found. Install the NVIDIA proprietary driver (>=525).")

def check_driver_and_gpu() -> str:
    """Returns compute capability like '6.1'."""
    step("Checking NVIDIA driver and GPU")
    drv = sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader",
             capture=True).splitlines()[0].strip()
    drv_major = int(drv.split(".")[0])
    if drv_major < 525:
        die(f"Driver {drv} is too old for CUDA 12.1. Need >=525. Update NVIDIA driver first.")
    ok(f"driver {drv}  (>=525, supports CUDA 12.1)")

    line = sh("nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader",
              capture=True).splitlines()[0]
    name, cap = [x.strip() for x in line.split(",")]
    ok(f"GPU: {name}, compute capability {cap} (sm_{cap.replace('.', '')})")
    return cap

# ---------- env setup ------------------------------------------------------- #

def env_exists(name: str) -> bool:
    out = sh("conda env list", capture=True)
    return any(line.split() and line.split()[0] == name for line in out.splitlines())

def create_or_update_env(env_name: str, env_yml: str | None, use_existing: bool):
    step(f"Setting up conda env: {env_name}")
    exists = env_exists(env_name)

    if env_yml:
        if not Path(env_yml).is_file():
            die(f"--env-yml file not found: {env_yml}")
        if exists and not use_existing:
            warn(f"env '{env_name}' already exists; updating from yml")
            sh(f"conda env update -n {env_name} -f {env_yml}")
        elif exists and use_existing:
            ok(f"using existing env '{env_name}' (skipping yml import)")
        else:
            sh(f"conda env create -n {env_name} -f {env_yml}")
        ok("env created/updated from yml")
    else:
        if exists:
            if not use_existing:
                die(f"env '{env_name}' already exists. "
                    f"Pass --use-existing to add packages to it, or pick a different --env-name.")
            ok(f"using existing env '{env_name}'")
        else:
            sh(f"conda create -n {env_name} python=3.12 -y")
            ok("fresh env created with python 3.12")

def env_prefix(env_name: str) -> str:
    """Absolute path to the env (e.g. /home/x/.conda/envs/fireants).
    `conda run` does NOT set $CONDA_PREFIX to the env path -- it inherits
    from the parent shell -- so we ask Python inside the env for sys.prefix."""
    p = subprocess.run(
        ["conda", "run", "-n", env_name, "python", "-c",
         "import sys; print(sys.prefix)"],
        capture_output=True, text=True, check=True,
    )
    return p.stdout.strip()

def env_has_executable(env_name: str, exe: str) -> bool:
    """True iff <env-prefix>/bin/<exe> exists and is executable."""
    return os.access(os.path.join(env_prefix(env_name), "bin", exe), os.X_OK)

def install_pip_packages(env_name: str):
    """PyTorch + fireants come from pip, not conda, because we need the
    cu121 wheel and a specific fireants version."""
    step("Installing PyTorch (cu121) and fireants")
    if env_has_module(env_name, "torch"):
        ok("torch already installed (skipping)")
    else:
        in_env(env_name,
               "pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121")
        ok("torch installed")

    if env_has_module(env_name, "fireants"):
        ok("fireants already installed (skipping)")
    else:
        # fireants 1.4.0 hard-pins `simpleitk==2.2.1`, which has been
        # un-published from PyPI for python >=3.12 (only 2.1.0 and 2.3.0+
        # have wheels now).  Install a compatible SimpleITK + the other
        # runtime deps first, then install fireants with --no-deps so pip
        # doesn't insist on 2.2.1.  Any SimpleITK 2.x works at runtime --
        # this was verified on a working install with SimpleITK 2.5.2.
        info("Installing fireants runtime deps (working around simpleitk==2.2.1 pin)")
        in_env(env_name,
               "pip install 'simpleitk>=2.3,<3' hydra-core typing "
               "matplotlib nibabel 'numpy<2' pandas pytest "
               "scikit-image scipy tqdm")
        in_env(env_name, "pip install --no-deps fireants==1.4.0")
        ok("fireants installed (with SimpleITK >=2.3 instead of pinned 2.2.1)")

def install_compiler_stack(env_name: str):
    """cuda-toolkit -> nvcc.  gcc 12 -> CUDA-compatible host compiler."""
    step("Installing CUDA toolkit (nvcc) into the env")
    if env_has_executable(env_name, "nvcc"):
        ok("nvcc already in env (skipping cuda-toolkit install)")
    else:
        # Use NVIDIA's own channel for the exact 12.1.1 build.
        sh(f"conda install -n {env_name} -c 'nvidia/label/cuda-12.1.1' cuda-toolkit -y")
        ok("cuda-toolkit 12.1 installed")

    step("Installing gcc/gxx 12 into the env (CUDA 12.1 needs gcc <= 13)")
    if env_has_executable(env_name, "gcc"):
        ok("env-local gcc already installed (skipping)")
    else:
        sh(f"conda install -n {env_name} -c conda-forge gcc=12 gxx=12 -y")
        ok("gcc/gxx 12 installed")

# ---------- fused_ops build ------------------------------------------------- #

def clone_fireants(work_dir: Path) -> Path:
    step("Cloning FireANTs source (for fused_ops sources)")
    repo = work_dir / "FireANTs"
    if repo.is_dir():
        ok(f"already cloned at {repo}")
    else:
        work_dir.mkdir(parents=True, exist_ok=True)
        sh(f"git clone https://github.com/rohitrango/FireANTs.git {repo}")
    return repo

def build_fused_ops(env_name: str, repo: Path, compute_cap: str):
    step(f"Building fireants_fused_ops for compute capability {compute_cap}")
    fused_dir = repo / "fused_ops"
    if not fused_dir.is_dir():
        die(f"fused_ops dir missing: {fused_dir}")

    # Two separate hazards when building torch CUDA extensions inside conda:
    #
    #   1. `conda run -n <env>` doesn't isolate the env -- it inherits (and
    #      keeps) anything the parent's base-env activate.d scripts set.
    #      If base conda also has gcc installed, CC/CXX may already point at
    #      /opt/miniconda3/bin/x86_64-conda-linux-gnu-c++ (a gcc >=13), and
    #      the base activate scripts re-set them even if we pass CC/CXX in
    #      the subprocess env.  CUDA 12.1 requires gcc <=12 -> build fails.
    #
    #   2. torch's cpp_extension passes `-ccbin <CXX>` to nvcc inline.  If
    #      we use NVCC_PREPEND_FLAGS to inject our own -ccbin, it lands
    #      BEFORE torch's and loses (nvcc's "last -ccbin wins").  We have
    #      to APPEND instead.
    #
    # Workaround: bypass `conda run` entirely, invoke the env's python by
    # absolute path, and pass an explicit, minimal env= with the env's
    # compilers and PATH.  Use NVCC_APPEND_FLAGS so our -ccbin overrides
    # whatever torch picked.
    prefix = env_prefix(env_name)
    env_bin = os.path.join(prefix, "bin")
    env_python = os.path.join(env_bin, "python")
    if not os.access(env_python, os.X_OK):
        die(f"env python not found at {env_python}")

    # Prefer the conda-forge toolchain wrappers (x86_64-conda-linux-gnu-gcc)
    # when present -- they're what torch's cpp_extension looks for first.
    def pick(*names):
        for n in names:
            path = os.path.join(env_bin, n)
            if os.access(path, os.X_OK):
                return path
        return None
    cc  = pick("x86_64-conda-linux-gnu-cc", "x86_64-conda-linux-gnu-gcc", "gcc")
    cxx = pick("x86_64-conda-linux-gnu-c++", "x86_64-conda-linux-gnu-g++", "g++")
    if cc is None or cxx is None:
        die(f"No gcc/g++ found in {env_bin}.  "
            f"Rerun with --use-existing after ensuring gcc=12/gxx=12 are installed.")

    info(f"env python            -> {env_python}")
    info(f"CUDA_HOME             -> {prefix}")
    info(f"TORCH_CUDA_ARCH_LIST  -> {compute_cap}")
    info(f"CC                    -> {cc}")
    info(f"CXX                   -> {cxx}")
    info("(bypassing `conda run` to avoid base-env activate.d overriding CC/CXX)")

    # Strip any /*miniconda*/bin entries from the inherited PATH so they
    # can't be discovered by shutil.which() fallbacks.
    parent_path = os.environ.get("PATH", "")
    cleaned_path = os.pathsep.join(
        p for p in parent_path.split(os.pathsep)
        if p and "miniconda" not in p.lower() and "anaconda" not in p.lower()
    )
    build_env = {
        # Carry over only the bits we need; do NOT splat os.environ -- that
        # would pull CC/CXX/CONDA_* that the base env set for its own gcc.
        "HOME":                 os.environ.get("HOME", ""),
        "USER":                 os.environ.get("USER", ""),
        "LANG":                 os.environ.get("LANG", "C.UTF-8"),
        "TERM":                 os.environ.get("TERM", "xterm"),
        "CUDA_HOME":            prefix,
        "TORCH_CUDA_ARCH_LIST": compute_cap,
        "CC":                   cc,
        "CXX":                  cxx,
        # Append our -ccbin so it wins over torch's (nvcc: last wins).
        "NVCC_APPEND_FLAGS":    f"-ccbin {cxx}",
        # Env bin first; base-conda bins removed.
        "PATH":                 env_bin + os.pathsep + cleaned_path,
    }

    # Run setup.py directly with the env's python -- no shell, no conda run.
    for subcmd in (["build_ext"], ["install"]):
        p = subprocess.run(
            [env_python, "setup.py", *subcmd],
            cwd=str(fused_dir),
            env=build_env,
        )
        if p.returncode != 0:
            die(f"fused_ops `setup.py {' '.join(subcmd)}` failed (exit {p.returncode}). "
                f"See output above.")
    ok("build + install complete")

def verify(env_name: str):
    step("Verifying fireants_fused_ops loads and runs")
    # Run python with the test script as a single argv element -- no shell
    # quoting involved, so it can't be silently broken.
    test = (
        "import torch\n"                       # must import torch FIRST so libc10 loads
        "import fireants_fused_ops as ffo\n"
        "from fireants.interpolator import fireants_interpolator as fi\n"
        "img = torch.randn(1,1,16,16,16).cuda()\n"
        "disp = torch.randn(1,16,16,16,3).cuda()*0.01\n"
        "out = fi(img, grid=disp)\n"
        "print('OK:', out.shape, ffo.__file__)\n"
    )
    p = subprocess.run(
        ["conda", "run", "-n", env_name, "--no-capture-output",
         "python", "-c", test],
        capture_output=True, text=True,
    )
    out = (p.stdout + p.stderr).strip()
    if p.returncode == 0 and "OK:" in out:
        ok([ln for ln in out.splitlines() if ln.startswith("OK:")][-1])
    else:
        die(f"verification failed (exit {p.returncode}):\n{out}")

# ---------- main ------------------------------------------------------------ #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env-name", default="fireants",
                    help="Name of the conda env to create/use. Default: fireants")
    ap.add_argument("--env-yml", default=None,
                    help="Optional: path to a `conda env export --from-history` yml "
                         "to bootstrap the env from. fused_ops still gets built locally.")
    ap.add_argument("--use-existing", action="store_true",
                    help="Don't error if --env-name already exists; install into it.")
    ap.add_argument("--work-dir", default=str(Path.home() / "fireants_build"),
                    help="Where to clone the FireANTs repo. Default: ~/fireants_build")
    args = ap.parse_args()

    print(f"\n{BOLD}FireANTs + fused_ops auto-installer{END}")
    print(f"  env-name:    {args.env_name}")
    print(f"  env-yml:     {args.env_yml or '(none -- creating fresh env)'}")
    print(f"  use-existing: {args.use_existing}")
    print(f"  work-dir:    {args.work_dir}")

    check_platform()
    cap = check_driver_and_gpu()
    create_or_update_env(args.env_name, args.env_yml, args.use_existing)
    install_pip_packages(args.env_name)
    install_compiler_stack(args.env_name)
    repo = clone_fireants(Path(args.work_dir))
    build_fused_ops(args.env_name, repo, cap)
    verify(args.env_name)

    print(f"\n{GREEN}{BOLD}Done.{END}  Activate with:  conda activate {args.env_name}\n")

if __name__ == "__main__":
    main()
