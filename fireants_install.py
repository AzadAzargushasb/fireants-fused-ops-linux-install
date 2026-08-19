#!/usr/bin/env python3
"""
fireants_install.py  --  AUTO-INSTALLER  (will install / build things!)

Sets up a working FireANTs + fireants_fused_ops environment from scratch.

  Usage:
    # (recommended) minimal env built from the bundled environment.yml
    python fireants_install.py --env-name fireants

    # ...plus the neuroimaging analysis stack (pingouin/nipype/nilearn/...)
    python fireants_install.py --env-name fireants --with-analysis-stack

    # update an env that already exists
    python fireants_install.py --env-name pre_env --use-existing

    # bootstrap from some other yml (e.g. reproduce another machine exactly)
    python fireants_install.py --env-yml pre_env_from_history.yml

  What it does (in order):
    1. Verifies you have conda + an NVIDIA GPU + a driver new enough.
    2. Detects this machine's GPU compute capability and picks a CUDA lane
       (which CUDA toolkit + which PyTorch build actually support that GPU).
    3. Creates / updates a conda env (minimal environment.yml by default).
    4. Inside that env, installs the lane's PyTorch, fireants, the matching
       CUDA toolkit (nvcc) and gcc 12.
    5. Clones https://github.com/rohitrango/FireANTs and builds fused_ops
       for that compute capability (the .so is GPU-arch-specific).
    6. Registers a Jupyter kernel so JupyterLab can see the env.
    7. Verifies `import fireants_fused_ops` succeeds and runs.

------------------------------------------------------------------------------
WHY THIS IS NOT A ONE-LINER
------------------------------------------------------------------------------
fused_ops is a CUDA C++ extension.  Building it requires four things that
must all match each other:

  * GPU compute    -- Each NVIDIA GPU has a "compute capability" (6.1 for a
    capability       GTX 1080, 8.6 for an RTX 3090, 12.0 for an RTX 5060).
                     This decides EVERYTHING else, so we detect it first.

  * nvcc           -- NVIDIA's CUDA compiler, from the "CUDA toolkit".  Only
                      CUDA >= 12.8 can emit code for Blackwell (sm_120).

  * PyTorch build  -- Must be built against the same CUDA major/minor, AND
                      must actually contain kernels for your GPU's arch.
                      cu121 wheels stop at sm_90; Blackwell needs cu128.

  * gcc / g++      -- nvcc invokes it for the CPU-side parts.  Modern distros
                      ship gcc 14/15 which produces obscure pybind11 errors --
                      the #1 cause of "fused_ops won't build".  We install
                      gcc 12, which both CUDA 12.1 and 12.8 accept.

The driver version (the "CUDA Version: 13.0" you see in `nvidia-smi`) is
just the *ceiling* the driver can run -- it does NOT mean CUDA 13 toolkit
is installed.  This script ignores it except to verify it's new enough.
"""

from __future__ import annotations
import argparse
import glob
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# ---------- pretty output --------------------------------------------------- #

GREEN = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"; BOLD = "\033[1m"; END = "\033[0m"

def step(msg):  print(f"\n{BOLD}==>{END} {msg}")
def info(msg):  print(f"    {msg}")
def ok(msg):    print(f"    {GREEN}[OK]{END}  {msg}")
def warn(msg):  print(f"    {YELLOW}[WARN]{END} {msg}")
def die(msg, code=1):
    print(f"\n{RED}[FATAL]{END} {msg}\n"); sys.exit(code)

# ---------- GPU / CUDA lane matrix ------------------------------------------ #
#
# There is no single CUDA+PyTorch stack that covers every NVIDIA GPU:
#
#   * The cu121 wheel index ENDS at torch 2.5.1, and those builds contain
#     kernels only for sm_50..sm_90.  Nothing newer will ever be published
#     for cu121.
#   * Blackwell (RTX 50-series, sm_120) needs CUDA >= 12.8 to compile at all,
#     and PyTorch only gained Blackwell kernels in 2.7.0 (cu128 builds).
#   * The cu128 builds in turn dropped Maxwell/Pascal, so a GTX 1080 and an
#     RTX 5060 genuinely cannot share one stack.
#
# So: pick the lane from the detected compute capability.  Adding a lane
# later (e.g. cu130 for a future arch) is a one-line edit here.
#
# NOTE: fireants_check.py carries a copy of this table so that each script
# stays standalone/copy-pasteable.  Keep the two in sync.

@dataclass
class Lane:
    cap_lo: float               # inclusive
    cap_hi: float               # exclusive
    cuda_label: str             # conda channel label providing cuda-toolkit
    cuda_version: str           # "12.1" -- what nvcc --version must report
    torch_versions: list        # newest first; later entries are build fallbacks
    torch_index: str            # pip --index-url
    blurb: str

CUDA_LANES = [
    Lane(5.0, 10.0, "nvidia/label/cuda-12.1.1", "12.1",
         ["2.5.1"], "https://download.pytorch.org/whl/cu121",
         "Maxwell..Hopper (sm_50..sm_90): GTX 10xx, RTX 20/30/40xx, A100, H100"),
    Lane(10.0, 99.0, "nvidia/label/cuda-12.8.2", "12.8",
         ["2.11.0", "2.7.0"], "https://download.pytorch.org/whl/cu128",
         "Blackwell (sm_100/sm_120): RTX 50-series, B100/B200"),
]

# Minimum NVIDIA driver per lane (the CUDA minor-version compatibility floor).
MIN_DRIVER = {"12.1": 525, "12.8": 525}

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

def select_lane(cap: str) -> Lane:
    c = float(cap)
    for lane in CUDA_LANES:
        if lane.cap_lo <= c < lane.cap_hi:
            return lane
    die(f"GPU compute capability {cap} ({cap_to_sm(cap)}) is not supported.\n"
        f"        FireANTs needs a PyTorch build containing kernels for your GPU;\n"
        f"        the oldest supported architecture is sm_50 (Maxwell, 2014).")

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

# ---------- conda robustness ------------------------------------------------ #
#
# conda 26.3.x ships conda-libmamba-solver 26.4.0, the release that turned
# SHARDED REPODATA on by default.  That code path caches repodata shards in
# a SQLite DB under <pkgs_dir>/cache/repodata_shards.db, and in 26.4.0 it
# opens that DB with neither WAL mode nor a busy_timeout.  Two threads in the
# same conda process (a cache reader and a network writer) then race on it,
# and past sqlite's default 5s timeout the reader dies with
#
#     sqlite3.OperationalError: database is locked
#
# ...which aborts the whole `conda env create`.  Upstream fixed it in 26.4.1
# (WAL + longer busy timeout) and 26.4.2 (serialized access):
#     https://github.com/conda/conda-libmamba-solver/issues/925
#
# On shared machines base conda is usually read-only, so users can't just
# upgrade.  We detect the affected version and set the documented opt-out
# instead, then fall back further if conda still fails.

BROKEN_SOLVER_LO = (26, 4, 0)
BROKEN_SOLVER_HI = (26, 4, 1)      # exclusive: 26.4.1 contains the fix

SQLITE_SIGNATURES = (
    "database is locked",
    "sqlite3.operationalerror",
    "repodata_shards",
    "file is not a database",
    "database disk image is malformed",
)
NETWORK_SIGNATURES = (
    "condahttperror",
    "connection broken",
    "connectionerror",
    "read timed out",
    "connection reset",
    "temporary failure in name resolution",
)
NET_BACKOFF = (2, 8, 30)

_conda_info_cache: dict = {}

def conda_info() -> dict:
    """Cached `conda info --json`."""
    global _conda_info_cache
    if not _conda_info_cache:
        p = subprocess.run(["conda", "info", "--json"],
                           capture_output=True, text=True)
        try:
            _conda_info_cache = json.loads(p.stdout)
        except (ValueError, TypeError):
            _conda_info_cache = {"_unavailable": True}
    return _conda_info_cache

def solver_version() -> tuple | None:
    """(major, minor, patch) of conda-libmamba-solver, or None if unknown."""
    ua = (conda_info().get("solver") or {}).get("user_agent", "")
    m = re.search(r"conda-libmamba-solver/(\d+)\.(\d+)\.(\d+)", ua)
    return tuple(int(g) for g in m.groups()) if m else None

_solver_warned = False

def solver_workaround_env() -> dict:
    """Env overrides needed to dodge known conda bugs on THIS machine."""
    global _solver_warned
    ver = solver_version()
    if ver is None or not (BROKEN_SOLVER_LO <= ver < BROKEN_SOLVER_HI):
        return {}
    if not _solver_warned:                       # say it once, not per command
        _solver_warned = True
        vs = ".".join(str(x) for x in ver)
        warn(f"conda-libmamba-solver {vs} has a sharded-repodata SQLite race "
             f"(upstream #924/#925)")
        if conda_info().get("root_writable"):
            info("real fix:  conda update -n base -c conda-forge conda-libmamba-solver")
        else:
            info("base conda is read-only here, so applying the documented opt-out instead")
        info("using CONDA_PLUGINS_USE_SHARDED_REPODATA=false")
    # Unknown CONDA_PLUGINS_* vars are silently ignored by older conda, so
    # this is safe even if the setting doesn't exist.
    return {"CONDA_PLUGINS_USE_SHARDED_REPODATA": "false"}

def purge_shards_cache():
    """Delete the sharded-repodata SQLite caches. They are pure cache: conda
    rebuilds them on demand. Also clears the corrupt-DB case (upstream #823)."""
    removed = 0
    for d in conda_info().get("pkgs_dirs", []):
        for path in glob.glob(os.path.join(d, "cache", "repodata_shards*.db*")):
            try:
                os.unlink(path)
                removed += 1
            except OSError:
                pass          # read-only shared pkgs dir -- nothing we can do
    if removed:
        info(f"removed {removed} stale repodata_shards cache file(s)")

def _run_streaming(cmd: str, extra_env: dict) -> tuple[int, str]:
    """Run cmd, echoing output live AND capturing it for error matching."""
    env = dict(os.environ)
    env.update(extra_env)
    p = subprocess.Popen(cmd, shell=True, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    chunks = []
    for line in p.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        chunks.append(line)
    p.wait()
    return p.returncode, "".join(chunks)

def run_conda(cmd: str) -> str:
    """Run a conda command, surviving the failures conda can't survive itself.

    Escalation ladder:
      0. as-is (plus any version-specific workaround)
      1. + sharded repodata disabled     (on SQLite errors)
      2. + classic solver                (still failing -- no SQLite path at all)
      transient network errors retry with backoff at whatever rung we're on.
    """
    overrides = solver_workaround_env()
    rung, net_tries, last_out = 0, 0, ""
    while True:
        env = dict(overrides)
        if rung >= 1:
            env["CONDA_PLUGINS_USE_SHARDED_REPODATA"] = "false"
        if rung >= 2:
            env["CONDA_SOLVER"] = "classic"
        shown = " ".join(f"{k}={v}" for k, v in sorted(env.items()))
        info(f"$ {shown + ' ' if shown else ''}{cmd}")
        rc, out = _run_streaming(cmd, env)
        if rc == 0:
            return out
        last_out, low = out, out.lower()

        if any(s in low for s in SQLITE_SIGNATURES) and rung < 2:
            rung += 1
            warn("conda's sharded-repodata SQLite cache failed "
                 "(known conda-libmamba-solver bug -- see README)")
            purge_shards_cache()
            info("retrying with " + ("sharded repodata disabled" if rung == 1
                                     else "the classic solver"))
            continue

        if any(s in low for s in NETWORK_SIGNATURES) and net_tries < len(NET_BACKOFF):
            delay = NET_BACKOFF[net_tries]
            net_tries += 1
            warn(f"transient network error; retry {net_tries}/{len(NET_BACKOFF)} "
                 f"in {delay}s")
            time.sleep(delay)
            continue
        break

    die(f"conda command failed after all fallbacks:\n  {cmd}\n\n"
        f"Last 20 lines:\n"
        + "\n".join(last_out.splitlines()[-20:])
        + "\n\n        Try manually:\n"
          "          rm -f ~/.conda/pkgs/cache/repodata_shards*.db*\n"
          "          CONDA_PLUGINS_USE_SHARDED_REPODATA=false CONDA_SOLVER=classic \\\n"
          f"            {cmd}")

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
    ver = solver_version()
    if ver:
        info(f"solver: conda-libmamba-solver {'.'.join(str(x) for x in ver)}")
    if shutil.which("nvidia-smi") is None:
        die("nvidia-smi not found. Install the NVIDIA proprietary driver.")

def gpu_compute_cap() -> str:
    """Compute capability of GPU 0, e.g. '6.1' / '8.6' / '12.0'.

    `--query-gpu=compute_cap` needs a reasonably recent nvidia-smi; fall back
    to the GPU-name lookup only if it's unavailable."""
    rc = subprocess.run(
        "nvidia-smi --query-gpu=compute_cap --format=csv,noheader",
        shell=True, capture_output=True, text=True)
    cap = rc.stdout.strip().splitlines()[0].strip() if rc.stdout.strip() else ""
    if rc.returncode == 0 and re.fullmatch(r"\d+\.\d+", cap):
        return cap
    die("could not read GPU compute capability from nvidia-smi.\n"
        "        Your nvidia-smi may be too old (needs --query-gpu=compute_cap).\n"
        "        Look your GPU up at https://developer.nvidia.com/cuda-gpus and\n"
        "        re-run with the value forced, e.g.:  --compute-cap 8.6")

def check_driver_and_gpu(forced_cap: str | None) -> tuple[str, Lane]:
    """Returns (compute capability, lane)."""
    step("Checking NVIDIA driver and GPU")
    drv = sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader",
             capture=True).splitlines()[0].strip()
    drv_major = int(drv.split(".")[0])

    cap = forced_cap or gpu_compute_cap()
    name = sh("nvidia-smi --query-gpu=name --format=csv,noheader",
              capture=True).splitlines()[0].strip()
    ok(f"GPU: {name}, compute capability {cap} ({cap_to_sm(cap)})")

    lane = select_lane(cap)
    need_drv = MIN_DRIVER.get(lane.cuda_version, 525)
    if drv_major < need_drv:
        die(f"Driver {drv} is too old for CUDA {lane.cuda_version}. "
            f"Need >={need_drv}. Update the NVIDIA driver first.")
    ok(f"driver {drv}  (>={need_drv}, supports CUDA {lane.cuda_version})")

    step("Selected CUDA lane")
    info(f"target GPUs   -> {lane.blurb}")
    ok(f"CUDA toolkit  -> {lane.cuda_version}  (conda: {lane.cuda_label})")
    ok(f"PyTorch       -> {lane.torch_versions[0]} from {lane.torch_index}")
    if len(lane.torch_versions) > 1:
        info(f"build fallbacks: torch {', '.join(lane.torch_versions[1:])}")
    return cap, lane

# ---------- env setup ------------------------------------------------------- #

def env_exists(name: str) -> bool:
    out = sh("conda env list", capture=True)
    return any(line.split() and line.split()[0] == name for line in out.splitlines())

def create_or_update_env(env_name: str, env_yml: str | None, use_existing: bool):
    step(f"Setting up conda env: {env_name}")
    exists = env_exists(env_name)

    # Default to the bundled minimal environment.yml.  The old default was a
    # 350-package `conda env export` dump, which took minutes to solve, pinned
    # every package to one machine's versions, and dragged in the Anaconda
    # `defaults` channel (Terms-of-Service gate).  None of that is needed.
    if env_yml is None:
        default_yml = SCRIPT_DIR / "environment.yml"
        if default_yml.is_file():
            env_yml = str(default_yml)
            info(f"using bundled minimal env spec: {default_yml.name}")

    if env_yml:
        if not Path(env_yml).is_file():
            die(f"--env-yml file not found: {env_yml}")
        if exists and use_existing:
            ok(f"using existing env '{env_name}' (skipping yml import)")
        elif exists:
            warn(f"env '{env_name}' already exists; updating from yml")
            run_conda(f"conda env update -n {env_name} -f {shlex.quote(env_yml)}")
        else:
            run_conda(f"conda env create -n {env_name} -f {shlex.quote(env_yml)}")
        ok("env created/updated from yml")
    else:
        if exists:
            if not use_existing:
                die(f"env '{env_name}' already exists. "
                    f"Pass --use-existing to add packages to it, or pick a different --env-name.")
            ok(f"using existing env '{env_name}'")
        else:
            run_conda(f"conda create -n {env_name} python=3.12 -y")
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

# ---------- pip packages ---------------------------------------------------- #

def constraints_arg() -> str:
    """`-c constraints.txt` if the file is next to this script, else ''.

    One file governs numpy (and anything else we need to bound) across every
    pip call, instead of repeating pins in a dozen places."""
    c = SCRIPT_DIR / "constraints.txt"
    return f"-c {shlex.quote(str(c))} " if c.is_file() else ""

def torch_status(env_name: str) -> dict | None:
    """What torch is installed in the env, and which GPU arches it supports?

    Returns None if torch isn't importable. This is deliberately NOT a
    presence check: an env can have a perfectly healthy torch that simply has
    no kernels for this machine's GPU (exactly what happens when a cu121 build
    meets an RTX 50-series card), and 'already installed, skipping' would then
    leave the env permanently broken."""
    probe = ("import json,torch;"
             "print(json.dumps({'version':torch.__version__,"
             "'cuda':torch.version.cuda,'arches':torch.cuda.get_arch_list()}))")
    p = subprocess.run(["conda", "run", "-n", env_name, "python", "-c", probe],
                       capture_output=True, text=True)
    if p.returncode != 0:
        return None
    for line in p.stdout.splitlines():          # skip any `conda run` chatter
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                pass
    return None

def install_torch(env_name: str, lane: Lane, version: str):
    in_env(env_name,
           f"pip install {constraints_arg()}torch=={version} "
           f"--index-url {lane.torch_index}")

def install_pip_packages(env_name: str, lane: Lane, cap: str,
                         fireants_version: str, with_analysis: bool):
    """PyTorch + fireants come from pip, not conda, because we need a specific
    CUDA-flavoured wheel and a specific fireants version."""
    step(f"Installing PyTorch {lane.torch_versions[0]} ({lane.torch_index.rsplit('/', 1)[-1]})")
    want_sm = cap_to_sm(cap)
    st = torch_status(env_name)
    if st is None:
        install_torch(env_name, lane, lane.torch_versions[0])
        ok("torch installed")
    elif not arch_supported(cap, st["arches"]):
        warn(f"installed torch {st['version']} (CUDA {st['cuda']}) has no {want_sm} "
             f"kernels -- it supports: {' '.join(st['arches'])}")
        info(f"replacing it with torch {lane.torch_versions[0]} for this GPU")
        install_torch(env_name, lane, lane.torch_versions[0])
        ok("torch replaced")
    else:
        ok(f"torch {st['version']} already covers {want_sm} (skipping)")

    step(f"Installing fireants {fireants_version}")
    if env_has_module(env_name, "fireants"):
        ok("fireants already installed (skipping)")
    else:
        # fireants hard-pins `simpleitk==2.2.1`, which has been un-published
        # from PyPI for python >=3.12 (only 2.1.0 and 2.3.0+ have wheels now).
        # Install a compatible SimpleITK + the other runtime deps first, then
        # install fireants with --no-deps so pip doesn't insist on 2.2.1.
        # Any SimpleITK 2.x works at runtime -- verified with SimpleITK 2.5.2.
        info("installing fireants runtime deps (works around the simpleitk==2.2.1 pin)")
        reqs = SCRIPT_DIR / "requirements-fireants.txt"
        if reqs.is_file():
            in_env(env_name, f"pip install {constraints_arg()}-r {shlex.quote(str(reqs))}")
        else:
            in_env(env_name,
                   f"pip install {constraints_arg()}'simpleitk>=2.3,<3' hydra-core "
                   "matplotlib nibabel 'numpy<2' pandas pytest scikit-image scipy tqdm")
        in_env(env_name,
               f"pip install {constraints_arg()}--no-deps fireants=={fireants_version}")
        ok(f"fireants {fireants_version} installed (SimpleITK >=2.3 instead of pinned 2.2.1)")

    if with_analysis:
        step("Installing the neuroimaging analysis stack")
        reqs = SCRIPT_DIR / "requirements-analysis.txt"
        if not reqs.is_file():
            die(f"--with-analysis-stack needs {reqs}, which is missing")
        in_env(env_name, f"pip install {constraints_arg()}-r {shlex.quote(str(reqs))}")
        ok("analysis stack installed")

def install_compiler_stack(env_name: str, lane: Lane):
    """cuda-toolkit -> nvcc.  gcc 12 -> CUDA-compatible host compiler."""
    step(f"Installing CUDA toolkit {lane.cuda_version} (nvcc) into the env")
    have = nvcc_version(env_name)
    if have and have.startswith(lane.cuda_version):
        ok(f"nvcc {have} already in env (skipping cuda-toolkit install)")
    else:
        if have:
            # Same trap as torch: an env can have a working nvcc that simply
            # can't emit code for this GPU (CUDA 12.1 knows nothing of sm_120).
            warn(f"env has nvcc {have}, but this GPU needs CUDA {lane.cuda_version}")
        run_conda(f"conda install -n {env_name} -c {shlex.quote(lane.cuda_label)} "
                  f"cuda-toolkit -y")
        ok(f"cuda-toolkit {lane.cuda_version} installed")

    step("Installing gcc/gxx 12 into the env (valid host compiler for CUDA 12.x)")
    if env_has_executable(env_name, "gcc"):
        ok("env-local gcc already installed (skipping)")
    else:
        run_conda(f"conda install -n {env_name} -c conda-forge gcc=12 gxx=12 -y")
        ok("gcc/gxx 12 installed")

def nvcc_version(env_name: str) -> str | None:
    """'12.1' / '12.8' for the env's nvcc, or None if there isn't one."""
    nvcc = os.path.join(env_prefix(env_name), "bin", "nvcc")
    if not os.access(nvcc, os.X_OK):
        return None
    p = subprocess.run([nvcc, "--version"], capture_output=True, text=True)
    m = re.search(r"release (\d+\.\d+)", p.stdout)
    return m.group(1) if m else None

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

def clean_build_dir(fused_dir: Path):
    """Stale objects compiled against a different torch will poison a rebuild."""
    for p in [fused_dir / "build", *fused_dir.glob("*.egg-info")]:
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
            info(f"removed {p}")

def build_env_for(env_name: str, compute_cap: str) -> dict:
    """The minimal, sanitised environment a torch CUDA extension build needs.

    Two separate hazards when building torch CUDA extensions inside conda:

      1. `conda run -n <env>` doesn't isolate the env -- it inherits (and
         keeps) anything the parent's base-env activate.d scripts set.
         If base conda also has gcc installed, CC/CXX may already point at
         /opt/miniconda3/bin/x86_64-conda-linux-gnu-c++ (a gcc >=13), and
         the base activate scripts re-set them even if we pass CC/CXX in
         the subprocess env.  CUDA 12.x wants gcc <=13 -> build fails.

      2. torch's cpp_extension passes `-ccbin <CXX>` to nvcc inline.  If
         we use NVCC_PREPEND_FLAGS to inject our own -ccbin, it lands
         BEFORE torch's and loses (nvcc's "last -ccbin wins").  We have
         to APPEND instead.

    Workaround: bypass `conda run` entirely, invoke the env's python by
    absolute path, and pass an explicit, minimal env= with the env's
    compilers and PATH.  Use NVCC_APPEND_FLAGS so our -ccbin overrides
    whatever torch picked.
    """
    prefix = env_prefix(env_name)
    env_bin = os.path.join(prefix, "bin")

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
    return {
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

def _try_build(env_name: str, fused_dir: Path, compute_cap: str) -> int:
    env_python = os.path.join(env_prefix(env_name), "bin", "python")
    if not os.access(env_python, os.X_OK):
        die(f"env python not found at {env_python}")
    build_env = build_env_for(env_name, compute_cap)
    info(f"env python            -> {env_python}")
    for subcmd in (["build_ext"], ["install"]):
        p = subprocess.run([env_python, "setup.py", *subcmd],
                           cwd=str(fused_dir), env=build_env)
        if p.returncode != 0:
            return p.returncode
    return 0

def build_fused_ops(env_name: str, repo: Path, compute_cap: str, lane: Lane):
    step(f"Building fireants_fused_ops for compute capability {compute_cap}")
    fused_dir = repo / "fused_ops"
    if not fused_dir.is_dir():
        die(f"fused_ops dir missing: {fused_dir}")

    # FireANTs' fused_ops sources are known-good against torch 2.5.1.  On lanes
    # that need a newer torch (Blackwell), the C++ extension API may have moved
    # under them, so fall back through the lane's older candidates on failure.
    for i, tv in enumerate(lane.torch_versions):
        if i > 0:
            warn(f"fused_ops failed to build against torch {lane.torch_versions[i-1]}")
            info(f"falling back to torch {tv} and rebuilding")
            clean_build_dir(fused_dir)
            install_torch(env_name, lane, tv)
        rc = _try_build(env_name, fused_dir, compute_cap)
        if rc == 0:
            ok(f"build + install complete (torch {tv})")
            return
    die(f"fused_ops failed to build against every candidate torch "
        f"({', '.join(lane.torch_versions)}).  See the compiler output above.\n\n"
        f"        This does NOT mean FireANTs is unusable on this machine: the\n"
        f"        fused ops are an optional speedup.  Plain FireANTs works with\n"
        f"        the torch that is now installed -- verify with:\n"
        f"          conda run -n {env_name} python -c "
        f"\"import torch,fireants; print(torch.cuda.is_available())\"\n"
        f"        ...and please report the compiler error upstream at\n"
        f"        https://github.com/rohitrango/FireANTs/issues")

# ---------- jupyter --------------------------------------------------------- #

def register_kernel(env_name: str):
    """Make the env visible to JupyterLab.

    `--user` writes ~/.local/share/jupyter/kernels/<name>/, which is on the
    default JUPYTER_PATH -- so *whichever* jupyter you launch finds it, even
    one shipped by something else entirely (e.g. FSL's /opt/fsl/bin/jupyter-lab)."""
    step("Registering Jupyter kernel")
    if not env_has_module(env_name, "ipykernel"):
        in_env(env_name, f"pip install {constraints_arg()}ipykernel")
    in_env(env_name,
           f"python -m ipykernel install --user --name {shlex.quote(env_name)} "
           f"--display-name {shlex.quote(f'Python ({env_name})')}")
    ok(f"kernel 'Python ({env_name})' registered "
       f"(restart JupyterLab to see it)")

# ---------- verification ---------------------------------------------------- #

ANALYSIS_IMPORTS = """
import pingouin, pandas, nibabel, numpy, scipy.signal, tkinter, joblib
import statsmodels.stats.multitest, sklearn.linear_model, nilearn.maskers
import plotly, kaleido, networkx
import nipype.interfaces.fsl as fsl
fsl.FLIRT()
print('OK-ANALYSIS')
"""

def verify(env_name: str, with_analysis: bool):
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

    if with_analysis:
        step("Verifying the analysis stack imports")
        p = subprocess.run(
            ["conda", "run", "-n", env_name, "python", "-c", ANALYSIS_IMPORTS],
            capture_output=True, text=True,
        )
        out = (p.stdout + p.stderr).strip()
        if p.returncode == 0 and "OK-ANALYSIS" in out:
            ok("all analysis-stack imports succeeded")
        else:
            # nipype needs FSL on PATH; that's an environment issue, not an
            # install failure, so don't nuke an otherwise-good install over it.
            warn(f"analysis-stack import check failed:\n{out}")
            if "fsl" in out.lower():
                info("if this is FSL: set FSLDIR and put $FSLDIR/bin on PATH")

# ---------- main ------------------------------------------------------------ #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env-name", default="fireants",
                    help="Name of the conda env to create/use. Default: fireants")
    ap.add_argument("--env-yml", default=None,
                    help="Path to a conda env yml to bootstrap from. "
                         "Default: the bundled minimal environment.yml.")
    ap.add_argument("--use-existing", action="store_true",
                    help="Don't error if --env-name already exists; install into it.")
    ap.add_argument("--work-dir", default=str(Path.home() / "fireants_build"),
                    help="Where to clone the FireANTs repo. Default: ~/fireants_build")
    ap.add_argument("--fireants-version", default="1.4.0",
                    help="fireants release to install. Default: 1.4.0")
    ap.add_argument("--with-analysis-stack", action="store_true",
                    help="Also install the neuroimaging analysis stack "
                         "(pingouin, nipype, nilearn, statsmodels, plotly, jupyter...)")
    ap.add_argument("--no-kernel", action="store_true",
                    help="Skip registering the env as a Jupyter kernel.")
    ap.add_argument("--compute-cap", default=None,
                    help="Force the GPU compute capability (e.g. 8.6) instead of "
                         "asking nvidia-smi. For old drivers or cross-building.")
    args = ap.parse_args()

    print(f"\n{BOLD}FireANTs + fused_ops auto-installer{END}")
    print(f"  env-name:       {args.env_name}")
    print(f"  env-yml:        {args.env_yml or '(bundled environment.yml)'}")
    print(f"  use-existing:   {args.use_existing}")
    print(f"  work-dir:       {args.work_dir}")
    print(f"  fireants:       {args.fireants_version}")
    print(f"  analysis stack: {args.with_analysis_stack}")

    check_platform()
    cap, lane = check_driver_and_gpu(args.compute_cap)
    create_or_update_env(args.env_name, args.env_yml, args.use_existing)
    install_pip_packages(args.env_name, lane, cap,
                         args.fireants_version, args.with_analysis_stack)
    install_compiler_stack(args.env_name, lane)
    repo = clone_fireants(Path(args.work_dir))
    build_fused_ops(args.env_name, repo, cap, lane)
    if not args.no_kernel:
        register_kernel(args.env_name)
    verify(args.env_name, args.with_analysis_stack)

    print(f"\n{GREEN}{BOLD}Done.{END}  Activate with:  conda activate {args.env_name}\n")

if __name__ == "__main__":
    main()
