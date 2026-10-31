"""
Thread count from THREADS or the cgroup CPU quota, set in the environment
before numpy is imported (OMP / OpenBLAS / MKL / numba read these at import).
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")
FALLBACK = 4   # no quota (a dev machine): never the host's core count


def cgroup_cpus() -> int | None:
    """CPUs from the cgroup v2 cpu.max (or v1 cfs quota), rounded up; None without a quota."""
    try:
        text = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if text and text[0] != "max":
            return max(1, math.ceil(int(text[0]) / int(text[1])))
        return None
    except (OSError, ValueError, IndexError):
        pass
    try:
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if quota > 0 and period > 0:
            return max(1, math.ceil(quota / period))
    except (OSError, ValueError):
        pass
    return None


def configure(threads: int) -> tuple[int, str]:
    """Set the thread environment variables; returns (threads, where the number came from)."""
    if threads > 0:
        n, source = threads, "THREADS"
    else:
        quota = cgroup_cpus()
        if quota is not None:
            n, source = quota, "cgroup cpu quota"
        else:
            try:
                affinity = len(os.sched_getaffinity(0))
            except (AttributeError, OSError):
                affinity = os.cpu_count() or 1
            n, source = min(FALLBACK, affinity), f"no cpu quota, min({FALLBACK}, {affinity} cpus)"
    for var in THREAD_VARS:
        # numba cannot change its thread count once loaded (a test harness that imported scanpy
        # before main() runs); in a container run nothing imports it before this point.
        if var == "NUMBA_NUM_THREADS" and "numba" in sys.modules:
            continue
        os.environ[var] = str(n)
    return n, source
