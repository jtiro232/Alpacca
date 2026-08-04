# Alpaccaroo - the only module that branches on the host OS.
# MIT License. See LICENSE.
"""Host-OS shims for the things POSIX hands us for free.

The engine asks for a capability; this module supplies the best available
implementation and returns a neutral answer when the host has none. Every
OS-specific line in Alpaccaroo lives here, so `gguf.py` and `kernels.py` stay
platform-free and read the same everywhere.

In both functions the POSIX path comes first and returns immediately, so
Linux behaviour is exactly what it was before this module existed: the
fallbacks below it are only ever reached on hosts that lack the syscall.
"""

from __future__ import annotations

import ctypes
import mmap
import os

_CHUNK = 1 << 24  # 16 MiB: large enough that read() syscall overhead vanishes


def prefetch(mm: "mmap.mmap", fh) -> None:
    """Advise that the whole mapping is about to be read sequentially.

    Linux/BSD: madvise, which faults the mapping in asynchronously and
    costs nothing. Windows has no madvise (`mmap.mmap` has no such method),
    so we stream the file once instead: the loader's per-tensor touches
    then hit the page cache rather than taking a hard fault per 4 KiB page.
    Measured on a 4.58 GiB Q4_K_M model, cold: a 0.96 s read cuts
    Model.load from 14.8 s to 11.4 s.
    """
    try:
        mm.madvise(mmap.MADV_SEQUENTIAL)
        mm.madvise(mmap.MADV_WILLNEED)
        return
    except (AttributeError, ValueError, OSError):
        pass  # no madvise here; fall through to the read-based warm-up
    try:
        pos = fh.tell()
        fh.seek(0)
        while fh.read(_CHUNK):
            pass
        fh.seek(pos)
    except (OSError, ValueError):
        pass  # purely advisory: a failed warm-up only costs speed


def physical_cores() -> int:
    """Physical core count, or 0 when it cannot be determined.

    Decode is memory-bandwidth-bound and SMT siblings contend for the same
    load ports: measured on a 6C/12T Ryzen, 6 threads decode 9-16% faster
    than 12. Linux exposes the topology in sysfs; Windows answers the same
    question through GetLogicalProcessorInformation. Anywhere else we
    return 0 and leave the caller's default (all logical CPUs) alone
    rather than guess.
    """
    try:
        cores = set()
        base = "/sys/devices/system/cpu"
        for name in os.listdir(base):
            if not name.startswith("cpu") or not name[3:].isdigit():
                continue
            try:
                with open(f"{base}/{name}/topology/core_id") as f:
                    core = f.read().strip()
                with open(f"{base}/{name}/topology/physical_package_id") as f:
                    pkg = f.read().strip()
            except OSError:
                continue
            cores.add((pkg, core))
        if cores:
            return len(cores)
    except Exception:
        pass
    return _physical_cores_windows()


def available_cpus() -> set[int]:
    """The logical CPUs this process may actually be scheduled on.

    A pinned process still reads the whole machine out of sysfs, so a pool
    sized from `physical_cores()` oversubscribes its own partition - four
    cores' worth of CPU running twelve threads. `sched_getaffinity` is the
    only thing that knows. Hosts without it get every CPU the OS reports,
    which is exactly the answer callers had before this existed.
    """
    try:
        return set(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return set(range(os.cpu_count() or 1))


def _linux_core_tiers() -> list[tuple[int, int]] | None:
    """[(max_kHz, physical core count)] descending, for schedulable CPUs.

    Returns None - never a guess - when sysfs cannot answer, so the caller
    keeps its previous behaviour on hosts with no cpufreq.
    """
    base = "/sys/devices/system/cpu"
    allowed = available_cpus()
    speeds: dict[tuple[str, str], int] = {}
    try:
        names = os.listdir(base)
    except OSError:
        return None
    for name in names:
        if not name.startswith("cpu") or not name[3:].isdigit():
            continue
        if int(name[3:]) not in allowed:
            continue
        try:
            with open(f"{base}/{name}/topology/core_id") as f:
                core = f.read().strip()
            with open(f"{base}/{name}/topology/physical_package_id") as f:
                pkg = f.read().strip()
        except OSError:
            continue
        try:
            with open(f"{base}/{name}/cpufreq/cpuinfo_max_freq") as f:
                khz = int(f.read().strip())
        except (OSError, ValueError):
            return None  # no per-core speed here; do not invent one
        key = (pkg, core)
        speeds[key] = max(speeds.get(key, 0), khz)
    if not speeds:
        return None
    tiers: dict[int, int] = {}
    for khz in speeds.values():
        tiers[khz] = tiers.get(khz, 0) + 1
    return sorted(tiers.items(), reverse=True)


def worker_cores() -> int:
    """Physical cores worth putting in the bandwidth-bound decode pool.

    `physical_cores()` answers what the machine has. This answers what is
    worth using, and on a hybrid CPU those differ. `prange` splits a row
    loop into equal static chunks, so a matvec finishes when its SLOWEST
    thread does: one core at half the clock of the rest does not add half a
    core of throughput, it stalls every other thread waiting for its chunk.

    Tiers are admitted fastest-first, and a tier of `k` cores at relative
    speed `f` is kept only while it beats the `n` cores already admitted:

        work/(n+k) / f  <  work/n     <=>     f > n/(n+k)

    A homogeneous machine has one tier at f = 1 and is unchanged, which is
    every non-hybrid host. Max frequency is a coarse stand-in for per-core
    throughput - it ignores the IPC and memory-latency gaps between core
    types - but it is the only per-core signal sysfs offers, and it points
    the right way.

    Measured, qwen2.5-3B Q4_K_M, Meteor Lake 2P+8E+2LP-E (4.3/3.6/2.1 GHz),
    warm decode tok/s, forward and reverse sweeps:

        threads     4      6      8     10     12(all)   14(logical)
        fwd       9.54  12.05  14.76  15.07     14.14      12.45
        rev       9.09  12.22  14.39  15.23     13.58      12.54

    This rule returns 10 there - the two 2.1 GHz LP-E cores excluded - for
    +9.3% over the 12 that `physical_cores()` reports.
    """
    tiers = _linux_core_tiers()
    if not tiers:
        return physical_cores()
    return cores_from_tiers(tiers)


def core_tiers() -> list[tuple[int, int]]:
    """The machine's speed tiers as [(max_kHz, cores)] descending, or [].

    Exists so callers can explain a core count without reaching into this
    module's OS-specific internals - every such line belongs in here.
    """
    return _linux_core_tiers() or []


def cores_from_tiers(tiers: list[tuple[int, int]]) -> int:
    """The admission rule of `worker_cores`, as a pure function.

    Split out so it can be tested against recorded machines instead of
    requiring one. `tiers` is [(speed, core count)] descending.
    """
    if not tiers:
        return 0
    top = tiers[0][0]
    kept = 0
    for khz, count in tiers:
        # keep while f > n/(n+k), i.e. khz*(n+k) > top*n; the first tier
        # is the baseline and is always admitted
        if kept and khz * (kept + count) <= top * kept:
            break
        kept += count
    return kept


class _CacheDescriptor(ctypes.Structure):
    _fields_ = [("Level", ctypes.c_ubyte), ("Associativity", ctypes.c_ubyte),
                ("LineSize", ctypes.c_ushort), ("Size", ctypes.c_ulong),
                ("Type", ctypes.c_ulong)]


class _ProcessorInfoUnion(ctypes.Union):
    _fields_ = [("ProcessorCore", ctypes.c_ubyte), ("NumaNode", ctypes.c_ulong),
                ("Cache", _CacheDescriptor), ("Reserved", ctypes.c_ulonglong * 2)]


class _ProcessorInfo(ctypes.Structure):
    _fields_ = [("ProcessorMask", ctypes.c_size_t),
                ("Relationship", ctypes.c_ulong),
                ("u", _ProcessorInfoUnion)]


_RELATION_PROCESSOR_CORE = 0
_ERROR_INSUFFICIENT_BUFFER = 122


def _physical_cores_windows() -> int:
    """One entry per physical core is tagged RelationProcessorCore.

    The non-Ex call tops out at one 64-CPU processor group, which is why a
    zero result falls back to the caller's default rather than to a wrong
    number: on a >64-thread host we would rather not tune than mistune.
    """
    if not hasattr(ctypes, "WinDLL"):
        return 0
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        size = ctypes.c_ulong(0)
        kernel32.GetLogicalProcessorInformation(None, ctypes.byref(size))
        if ctypes.get_last_error() != _ERROR_INSUFFICIENT_BUFFER:
            return 0
        count = size.value // ctypes.sizeof(_ProcessorInfo)
        buf = (_ProcessorInfo * count)()
        if not kernel32.GetLogicalProcessorInformation(buf, ctypes.byref(size)):
            return 0
        return sum(1 for e in buf if e.Relationship == _RELATION_PROCESSOR_CORE)
    except Exception:
        return 0
