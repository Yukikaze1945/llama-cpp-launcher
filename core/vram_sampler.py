"""GPU memory sampling for the VRAM predictor — no new dependencies.

Two backends, tried in this order (plan layer 4):

* **ctypes → NVML**, calling `nvmlDeviceGetMemoryInfo_v2` when the installed
  library has it. v2 is preferred because under WDDM the driver carves a
  *reserved* block out of `total`, and only the v2 struct reports it; with the
  v1 call `total != free + used` and the difference looks like phantom usage.
* **nvidia-smi** as a fallback, parsed from a single CSV query. It is ~10x
  slower to start, so it is only for showing free VRAM and a coarse used
  baseline — the caller decides whether a run is still measurable.

Nothing here is Qt-aware and nothing raises: `read()` returns None and keeps
the reason in `last_error`, because a missing GPU driver must degrade the panel
to "unavailable", not break the launcher.

Windows note: per-process VRAM is not attributable here (nvidia-smi reports
N/A for processes on this driver), so the predictor uses *whole-card* `used`
deltas and the `noisy` check below guards against anything else allocating
during the run.
"""
from __future__ import annotations

import ctypes
import os
import shutil
import statistics
import subprocess
from dataclasses import dataclass

MIB = 1 << 20
GIB = 1 << 30

#: Target cadence for the UI sampler thread; outside this band the polling
#: either misses short peaks (>100 ms) or wastes the GPU driver's time (<50 ms).
SAMPLE_INTERVAL_MS = 75

#: A baseline that moved more than this between "before start" and "after stop"
#: means someone else touched the card while the server ran, so the measured
#: peak is not this run's and must not enter the learner (spec: max(256 MiB,
#: 2 % of GPU total)).
NOISY_FLOOR_BYTES = 256 * MIB
NOISY_RATIO = 0.02

_NVML_NAMES = ("nvml.dll", "libnvidia-ml.so.1", "libnvidia-ml.so")


@dataclass(frozen=True)
class GpuMemory:
    total_bytes: int = 0
    free_bytes: int = 0
    used_bytes: int = 0
    name: str = ""
    source: str = ""          # "nvml" | "nvidia-smi"


# --------------------------------------------------------------------------
# NVML via ctypes
# --------------------------------------------------------------------------

class _MemoryV1(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong),
                ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong)]


class _MemoryV2(ctypes.Structure):
    # The C header puts `version` first and the driver insists on the
    # struct-version word (sizeof | (ver << 24)), so a zero here is rejected
    # with NVML_ERROR_STRUCT_SIZE_INVALID rather than treated as "unknown".
    _fields_ = [("version", ctypes.c_uint),
                ("total", ctypes.c_ulonglong),
                ("reserved", ctypes.c_ulonglong),
                ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong)]


_NVML_OK = 0


def _find_nvml_library() -> str | None:
    """Path of an NVML library to load, or None.

    Windows ships it next to the driver (`C:\\Windows\\System32\\nvml.dll`), so
    the plain name resolves through the normal search path. On Linux the
    SONAME is tried directly, then the common library dirs, because a
    non-standard install may not be in ldconfig's cache.
    """
    if os.name == "nt":
        return "nvml.dll"
    for name in _NVML_NAMES[1:]:
        try:
            ctypes.CDLL(name)
            return name
        except OSError:
            continue
    for root in ("/usr/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/lib"):
        for name in _NVML_NAMES[1:]:
            cand = os.path.join(root, name)
            if os.path.exists(cand):
                return cand
    return None


def _load_nvml():
    """Load NVML, or None if there is no NVIDIA driver here."""
    lib = _find_nvml_library()
    if not lib:
        return None
    try:
        return ctypes.CDLL(lib)
    except OSError:
        return None


class _NvmlBackend:
    source = "nvml"

    def __init__(self, index: int):
        self._lib = _load_nvml()
        self._handle = None
        self._name = ""
        self._memory_v2 = False
        self._used_v2_word = 0
        self.error = ""
        if self._lib is None:
            self.error = "nvml library not found"
            return
        try:
            if self._lib.nvmlInit_v2() != _NVML_OK:
                self.error = "nvmlInit_v2 failed"
                self._lib = None
                return
        except AttributeError:
            self.error = "nvmlInit_v2 missing"
            self._lib = None
            return
        handle = ctypes.c_void_p()
        rc = -1
        for fn in ("nvmlDeviceGetHandleByIndex_v2", "nvmlDeviceGetHandleByIndex"):
            try:
                rc = getattr(self._lib, fn)(index, ctypes.byref(handle))
            except AttributeError:
                continue
            if rc == _NVML_OK:
                self._handle = handle
                break
        if self._handle is None:
            self.error = f"no NVML device at index {index} (rc={rc})"

        # Probe v2 once at open(): the version word is sizeof | (2 << 24), the
        # same encoding pynvml uses, and a driver without the v2 call simply
        # keeps answering through the v1 struct.
        probe = _MemoryV2()
        probe.version = ctypes.sizeof(_MemoryV2) | (2 << 24)
        try:
            if self._lib.nvmlDeviceGetMemoryInfo_v2(self._handle,
                                                    ctypes.byref(probe)) == _NVML_OK:
                self._memory_v2 = True
                self._used_v2_word = probe.version
        except AttributeError:
            pass
        buf = ctypes.create_string_buffer(80)
        try:
            if self._lib.nvmlDeviceGetName(self._handle, buf, 80) == _NVML_OK:
                self._name = buf.value.decode("utf-8", "replace")
        except AttributeError:
            pass

    def read(self) -> GpuMemory | None:
        if self._lib is None or self._handle is None:
            return None
        if self._memory_v2:
            m = _MemoryV2()
            m.version = self._used_v2_word
            try:
                if self._lib.nvmlDeviceGetMemoryInfo_v2(self._handle, ctypes.byref(m)) != _NVML_OK:
                    self._memory_v2 = False      # fall through to v1 below
                else:
                    return GpuMemory(m.total, m.free, m.used, self._name, self.source)
            except AttributeError:
                self._memory_v2 = False
        m1 = _MemoryV1()
        try:
            if self._lib.nvmlDeviceGetMemoryInfo(self._handle, ctypes.byref(m1)) != _NVML_OK:
                return None
        except (AttributeError, OSError):
            return None
        return GpuMemory(m1.total, m1.free, m1.used, self._name, self.source)

    def close(self):
        if self._lib is not None:
            try:
                self._lib.nvmlShutdown()
            except (AttributeError, OSError):
                pass
        self._lib = None
        self._handle = None


# --------------------------------------------------------------------------
# nvidia-smi fallback
# --------------------------------------------------------------------------

_SMI_ARGS = ("--query-gpu=index,name,memory.total,memory.free,memory.used",
             "--format=csv,noheader,nounits")


def _run_nvidia_smi(timeout: float) -> str | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        for cand in (r"C:\Windows\System32\nvidia-smi.exe", "/usr/bin/nvidia-smi"):
            if os.path.exists(cand):
                exe = cand
                break
    if not exe:
        return None
    kwargs = {}
    if os.name == "nt":
        # A GUI build has no console; without this flag every poll would flash
        # a window.
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    try:
        out = subprocess.run([exe, *_SMI_ARGS], capture_output=True, text=True,
                             timeout=timeout, encoding="utf-8", errors="replace",
                             **kwargs)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _parse_smi(stdout: str | None, index: int) -> GpuMemory | None:
    if not stdout:
        return None
    for line in stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            if int(parts[0]) != index:
                continue
            total, free, used = (int(float(parts[i])) for i in (2, 3, 4))
        except ValueError:
            continue
        return GpuMemory(total * MIB, free * MIB, used * MIB, parts[1], "nvidia-smi")
    return None


class _SmiBackend:
    source = "nvidia-smi"

    def __init__(self, index: int, timeout: float = 2.0):
        self._index = index
        self._timeout = timeout
        self.error = ""

    def read(self) -> GpuMemory | None:
        sample = _parse_smi(_run_nvidia_smi(self._timeout), self._index)
        if sample is None:
            self.error = "nvidia-smi gave no usable row"
        return sample

    def close(self):
        return None


# --------------------------------------------------------------------------
# the object the UI holds
# --------------------------------------------------------------------------

class VramSampler:
    """One open handle, polled for `used`/`free`.

    `open()` resolves the backend lazily so importing this module never touches
    the driver — tests construct it, and a machine with no NVIDIA GPU just ends
    up with `status() == "none"`.
    """

    def __init__(self, index: int = 0, allow_smi: bool = True):
        self._index = index
        self._allow_smi = allow_smi
        self._backend = None
        self.last_error = ""

    def open(self) -> bool:
        if self._backend is not None:
            return True
        backends = [_NvmlBackend(self._index)]
        if self._allow_smi:
            backends.append(_SmiBackend(self._index))
        for backend in backends:
            # One probe read decides it: a handle that exists but cannot report
            # memory is useless to the predictor.
            if backend.read() is not None:
                self._backend = backend
                self.last_error = ""
                return True
            backend.close()
            if backend.error:
                self.last_error = backend.error
        self.last_error = self.last_error or "no GPU memory source"
        return False

    def status(self) -> str:
        return self._backend.source if self._backend else "none"

    def read(self) -> GpuMemory | None:
        if self._backend is None and not self.open():
            return None
        sample = self._backend.read()
        if sample is None:
            self.last_error = self.last_error or "read failed"
        return sample

    def close(self):
        if self._backend is not None:
            self._backend.close()
            self._backend = None


def read_gpu_memory(index: int = 0) -> GpuMemory | None:
    """One-shot read for callers that are not polling (the panel's idle label)."""
    sampler = VramSampler(index)
    try:
        return sampler.read()
    finally:
        sampler.close()


# --------------------------------------------------------------------------
# baselines and the noisy rule
# --------------------------------------------------------------------------

def median_used(samples: list[int]) -> int:
    """Median of polled `used` values — a single spike must not set the baseline."""
    vals = [int(s) for s in samples if s is not None]
    if not vals:
        return 0
    return int(statistics.median(vals))


def noisy_threshold_bytes(gpu_total_bytes: int) -> int:
    total = int(gpu_total_bytes or 0)
    return max(NOISY_FLOOR_BYTES, int(NOISY_RATIO * total))


def is_noisy(baseline_before: int, baseline_after: int, gpu_total_bytes: int) -> bool:
    """True when the card moved under the run, so its peak is not ours.

    Compares the two baselines, not the peak: a legitimate run leaves the card
    roughly where it started, so any drift of the *floor* is somebody else's
    allocation.
    """
    drift = abs(int(baseline_after) - int(baseline_before))
    return drift > noisy_threshold_bytes(gpu_total_bytes)
