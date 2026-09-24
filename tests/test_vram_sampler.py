# -*- coding: utf-8 -*-
"""The GPU memory sampler: two backends, one whole-card `used` signal.

The predictor's only window onto reality is this module, so the tests are about
failure behaviour: what happens when NVML is absent, when the driver rejects the
v2 struct, when nvidia-smi prints a row for another GPU, and when nothing at all
is readable. A machine without an NVIDIA card must be able to run this file, so
every test drives a fake library or fake stdout instead of the local driver.
"""
import ctypes
import types

import pytest

from core import vram_sampler as vs

MIB = vs.MIB


class _Ctypes:
    """ctypes as core.vram_sampler sees it: `byref` becomes identity so a fake
    NVML can be handed the struct itself and fill it in, and `CDLL` can be
    swapped without patching the interpreter-wide attribute."""

    def __init__(self, cdll=None):
        self._real = ctypes
        self._cdll = cdll

    def __getattr__(self, name):
        return getattr(self._real, name)

    @staticmethod
    def byref(obj):
        return obj

    def CDLL(self, name):
        if self._cdll is None:
            return self._real.CDLL(name)
        return self._cdll(name)


class _FakeOs:
    """A stand-in for `os` inside this one module.

    Patching the real `os.name` selects PosixPath process-wide and pytest dies
    in its own teardown, so the sampler's `os` reference gets replaced instead.
    """

    def __init__(self, name, exists=None):
        self.name = name
        self.sep = "\\" if name == "nt" else "/"
        self.path = types.SimpleNamespace(
            exists=exists or (lambda p: False),
            join=lambda *parts: self.sep.join(str(p).rstrip(self.sep) for p in parts),
        )


class FakeNvml:
    """Enough of the NVML C API for _NvmlBackend, recording what it was asked.

    `missing` lists entry points this nvml.dll does not export: the sampler only
    ever sees a raised AttributeError there, which is what an old driver looks
    like from inside its try/except blocks.
    """

    def __init__(self, *, total=24 * 1024 * MIB, free=20 * 1024 * MIB,
                 used=4 * 1024 * MIB, v2=True, v2_rc=0, name=b"RTX 4090",
                 init_rc=0, handle_rc=0, missing=()):
        self.values = (total, free, used)
        self.v2 = v2
        self.v2_rc = v2_rc
        self.name = name
        self.init_rc = init_rc
        self.handle_rc = handle_rc
        self.missing = set(missing)
        self.version_words = []
        self.calls = []
        self.shutdowns = 0

    def _sym(self, name):
        if name in self.missing:
            raise AttributeError(name)

    def nvmlInit_v2(self):
        self._sym("nvmlInit_v2")
        self.calls.append("init")
        return self.init_rc

    def nvmlDeviceGetHandleByIndex_v2(self, index, out):
        self._sym("nvmlDeviceGetHandleByIndex_v2")
        self.calls.append(("handle", "v2", index))
        if self.handle_rc:
            return self.handle_rc
        out.value = 0x1234
        return 0

    def nvmlDeviceGetHandleByIndex(self, index, out):
        self._sym("nvmlDeviceGetHandleByIndex")
        self.calls.append(("handle", "v1", index))
        if self.handle_rc:
            return self.handle_rc
        out.value = 0x1234
        return 0

    def _fill(self, m):
        m.total, m.free, m.used = self.values

    def nvmlDeviceGetMemoryInfo_v2(self, handle, m):
        self._sym("nvmlDeviceGetMemoryInfo_v2")
        self.calls.append("v2")
        self.version_words.append(m.version)
        if not self.v2 or self.v2_rc:
            return self.v2_rc or 9          # NVML_ERROR_STRUCT_SIZE_INVALID etc.
        self._fill(m)
        return 0

    def nvmlDeviceGetMemoryInfo(self, handle, m):
        self._sym("nvmlDeviceGetMemoryInfo")
        self.calls.append("v1")
        self._fill(m)
        return 0

    def nvmlDeviceGetName(self, handle, buf, length):
        self._sym("nvmlDeviceGetName")
        buf.value = self.name
        return 0

    def nvmlShutdown(self):
        self.shutdowns += 1


@pytest.fixture
def nvml(monkeypatch):
    """Install a fake NVML (and a byref that a fake can fill). Returns a setter
    so a test can swap in a differently-behaving library."""
    holder = {}

    def install(**kwargs):
        lib = FakeNvml(**kwargs)
        holder["lib"] = lib
        monkeypatch.setattr(vs, "_load_nvml", lambda: lib)
        monkeypatch.setattr(vs, "ctypes", _Ctypes())
        return lib

    holder["install"] = install
    yield holder


# --------------------------------------------------------------------------
# the v2 struct and its version word
# --------------------------------------------------------------------------

def test_v2_layout_matches_nvml_memory_v2_and_carries_reserved():
    names = [name for name, _type in vs._MemoryV2._fields_]
    assert names == ["version", "total", "reserved", "free", "used"]
    assert [name for name, _type in vs._MemoryV1._fields_] == ["total", "free", "used"]


def test_version_word_is_sizeof_or_two_shifted_into_the_high_byte():
    # The driver rejects a zero version with NVML_ERROR_STRUCT_SIZE_INVALID, so
    # the encoding pynvml uses is what a real nvml.dll accepts.
    expected = ctypes.sizeof(vs._MemoryV2) | (2 << 24)
    probe = vs._MemoryV2()
    probe.version = expected
    assert probe.version >> 24 == 2
    assert probe.version & 0xFFFFFF == ctypes.sizeof(vs._MemoryV2)


def test_open_probes_v2_with_the_version_word_and_reads_through_v2(nvml):
    lib = nvml["install"]()
    backend = vs._NvmlBackend(0)
    assert backend._memory_v2 is True
    assert lib.version_words[0] == ctypes.sizeof(vs._MemoryV2) | (2 << 24)
    sample = backend.read()
    assert sample.source == "nvml"
    assert sample.name == "RTX 4090"
    assert sample.total_bytes == 24 * 1024 * MIB
    assert sample.used_bytes == 4 * 1024 * MIB
    assert "v1" not in lib.calls                     # v2 served the read


def test_the_probed_version_word_is_reused_for_every_read(nvml):
    lib = nvml["install"]()
    backend = vs._NvmlBackend(0)
    backend.read()
    backend.read()
    assert set(lib.version_words) == {ctypes.sizeof(vs._MemoryV2) | (2 << 24)}


def test_a_driver_that_rejects_v2_falls_back_to_v1_once(nvml):
    lib = nvml["install"](v2_rc=3)
    backend = vs._NvmlBackend(0)
    assert backend._memory_v2 is False               # the probe already failed
    sample = backend.read()
    assert sample.used_bytes == 4 * 1024 * MIB
    assert lib.calls.count("v2") == 1                # probed at open(), not per poll
    assert "v1" in lib.calls


def test_a_v2_failure_at_read_time_degrades_instead_of_reporting_nothing(nvml):
    lib = nvml["install"]()
    backend = vs._NvmlBackend(0)
    assert backend.read() is not None
    lib.v2 = False                                   # driver stops answering v2
    sample = backend.read()
    assert sample is not None and "v1" in lib.calls
    assert backend._memory_v2 is False


def test_an_nvml_library_without_the_v2_entry_point_still_reads(nvml):
    nvml["install"](missing=["nvmlDeviceGetMemoryInfo_v2"])
    backend = vs._NvmlBackend(0)
    assert backend._memory_v2 is False
    assert backend.read().source == "nvml"


def test_the_pre_v2_handle_entry_point_is_used_when_the_v2_one_is_absent(nvml):
    lib = nvml["install"](missing=["nvmlDeviceGetHandleByIndex_v2"])
    backend = vs._NvmlBackend(2)
    assert backend.error == ""
    assert ("handle", "v1", 2) in lib.calls
    assert backend.read().used_bytes == 4 * 1024 * MIB


def test_a_library_without_a_name_entry_point_still_reports_memory(nvml):
    nvml["install"](missing=["nvmlDeviceGetName"])
    backend = vs._NvmlBackend(0)
    sample = backend.read()
    assert sample.name == "" and sample.source == "nvml"


def test_missing_init_leaves_a_reason_and_no_sample(nvml):
    nvml["install"](missing=["nvmlInit_v2"])
    backend = vs._NvmlBackend(0)
    assert backend.error == "nvmlInit_v2 missing"
    assert backend._lib is None
    assert backend.read() is None


def test_init_failure_is_an_error_not_an_exception(nvml):
    lib = nvml["install"](init_rc=15)
    backend = vs._NvmlBackend(0)
    assert backend.error == "nvmlInit_v2 failed"
    assert backend.read() is None


def test_no_device_at_the_index_names_the_index(nvml):
    lib = nvml["install"](handle_rc=100)
    backend = vs._NvmlBackend(0)
    assert "index 0" in backend.error and "rc=100" in backend.error
    assert backend.read() is None


def test_close_shuts_nvml_down_once_and_is_idempotent(nvml):
    lib = nvml["install"]()
    backend = vs._NvmlBackend(0)
    backend.close()
    assert lib.shutdowns == 1
    backend.close()
    assert lib.shutdowns == 1


def test_no_nvml_library_at_all_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(vs, "_load_nvml", lambda: None)
    backend = vs._NvmlBackend(0)
    assert backend.error == "nvml library not found"
    assert backend.read() is None


def test_find_nvml_library_returns_the_windows_name_without_loading(monkeypatch):
    def never(name):
        raise AssertionError("must not probe on Windows")

    monkeypatch.setattr(vs, "os", _FakeOs("nt"))
    monkeypatch.setattr(vs, "ctypes", _Ctypes(cdll=never))
    assert vs._find_nvml_library() == "nvml.dll"


def test_find_nvml_library_on_linux_tries_the_soname_then_known_dirs(monkeypatch):
    loaded = []

    def fake_cdll(name):
        loaded.append(name)
        raise OSError("cannot open")

    monkeypatch.setattr(vs, "os", _FakeOs(
        "posix", exists=lambda p: p == "/usr/lib64/libnvidia-ml.so.1"))
    monkeypatch.setattr(vs, "ctypes", _Ctypes(cdll=fake_cdll))
    assert vs._find_nvml_library() == "/usr/lib64/libnvidia-ml.so.1"
    assert loaded == ["libnvidia-ml.so.1", "libnvidia-ml.so"]


def test_find_nvml_library_gives_up_with_none(monkeypatch):
    monkeypatch.setattr(vs, "os", _FakeOs("posix"))
    monkeypatch.setattr(vs, "ctypes", _Ctypes(cdll=lambda name: (_ for _ in ()).throw(OSError())))
    assert vs._find_nvml_library() is None


# --------------------------------------------------------------------------
# nvidia-smi fallback
# --------------------------------------------------------------------------

SMI_TWO_GPUS = ("0, NVIDIA GeForce RTX 4090, 24576, 20480, 4096\n"
                "1, NVIDIA RTX 6000, 49152, 48000, 1152\n")


def test_smi_row_is_converted_from_mib_to_bytes():
    sample = vs._parse_smi(SMI_TWO_GPUS, 0)
    assert sample.total_bytes == 24576 * MIB
    assert sample.free_bytes == 20480 * MIB
    assert sample.used_bytes == 4096 * MIB
    assert sample.name == "NVIDIA GeForce RTX 4090"
    assert sample.source == "nvidia-smi"


def test_smi_picks_the_requested_index():
    assert vs._parse_smi(SMI_TWO_GPUS, 1).used_bytes == 1152 * MIB
    assert vs._parse_smi(SMI_TWO_GPUS, 2) is None


def test_smi_skips_rows_it_cannot_use_instead_of_failing_the_query():
    broken = ("0, card, 24576, N/A, 4096\n"
              "short, line\n"
              "1, RTX 6000, 49152, 48000, 1152\n")
    assert vs._parse_smi(broken, 0) is None
    assert vs._parse_smi(broken, 1).name == "RTX 6000"


def test_smi_accepts_padded_and_fractional_numbers():
    sample = vs._parse_smi(" 0 , card , 24576.0 , 20000.4 ,  4575.6 \n", 0)
    assert sample.used_bytes == 4575 * MIB          # truncated, never rounded up


@pytest.mark.parametrize("stdout", [None, "", "   \n", "\n"])
def test_smi_with_no_output_reads_nothing(stdout):
    assert vs._parse_smi(stdout, 0) is None


def test_smi_query_asks_for_csv_without_units():
    assert "noheader" in " ".join(vs._SMI_ARGS)
    assert "nounits" in " ".join(vs._SMI_ARGS)
    assert "memory.used" in " ".join(vs._SMI_ARGS)


class _Completed:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def test_smi_backend_reads_and_reports_index_zero(monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["kwargs"] = kwargs
        return _Completed(stdout=SMI_TWO_GPUS)

    monkeypatch.setattr(vs.shutil, "which", lambda exe: r"C:\Windows\System32\nvidia-smi.exe")
    monkeypatch.setattr(vs.subprocess, "run", fake_run)
    sample = vs._SmiBackend(0).read()
    assert sample.used_bytes == 4096 * MIB
    assert seen["cmd"][0].endswith("nvidia-smi.exe")
    assert seen["kwargs"]["timeout"] == 2.0


def test_smi_never_opens_a_console_window_on_windows(monkeypatch):
    # The packaged launcher is a windowed exe: without CREATE_NO_WINDOW every
    # poll would flash a console at the user.
    seen = {}

    def fake_run(cmd, **kwargs):
        seen.update(kwargs)
        return _Completed(stdout=SMI_TWO_GPUS)

    monkeypatch.setattr(vs, "os", _FakeOs("nt"))
    monkeypatch.setattr(vs.shutil, "which", lambda exe: "nvidia-smi")
    monkeypatch.setattr(vs.subprocess, "run", fake_run)
    assert vs._parse_smi(vs._run_nvidia_smi(2.0), 0) is not None
    assert seen["creationflags"] & 0x08000000


def test_smi_needs_no_console_flag_off_windows(monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen.update(kwargs)
        return _Completed(stdout=SMI_TWO_GPUS)

    monkeypatch.setattr(vs, "os", _FakeOs("posix"))
    monkeypatch.setattr(vs.shutil, "which", lambda exe: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(vs.subprocess, "run", fake_run)
    vs._run_nvidia_smi(2.0)
    assert "creationflags" not in seen


def test_smi_gets_no_output_when_the_tool_is_missing(monkeypatch):
    monkeypatch.setattr(vs, "os", _FakeOs("posix"))
    monkeypatch.setattr(vs.shutil, "which", lambda exe: None)
    assert vs._run_nvidia_smi(2.0) is None


def test_smi_falls_back_to_the_known_install_paths(monkeypatch):
    monkeypatch.setattr(vs, "os", _FakeOs("posix", exists=lambda p: p == "/usr/bin/nvidia-smi"))
    monkeypatch.setattr(vs.shutil, "which", lambda exe: None)
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return _Completed(stdout=SMI_TWO_GPUS)

    monkeypatch.setattr(vs.subprocess, "run", fake_run)
    assert vs._parse_smi(vs._run_nvidia_smi(2.0), 0) is not None
    assert seen["cmd"][0] == "/usr/bin/nvidia-smi"


@pytest.mark.parametrize("completed", [
    _Completed(returncode=1, stdout=SMI_TWO_GPUS),
    _Completed(returncode=0, stdout="ERROR: No running WMI instance.\n"),
])
def test_smi_failure_or_garbage_yields_no_sample(monkeypatch, completed):
    monkeypatch.setattr(vs.shutil, "which", lambda exe: "nvidia-smi")
    monkeypatch.setattr(vs.subprocess, "run", lambda cmd, **kw: completed)
    assert vs._parse_smi(vs._run_nvidia_smi(2.0), 0) is None


def test_smi_driver_hang_is_a_timeout_not_a_frozen_ui(monkeypatch):
    def boom(cmd, **kwargs):
        raise vs.subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 1))

    monkeypatch.setattr(vs.shutil, "which", lambda exe: "nvidia-smi")
    monkeypatch.setattr(vs.subprocess, "run", boom)
    assert vs._run_nvidia_smi(0.01) is None


def test_smi_permission_error_is_swallowed(monkeypatch):
    def boom(cmd, **kwargs):
        raise OSError("denied")

    monkeypatch.setattr(vs.shutil, "which", lambda exe: "nvidia-smi")
    monkeypatch.setattr(vs.subprocess, "run", boom)
    assert vs._run_nvidia_smi(2.0) is None


# --------------------------------------------------------------------------
# backend resolution
# --------------------------------------------------------------------------

class _Stub:
    def __init__(self, source, sample=None, error=""):
        self.source = source
        self._sample = sample
        self.error = error
        self.reads = 0
        self.closed = 0

    def read(self):
        self.reads += 1
        return self._sample

    def close(self):
        self.closed += 1


def test_nvml_wins_and_the_slow_smi_path_is_never_polled(monkeypatch):
    ok = _Stub("nvml", sample=vs.GpuMemory(1, 2, 3, "card", "nvml"))
    monkeypatch.setattr(vs, "_NvmlBackend", lambda index: ok)
    smi = _Stub("nvidia-smi", sample=vs.GpuMemory(1, 2, 3, "card", "nvidia-smi"))
    monkeypatch.setattr(vs, "_SmiBackend", lambda index, timeout=2.0: smi)
    sampler = vs.VramSampler(0)
    assert sampler.open() is True
    assert sampler.status() == "nvml"
    assert smi.reads == 0                             # a subprocess per poll is the cost
    assert smi.closed == 0                             # left alone, not torn down twice
    assert sampler.last_error == ""


def test_a_handle_that_cannot_report_memory_is_not_a_source(monkeypatch):
    dead = _Stub("nvml", error="no NVML device at index 0 (rc=100)")
    good = _Stub("nvidia-smi", sample=vs.GpuMemory(10, 5, 5, "card", "nvidia-smi"))
    monkeypatch.setattr(vs, "_NvmlBackend", lambda index: dead)
    monkeypatch.setattr(vs, "_SmiBackend", lambda index, timeout=2.0: good)
    sampler = vs.VramSampler(0)
    assert sampler.open() is True
    assert sampler.status() == "nvidia-smi"
    assert dead.closed == 1
    assert sampler.last_error == ""              # a working source clears the reason


def test_all_backends_failing_reports_the_reason_and_stays_quiet(monkeypatch):
    nvml_stub = _Stub("nvml", error="nvml library not found")
    smi_stub = _Stub("nvidia-smi", error="nvidia-smi gave no usable row")
    monkeypatch.setattr(vs, "_NvmlBackend", lambda index: nvml_stub)
    monkeypatch.setattr(vs, "_SmiBackend", lambda index, timeout=2.0: smi_stub)
    sampler = vs.VramSampler(0)
    assert sampler.open() is False
    assert sampler.status() == "none"
    assert sampler.read() is None
    assert sampler.last_error == "nvidia-smi gave no usable row"
    assert sampler.open() is False                    # a later poll may retry


def test_allow_smi_false_keeps_the_panel_off_the_slow_path(monkeypatch):
    nvml_stub = _Stub("nvml", error="nvml library not found")
    built = []
    monkeypatch.setattr(vs, "_NvmlBackend", lambda index: nvml_stub)
    monkeypatch.setattr(vs, "_SmiBackend",
                        lambda index, timeout=2.0: built.append(index) or _Stub("nvidia-smi"))
    sampler = vs.VramSampler(0, allow_smi=False)
    assert sampler.open() is False
    assert built == []


def test_open_is_idempotent_and_reopens_after_close(monkeypatch):
    stub = _Stub("nvml", sample=vs.GpuMemory(1, 2, 3, "card", "nvml"))
    monkeypatch.setattr(vs, "_NvmlBackend", lambda index: stub)
    sampler = vs.VramSampler(0)
    assert sampler.open() and sampler.open()
    assert stub.reads == 1                        # the probe read, once
    sampler.close()
    assert stub.closed == 1 and sampler.status() == "none"
    assert sampler.open() is True


def test_read_opens_lazily_and_keeps_last_sample_failing(monkeypatch):
    stub = _Stub("nvml", sample=vs.GpuMemory(1, 2, 3, "card", "nvml"))
    monkeypatch.setattr(vs, "_NvmlBackend", lambda index: stub)
    sampler = vs.VramSampler(0)
    assert sampler.status() == "none"             # nothing touched at construction
    assert sampler.read().source == "nvml"
    stub._sample = None
    assert sampler.read() is None


def test_read_gpu_memory_is_one_shot_and_closes(monkeypatch):
    stub = _Stub("nvml", sample=vs.GpuMemory(1 << 30, 1, 1, "card", "nvml"))
    monkeypatch.setattr(vs, "_NvmlBackend", lambda index: stub)
    sample = vs.read_gpu_memory(0)
    assert sample.total_bytes == 1 << 30
    assert stub.closed == 1


def test_a_machine_with_no_gpu_source_reports_none_and_a_reason(monkeypatch):
    monkeypatch.setattr(vs, "_NvmlBackend", lambda index: _Stub("nvml", error="none here"))
    monkeypatch.setattr(vs, "_SmiBackend", lambda index, timeout=2.0: _Stub("nvidia-smi"))
    sampler = vs.VramSampler(3)
    assert sampler.read() is None
    assert sampler.last_error == "none here"          # a silent backend still explains itself


def test_a_failing_backend_that_says_nothing_still_leaves_a_reason(monkeypatch):
    monkeypatch.setattr(vs, "_NvmlBackend", lambda index: _Stub("nvml"))
    monkeypatch.setattr(vs, "_SmiBackend", lambda index, timeout=2.0: _Stub("nvidia-smi"))
    sampler = vs.VramSampler(0)
    assert sampler.open() is False
    assert sampler.last_error == "no GPU memory source"


def test_a_failed_open_is_retried_by_the_next_read(monkeypatch):
    # A driver that initialises late must not cost the user a restart.
    bad = _Stub("nvml", error="nvml library not found")
    good = _Stub("nvml", sample=vs.GpuMemory(1, 2, 3, "card", "nvml"))
    backends = iter([bad, good])
    monkeypatch.setattr(vs, "_NvmlBackend", lambda index: next(backends))
    monkeypatch.setattr(vs, "_SmiBackend", lambda index, timeout=2.0: _Stub("nvidia-smi"))
    sampler = vs.VramSampler(0, allow_smi=False)
    assert sampler.read() is None
    assert sampler.read().source == "nvml"
    assert sampler.last_error == ""


def test_sample_fields_have_defaults_so_an_partial_read_is_still_a_sample():
    empty = vs.GpuMemory()
    assert (empty.total_bytes, empty.free_bytes, empty.used_bytes) == (0, 0, 0)
    assert empty.name == "" and empty.source == ""


def test_poll_interval_sits_in_the_50_to_100ms_band():
    assert 50 <= vs.SAMPLE_INTERVAL_MS <= 100


# --------------------------------------------------------------------------
# baselines and the noisy rule
# --------------------------------------------------------------------------

@pytest.mark.parametrize("samples,expected", [
    ([], 0),
    ([None], 0),
    ([500], 500),
    ([400, 500, 600], 500),
    ([500, 400, 600, 500], 500),        # even count: the mean of the middle two
    ([500, 900, 500, 500, 500], 500),   # one spike must not set the baseline
    ([500, 900, 500, 500, 500, 500], 500),
    ([None, 500, None, 600], 550),
])
def test_median_used_ignores_gaps_and_spikes(samples, expected):
    assert vs.median_used(samples) == expected


@pytest.mark.parametrize("total,expected", [
    (0, vs.NOISY_FLOOR_BYTES),
    (8 * 1024 * MIB, vs.NOISY_FLOOR_BYTES),          # 2 % = 160 MiB < the floor
    (16 * 1024 * MIB, int(0.02 * 16 * 1024 * MIB)),  # 320 MiB beats the floor
    (24 * 1024 * MIB, int(0.02 * 24 * 1024 * MIB)),
    (None, vs.NOISY_FLOOR_BYTES),
])
def test_noisy_threshold_is_two_percent_with_a_256mib_floor(total, expected):
    assert vs.noisy_threshold_bytes(total) == max(vs.NOISY_FLOOR_BYTES, expected)


def test_a_baseline_that_only_reached_the_threshold_is_still_clean():
    thr = vs.noisy_threshold_bytes(16 * 1024 * MIB)
    assert vs.is_noisy(1000, 1000 + thr, 16 * 1024 * MIB) is False
    assert vs.is_noisy(1000, 1000 + thr + 1, 16 * 1024 * MIB) is True


def test_noise_is_measured_in_both_directions():
    # Another process freeing memory moves the floor just as much.
    big = 200 * 1024 * MIB
    assert vs.is_noisy(big, 0, 16 * 1024 * MIB) is True
    assert vs.is_noisy(0, big, 16 * 1024 * MIB) is True


def test_the_noisy_rule_compares_baselines_not_the_peak():
    # A run that reserved 8 GiB and gave it all back is clean; the peak is ours.
    assert vs.is_noisy(2 * 1024 * MIB, 2 * 1024 * MIB + 10 * MIB, 16 * 1024 * MIB) is False
