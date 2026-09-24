# -*- coding: utf-8 -*-
"""Wiring between the sampler thread, the learner and the read-only card.

The spec's safety rules are behavioural, not arithmetic, so they need a harness
of their own:

* "预测绝对不能自动修改参数，只有用户点击按钮后才能修改 kvmem_budget" — a refresh
  must never reach the `--kvmem-budget` widget; only a click may.
* "OOM 默认不持久化，只在当前 UI 标记" — the rejected run leaves no trace on disk
  but is visible in the note line.
* the note has one writer per refresh, so the label ordering in
  `_on_run_finished` / `_on_apply_budget` is load-bearing and tested.

No real GPU, thread or model file is involved: the window is a stub exposing the
four hooks `VramPredictor` is documented to use, and the sampler is fed
`GpuMemory` values by hand.
"""
import time
from contextlib import ExitStack

import pytest

from PyQt6.QtCore import QObject
from PyQt6.QtWidgets import QLineEdit

from core import vram_estimator as ve
from core import vram_learning as vl
from core import vram_sampler as vs
from tests.conftest import silenced_qt_method
from ui import vram_monitor as vm
from ui.log_parser import parse_log_line

GIB = ve.GIB

KV_BYTES_LINE = (
    "0.10 I kvmem: KVMEM_KV_BYTES bytes=1853882368 cells=53248 slots=416 "
    "budget=36864 pool=53248 ratio=0.50 high=0.45 low=0.35 cap_blocks=1920 "
    "gpu_total=34359738368 block_bytes=4456448")
MODEL_BUFFER_LINE = "load_tensors:             CUDA0 model buffer size = 11545.60 MiB"
SLOT_POOL_LINE = (
    "0.12 I kv_unified: load: KVMem slot-pool cells=53248 slots=416 "
    "block_tokens=128 budget=36864 gen_reserve=16384 sink_blocks=1 "
    "method=retrieval harvest_v=0 type_k=q8_0 type_v=q8_0 n_embd_k=1024 "
    "attn_layers=16")


def _geometry(**overrides):
    # A 4 GiB card of weights on a 16 GiB GPU with 10 GiB free: enough headroom
    # that a safe budget exists, so a missing number is a bug and not physics.
    base = dict(arch="qwen35", block_count=64, attn_layers=16, n_k=1024, n_v=1024,
                n_embd=2048, mtp_layers=1, n_k_mtp=1024, n_v_mtp=1024,
                has_nextn=True, data_span_bytes=4 * GIB)
    base.update(overrides)
    return ve.ModelGeometry(**base)


def _values(**overrides):
    base = dict(model="", mmproj="", ctx_size=262144, batch_size=512,
                n_gpu_layers=-1, kvmem_enabled=True, kvmem_budget=36864,
                kvmem_gen_reserve=16384, kvmem_block_tokens=128,
                kvmem_gpu_ratio=0.5, cache_type_k="q8_0", cache_type_v="q8_0",
                spec_type="draft-mtp", spec_kv_dtype="f16", mmproj_offload=False)
    base.update(overrides)
    return base


class FakeBudgetWidget:
    """The `--kvmem-budget` control: records every write, however it arrives."""

    def __init__(self):
        self.written = []

    def setValue(self, value):
        self.written.append(int(value))


class FakeAdvanced:
    def __init__(self, tab="kvmem"):
        self.tab = tab
        self.mounted = None
        self._retranslate_extras = []
        self.budget = FakeBudgetWidget()
        self.model_edit = None
        self.mmproj_edit = None
        self.tabs = None

    def add_tab_header(self, key, widget):
        if key != self.tab:
            return False
        self.mounted = widget
        return True

    def param_widget(self, key):
        return {"kvmem_budget": self.budget, "model": self.model_edit,
                "mmproj": self.mmproj_edit}.get(key)


class FakeWindow(QObject):
    """The four hooks `VramPredictor` documents, and nothing else.

    A QObject because the predictor takes the window as its Qt parent, which is
    what keeps the monitor thread alive for exactly as long as the window.
    """

    def __init__(self, values=None, tab="kvmem"):
        super().__init__()
        self.advanced_panel = FakeAdvanced(tab)
        self.work_dir = "."
        self.engine = None
        self.values = _values(**(values or {}))
        self.error_report = []

    def _get_current_values(self):
        return dict(self.values)

    def _server_binary(self):
        return ""

    def _engine_error_report(self):
        return self.error_report


class CountingStore(vl.VramLearningStore):
    def __init__(self, path):
        super().__init__(path)
        self.saves = 0

    def save(self):
        self.saves += 1
        return super().save()


@pytest.fixture
def monitor():
    m = vm.VramMonitor(0)
    m.status = "nvml"
    m._measure = vl.PeakMeasure(16 * GIB)
    return m


def _feed(monitor, used, total=16 * GIB, free=8 * GIB, source="nvml", n=1):
    for _ in range(n):
        monitor._consume(vs.GpuMemory(total_bytes=total, free_bytes=free,
                                      used_bytes=int(used), name="RTX 4090",
                                      source=source))


@pytest.fixture
def predictor(tmp_path, monkeypatch, request):
    """An installed predictor: card mounted, no thread, no geometry worker.

    `silenced_qt_method` for `start`, not monkeypatch: QThread owns that method in
    C++, and monkeypatch restores an inherited Qt method in a form that binds no
    `self` — which then breaks any later test that really starts a monitor.
    """
    stack = ExitStack()
    stack.enter_context(silenced_qt_method(vm.VramMonitor, "start"))
    request.addfinalizer(stack.close)
    store = CountingStore(tmp_path / "vram_learning.json")
    monkeypatch.setattr(vm.vl, "read_state", lambda path=None: store)
    window = FakeWindow()
    p = vm.VramPredictor(window, window._get_current_values)
    p._store = store
    assert p.install() is True
    p._geometry = _geometry()
    p._fingerprint = "qwen35-64-1234"
    p._sample = vs.GpuMemory(total_bytes=16 * GIB, free_bytes=10 * GIB,
                             used_bytes=6 * GIB, name="NVIDIA GeForce RTX 4090",
                             source="nvml")
    p._monitor.status = "nvml"
    store.saves = 0
    return p


@pytest.fixture
def watched_predictor(tmp_path, monkeypatch, request):
    """A predictor on a window that really owns the model line edit.

    `predictor` answers param_widget("model") with None — the case of a page
    without that control — so the watch that a restored preset depends on can
    only be exercised against a live Qt signal.
    """
    stack = ExitStack()
    stack.enter_context(silenced_qt_method(vm.VramMonitor, "start"))
    stack.enter_context(silenced_qt_method(vm._GeometryWorker, "start"))
    request.addfinalizer(stack.close)
    store = vl.VramLearningStore(tmp_path / "vram_learning.json")
    monkeypatch.setattr(vm.vl, "read_state", lambda path=None: store)
    window = FakeWindow()
    window.advanced_panel.model_edit = QLineEdit()
    window.advanced_panel.mmproj_edit = QLineEdit()
    # Like the real panel, the window reads these controls back as parameters.
    for edit, key in ((window.advanced_panel.model_edit, "model"),
                      (window.advanced_panel.mmproj_edit, "mmproj")):
        edit.textChanged.connect(lambda text, k=key: window.values.__setitem__(k, text))
    p = vm.VramPredictor(window, window._get_current_values)
    assert p.install() is True
    return p


def _meas(**overrides):
    base = dict(peak_bytes=4 * GIB, baseline_bytes=2 * GIB, tail_bytes=2 * GIB,
                gpu_total_bytes=16 * GIB, free_bytes=9 * GIB,
                device_name="NVIDIA GeForce RTX 4090", source="nvml",
                usable=True, reason="")
    base.update(overrides)
    return vm.RunMeasurement(**base)


# --------------------------------------------------------------------------
# RunMeasurement: how much of a peak may be trusted
# --------------------------------------------------------------------------

def test_only_nvml_cadence_counts_as_a_reliable_peak():
    assert _meas(source="nvml").reliable is True
    assert _meas(source="nvidia-smi").reliable is False
    assert _meas(source="none").reliable is False


def test_a_coarse_run_is_displayed_but_never_taught_from():
    meas = _meas(source="nvidia-smi")
    assert meas.usable is True and meas.learnable is False
    assert meas.reject_label == "coarse"


def test_a_verdict_reason_beats_the_cadence_label():
    meas = _meas(usable=False, reason="noisy", source="nvidia-smi")
    assert meas.reject_label == "noisy"
    assert _meas(usable=True, source="nvml").reject_label == ""


def test_learnable_needs_both_a_valid_sample_and_a_fine_cadence():
    assert _meas().learnable is True
    assert _meas(usable=False).learnable is False
    assert _meas(source="nvidia-smi").learnable is False


# --------------------------------------------------------------------------
# the polling thread's bookkeeping (driven by hand, no thread started)
# --------------------------------------------------------------------------

def test_the_labels_are_throttled_but_the_peak_tracker_sees_every_sample(monitor):
    seen = []
    monitor.sample_ready.connect(lambda s: seen.append(s))
    monitor.start_baseline()
    monitor.begin_run()
    monitor._last_emit = 0.0
    _feed(monitor, 3 * GIB)
    first = len(seen)
    _feed(monitor, 9 * GIB)
    _feed(monitor, 9 * GIB)
    assert len(seen) == first                  # inside the throttle window
    assert monitor._measure._peak == 9 * GIB    # the spike was still recorded
    assert monitor._measure._run_samples == 3


def test_the_baseline_needs_its_samples_before_a_run_is_measurable(monitor):
    monitor.start_baseline()
    _feed(monitor, 2 * GIB, n=4)
    assert monitor._measure.baseline_ready is False
    _feed(monitor, 2 * GIB)
    assert monitor._measure.baseline_ready is True
    assert monitor._measure.baseline_bytes() == 2 * GIB


def test_arming_before_the_first_sample_still_knows_the_card_size(monitor):
    monitor._last_total = 0
    monitor._measure = vl.PeakMeasure(0)
    monitor.start_baseline()
    _feed(monitor, 2 * GIB, total=24 * GIB)
    assert monitor._measure.gpu_total_bytes == 24 * GIB
    assert monitor._last_total == 24 * GIB


def test_the_run_finishes_once_the_tail_baseline_is_full(monitor):
    finished = []
    monitor.run_finished.connect(finished.append)
    monitor.start_baseline()
    _feed(monitor, 2 * GIB, n=5)
    monitor.begin_run()
    _feed(monitor, 6 * GIB, n=3)
    monitor.end_run()
    _feed(monitor, 2 * GIB, n=4)
    assert finished == []                       # tail incomplete: keep waiting
    _feed(monitor, 2 * GIB)
    assert len(finished) == 1
    meas = finished[0]
    assert meas.peak_bytes == 4 * GIB           # 6 GiB peak minus the 2 GiB floor
    assert meas.usable is True and meas.source == "nvml"
    assert meas.gpu_total_bytes == 16 * GIB
    assert meas.device_name == "RTX 4090"
    _feed(monitor, 2 * GIB, n=5)
    assert len(finished) == 1                   # reported once, not per poll


def test_a_dead_process_stops_waiting_after_the_tail_deadline(monitor):
    finished = []
    monitor.run_finished.connect(finished.append)
    monitor.start_baseline()
    _feed(monitor, 2 * GIB, n=5)
    monitor.begin_run()
    _feed(monitor, 6 * GIB)
    monitor.end_run()
    monitor._tail_deadline = time.monotonic() - 1
    _feed(monitor, 2 * GIB)
    assert len(finished) == 1
    assert finished[0].usable is False
    assert finished[0].reason == "no_tail"
    assert monitor._measure.phase == "done"


def test_a_peak_that_never_rose_above_the_floor_is_not_a_measurement(monitor):
    finished = []
    monitor.run_finished.connect(finished.append)
    monitor.start_baseline()
    _feed(monitor, 2 * GIB, n=5)
    monitor.begin_run()
    _feed(monitor, 2 * GIB)
    monitor.end_run()
    _feed(monitor, 2 * GIB, n=5)
    assert finished[0].peak_bytes == 0
    assert finished[0].usable is False
    assert finished[0].reason == "no_growth"


def test_another_process_moving_the_floor_invalidates_the_peak(monitor):
    finished = []
    monitor.run_finished.connect(finished.append)
    monitor.start_baseline()
    _feed(monitor, 2 * GIB, n=5)
    monitor.begin_run()
    _feed(monitor, 8 * GIB)
    monitor.end_run()
    _feed(monitor, 6 * GIB, n=5)                # the card never came back down
    assert finished[0].usable is False
    assert finished[0].reason == "noisy"


def test_starting_a_baseline_without_a_gpu_source_is_refused(monitor):
    monitor.status = "none"
    assert monitor.start_baseline() is False


def test_the_launch_delay_follows_the_backend_in_use(monitor):
    monitor.status = "nvml"
    assert monitor.baseline_window_ms() == (vl.PeakMeasure.BASELINE_SAMPLES + 1) * 75
    monitor.status = "nvidia-smi"
    assert monitor.baseline_window_ms() == (vl.PeakMeasure.BASELINE_SAMPLES + 1) * 1000


def test_the_poll_period_is_the_one_the_docstring_promises():
    assert vm._POLL_MS == {"nvml": 75, "nvidia-smi": 1000}
    assert 50 <= vm._POLL_MS["nvml"] <= 100


def test_stop_polling_before_the_thread_ever_ran_is_a_no_op(monitor):
    # Qt ignores requestInterruption() on a thread that is not running, so this
    # must never be the only thing standing between us and a finished poller.
    monitor.stop_polling()
    assert monitor.isInterruptionRequested() is False
    assert monitor.isRunning() is False


def test_the_thread_polls_until_asked_to_stop_and_then_finishes():
    class FakeSampler:
        def __init__(self):
            self.reads = 0
            self.last_error = ""

        def open(self):
            return True

        def status(self):
            return "nvml"

        def read(self):
            self.reads += 1
            return vs.GpuMemory(total_bytes=16 * GIB, free_bytes=14 * GIB,
                                used_bytes=2 * GIB, name="RTX 4090", source="nvml")

        def close(self):
            pass

    sampler = FakeSampler()
    monitor = vm.VramMonitor(0)
    monitor._sampler = sampler
    monitor.start()
    deadline = time.monotonic() + 5.0
    while sampler.reads < 3 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert sampler.reads >= 3                      # the loop is really polling
    monitor.stop_polling()
    assert monitor.wait(3000) is True              # exiting is the interruption's job
    assert monitor.isRunning() is False


def test_sleep_until_waits_for_the_deadline_and_not_for_a_whole_poll(monitor):
    started = time.monotonic()
    monitor._sleep_until(started + 0.1)
    waited = time.monotonic() - started
    assert 0.08 <= waited < 1.0


# --------------------------------------------------------------------------
# installation: the card belongs to the kvmem page and nowhere else
# --------------------------------------------------------------------------

def test_install_mounts_the_card_on_the_kvmem_page(predictor):
    from ui.vram_panel import VramPanel
    assert isinstance(predictor._window.advanced_panel.mounted, VramPanel)
    assert predictor._panel is predictor._window.advanced_panel.mounted


def test_install_fails_cleanly_when_the_page_is_not_there(tmp_path, monkeypatch):
    with silenced_qt_method(vm.VramMonitor, "start"):
        monkeypatch.setattr(vm.vl, "read_state",
                            lambda path=None: vl.VramLearningStore(tmp_path / "x.json"))
        window = FakeWindow(tab="other")
        p = vm.VramPredictor(window, window._get_current_values)
        assert p.install() is False
        assert p._panel is None
        assert window.advanced_panel.mounted is None
        p.refresh()                             # every entry point stays inert
        p.note_state("running")
        assert p.arm_launch() == 0
        p.shutdown()


def test_a_language_switch_reretangles_the_card_and_repredicts(predictor):
    calls = []
    predictor.refresh = lambda: calls.append(1)      # instance attr shadows the method
    predictor.retranslate()
    assert calls == [1]
    extras = predictor._window.advanced_panel._retranslate_extras
    assert len(extras) == 1
    extras[0](None)                                  # the window's own hook is callable
    assert calls == [1, 1]


# --------------------------------------------------------------------------
# refresh: numbers on the card, nothing into the command
# --------------------------------------------------------------------------

def test_a_refresh_fills_the_card_and_leaves_the_budget_widget_alone(predictor):
    predictor.refresh()
    panel = predictor._panel
    assert panel._values["predicted"].text() != "—"
    assert panel._values["safe"].text() != "—"
    assert panel._values["budget"].text().isdigit()
    assert panel.apply_btn.isEnabled() is True
    assert panel.note.text() == ""
    assert predictor._window.advanced_panel.budget.written == []


def test_refreshing_a_hundred_times_still_writes_nothing(predictor):
    for _ in range(100):
        predictor.refresh()
    predictor._panel.apply_btn.click()
    assert predictor._window.advanced_panel.budget.written == [
        int(predictor._panel._values["budget"].text())]


def test_clicking_apply_writes_the_advised_budget_once_and_says_so(predictor):
    predictor.refresh()
    advised = int(predictor._panel._values["budget"].text())
    predictor._panel.apply_btn.click()
    assert predictor._window.advanced_panel.budget.written == [advised]
    assert predictor._panel.note.text() == "建议 Budget 已写入 --kvmem-budget 控件"


def test_a_click_survives_a_refresh_because_the_note_writer_goes_last(predictor):
    predictor.refresh()
    predictor._panel.apply_btn.click()
    assert "已写入" in predictor._panel.note.text()
    assert predictor._geometry is not None          # refresh() ran inside the handler


def test_a_missing_budget_widget_means_no_write_and_no_crash(predictor):
    predictor._window.advanced_panel.budget = None
    predictor.refresh()
    predictor._on_apply_budget(32768)
    assert predictor._panel.note.text() != "建议 Budget 已写入 --kvmem-budget 控件"


def test_kvmem_off_explains_that_the_pool_is_not_counted(predictor):
    predictor._window.values["kvmem_enabled"] = False
    predictor.refresh()
    assert "KVMem 未启用" in predictor._panel.note.text()
    assert predictor._panel._values["budget"].text() == "—"   # no advice without a pool


def test_a_model_with_no_structure_says_the_prediction_is_untrusted(predictor):
    predictor._geometry = _geometry(attn_layers=0, n_k=0, n_v=0, block_count=0)
    predictor.refresh()
    note = predictor._panel.note.text()
    assert "模型结构缺少" in note
    assert "kv_row_bytes" in note


def test_partial_offload_is_marked_as_conservative_not_as_precise(predictor):
    predictor._window.values["n_gpu_layers"] = 8
    predictor.refresh()
    assert "部分层卸载" in predictor._panel.note.text()


def test_a_card_too_small_for_the_smallest_pool_offers_no_budget(predictor):
    predictor._sample = vs.GpuMemory(total_bytes=16 * GIB, free_bytes=1 * GIB,
                                     used_bytes=15 * GIB, name="RTX 4090",
                                     source="nvml")
    predictor.refresh()
    assert predictor._panel._values["budget"].text() == "无安全值"
    assert predictor._panel.apply_btn.isEnabled() is False
    assert "没有安全 Budget" in predictor._panel.note.text()


def test_an_advice_clamped_by_the_ratio_says_which_control_limits_it(predictor, monkeypatch):
    def clamped(inp, free, predict):
        return ve.BudgetAdvice(budget_safe=1024, pool_cells=2048, slots=16,
                               free_bytes=free, v_hat_bytes=8 * GIB, u_bytes=GIB,
                               v_safe_bytes=9 * GIB, headroom_bytes=1 * GIB,
                               feasible=True, clamped=True, at_upper_bound=False,
                               at_lower_bound=False, reason="ratio_clamped")

    monkeypatch.setattr(ve, "solve_safe_budget", clamped)
    predictor.refresh()
    assert "gpu-ratio" in predictor._panel.note.text()
    assert predictor._panel.apply_btn.isEnabled() is True


def test_no_sample_means_no_advice_rather_than_an_infinite_one(predictor):
    predictor._sample = None
    predictor.refresh()
    assert predictor._panel._values["predicted"].text() != "—"
    assert predictor._panel._values["budget"].text() == "—"


def test_the_geometry_is_parsed_off_the_gui_thread_only_for_a_real_path(predictor):
    predictor._window.values["model"] = "Z:\\definitely\\not\\here.gguf"
    before = predictor._seq
    predictor.request_geometry()
    assert predictor._seq == before + 1
    assert predictor._worker is None                 # the path does not exist
    predictor._window.values["model"] = ""
    predictor.request_geometry()
    assert "尚未选择模型文件" in predictor._panel.note.text()


def test_a_geometry_result_from_a_superseded_request_is_dropped(predictor):
    predictor.refresh()
    predictor._on_geometry(999, _geometry(block_count=1), None, "fp")
    assert predictor._geometry.block_count == 64
    predictor._on_geometry(predictor._seq, _geometry(block_count=7), None, "fp2")
    assert predictor._geometry.block_count == 7
    assert predictor._fingerprint == "fp2"


def test_a_failed_parse_clears_the_geometry_and_explains_itself(predictor):
    predictor.refresh()
    predictor._on_geometry_err(predictor._seq, "not a GGUF file")
    assert predictor._geometry is None
    assert "模型结构解析失败" in predictor._panel.note.text()
    assert "not a GGUF file" in predictor._panel.note.text()
    predictor.refresh()
    # A refresh must not turn a finished failure back into "still working". That
    # repaint is what made a card that would never fill in look like a slow one.
    assert "模型结构解析失败" in predictor._panel.note.text()
    assert predictor._panel.note.text() != "正在解析模型结构…"


def test_a_model_written_without_a_click_still_gets_parsed(watched_predictor,
                                                           tmp_path):
    """The startup path the packaged launcher actually takes.

    `_restore_last_preset()` writes `-m` into the line edit after the card was
    installed, and only the browser's click and `--ctx-size` reach
    `_update_model_info()` — so before the watch, the exe's card sat on
    正在解析模型结构… for the whole session.
    """
    p = watched_predictor
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    before = p._seq
    p._window.advanced_panel.model_edit.setText(str(model))
    assert p._seq == before + 1
    assert p._worker is not None
    assert p._geometry_source == (str(model), "")


def test_the_same_file_is_not_parsed_again_for_another_parameter(watched_predictor,
                                                                tmp_path):
    p = watched_predictor
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    p._window.advanced_panel.model_edit.setText(str(model))
    worker = p._worker
    p.model_changed()                       # e.g. a --ctx-size refresh
    assert p._worker is worker              # no second thread for the same files
    other = tmp_path / "n.gguf"
    other.write_bytes(b"GGUF")
    p._window.advanced_panel.model_edit.setText(str(other))
    assert p._worker is not worker


def test_a_late_mmproj_is_parsed_too(watched_predictor, tmp_path):
    """--mmproj restored with the preset reaches the vision-tower estimate: the
    model path never changes, so only the pair can say the request is new."""
    p = watched_predictor
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    p._window.advanced_panel.model_edit.setText(str(model))
    worker = p._worker
    mmproj = tmp_path / "v.gguf"
    mmproj.write_bytes(b"GGUF")
    p._window.advanced_panel.mmproj_edit.setText(str(mmproj))
    assert p._worker is not worker
    assert p._geometry_source == (str(model), str(mmproj))


def test_a_failed_parse_releases_the_file_so_the_next_edit_retries(watched_predictor,
                                                                  tmp_path):
    p = watched_predictor
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    p._window.values["model"] = str(model)
    p.request_geometry()
    first = p._worker
    p._on_geometry_err(p._seq, "locked by the server")
    assert p._geometry_source == ("", "")
    p.request_geometry()
    assert p._worker is not first             # retried, not swallowed


# --------------------------------------------------------------------------
# what a finished run may and may not teach
# --------------------------------------------------------------------------

def test_a_good_run_learns_and_persists(predictor):
    predictor.refresh()
    predictor._run_values = _values()
    predictor._run_flags = vl.RunOutcome(ready=True, clean_stop=True)
    predictor._on_run_finished(_meas())
    assert predictor._store.saves == 1
    assert len(predictor._store.profiles) == 1
    profile = next(iter(predictor._store.profiles.values()))
    assert len(profile.samples) == 1
    assert profile.samples[0]["peak_gib"] == pytest.approx(4.0, abs=1e-6)


def test_the_note_of_a_learned_run_is_the_predictors_own(predictor):
    predictor._run_values = _values()
    predictor._run_flags = vl.RunOutcome(ready=True, clean_stop=True)
    predictor._on_run_finished(_meas())
    assert predictor._panel.note.text() == ""


def test_oom_is_marked_on_screen_and_never_written_to_disk(predictor, tmp_path):
    predictor._run_values = _values()
    predictor._run_flags = vl.RunOutcome(ready=True, clean_stop=True, oom=True)
    predictor._on_run_finished(_meas(peak_bytes=15 * GIB))
    assert predictor._panel.note.text() == "本次以显存不足结束，未用于学习"
    assert predictor._store.saves == 0
    assert predictor._store.profiles == {}
    assert not (tmp_path / "vram_learning.json").exists()


def test_a_coarse_run_is_shown_but_not_learned(predictor):
    predictor._run_values = _values()
    predictor._run_flags = vl.RunOutcome(ready=True, clean_stop=True)
    predictor._on_run_finished(_meas(source="nvidia-smi"))
    assert predictor._panel.note.text() == "本次仅有粗粒度采样，未用于学习"
    assert predictor._store.saves == 0


def test_a_noisy_run_is_rejected_by_its_reason(predictor):
    predictor._run_values = _values()
    predictor._run_flags = vl.RunOutcome(ready=True, clean_stop=True)
    predictor._on_run_finished(_meas(usable=False, reason="noisy"))
    assert predictor._panel.note.text() == "本次显存被其他进程扰动，未用于学习"


def test_a_failed_startup_is_not_a_measurement(predictor):
    predictor._run_values = _values()
    predictor._run_flags = vl.RunOutcome(ready=False, clean_stop=False)
    predictor._on_run_finished(_meas())
    assert predictor._panel.note.text() == "本次未启动就绪，未用于学习"
    assert predictor._store.saves == 0


def test_a_param_error_rejected_by_the_engine_never_teaches(predictor):
    predictor._window.values = _values()
    predictor._window.error_report = [{"param": "--kvmem-budget", "msg": "bad"}]
    predictor.begin_run()
    predictor.note_state("running")
    predictor.end_run({}, [])
    assert predictor._run_flags.param_error is True
    assert predictor._run_flags.ready is True
    predictor._on_run_finished(_meas())
    assert predictor._panel.note.text() == "本次参数被引擎拒绝，未用于学习"
    assert predictor._store.saves == 0


def test_a_run_with_no_geometry_yet_reports_no_measurement(predictor):
    predictor._geometry = None
    predictor._run_values = _values()
    predictor._run_flags = vl.RunOutcome(ready=True, clean_stop=True)
    predictor._on_run_finished(_meas())
    assert predictor._panel.note.text() == "本次没有可用测量，未用于学习"
    assert predictor._store.profiles == {}


def test_the_learning_profile_names_the_card_that_was_measured(predictor):
    predictor._run_values = _values()
    predictor._run_flags = vl.RunOutcome(ready=True, clean_stop=True)
    predictor._on_run_finished(_meas())
    key = next(iter(predictor._store.profiles))
    assert "RTX 4090" in key.replace("GeForce ", "")
    assert str(16 * GIB) in key


def test_a_run_that_ended_before_the_first_poll_still_names_its_card(predictor):
    predictor._sample = None
    predictor._run_values = _values()
    predictor._run_flags = vl.RunOutcome(ready=True, clean_stop=True)
    predictor._on_run_finished(_meas())
    key = next(iter(predictor._store.profiles))
    assert "unknown|0" not in key
    assert str(16 * GIB) in key


def test_the_stored_prediction_moves_after_a_learning_run(predictor):
    predictor._run_values = _values()
    predictor._run_flags = vl.RunOutcome(ready=True, clean_stop=True)
    predictor.refresh()
    before = predictor._panel._values["safe"].text()
    predictor._on_run_finished(_meas(peak_bytes=8 * GIB))
    after = predictor._panel._values["safe"].text()
    assert before != after
    assert "小样本上界" in predictor._panel._values["confidence"].text()


# --------------------------------------------------------------------------
# the four lines the main window delegates
# --------------------------------------------------------------------------

def test_the_run_captures_the_logged_totals_for_itself_and_rates_for_next_time(predictor):
    info = {}
    for line in (KV_BYTES_LINE, SLOT_POOL_LINE, MODEL_BUFFER_LINE):
        parse_log_line(line, info)
    predictor._window.values = _values()
    predictor.end_run(info, ["server ready"])
    assert predictor._actual_run["kv_bytes"] == 1853882368
    assert "kv_bytes" not in predictor._rate_actual
    assert predictor._rate_actual["kv_per_token_bytes"] == 34816
    assert predictor._rate_actual["model_vram_bytes"] > 0


def test_the_established_rate_survives_into_the_next_refresh(predictor):
    # The rate needs two lines: `block_bytes` from the KV report and
    # `block_tokens` from the slot-pool report.
    info = {}
    for line in (KV_BYTES_LINE, SLOT_POOL_LINE):
        parse_log_line(line, info)
    predictor.end_run(info, [])
    predictor._window.values["kvmem_budget"] = 73728
    predictor.refresh()
    est = ve.estimate(predictor._inputs(predictor._window.values))
    assert est.kv_per_token_bytes == 34816
    assert est.kv_bytes == (73728 + 16384) * 34816


def test_begin_run_drops_the_previous_run_state(predictor):
    info = {}
    parse_log_line(KV_BYTES_LINE, info)
    predictor.end_run(info, [])
    predictor.begin_run()
    assert predictor._actual_run == {}
    assert predictor._rate_actual == {}
    assert predictor._run_flags is None
    assert predictor._saw_ready is False
    assert predictor._final_state == ""


def test_only_ready_and_the_two_terminal_states_are_remembered(predictor):
    for state in ("starting", "log", "idle"):
        predictor.note_state(state)
    assert predictor._saw_ready is False
    predictor.note_state("running")
    assert predictor._saw_ready is True
    predictor.note_state("error")
    assert predictor._final_state == "error"
    predictor.note_state("stopped")
    assert predictor._final_state == "stopped"
    predictor.end_run({}, [])
    assert predictor._run_flags.clean_stop is True
    assert predictor._run_flags.ready is True


def test_arm_launch_waits_one_baseline_window_after_arm(predictor):
    predictor._monitor.status = "nvml"
    predictor._monitor._last_total = 16 * GIB
    wait_ms = predictor.arm_launch()
    assert wait_ms == (vl.PeakMeasure.BASELINE_SAMPLES + 1) * 75
    assert predictor._monitor._measure.phase == "baseline"


def test_arm_launch_does_not_delay_a_launch_when_there_is_no_gpu(predictor):
    predictor._monitor.status = "none"
    assert predictor.arm_launch() == 0


def test_a_sample_from_the_thread_lands_on_the_card(predictor):
    sample = vs.GpuMemory(total_bytes=24 * GIB, free_bytes=22 * GIB,
                          used_bytes=2 * GIB, name="RTX 4090", source="nvml")
    predictor._on_sample(sample)
    assert predictor._panel._values["free"].text() == "22.00 GiB / 24.00 GiB"
    assert predictor._sample is sample


def test_a_backend_that_died_reports_itself_on_the_card(predictor):
    predictor._on_status("none", "nvidia-smi gave no usable row")
    assert "nvidia-smi gave no usable row" in predictor._panel.note.text()
    assert predictor._panel.apply_btn.isEnabled() is False


def test_a_backend_that_came_back_repredicts(predictor):
    predictor.refresh()
    before = predictor._panel._values["predicted"].text()
    predictor._sample = vs.GpuMemory(total_bytes=24 * GIB, free_bytes=23 * GIB,
                                     used_bytes=1 * GIB, name="RTX 4090",
                                     source="nvml")
    predictor._on_status("nvml", "")
    assert predictor._panel._values["predicted"].text() == before
    assert predictor._panel._values["budget"].text() != "无安全值"


def test_shutdown_leaves_no_thread_behind(predictor):
    predictor.shutdown()
    assert predictor._monitor.isRunning() is False
    predictor.shutdown()


def test_model_changed_asks_for_a_fresh_geometry_without_blocking(predictor):
    before = predictor._seq
    predictor.model_changed()
    assert predictor._seq == before + 1
