"""Background GPU-memory polling for the VRAM panel.

A QThread around `core.vram_sampler` plus the Qt-free `PeakMeasure` bookkeeping:
this module decides *when* to look at the card and hands the samples to the
tracker, while `VramPredictor` (below) turns the numbers into a prediction, a
learning sample, and the text the panel shows. The main window contributes four
one-line delegations and a values callback — it never touches a label.

Why a thread at all: the nvidia-smi fallback costs 100-300 ms per read, which
would freeze the GUI at every poll. NVML reads are microseconds, so the same
loop is cheap there — only the interval changes.

The measured peak is only used for learning when the backend resolved it at
`nvml` cadence. A 1 s nvidia-smi poll can step over a short peak, and an
under-measured peak would shrink the safety bound — the wrong direction to err
in — so a coarse run is displayed as `coarse` and never taught from.
"""
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from core import vram_estimator as ve
from core import vram_learning as vl
from core import vram_sampler as vs
from core.i18n import t
from ui.log_parser import kvmem_actual_alloc
from ui.vram_panel import VramPanel

#: QLabel refreshes are throttled to this; the peak tracker still sees 100 % of
#: the samples, so a spike between two label updates cannot be lost.
DISPLAY_MIN_INTERVAL_S = 0.25

#: How long to keep sampling after the process is gone before giving up on the
#: tail baseline (a dead sampler must not leave the panel waiting forever).
TAIL_DEADLINE_S = 3.0

_POLL_MS = {"nvml": 75, "nvidia-smi": 1000}


@dataclass(frozen=True)
class RunMeasurement:
    """One run's whole-card peak, as reported to the main window."""
    peak_bytes: int = 0
    baseline_bytes: int = 0
    tail_bytes: int = 0
    gpu_total_bytes: int = 0
    free_bytes: int = 0
    device_name: str = ""
    source: str = ""              # "nvml" | "nvidia-smi" | "none"
    usable: bool = False
    reason: str = ""              # "" | no_baseline | no_samples | no_tail | noisy | no_growth

    @property
    def reliable(self) -> bool:
        """True when the cadence was fine enough for the peak to be trusted."""
        return self.source == "nvml"

    @property
    def learnable(self) -> bool:
        return bool(self.usable and self.reliable)

    @property
    def reject_label(self) -> str:
        if self.reason:
            return self.reason
        if not self.reliable:
            return "coarse"
        return ""


class VramMonitor(QThread):
    sample_ready = pyqtSignal(object)         # vs.GpuMemory
    status_changed = pyqtSignal(str, str)     # backend ("nvml"/"nvidia-smi"/"none"), reason
    run_finished = pyqtSignal(object)         # RunMeasurement

    def __init__(self, index: int = 0, parent=None):
        super().__init__(parent)
        self._sampler = vs.VramSampler(index)
        self._measure = vl.PeakMeasure()
        self._lock = threading.Lock()
        self._last_emit = 0.0
        self._tail_deadline = 0.0
        self._last_total = 0
        self.status = "none"
        self.status_reason = ""

    # -- lifecycle ---------------------------------------------------------
    def run(self):
        if not self._sampler.open():
            self.status = "none"
            self.status_reason = self._sampler.last_error
            self.status_changed.emit("none", self.status_reason)
            return
        self.status = self._sampler.status()
        self.status_changed.emit(self.status, "")
        period = _POLL_MS.get(self.status, 250) / 1000.0
        while not self.isInterruptionRequested():
            started = time.monotonic()
            sample = self._sampler.read()
            if sample is not None:
                self._consume(sample)
            self._sleep_until(started + period)

    def stop_polling(self):
        self.requestInterruption()

    def _sleep_until(self, deadline: float):
        while time.monotonic() < deadline and not self.isInterruptionRequested():
            self.msleep(20)

    # -- run bookkeeping (called from the GUI thread) --------------------
    def start_baseline(self) -> bool:
        """Arm the pre-start baseline. False when there is nothing to measure."""
        if self.status == "none":
            return False
        with self._lock:
            self._measure = vl.PeakMeasure(self._last_total)
            self._measure.start_baseline()
        return True

    def baseline_window_ms(self) -> int:
        """How long `start_baseline()` needs to fill up — the launch delay."""
        period = _POLL_MS.get(self.status, 250)
        return (vl.PeakMeasure.BASELINE_SAMPLES + 1) * period

    def begin_run(self):
        with self._lock:
            self._measure.begin_run()

    def end_run(self):
        with self._lock:
            self._measure.end_run()
            if self._measure.phase == "after":
                self._tail_deadline = time.monotonic() + TAIL_DEADLINE_S

    # -- the polling side --------------------------------------------------
    def _consume(self, sample: vs.GpuMemory):
        self._last_total = int(sample.total_bytes) or self._last_total
        with self._lock:
            phase = self._measure.phase
            if not self._measure.gpu_total_bytes:
                # Arming before the first sample lands must not lose the card
                # size: the noisy threshold is derived from it.
                self._measure.gpu_total_bytes = self._last_total
            self._measure.add(sample.used_bytes)
            finished = None
            if phase == "after":
                if self._measure.tail_ready or time.monotonic() >= self._tail_deadline:
                    self._measure.phase = "done"
                    finished = self._measurement(sample)
        if finished is not None:
            self.run_finished.emit(finished)
        now = time.monotonic()
        if now - self._last_emit >= DISPLAY_MIN_INTERVAL_S:
            self._last_emit = now
            self.sample_ready.emit(sample)

    def _measurement(self, sample: vs.GpuMemory) -> RunMeasurement:
        m = self._measure
        usable, reason = m.verdict()
        return RunMeasurement(
            peak_bytes=m.peak_bytes(), baseline_bytes=m.baseline_bytes(),
            tail_bytes=m.tail_bytes(), gpu_total_bytes=m.gpu_total_bytes,
            free_bytes=sample.free_bytes, device_name=sample.name,
            source=sample.source, usable=usable, reason=reason,
        )


# --------------------------------------------------------------------------
# model geometry, parsed off the GUI thread
# --------------------------------------------------------------------------

#: Running parse threads. A QThread destroyed while it runs takes the process
#: with it, so a window that closes mid-parse must not be the last reference;
#: each worker removes itself on finish.
_active_geometry_workers: set = set()


class _GeometryWorker(QThread):
    """parse_gguf reads the tensor table, which the KV geometry needs."""
    finished_ok = pyqtSignal(int, object, object, str)   # seq, geometry, mmproj, fingerprint
    finished_err = pyqtSignal(int, str)

    def __init__(self, seq, model_path, mmproj_path="", parent=None):
        super().__init__(parent)
        self._seq = seq
        self._model_path = str(model_path or "")
        self._mmproj_path = str(mmproj_path or "")

    def run(self):
        try:
            from gguf.parser import parse_gguf
            info = parse_gguf(self._model_path)
            geo = ve.derive_model_geometry(info)
            mm_geo = None
            if self._mmproj_path:
                try:
                    mm_geo = ve.derive_model_geometry(parse_gguf(self._mmproj_path))
                except Exception:
                    mm_geo = None        # a missing vision tower must not cost the KV math
            stats = getattr(info, "stats", None)
            fp = vl.model_fingerprint(
                geo.arch, geo.block_count, len(getattr(info, "tensors", []) or ()),
                int(getattr(info, "file_size", 0) or 0),
                str(getattr(stats, "dominant_type_name", "") or ""), geo.has_nextn)
            self.finished_ok.emit(self._seq, geo, mm_geo, fp)
        except Exception as e:
            self.finished_err.emit(self._seq, str(e))


# --------------------------------------------------------------------------
# the controller: sampling -> estimate -> learn -> the read-only card
# --------------------------------------------------------------------------

class VramPredictor(QObject):
    """Everything the VRAM card does, except the four lines the window calls.

    `values_provider` is the window's current-parameter reader, so the predictor
    never reaches into widgets except through `param_widget()` to write the one
    value the user approved with a click.
    """

    def __init__(self, window, values_provider, parent=None):
        super().__init__(parent or window)
        self._window = window
        self._read_values = values_provider
        self._monitor = VramMonitor(parent=self)
        self._panel = None
        self._store = vl.read_state()
        self._geometry = None
        self._mmproj_geometry = None
        self._fingerprint = ""
        #: The two files `_geometry` describes (empty while pending/failed).
        self._geometry_source = ("", "")
        #: Raw message of the last failed parse; "" when nothing went wrong.
        self._geometry_error = ""
        self._seq = 0
        self._worker = None
        self._sample = None
        self._build_id = ""
        self._saw_ready = False
        self._final_state = ""
        self._run_flags = None
        self._run_values = {}
        self._actual_run = {}
        self._rate_actual = {}

    # -- installation ----------------------------------------------------
    def install(self) -> bool:
        """Mount the card on the KVMem page. False when that page is not there."""
        panel = VramPanel()
        if not self._window.advanced_panel.add_tab_header("kvmem", panel):
            panel.deleteLater()
            return False
        self._panel = panel
        panel.apply_budget.connect(self._on_apply_budget)
        self._monitor.sample_ready.connect(self._on_sample)
        self._monitor.status_changed.connect(self._on_status)
        self._monitor.run_finished.connect(self._on_run_finished)
        self._build_id = self._engine_build()
        tabs = getattr(self._window.advanced_panel, "tabs", None)
        if tabs is not None:
            tabs.currentChanged.connect(lambda _i: self.refresh())
        # The panel's documented extension list: a language switch retranslates
        # the card without the main window knowing this feature exists.
        self._window.advanced_panel._retranslate_extras.append(lambda _p: self.retranslate())
        self._monitor.start()
        self.request_geometry()
        self._watch_model_control()
        return True

    def _watch_model_control(self):
        """Re-parse when a model or mmproj arrives without a browser click.

        The window's own "the model row changed" hub is `_update_model_info()`,
        and it is reached from the browser's click and from `--ctx-size` — not
        from a preset restored at startup, which writes `-m` straight into the
        widget. Without this the card sits on 正在解析模型结构… for the whole
        session, because at install() time the model really is not there yet.
        """
        for key in ("model", "mmproj"):
            changed = getattr(self._window.advanced_panel.param_widget(key),
                              "textChanged", None)
            if changed is not None:
                changed.connect(lambda _text: self.request_geometry())

    def shutdown(self):
        self._monitor.stop_polling()
        self._monitor.wait(2000)
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.wait(3000)

    def retranslate(self):
        if self._panel is not None:
            self._panel.retranslate()
            self.refresh()

    # -- the window's four delegation points -----------------------------
    def note_state(self, state: str):
        """Fed every runner state change; only the two that matter are kept."""
        if state == "running":
            self._saw_ready = True
        if state in ("stopped", "error"):
            self._final_state = state

    def arm_launch(self) -> int:
        """Take the pre-start baseline; returns the ms to wait before spawning."""
        if self._panel is None or not self._monitor.start_baseline():
            return 0
        return self._monitor.baseline_window_ms()

    def begin_run(self):
        self._saw_ready = False
        self._final_state = ""
        self._run_flags = None
        self._run_values = {}
        self._actual_run = {}
        self._rate_actual = {}
        self._monitor.begin_run()

    def end_run(self, runtime_info, log_lines):
        """Called while the run's parsed info is still alive (before the wipe)."""
        self._run_values = dict(self._read_values() or {})
        self._actual_run = kvmem_actual_alloc(runtime_info or {})
        # The logged totals belong to the budget that ran; only the rates
        # survive into the next prediction.
        self._rate_actual = kvmem_actual_alloc(runtime_info or {}, rate_only=True)
        self._run_flags = vl.RunOutcome(
            ready=self._saw_ready,
            clean_stop=self._final_state == "stopped",
            oom=vl.detect_oom(log_lines),
            param_error=self._reported_param_error())
        self._monitor.end_run()

    def model_changed(self):
        self.request_geometry()

    def _pending_note(self) -> str:
        """Why the card still has no numbers. A failure is not "still working"."""
        if self._geometry_error:
            return t("模型结构解析失败：{m}", m=self._geometry_error)
        return t("正在解析模型结构…")

    # -- prediction ------------------------------------------------------
    def refresh(self):
        if self._panel is None:
            return
        if self._geometry is None:
            self._panel.set_note(self._pending_note())
            return
        values = self._read_values() or {}
        inp = self._inputs(values)
        est = ve.estimate(inp)
        key = self._profile_key()
        batch = self._batch(values)
        pred = self._store.predict(est, key=key, batch_size=batch,
                                   gpu_total_bytes=self._total_bytes())
        advice = None
        free = int(getattr(self._sample, "free_bytes", 0) or 0)
        if free > 0 and values.get("kvmem_enabled", True):
            advice = ve.solve_safe_budget(
                inp, free, lambda e: self._predict_pair(e, key, batch))
        self._panel.set_prediction(pred, advice)
        self._panel.set_note(self._note(est, values, advice))

    def _predict_pair(self, est, key, batch):
        p = self._store.predict(est, key=key, batch_size=batch,
                                gpu_total_bytes=self._total_bytes())
        return p.v_hat_bytes, p.u_bytes

    def _inputs(self, values, actual=None):
        kv = values.get("kv_dtype") or "f16"
        sample = self._sample
        return ve.VramInputs(
            geometry=self._geometry,
            mmproj_geometry=self._mmproj_geometry,
            mmproj_offload=bool(values.get("mmproj_offload", True)),
            ctx_size=int(values.get("ctx_size") or 0),
            batch_size=self._batch(values),
            n_gpu_layers=int(values.get("n_gpu_layers") if values.get("n_gpu_layers")
                             is not None else -1),
            kvmem_enabled=bool(values.get("kvmem_enabled", True)),
            budget=int(values.get("kvmem_budget") or 0),
            gen_reserve=int(values.get("kvmem_gen_reserve") or 0),
            block_tokens=int(values.get("kvmem_block_tokens") or 128),
            gpu_ratio=float(values.get("kvmem_gpu_ratio") or 0.0),
            gpu_total_bytes=int(getattr(sample, "total_bytes", 0) or 0) or None,
            type_k=values.get("cache_type_k") or kv,
            type_v=values.get("cache_type_v") or kv,
            spec_type=values.get("spec_type") or "none",
            spec_kv_dtype=values.get("spec_kv_dtype") or "f16",
            actual=self._rate_actual if actual is None else actual)

    @staticmethod
    def _batch(values) -> int:
        return int(values.get("batch_size") or 512)

    def _total_bytes(self) -> int:
        return int(getattr(self._sample, "total_bytes", 0) or 0)

    def _card(self, meas=None):
        """(name, total_bytes): the live sample, else the finished run's reading.

        A run that ended before the first poll would otherwise teach an
        `unknown|0` profile that no later prediction can match.
        """
        sample = self._sample
        name = str(getattr(sample, "name", "") or "")
        total = int(getattr(sample, "total_bytes", 0) or 0)
        if meas is not None:
            name = name or str(meas.device_name or "")
            total = total or int(meas.gpu_total_bytes or 0)
        return name, total

    def _profile_key(self, meas=None) -> str:
        name, total = self._card(meas)
        return vl.profile_key(name, total, self._build_id, self._fingerprint)

    def _engine_build(self) -> str:
        eng = getattr(self._window, "engine", None)
        if eng is None or not getattr(eng, "read_identity", None):
            return ""
        try:
            path = self._window._server_binary()
            identity = eng.identity(path) if path else None
            return (eng.describe_identity(identity) if identity else "") or ""
        except Exception:
            return ""

    def _note(self, est, values, advice=None) -> str:
        """The one note line, most-important reason first: the panel has a single
        label and every writer would otherwise overwrite the last."""
        if not values.get("kvmem_enabled", True):
            return t("KVMem 未启用：本次预测不含 KV 池")
        if "no_gguf" in est.degraded:
            return t("模型结构未能解析，权重按 0 计（预测不可信）")
        if "partial_offload" in est.degraded:
            return t("部分层卸载无法按层切分权重，已按整模型计入（偏保守）")
        if est.missing:
            return t("模型结构缺少 {m}，预测为降级值", m=", ".join(est.missing))
        reason = getattr(advice, "reason", "")
        if reason == "nothing_fits":
            return t("当前空闲显存放不下最小的 KV 池，没有安全 Budget")
        if reason == "ratio_clamped":
            return t("建议值已受 --kvmem-gpu-ratio 上限约束")
        return ""

    # -- slots -----------------------------------------------------------
    def _on_sample(self, sample):
        self._sample = sample
        if self._panel is not None:
            self._panel.set_sample(sample)

    def _on_status(self, status, reason):
        if self._panel is not None:
            self._panel.set_status(status, reason)
        if status != "none":
            self.refresh()

    def _on_geometry(self, seq, geometry, mmproj_geometry, fingerprint):
        if seq != self._seq:
            return                    # the user switched models while this ran
        self._geometry = geometry
        self._mmproj_geometry = mmproj_geometry
        self._fingerprint = fingerprint
        self._geometry_error = ""
        self.refresh()

    def _on_geometry_err(self, seq, msg):
        if seq != self._seq:
            return
        self._geometry = None
        # The path is released again so the next edit retries: a file that was
        # locked by a running server is worth a second look.
        self._geometry_source = ("", "")
        self._geometry_error = msg[:160]
        if self._panel is not None:
            self._panel.set_note(self._pending_note())

    def _on_apply_budget(self, budget):
        """The only place this feature writes a parameter — on the user's click."""
        widget = self._window.advanced_panel.param_widget("kvmem_budget")
        if widget is None:
            return
        widget.setValue(int(budget))
        # refresh() re-derives the note line, so the confirmation goes last.
        self.refresh()
        self._panel.set_note(t("建议 Budget 已写入 --kvmem-budget 控件"))

    def _on_run_finished(self, meas):
        if self._panel is None:
            return
        values = self._run_values or (self._read_values() or {})
        if self._geometry is None:
            self.refresh()
            self._panel.show_learn_result(False, "no_measurement")
            return
        est = ve.estimate(self._inputs(values, actual=self._actual_run))
        flags = self._run_flags or vl.RunOutcome()
        outcome = replace(flags, sample_valid=meas.learnable,
                          sample_reason=meas.reject_label)
        gpu_name, gpu_total = self._card(meas)
        res = self._store.record(
            est, key=self._profile_key(meas), measured_peak_bytes=meas.peak_bytes,
            outcome=outcome, batch_size=self._batch(values),
            gpu_total_bytes=gpu_total,
            gpu_name=gpu_name,
            engine_build=self._build_id, model_fp=self._fingerprint,
            extra={"ctx": int(values.get("ctx_size") or 0),
                   "budget": int(values.get("kvmem_budget") or 0),
                   "reserve": int(values.get("kvmem_gen_reserve") or 0),
                   "ratio": float(values.get("kvmem_gpu_ratio") or 0.0)})
        if res.learned:
            self._store.save()
        # The note label has one owner per refresh, so the fresh prediction is
        # rendered first and the verdict on the run that just ended lands on top.
        self.refresh()
        self._panel.show_learn_result(res.learned, res.reason)

    def _reported_param_error(self) -> bool:
        try:
            return any(h.get("param") for h in (self._window._engine_error_report() or []))
        except Exception:
            return False

    # -- geometry requests ------------------------------------------------
    def request_geometry(self):
        values = self._read_values() or {}
        self._seq += 1
        seq = self._seq
        model = self._abs(values.get("model"))
        mmproj = self._abs(values.get("mmproj"))
        if not model:
            self._geometry = None
            self._geometry_source = ("", "")
            self._geometry_error = ""
            self._fingerprint = ""
            if self._panel is not None:
                self._panel.set_note(t("尚未选择模型文件"))
            return
        if (model, mmproj) == self._geometry_source:
            # The same two files: `textChanged` also fires when a preset re-writes
            # a path the card already parsed, and a parse thread per keystroke of
            # a typed path is not what the KV geometry needs.
            return
        self._geometry_source = (model, mmproj)
        self._geometry_error = ""
        worker = _GeometryWorker(seq, model, mmproj, self)
        worker.finished_ok.connect(self._on_geometry)
        worker.finished_err.connect(self._on_geometry_err)
        worker.finished.connect(lambda: _active_geometry_workers.discard(worker))
        _active_geometry_workers.add(worker)
        self._worker = worker
        worker.start()

    def _abs(self, path) -> str:
        """Resolve a model path the way the launcher launches it (A8 base)."""
        if not path:
            return ""
        p = Path(str(path))
        if not p.is_absolute():
            p = Path(self._window.work_dir) / p
        return str(p) if p.exists() else ""
