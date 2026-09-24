"""The read-only KVMem VRAM card: what the predictor thinks, and one button.

Everything on it is a QLabel. The single way it changes the launch command is
`apply_budget`, which the main window answers by writing the *user's* click into
the `--kvmem-budget` control — a prediction never reaches argv by itself, and a
missing GPU driver only ever costs the panel its numbers.
"""
from PyQt6.QtWidgets import (
    QWidget, QGridLayout, QHBoxLayout, QVBoxLayout, QLabel, QPushButton
)

from PyQt6.QtCore import pyqtSignal

from core.i18n import t
from core import vram_estimator as ve

GIB = ve.GIB

#: (label literal, attribute name) in display order.
FIELDS = (
    ("预计峰值", "predicted"),
    ("经验安全峰值", "safe"),
    ("当前空闲", "free"),
    ("安全余量", "headroom"),
    ("建议 KVMem Budget", "budget"),
    ("成功样本数", "samples"),
    ("置信度", "confidence"),
)

_CONFIDENCE_TEXT = {"low": "低", "medium": "中", "high": "高"}

#: Which bound actually set U (core.vram_learning.safety_bound).
_BAND_TEXT = {
    "structural": "结构估算（首次运行）",
    "small_sample": "小样本上界",
    "empirical": "经验 P95",
}

#: Why a finished run was not allowed to teach the model. OOM in particular is
#: shown but never persisted (spec: 只在当前 UI 标记).
_REJECT_TEXT = {
    "oom": "本次以显存不足结束，未用于学习",
    "startup_failed": "本次未启动就绪，未用于学习",
    "param_error": "本次参数被引擎拒绝，未用于学习",
    "unclean_stop": "本次非正常结束，未用于学习",
    "noisy": "本次显存被其他进程扰动，未用于学习",
    "coarse": "本次仅有粗粒度采样，未用于学习",
    "no_baseline": "本次启动前未采到基线，未用于学习",
    "no_samples": "本次未采到运行样本，未用于学习",
    "no_tail": "本次结束后未采到基线，未用于学习",
    "no_growth": "本次显存未增长，未用于学习",
    "no_measurement": "本次没有可用测量，未用于学习",
    "implausible": "本次测量明显失真，未用于学习",
    "non_finite": "本次测量无效，未用于学习",
    "profile_mismatch": "本次与已学习配置不符，未用于学习",
    "rls_unstable": "本次学习被跳过",
    "rejected": "本次未用于学习",
}

_HINT = "预测仅供参考，只有点击按钮才会写入 --kvmem-budget"


def _gib_text(value_bytes) -> str:
    if not value_bytes:
        return "—"
    return f"{ve.gib(int(value_bytes)):.2f} GiB"


class VramPanel(QWidget):
    apply_budget = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._budget_advice = 0
        outer = QHBoxLayout(self)
        outer.setContentsMargins(2, 0, 2, 0)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(2)
        self._values = {}
        self._captions = {}
        self._labels = {key: label for label, key in FIELDS}
        for row, (label, key) in enumerate(FIELDS):
            cap = QLabel(t(label), self)
            cap.setStyleSheet("color: #6b7280; font-size: 11px;")
            val = QLabel("—", self)
            val.setStyleSheet("color: #cdd6f4; font-size: 11px;")
            grid.addWidget(cap, row, 0)
            grid.addWidget(val, row, 1)
            self._captions[key] = cap
            self._values[key] = val
        outer.addLayout(grid, 1)

        side = QVBoxLayout()
        side.setContentsMargins(0, 0, 0, 0)
        self.hint = QLabel(t(_HINT), self)
        self.hint.setStyleSheet("color: #6b7280; font-size: 11px;")
        self.hint.setWordWrap(True)
        self.note = QLabel("", self)
        self.note.setStyleSheet("color: #d97706; font-size: 11px;")
        self.note.setWordWrap(True)
        side.addWidget(self.hint)
        side.addWidget(self.note)
        side.addStretch()
        outer.addLayout(side, 1)

        self.apply_btn = QPushButton(t("采用建议 Budget"), self)
        self.apply_btn.setFixedHeight(24)
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._on_apply)
        outer.addWidget(self.apply_btn)

    # -- slots fed by the main window -------------------------------------
    def set_sample(self, sample):
        """vs.GpuMemory from the monitor thread, or None when it is gone."""
        if sample is None:
            self._values["free"].setText(t("不可用"))
            return
        free = _gib_text(sample.free_bytes)
        # `_gib_text` renders an unknown size as "—", which is a string and so
        # always true: the denominator is dropped on the byte count.
        if int(getattr(sample, "total_bytes", 0) or 0) > 0:
            self._values["free"].setText(f"{free} / {_gib_text(sample.total_bytes)}")
        else:
            self._values["free"].setText(free)

    def set_status(self, status: str, reason: str = ""):
        if status == "none":
            self.set_note(t("显存采样不可用：{r}", r=reason or t("未找到 NVML 或 nvidia-smi")))
            self.apply_btn.setEnabled(False)

    def set_prediction(self, prediction, advice=None):
        """core.vram_learning.Prediction + optional core.vram_estimator.BudgetAdvice.

        Labels and the button only — the note line belongs to the caller that can
        see every reason at once (`VramPredictor.refresh`), so one writer wins.
        """
        if prediction is None:
            return
        self._values["predicted"].setText(_gib_text(prediction.v_hat_bytes))
        self._values["safe"].setText(_gib_text(prediction.v_safe_bytes))
        n = int(prediction.n or 0)
        self._values["samples"].setText(t("{n} 次", n=n) if n else t("尚无"))
        conf = _CONFIDENCE_TEXT.get(prediction.confidence, prediction.confidence or "—")
        band = _BAND_TEXT.get(prediction.band, prediction.band)
        self._values["confidence"].setText(f"{t(conf)} · {t(band)}" if band else t(conf))
        if advice is not None:
            self._budget_advice = int(advice.budget_safe or 0)
            self._values["budget"].setText(str(self._budget_advice) if self._budget_advice
                                           else t("无安全值"))
            self._values["headroom"].setText(_gib_text(advice.headroom_bytes))
            self.apply_btn.setEnabled(self._budget_advice > 0)

    def show_learn_result(self, learned: bool, reason: str = ""):
        """Called after a run ends: OOM and friends are marked, never stored.

        A run that learned needs no mark — whatever `refresh()` just said about
        the estimate stands.
        """
        if not learned:
            self.set_note(t(_REJECT_TEXT.get(reason, _REJECT_TEXT["rejected"])))

    def set_note(self, text: str):
        self.note.setText(text)

    # -- the only mutation this widget can cause --------------------------
    def _on_apply(self):
        if self._budget_advice > 0:
            self.apply_budget.emit(self._budget_advice)

    def retranslate(self):
        for key, cap in self._captions.items():
            cap.setText(t(self._labels[key]))
        self.hint.setText(t(_HINT))
        self.apply_btn.setText(t("采用建议 Budget"))
