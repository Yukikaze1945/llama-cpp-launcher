# -*- coding: utf-8 -*-
"""The KVMem VRAM card as a widget: seven read-only rows and one button.

The spec's two hard UI rules live here. The card must never reach a parameter by
itself ("预测绝对不能自动修改参数，只有用户点击按钮后才能修改 kvmem_budget"), and a run
that was not allowed to teach the model must say so on screen without being
written to disk. Both break easily — a `set_prediction()` that also writes the
note, a button enabled before advice arrives — so they are pinned rather than
trusted.

`core.i18n.t()` resolves at call time, so a language test has to switch the
language before calling the setter that renders.
"""
import inspect
from dataclasses import replace

import pytest

from PyQt6.QtWidgets import QAbstractButton, QLineEdit, QSpinBox

from core import i18n as I
from core import vram_estimator as ve
from core import vram_learning as vl
from core import vram_sampler as vs
from ui import vram_monitor as vm
from ui import vram_panel as vp

GIB = ve.GIB
MIB = ve.MIB


def en(zh: str) -> str:
    """The English side of a literal, straight from the table."""
    return I._EN[zh]


@pytest.fixture
def panel():
    w = vp.VramPanel()
    yield w
    w.deleteLater()


@pytest.fixture
def english():
    original = I.get_language()
    I.set_language("en")
    yield
    I.set_language(original)


def _pred(n=0, band="structural", confidence="low", v_hat=12.5, u=1.0):
    return vl.Prediction(v_hat_bytes=int(v_hat * GIB), u_bytes=int(u * GIB),
                         v_safe_bytes=int((v_hat + u) * GIB),
                         residual_hat_gib=0.0, n=n, band=band,
                         confidence=confidence, dominant=band)


def _advice(budget=32768, headroom_mib=1024, reason=""):
    return ve.BudgetAdvice(budget_safe=budget, pool_cells=budget + 16384,
                           slots=(budget + 16384) // 128, free_bytes=20 * GIB,
                           v_hat_bytes=12 * GIB, u_bytes=GIB, v_safe_bytes=13 * GIB,
                           headroom_bytes=int(headroom_mib * MIB),
                           feasible=budget > 0, clamped=False, at_upper_bound=False,
                           at_lower_bound=False, reason=reason)


def _sample(total_gib, free_gib, source="nvml"):
    return vs.GpuMemory(total_bytes=int(total_gib * GIB),
                        free_bytes=int(free_gib * GIB),
                        used_bytes=max(0, int((total_gib - free_gib) * GIB)),
                        name="NVIDIA GeForce RTX 4090", source=source)


def _text(panel, key):
    return panel._values[key].text()


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------

def test_the_card_shows_exactly_the_seven_specified_rows(panel):
    assert [cap.text() for cap in panel._captions.values()] == \
        [label for label, _key in vp.FIELDS]
    assert list(panel._captions) == [key for _label, key in vp.FIELDS]


def test_every_row_starts_as_a_dash_not_a_zero(panel):
    # "0.00 GiB" would read as a measurement; "—" reads as "nothing yet".
    assert [_text(panel, key) for _label, key in vp.FIELDS] == ["—"] * 7


def test_the_card_holds_no_editable_widget_besides_one_button(panel):
    assert panel.findChildren((QLineEdit, QSpinBox)) == []
    assert [b.text() for b in panel.findChildren(QAbstractButton)] == \
        [vp.t("采用建议 Budget")]


def test_the_button_starts_disabled_because_there_is_no_advice_yet(panel):
    assert panel.apply_btn.isEnabled() is False


# --------------------------------------------------------------------------
# prediction rows
# --------------------------------------------------------------------------

def test_a_prediction_fills_the_peak_and_confidence_rows(panel):
    panel.set_prediction(_pred())
    assert _text(panel, "predicted") == "12.50 GiB"
    assert _text(panel, "safe") == "13.50 GiB"
    assert _text(panel, "samples") == "尚无"
    assert _text(panel, "confidence") == "低 · 结构估算（首次运行）"
    assert _text(panel, "budget") == "—"                 # advice is separate


def test_the_sample_count_and_band_are_shown_the_way_a_user_reads_them(panel):
    panel.set_prediction(_pred(n=12, band="empirical", confidence="high"))
    assert _text(panel, "samples") == "12 次"
    assert _text(panel, "confidence") == "高 · 经验 P95"
    panel.set_prediction(_pred(n=1, band="small_sample", confidence="medium"))
    assert _text(panel, "confidence") == "中 · 小样本上界"


def test_an_unknown_band_or_confidence_is_printed_instead_of_invented(panel):
    panel.set_prediction(_pred(band="quantum", confidence="uncertain"))
    assert _text(panel, "confidence") == "uncertain · quantum"


def test_a_prediction_without_a_band_shows_only_the_confidence(panel):
    panel.set_prediction(replace(_pred(n=3), band=""))
    assert _text(panel, "confidence") == "低"


def test_no_safe_budget_says_so_and_keeps_the_button_off(panel):
    panel.set_prediction(_pred(), _advice(budget=0, reason="nothing_fits"))
    assert _text(panel, "budget") == "无安全值"
    assert _text(panel, "headroom") == "1.00 GiB"
    assert panel.apply_btn.isEnabled() is False


def test_a_usable_advice_shows_the_number_and_arms_the_button(panel):
    panel.set_prediction(_pred(), _advice(budget=30720, headroom_mib=2048))
    assert _text(panel, "budget") == "30720"
    assert _text(panel, "headroom") == "2.00 GiB"
    assert panel.apply_btn.isEnabled() is True


def test_arming_the_button_does_not_emit_anything_by_itself(panel):
    seen = []
    panel.apply_budget.connect(seen.append)
    panel.set_prediction(_pred(), _advice(budget=30720))
    assert seen == []


def test_a_new_advice_replaces_the_previous_one(panel):
    panel.set_prediction(_pred(), _advice(budget=30720))
    panel.set_prediction(_pred(), _advice(budget=0, reason="nothing_fits"))
    assert _text(panel, "budget") == "无安全值"
    assert panel.apply_btn.isEnabled() is False


def test_a_missing_prediction_keeps_the_last_rendered_numbers(panel):
    panel.set_prediction(_pred(v_hat=9.0), _advice(budget=1024))
    panel.set_prediction(None)
    assert _text(panel, "predicted") == "9.00 GiB"
    assert _text(panel, "budget") == "1024"


def test_a_prediction_never_writes_the_note_line(panel):
    """One note owner (VramPredictor.refresh), or two writers clobber each other."""
    panel.set_note("正在解析模型结构…")
    panel.set_prediction(_pred(), _advice())
    assert panel.note.text() == "正在解析模型结构…"


# --------------------------------------------------------------------------
# live sampling rows
# --------------------------------------------------------------------------

def test_a_sample_shows_free_over_total(panel):
    panel.set_sample(_sample(24, 20.5))
    assert _text(panel, "free") == "20.50 GiB / 24.00 GiB"


def test_a_lost_sampler_says_unavailable_rather_than_zero(panel):
    panel.set_sample(_sample(24, 20))
    panel.set_sample(None)
    assert _text(panel, "free") == "不可用"


def test_a_card_that_reports_no_size_shows_only_the_free_figure(panel):
    panel.set_sample(_sample(0, 2))
    assert _text(panel, "free") == "2.00 GiB"


def test_losing_the_backend_disables_the_button_and_names_the_reason(panel):
    panel.set_prediction(_pred(), _advice(budget=30720))
    panel.set_status("none", "nvml library not found")
    assert panel.apply_btn.isEnabled() is False
    assert "nvml library not found" in panel.note.text()


def test_losing_the_backend_without_a_reason_still_explains_itself(panel):
    panel.set_status("none", "")
    assert "NVML" in panel.note.text()
    assert panel.note.text() != "none"


@pytest.mark.parametrize("status", ["nvml", "nvidia-smi"])
def test_a_working_backend_leaves_the_note_alone(panel, status):
    panel.set_status("none", "gone")
    written = panel.note.text()
    assert "gone" in written                      # the reason is quoted, not swallowed
    panel.set_status(status, "")
    assert panel.note.text() == written           # only the note's owner clears it


# --------------------------------------------------------------------------
# why a finished run did not teach the model
# --------------------------------------------------------------------------

def test_a_run_that_learned_leaves_the_predictors_note_in_place(panel):
    panel.set_note("建议 Budget 已写入 --kvmem-budget 控件")
    panel.show_learn_result(True)
    assert panel.note.text() == "建议 Budget 已写入 --kvmem-budget 控件"


def test_oom_is_marked_on_screen(panel):
    panel.show_learn_result(False, "oom")
    assert panel.note.text() == "本次以显存不足结束，未用于学习"


@pytest.mark.parametrize("reason", sorted(vp._REJECT_TEXT))
def test_every_rejection_reason_has_its_own_line(panel, reason):
    panel.show_learn_result(False, reason)
    assert panel.note.text() == vp._REJECT_TEXT[reason]
    assert "本次" in panel.note.text()             # every line speaks about this run


def test_an_unexpected_reason_still_says_the_run_was_not_used(panel):
    panel.show_learn_result(False, "quantum_tunneling")
    assert panel.note.text() == vp._REJECT_TEXT["rejected"]


def test_the_reject_table_covers_every_reason_the_pipeline_can_emit():
    reasons = {"rejected"}
    for flags in (dict(oom=True), dict(ready=False),
                  dict(ready=True, param_error=True),
                  dict(ready=True, clean_stop=True, sample_valid=False,
                       sample_reason="noisy"),
                  dict(ready=True, clean_stop=False, sample_valid=True)):
        reasons.add(vl.RunOutcome(**flags).reject_reason)
    reasons.add(vm.RunMeasurement(usable=True, source="nvidia-smi").reject_label)
    reasons |= {"no_baseline", "no_samples", "no_tail", "noisy", "no_growth"}
    reasons |= {"no_measurement", "non_finite", "implausible", "profile_mismatch",
                "rls_unstable"}
    assert reasons == set(vp._REJECT_TEXT)


def test_the_hint_line_states_the_only_way_a_parameter_changes(panel):
    assert panel.hint.text() == vp._HINT
    assert "--kvmem-budget" in panel.hint.text()


# --------------------------------------------------------------------------
# the one mutation this widget can cause
# --------------------------------------------------------------------------

def test_clicking_apply_emits_the_advised_budget_once(panel):
    seen = []
    panel.apply_budget.connect(seen.append)
    panel.set_prediction(_pred(), _advice(budget=30720))
    panel.apply_btn.click()
    assert seen == [30720]


def test_a_forced_click_without_advice_emits_nothing(panel):
    # The enabled flag is the visible guard; the value check is the real one.
    seen = []
    panel.apply_budget.connect(seen.append)
    panel.set_prediction(_pred(), _advice(budget=0, reason="nothing_fits"))
    panel.apply_btn.setEnabled(True)
    panel.apply_btn.click()
    assert seen == []


def test_the_emitted_budget_is_always_the_latest_advice(panel):
    seen = []
    panel.apply_budget.connect(seen.append)
    panel.set_prediction(_pred(), _advice(budget=30720))
    panel.apply_btn.click()
    panel.set_prediction(_pred(), _advice(budget=8192))
    panel.apply_btn.click()
    assert seen == [30720, 8192]
    panel.set_prediction(_pred(), _advice(budget=0, reason="nothing_fits"))
    panel.apply_btn.setEnabled(True)
    panel.apply_btn.click()
    assert seen == [30720, 8192]                # the stale figure is gone


def test_the_panel_source_touches_no_engine_and_no_command(panel):
    source = inspect.getsource(vp.VramPanel)
    assert "argv" not in source and "setValue" not in source and "CommandBuilder" not in source
    assert "engine" not in inspect.getsource(vp).split("class VramPanel")[0]


# --------------------------------------------------------------------------
# language
# --------------------------------------------------------------------------

def test_retranslate_moves_every_caption_and_the_button(english, panel):
    panel.retranslate()
    assert [cap.text() for cap in panel._captions.values()] == \
        [en(label) for label, _key in vp.FIELDS]
    assert panel.apply_btn.text() == en("采用建议 Budget")
    assert panel.hint.text() == en(vp._HINT)


def test_retranslate_keeps_the_current_numbers(english, panel):
    panel.set_prediction(_pred(n=4), _advice(budget=30720))
    panel.retranslate()
    assert _text(panel, "predicted") == "12.50 GiB"
    assert _text(panel, "budget") == "30720"


def test_rows_written_while_english_are_english(english, panel):
    panel.set_prediction(_pred(n=3), _advice(budget=0))
    assert _text(panel, "samples") == "3 runs"
    assert _text(panel, "budget") == en("无安全值")
    panel.set_sample(None)
    assert _text(panel, "free") == en("不可用")


def test_rejection_lines_are_translated(english, panel):
    panel.show_learn_result(False, "noisy")
    assert panel.note.text() == en(vp._REJECT_TEXT["noisy"])


# --------------------------------------------------------------------------
# the literals tests/test_i18n.py's AST scan cannot see
# --------------------------------------------------------------------------

def _card_literals():
    """Every Chinese literal this module hands to t(), including by-name ones."""
    literals = [label for label, _key in vp.FIELDS]
    literals += [vp._HINT, "采用建议 Budget", "不可用", "尚无", "无安全值", "{n} 次",
                 "显存采样不可用：{r}", "未找到 NVML 或 nvidia-smi"]
    literals += list(vp._CONFIDENCE_TEXT.values())
    literals += list(vp._BAND_TEXT.values())
    literals += list(vp._REJECT_TEXT.values())
    return literals


def test_every_card_literal_has_an_english_entry():
    missing = [s for s in _card_literals() if s not in I._EN]
    assert not missing, f"missing from core/i18n._EN: {missing}"


def test_no_card_literal_is_translated_to_itself():
    flat = [s for s in _card_literals() if I._EN[s] == s]
    assert not flat, f"English entry equals the Chinese source: {flat}"
