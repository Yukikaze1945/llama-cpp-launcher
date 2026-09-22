# -*- coding: utf-8 -*-
"""Stage 6: the kvmem engine's control behaviour (ui/kvmem_linkage.py).

Everything here is the part of plan §5 that a parameter table cannot express:
the "do not send" sentinel (§5.2), KV pairing (§5.3), the thinking tri-state
(§5.4), chat-template exclusivity (§5.5), and gating/clamping (§5.6) — plus the
two acceptance items the plan attaches to them: an untouched sampling page emits
zero flags *through the widgets*, and switching language twice does not move a
single value.

The panel is built directly rather than through MainWindow: the hooks are the
subject, and a full window would add startup probe threads (see
tests/test_kvmem_engine_window.py for why those need a sandbox).
"""
import pytest

from core import engine as engine_mod
from core import i18n
from core import kvmem_params_schema as K
from core.i18n import t
from ui.advanced_panel import AdvancedPanel
from ui.command_builder import CommandBuilder
from ui import kvmem_linkage

SAMPLING_FLAGS = ["--temperature", "--top-p", "--top-k", "--min-p",
                  "--presence-penalty", "--frequency-penalty",
                  "--repeat-penalty", "--seed"]
SAMPLING_KEYS = ["temperature", "top_p", "top_k", "min_p", "presence_penalty",
                 "frequency_penalty", "repeat_penalty", "seed"]


@pytest.fixture
def panel():
    """A kvmem advanced panel with the linkage installed, as MainWindow builds it."""
    p = AdvancedPanel(defaults=K.fallback_defaults(),
                      chat_templates=[], schema=K)
    assert kvmem_linkage.apply_linkage(p, engine_mod.KVMEM) is True
    return p


@pytest.fixture
def cb():
    """A builder on this engine's baseline — the same is_default() gate the
    window uses, so argv assertions mean what the preview means."""
    return CommandBuilder(K.fallback_defaults(), params=K.PARAMS)


@pytest.fixture
def language():
    """Restore the UI language whatever a test switches it to."""
    previous = i18n.get_language()
    try:
        yield previous
    finally:
        i18n.set_language(previous)


def argv_for(panel, cb):
    baseline = K.fallback_defaults()
    state = dict(baseline)
    state.update(panel.get_values())
    # `model` is what makes this a startable command; it is not what is tested.
    state["model"] = "D:/models/m.gguf"
    return cb.build(state)


def flag_index(argv, flag):
    return argv.index(flag) if flag in argv else -1


# ---------------------------------------------------------------------------
# the seam itself
# ---------------------------------------------------------------------------

def test_linkage_is_inert_for_the_llama_panel():
    """A llama.cpp panel must not gain a single hook (plan §1's promise)."""
    from core import params_schema as L
    p = AdvancedPanel(defaults=L.fallback_defaults(), schema=L)
    assert kvmem_linkage.apply_linkage(p, engine_mod.LLAMA) is False
    assert kvmem_linkage.apply_linkage(p, engine_mod.KVMEM) is False
    assert (p._read_hooks, p._write_hooks, p._retranslate_extras) == ([], [], [])


def test_engine_capability_gates_the_call():
    """MainWindow asks the registry, never the schema module, whether to wire."""
    assert "param_linkage" in engine_mod.KVMEM.supports
    assert "param_linkage" not in engine_mod.LLAMA.supports


# ---------------------------------------------------------------------------
# §5.2 — the "do not send" sentinel
# ---------------------------------------------------------------------------

def test_untouched_panel_emits_no_sampling_flag(panel, cb):
    """Acceptance 6, first clause: never touching the page sends nothing."""
    argv = argv_for(panel, cb)
    assert not [f for f in SAMPLING_FLAGS if f in argv]


def test_sentinel_is_reachable_and_never_clamped(panel):
    """Acceptance 6, second clause: minimum() *is* the sentinel, so Qt cannot
    push an untouched value into the range the parser reads as a request."""
    for key in SAMPLING_KEYS + ["reasoning_budget"]:
        param = K.PARAMS_BY_KEY[key]
        widget = panel.param_widget(key)
        widget.setMinimum(param.min)          # what _build_param_row already did
        widget.setValue(param.min)
        assert widget.value() == pytest.approx(param.default), key
        assert panel.get_values()[key] == pytest.approx(param.default), key


def test_sentinel_carries_a_label_not_a_number(panel):
    """The value stays a number; only its display says what it means."""
    assert panel.param_widget("temperature").specialValueText() == t("引擎默认（不发送）")
    assert panel.param_widget("seed").specialValueText() == t("随机")
    assert panel.param_widget("reasoning_budget").specialValueText() == t("不限制（不发送）")


def test_special_value_text_survives_language_switch(panel, language):
    i18n.set_language("en")
    panel.retranslate_ui()
    english = panel.param_widget("temperature").specialValueText()
    assert english and "引擎" not in english
    i18n.set_language("zh")
    panel.retranslate_ui()
    assert panel.param_widget("temperature").specialValueText() == t("引擎默认（不发送）")


def test_typed_value_reaches_argv(panel, cb):
    """Acceptance 6, third clause: 0.7 the user types is 0.7 the server gets."""
    panel.param_widget("temperature").setValue(0.7)
    argv = argv_for(panel, cb)
    i = flag_index(argv, "--temperature")
    assert i >= 0 and argv[i + 1] == "0.70"
    assert not cb.is_default("temperature", panel.get_values())


def test_page_reset_returns_to_not_sending(panel, cb):
    """Acceptance 6, last clause: one click back to the sentinel state."""
    widget = panel.param_widget("temperature")
    widget.setValue(0.7)
    panel.param_widget("top_k").setValue(40)
    panel.param_widget("ctx_size").setValue(8192)
    assert "--temperature" in argv_for(panel, cb)

    panel._kvmem_reset_buttons["sampling"].click()
    argv = argv_for(panel, cb)
    assert not [f for f in SAMPLING_FLAGS if f in argv]

    panel._kvmem_reset_buttons["context"].click()
    assert panel.get_values()["ctx_size"] == K.PARAMS_BY_KEY["ctx_size"].default
    assert "--ctx-size" not in argv_for(panel, cb)


def test_page_reset_leaves_the_other_pages_alone(panel):
    """A reset is scoped to its page: the recipe on another tab survives."""
    panel.param_widget("kvmem_budget").setValue(24576)
    panel._kvmem_reset_buttons["sampling"].click()
    assert panel.get_values()["kvmem_budget"] == 24576


def test_model_and_server_pages_get_no_reset_button(panel):
    """Those pages hold paths, host and port — 'restore defaults' there would
    discard the user's setup rather than stop emitting flags."""
    assert set(panel._kvmem_reset_buttons) == set(kvmem_linkage.RESET_TABS)
    assert "model" not in panel._kvmem_reset_buttons
    assert "server" not in panel._kvmem_reset_buttons


# ---------------------------------------------------------------------------
# §5.3 — KV cache pairing
# ---------------------------------------------------------------------------

def test_quantized_k_pins_v(panel, cb):
    k = panel.param_widget("cache_type_k")
    v = panel.param_widget("cache_type_v")
    assert v.isEnabled()
    k.setCurrentText("q8_0")
    assert v.currentText() == "q8_0"
    assert not v.isEnabled()
    argv = argv_for(panel, cb)
    i = flag_index(argv, "-ctk")
    assert i >= 0 and argv[i + 1] == argv[flag_index(argv, "-ctv") + 1] == "q8_0"
    # the ordering rule from §5.3: --kv-dtype precedes an explicit pair
    assert flag_index(argv, "--kv-dtype") < i


def test_releasing_k_gives_v_back(panel):
    k = panel.param_widget("cache_type_k")
    v = panel.param_widget("cache_type_v")
    k.setCurrentText("f32")
    k.setCurrentText("")
    assert v.isEnabled()
    v.setCurrentText("q4_0")
    assert v.currentText() == "q4_0"


def test_kv_pairing_still_validates_after_the_linkage(panel):
    """The lock is not a substitute for the pre-start check (§5.3's last line):
    a value that arrives without going through the signal must still be caught.
    """
    values = K.fallback_defaults()
    values.update({"kv_dtype": "f32", "cache_type_k": "q8_0", "cache_type_v": "f16"})
    problems = dict(K.validate_params(values))
    assert "cache_type_v" in problems


# ---------------------------------------------------------------------------
# §5.4 — the thinking tri-state
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode,expect", [
    (K.THINKING_TEMPLATE_DEFAULT, []),
    (K.THINKING_ON, ["--enable-thinking"]),
    (K.THINKING_OFF, ["--no-think"]),
])
def test_thinking_combo_drives_exactly_one_flag(panel, cb, mode, expect):
    panel.param_widget("thinking_mode").setCurrentIndex(mode)
    argv = argv_for(panel, cb)
    got = [f for f in ("--enable-thinking", "--no-think") if f in argv]
    assert got == expect


def test_thinking_flags_are_derived_not_stored(panel):
    """No widget owns them, so get_values() is the only thing that keeps the
    pair in step with the combo the user actually touched."""
    values = panel.get_values()
    assert "enable_thinking" not in {p.key for p in K.UI_PARAMS}
    panel.param_widget("thinking_mode").setCurrentIndex(K.THINKING_OFF)
    values = panel.get_values()
    assert values["no_think"] is True and values["enable_thinking"] is False


def test_preset_of_raw_flags_moves_the_combo(panel):
    """Loading a preset writes flags (no control for them); the combo has to
    follow, or the next save would silently rewrite the user's choice."""
    panel.set_values({"no_think": True, "enable_thinking": False})
    assert panel.get_values()["thinking_mode"] == K.THINKING_OFF
    panel.set_values({"thinking_mode": K.THINKING_ON})
    assert panel.get_values()["no_think"] is False


def test_contradictory_pair_resolves_to_off(panel, cb):
    """A preset with both flags is a state the combo cannot express, so it came
    from a hand edit or an engine key collision. The restrictive reading wins,
    and — this is the point — the combo *shows* the resolution instead of
    deciding behind the user's back."""
    panel.set_values({"enable_thinking": True, "no_think": True})
    assert panel.param_widget("thinking_mode").currentIndex() == K.THINKING_OFF
    values = panel.get_values()
    assert values["no_think"] is True and values["enable_thinking"] is False
    argv = argv_for(panel, cb)
    assert "--no-think" in argv and "--enable-thinking" not in argv


# ---------------------------------------------------------------------------
# §5.5 — chat template exclusivity
# ---------------------------------------------------------------------------

def test_template_file_locks_the_inline_template(panel):
    tpl = panel.param_widget("chat_template")
    file_w = panel.param_widget("chat_template_file")
    assert tpl.isEnabled() and file_w.isEnabled()
    file_w.setText("D:/prompts/chat.jinja")
    assert not tpl.isEnabled() and file_w.isEnabled()
    file_w.setText("")
    assert tpl.isEnabled()


def test_inline_template_locks_the_file_row(panel):
    """The 浏览 button belongs to the row: a live button writing into a
    greyed-out field is worse than the field being greyed out."""
    from PyQt6.QtWidgets import QPushButton
    tpl = panel.param_widget("chat_template")
    file_w = panel.param_widget("chat_template_file")
    tpl.setEditText("{{ message.content }}")
    assert tpl.isEnabled() and not file_w.isEnabled()
    buttons = [b for b in file_w.parentWidget().findChildren(QPushButton)]
    assert buttons and all(not b.isEnabled() for b in buttons)


def test_a_preset_holding_both_stays_editable(panel):
    """Nothing is locked while both sides carry text — the validator refuses
    the start, and the user has to be able to reach one of the two fields."""
    tpl = panel.param_widget("chat_template")
    file_w = panel.param_widget("chat_template_file")
    panel.set_values({"chat_template": "abc", "chat_template_file": "D:/x.jinja"})
    assert tpl.isEnabled() and file_w.isEnabled()
    problems = dict(K.validate_params(panel.get_values()))
    assert "chat_template" in problems


# ---------------------------------------------------------------------------
# §5.6 — gating and clamping
# ---------------------------------------------------------------------------

def test_gating_never_changes_get_values(panel):
    """Plan §5.6's test: turning KVMem off must not drop the user's numbers
    from the preset they save — the controls grey out, they do not reset."""
    panel.set_values({"kvmem_budget": 24576, "kvmem_method": "recency",
                      "spec_type": "draft-mtp", "spec_draft_n_max": 5})
    before = panel.get_values()
    switch = panel.param_widget("kvmem_enabled")
    switch.setChecked(False)
    after = panel.get_values()
    # everything but the switch itself, which the test just moved
    assert {k: v for k, v in after.items() if k != "kvmem_enabled"} == \
           {k: v for k, v in before.items() if k != "kvmem_enabled"}
    assert after["kvmem_enabled"] is False
    assert not panel.param_widget("kvmem_budget").isEnabled()
    assert panel.param_widget("kv_dtype").isEnabled()   # not a kvmem_* param
    switch.setChecked(True)
    assert panel.param_widget("kvmem_budget").isEnabled()
    assert panel.get_values() == before


def test_spec_type_none_greys_out_the_spec_group(panel):
    spec = panel.param_widget("spec_type")
    spec.setCurrentText("none")
    assert not panel.param_widget("spec_draft_n_max").isEnabled()
    assert not panel.param_widget("spec_kv_dtype").isEnabled()
    assert spec.isEnabled()                             # the switch never gates itself
    spec.setCurrentText("draft-mtp")
    assert panel.param_widget("spec_draft_n_max").isEnabled()


def test_n_predict_ceiling_follows_ctx_size(panel):
    ctx, npr = panel.param_widget("ctx_size"), panel.param_widget("n_predict")
    ctx.setValue(4096)
    assert npr.maximum() == 4096
    # a preset that raises both must not be clamped against the *old* ceiling
    panel.set_values({"ctx_size": 262144, "n_predict": 16384})
    assert npr.value() == 16384 and ctx.value() == 262144


def test_n_predict_ceiling_never_eats_the_sentinel(panel):
    """-1 means "use the engine default" and must stay reachable at any ctx."""
    npr = panel.param_widget("n_predict")
    panel.param_widget("ctx_size").setValue(1)
    npr.setValue(-1)
    assert npr.value() == -1


# ---------------------------------------------------------------------------
# acceptance 9 — language switching must not move a value
# ---------------------------------------------------------------------------

def test_two_language_switches_keep_every_value(panel, language):
    panel.set_values({"temperature": 0.7, "top_k": 40, "ctx_size": 32768,
                      "kvmem_method": "recency", "thinking_mode": K.THINKING_ON,
                      "cache_type_k": "q5_0", "reasoning_effort": "high",
                      "no_ui": True, "chat_template_kwargs": '{"a": 1}'})
    before = panel.get_values()
    for lang in ("en", "zh", "en", "zh"):
        i18n.set_language(lang)
        panel.retranslate_ui()
    after = panel.get_values()
    assert after == before
    # and the derived pair still emits what the combo says
    assert "--enable-thinking" in argv_for(panel, CommandBuilder(
        K.fallback_defaults(), params=K.PARAMS))
