# -*- coding: utf-8 -*-
"""The kvmem engine's parameter *behaviour* — what the schema table cannot say.

`core/kvmem_params_schema.py` describes what each control is and when its flag
is emitted. Five things it cannot describe are installed here, on top of an
already-built `AdvancedPanel`, through the panel's neutral hook lists
(`_read_hooks` / `_write_hooks` / `_retranslate_extras`):

  * **sentinel wording** (§5.2): a sampling spin whose value *is* the "do not
    send" sentinel shows `引擎默认（不发送）` instead of `-1.00`, and a page
    button restores that state in one click. No value is ever rewritten, so an
    untouched panel still emits zero sampling flags.
  * **KV pairing** (§5.3): a quantized -ctk pins -ctv to the same type, because
    the binary aborts on a mismatched pair.
  * **the thinking tri-state** (§5.4): one combo in, two independent flags out.
  * **chat template exclusivity** (§5.5): `--chat-template` and
    `--chat-template-file` must not both reach argv.
  * **gating and clamping** (§5.6): `kvmem_*` / `spec_*` controls grey out with
    their switch, and `--n-predict` cannot exceed `--ctx-size`.

Two invariants every function here keeps, both pinned by
tests/test_kvmem_ui_linkage.py:

  1. **Gating never loses a value.** Everything is `setEnabled(False)`, never
     `setVisible(False)` and never a clear — unticking `--kvmem` must not drop
     the user's budget from the preset they save. `get_values()` returns an
     identical dict across any gate.
  2. **Nothing rewrites what the CommandBuilder sees**, except the documented
     thinking_mode -> enable_thinking/no_think derivation. A value the user
     typed reaches argv unchanged, or the stage-5 validator refuses to start.

llama.cpp never reaches this module: `Engine.supports` gates the call, and
`apply_linkage` re-checks the panel's schema module so a wrong call is inert.
"""
import logging

from PyQt6.QtWidgets import QPushButton, QLabel, QWidget, QHBoxLayout

from core.i18n import t
from core import kvmem_params_schema as schema

logger = logging.getLogger(__name__)

#: Spin whose minimum *is* the "do not send" sentinel -> label shown there.
#: Each entry must satisfy `Param.default == Param.min`: setSpecialValueText()
#: only fires at minimum(), so a drifted range would make the label unreachable
#: rather than wrong — and the user would keep reading "-1.00" as something
#: they can send. Checked here (warn + skip) and asserted by the schema test.
SPECIAL_VALUE_TEXTS = {
    "temperature": "引擎默认（不发送）",
    "top_p": "引擎默认（不发送）",
    "top_k": "引擎默认（不发送）",
    "min_p": "引擎默认（不发送）",
    "presence_penalty": "引擎默认（不发送）",
    "frequency_penalty": "引擎默认（不发送）",
    "repeat_penalty": "引擎默认（不发送）",
    "seed": "随机",
    "reasoning_budget": "不限制（不发送）",
}

#: Tabs that get a page-level "back to the binary's defaults" button. The model
#: and server tabs are deliberately excluded: their rows are paths, host and
#: port, where "restore defaults" would throw away the user's setup rather than
#: stop the page from contributing flags.
RESET_TABS = ("context", "sampling", "kvmem")

RESET_TEXT = "恢复引擎默认（不发送本页参数）"
RESET_TIP = ("把本页每一项参数恢复到服务器二进制自己的默认值。\n"
             "命令行只在取值与默认值不同时才发出参数，所以这一步之后本页不再贡献任何 flag。")
RESET_HINT = "本页参数等于服务器默认值时不会出现在命令行中"

#: (switch key, parameter prefix it owns). `kv_dtype` / `cache_type_*` / the
#: `--spec-*` family are not `kvmem_*` and do keep working with --no-kvmem, so
#: they stay live; N-grams and cache types under a disabled KVMem are still
#: emitted, which is what the binary does with them.
GATES = (
    ("kvmem_enabled", "kvmem_"),
    ("spec_type", "spec_"),
)


def _param(panel, key):
    return panel._PARAMS_BY_KEY.get(key)


def _widget(panel, key):
    """The control for a key, or None (hidden emitters, unknown keys)."""
    p = _param(panel, key)
    if p is None or not p.wattr:
        return None
    return getattr(panel, p.wattr, None)


def _row_buttons(widget):
    """Buttons sharing a row wrapper with `widget` (the 浏览 button of a file row).

    Identified by the wrapper's own layout — a row wrapper holds the field plus
    only push buttons. Anything else (a tab's form content, which *is* a
    QLineEdit's/QComboBox's parent for unwrapped rows and whose layout is full
    of labels) returns [], so a gate can never switch off the help buttons of a
    whole tab by accident.
    """
    parent = widget.parentWidget()
    if parent is None:
        return []
    layout = parent.layout()
    if layout is None:
        return []
    children = [layout.itemAt(i).widget() for i in range(layout.count())]
    children = [c for c in children if c is not None]
    if widget not in children:
        return []
    others = [c for c in children if c is not widget]
    return others if all(isinstance(c, QPushButton) for c in others) else []


def _set_row_enabled(panel, key, enabled):
    """Enable/disable one row (field and its buttons) without hiding it."""
    w = _widget(panel, key)
    if w is None:
        return
    w.setEnabled(enabled)
    for b in _row_buttons(w):
        b.setEnabled(enabled)


def _text_of(widget):
    """The widget's textual value, for both line edits and combos."""
    if hasattr(widget, "currentText"):
        return str(widget.currentText()).strip()
    return str(widget.text()).strip()


# ---------------------------------------------------------------------------
# 1. sentinel wording + page reset (§5.2)
# ---------------------------------------------------------------------------

def apply_special_value_texts(panel):
    """Label each "do not send" sentinel inside its spin box."""
    applied = []
    for key, text in SPECIAL_VALUE_TEXTS.items():
        p, w = _param(panel, key), _widget(panel, key)
        if p is None or w is None or not hasattr(w, "setSpecialValueText"):
            continue
        if p.default != p.min:
            logger.warning("kvmem linkage: %s 的哨兵值 %r 不是最小值 %r，跳过哨兵文案",
                           key, p.default, p.min)
            continue
        w.setSpecialValueText(t(text))
        applied.append((w, text))

    def retranslate(_panel):
        for w, text in applied:
            w.setSpecialValueText(t(text))

    panel._retranslate_extras.append(retranslate)


def _page_defaults(panel, tab_key):
    """The binary's own value for every control of one page."""
    return {p.key: p.default for p in schema.tab_params(tab_key)
            if p.widget and p.wattr}


def add_page_reset_buttons(panel):
    """One button per parameter page: write the binary defaults back into it."""
    buttons = getattr(panel, "_kvmem_reset_buttons", None)
    if buttons is None:
        buttons = panel._kvmem_reset_buttons = {}
    for tab_key in RESET_TABS:
        if not schema.tab_params(tab_key):
            continue
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(2, 0, 2, 0)
        hint = QLabel(t(RESET_HINT), row)
        hint.setStyleSheet("color: #6b7280; font-size: 11px;")
        btn = QPushButton(t(RESET_TEXT), row)
        btn.setToolTip(t(RESET_TIP))
        btn.setFixedHeight(24)
        btn.clicked.connect(
            lambda _=False, k=tab_key: panel.set_values(_page_defaults(panel, k)))
        lay.addWidget(hint)
        lay.addStretch()
        lay.addWidget(btn)
        if not panel.add_tab_header(tab_key, row):
            row.deleteLater()
            continue
        buttons[tab_key] = btn          # the row owns the widget; this is a name
        panel._retranslate_extras.append(
            lambda _p, b=btn, h=hint: (b.setText(t(RESET_TEXT)),
                                       b.setToolTip(t(RESET_TIP)),
                                       h.setText(t(RESET_HINT))))


# ---------------------------------------------------------------------------
# 2. KV cache pairing (§5.3)
# ---------------------------------------------------------------------------

def wire_kv_linkage(panel):
    """A quantized K pins V to the same type — the parser aborts on a mismatch."""
    k = _widget(panel, "cache_type_k")
    v = _widget(panel, "cache_type_v")
    if k is None or v is None:
        return
    v_param = _param(panel, "cache_type_v")

    def refresh():
        kv = _text_of(k)
        if kv in schema.QUANT_CACHE_TYPES:
            if _text_of(v) != kv:
                v.setCurrentText(kv)     # the only value this pairing accepts
            v.setEnabled(False)
            v.setToolTip(t("K 为量化类型时 V 必须与之一致"))
        else:
            v.setEnabled(True)
            v.setToolTip(t(v_param.tooltip) if v_param.tooltip else "")

    k.currentTextChanged.connect(refresh)
    refresh()
    panel._retranslate_extras.append(lambda _p: refresh())


# ---------------------------------------------------------------------------
# 3. the thinking tri-state (§5.4)
# ---------------------------------------------------------------------------

def wire_thinking_tristate(panel):
    """One combo in, two independent flags out — in both directions."""

    def read(_panel, values):
        mode = values.get("thinking_mode", schema.THINKING_TEMPLATE_DEFAULT)
        values["enable_thinking"] = mode == schema.THINKING_ON
        values["no_think"] = mode == schema.THINKING_OFF

    def write(_panel, values):
        if "thinking_mode" in values:
            mode = values["thinking_mode"]
        elif "enable_thinking" in values or "no_think" in values:
            # A preset carrying the flags but no combo state (or a key
            # collision with the other engine): derive the combo from them, so
            # the control shows what will really be sent. "Both set" is a state
            # this combo cannot produce, and the restrictive reading (thinking
            # off) wins — the important part is that the resolution is visible
            # in the combo rather than decided on the way to argv.
            mode = (schema.THINKING_OFF if values.get("no_think")
                    else schema.THINKING_ON if values.get("enable_thinking")
                    else schema.THINKING_TEMPLATE_DEFAULT)
            values["thinking_mode"] = mode
        else:
            return
        values["enable_thinking"] = mode == schema.THINKING_ON
        values["no_think"] = mode == schema.THINKING_OFF

    panel._read_hooks.append(read)
    panel._write_hooks.append(write)


# ---------------------------------------------------------------------------
# 4. chat template exclusivity (§5.5)
# ---------------------------------------------------------------------------

def wire_chat_template_exclusion(panel):
    """Grey out the empty side of the --chat-template / --chat-template-file pair.

    Only the *empty* side is ever disabled: a preset that carries both (which
    the validator refuses at start) still shows both values, so the user can
    clear one instead of staring at a locked box.
    """
    tpl = _widget(panel, "chat_template")
    file_w = _widget(panel, "chat_template_file")
    if tpl is None or file_w is None:
        return

    def refresh():
        has_tpl = bool(_text_of(tpl))
        has_file = bool(_text_of(file_w))
        _set_row_enabled(panel, "chat_template", not (has_file and not has_tpl))
        _set_row_enabled(panel, "chat_template_file", not (has_tpl and not has_file))

    for w, sig in ((tpl, "currentTextChanged"), (file_w, "textChanged")):
        signal = getattr(w, sig, None)
        if signal is not None:
            signal.connect(refresh)
    refresh()


# ---------------------------------------------------------------------------
# 5. gating and clamping (§5.6)
# ---------------------------------------------------------------------------

def _gate_is_on(panel, switch_key):
    """Is the switch in the state where its parameters mean something?"""
    switch = _widget(panel, switch_key)
    if switch is None:
        return True
    if hasattr(switch, "isChecked"):
        return bool(switch.isChecked())
    p = _param(panel, switch_key)
    inactive = p.skip_values if p is not None else ()
    return _text_of(switch) not in (inactive or ())


def wire_gates(panel):
    """setEnabled(False) on the controls a switch turns off — never setVisible."""
    for switch_key, prefix in GATES:
        if _widget(panel, switch_key) is None:
            continue
        gated = [key for key in panel._PARAMS_BY_KEY
                 if key.startswith(prefix) and key != switch_key
                 and _param(panel, key).wattr]

        def refresh(_=None, own=switch_key, keys=gated):
            on = _gate_is_on(panel, own)
            for key in keys:
                _set_row_enabled(panel, key, on)

        switch = _widget(panel, switch_key)
        for sig in ("toggled", "currentTextChanged"):
            signal = getattr(switch, sig, None)
            if signal is not None:
                signal.connect(refresh)
        refresh()


def wire_n_predict_clamp(panel):
    """--n-predict cannot be raised above --ctx-size (the server rejects it)."""
    ctx = _widget(panel, "ctx_size")
    npr = _widget(panel, "n_predict")
    npr_param = _param(panel, "n_predict")
    if ctx is None or npr is None:
        return

    def tooltip():
        base = t(npr_param.tooltip) if npr_param.tooltip else ""
        extra = t("上限随 --ctx-size 自动夹逼（当前上限 {n}）", n=npr.maximum())
        return f"{base}\n{extra}" if base else extra

    def clamp(value=None):
        try:
            top = int(ctx.value() if value is None else value)
        except (TypeError, ValueError):
            return
        npr.setMaximum(max(npr.minimum(), top))
        npr.setToolTip(tooltip())

    def before_write(_panel, values):
        # Widen the ceiling *before* any widget write: a preset that raises both
        # values would otherwise clamp n_predict against the old ctx.
        if "ctx_size" in values:
            try:
                npr.setMaximum(max(npr.minimum(), int(values["ctx_size"])))
            except (TypeError, ValueError):
                pass

    ctx.valueChanged.connect(clamp)
    panel._write_hooks.append(before_write)
    panel._retranslate_extras.append(lambda _p: npr.setToolTip(tooltip()))
    clamp()


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def apply_linkage(panel, engine=None):
    """Install every kvmem behaviour on a panel built from the kvmem schema.

    Returns True when the hooks went in. Calling it for another engine's panel
    does nothing at all, so the llama.cpp panel keeps its original behaviour.
    """
    if engine is not None and getattr(engine, "id", None) != schema.ENGINE_ID:
        return False
    if getattr(panel, "_schema", None) is not schema:
        return False
    apply_special_value_texts(panel)
    add_page_reset_buttons(panel)
    wire_thinking_tristate(panel)
    wire_kv_linkage(panel)
    wire_chat_template_exclusion(panel)
    wire_gates(panel)
    wire_n_predict_clamp(panel)
    return True
