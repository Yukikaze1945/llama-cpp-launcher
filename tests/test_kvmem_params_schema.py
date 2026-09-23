# -*- coding: utf-8 -*-
"""Structural tests for the kvmem engine schema (core/kvmem_params_schema.py).

These mirror the invariants tests/test_params_schema.py pins for the llama.cpp
engine — the second schema has to hold the same lines or the shared UI code
(panels, presets, i18n) starts behaving differently per engine.
"""
import json
import re

import pytest

from core import i18n as I
from core import kvmem_params_schema as K

_CJK = re.compile(r"[\u4e00-\u9fff]")

# Emit modes and widget kinds CommandBuilder._emit / AdvancedPanel._build_param_row
# actually implement. The kvmem engine must not need a new one (hard rule).
KNOWN_EMIT = {
    "diff", "diff_pos", "diff_ge0", "diff_nonempty", "diff_skip", "diff_join",
    "bool_pos", "bool_neg", "truthy", "prio", "index", "ngl", "ngl_draft",
    "none", "extra",
}
KNOWN_WIDGET = {
    "text", "password", "spin", "dspin", "check", "combo", "combo_index",
    "combo_edit", "file", "dir", "dir_text", "list", "checklist", "mtext",
    None,
}
# The "do not send" sentinels of the sampling page (plan §5.2).
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty",
                 "frequency_penalty", "repeat_penalty", "seed")


def test_schema_size_and_shape():
    assert len(K.PARAMS) == 52
    # enable_thinking / no_think are hidden emitters (the tri-state control is
    # thinking_mode), so the UI renders 50 rows.
    assert len(K.UI_PARAMS) == 50
    assert len(K.PARAMS_BY_KEY) == len(K.PARAMS)
    assert len(K.fallback_defaults()) == len(K.PARAMS)


def test_keys_and_wattrs_unique():
    keys = [p.key for p in K.PARAMS]
    assert len(keys) == len(set(keys))
    wattrs = [p.wattr for p in K.UI_PARAMS]
    assert len(wattrs) == len(set(wattrs))
    # A widget attribute must follow its key, or getattr(self, p.wattr) in the
    # panel reads another parameter's control.
    for p in K.UI_PARAMS:
        assert p.wattr == "adv_" + p.key or p.key == "chat_template_kwargs", p.key


def test_emit_modes_and_widgets_are_existing_ones():
    assert {p.emit for p in K.PARAMS} <= KNOWN_EMIT
    assert {p.widget for p in K.PARAMS} <= KNOWN_WIDGET


def test_tabs_are_consistent():
    assert set(K.TAB_ORDER) == {key for key, _ in K.TAB_TITLES}
    assert set(p.tab for p in K.PARAMS) == set(K.TAB_ORDER)
    # every tab key is one the llama engine does not use for something else
    # (AdvancedPanel special-cases tab_key == "model" and "gpu").
    assert "gpu" not in K.TAB_ORDER


def test_rows_unique_and_dense_per_tab():
    for tab in K.TAB_ORDER:
        rows = [p.row for p in K.tab_params(tab)]
        assert rows == sorted(rows)
        assert len(rows) == len(set(rows))
        assert None not in rows
    # hidden emitters carry no row
    for p in K.PARAMS:
        if p.wattr is None:
            assert p.row is None and p.label is None and p.widget is None


def test_defaults_are_the_binary_defaults():
    """`default` is the binary's value, so is_default() suppresses exactly the
    flags the server would have chosen anyway (plan §5.1)."""
    d = K.fallback_defaults()
    assert d["port"] == 8080          # not 18200 — see INITIAL_OVERRIDES
    assert d["ctx_size"] == 2048
    assert d["kvmem_block_tokens"] == 128
    assert d["kvmem_gen_reserve"] == 256
    assert d["kv_dtype"] == "q8_0"
    assert d["spec_type"] == "none"
    assert d["kvmem_mtp_state"] == "replay"
    assert d["reasoning_budget"] == -1
    assert d["mmproj_offload"] is True
    assert d["kvmem_enabled"] is True
    assert d["kvmem_harvest_v"] is False
    json.dumps(d, ensure_ascii=False)  # presets must stay serializable


def test_initial_overrides_actually_change_something():
    """A recipe value equal to the binary default would be silently dropped by
    is_default(); INITIAL_OVERRIDES only earns its keep by differing."""
    d = K.fallback_defaults()
    for key, val in K.INITIAL_OVERRIDES.items():
        assert key in d, key
        assert d[key] != val, f"INITIAL_OVERRIDES[{key}] equals the default"


def test_sampling_sentinels_are_reachable_not_clamped():
    """Plan §5.2: the Qt minimum equals the sentinel, so the untouched value can
    never be clamped into the legal range. The real range lives in LEGAL_RANGE."""
    for key in SAMPLING_KEYS:
        p = K.PARAMS_BY_KEY[key]
        assert p.default == p.min, f"{key}: sentinel must equal the spin minimum"
        if p.fmt == "f2":
            assert p.default < 0 or p.default == 0


def test_legal_range_keys_exist():
    for key in K.LEGAL_RANGE:
        assert key in K.PARAMS_BY_KEY, key
    # LEGAL_RANGE only covers params whose sentinel is outside the legal range
    for key in SAMPLING_KEYS:
        assert key in K.LEGAL_RANGE, key


def test_no_drift_keys_are_known():
    assert set(K.NO_DRIFT_KEYS) <= set(K.PARAMS_BY_KEY)
    assert "port" in K.NO_DRIFT_KEYS
    assert set(SAMPLING_KEYS) <= set(K.NO_DRIFT_KEYS)


def test_user_input_keys_are_known_and_disjoint_from_no_drift():
    """Two different exclusion reasons; a key must not need both."""
    assert set(K.USER_INPUT_PARAMS) <= set(K.PARAMS_BY_KEY)
    assert {"model", "mmproj", "ui_dir", "chat_template",
            "chat_template_kwargs", "extra_args"} <= set(K.USER_INPUT_PARAMS)
    assert not set(K.USER_INPUT_PARAMS) & set(K.NO_DRIFT_KEYS)


_VALUE_WIDGETS = {"text", "password", "spin", "dspin", "combo", "combo_index",
                  "combo_edit", "file", "dir", "dir_text", "list", "checklist",
                  "mtext"}


def test_value_flags_are_never_typed_as_switches():
    """help_flag_maps() buckets a param by `parser`: 'bool' means the flag is a
    switch, so it is indexed without a value type. Every control that holds a
    value names a value flag — the live probe (Tier 2) caught model / mmproj /
    ui_dir typed as switches, which would have made the --help index treat a
    path flag as a boolean."""
    for p in K.PARAMS:
        if p.flags and p.widget in _VALUE_WIDGETS:
            assert p.parser != "bool", f"{p.key}: {p.flags} takes a value"
        if p.widget == "check":
            assert p.parser == "bool", p.key
    value_map, flag_map, neg_map = K.help_flag_maps()
    for key in ("model", "mmproj", "ui_dir", "host", "port", "chat_template",
                "ctx_size", "temperature", "kvmem_budget"):
        assert key in value_map, key
        assert key not in flag_map and key not in neg_map, key
    for key in ("no_ui", "enable_thinking", "kvmem_harvest_v"):
        assert list(flag_map[key]) == [K.PARAMS_BY_KEY[key].flags[0]], key
    # Merged alias pairs are the neg bucket: one control, positive flag first
    # and its --no-* counterpart second (emit order matters for bool_neg).
    assert set(neg_map) == {p.key for p in K.PARAMS if p.emit == "bool_neg"}
    for key, flags in neg_map.items():
        p = K.PARAMS_BY_KEY[key]
        assert list(flags) == list(p.flags), key
        assert p.flag == list(flags), key
        assert len(flags) == 2 and flags[0].startswith("--") \
            and flags[1].startswith("--no-"), (key, flags)
    # UI-only / verbatim params stay invisible to --help parsing.
    for key in ("thinking_mode", "extra_args"):
        assert not K.PARAMS_BY_KEY[key].flags, key
        assert key not in value_map and key not in flag_map and key not in neg_map


def test_kv_dtype_is_emitted_before_explicit_cache_types():
    """kvmem resolves a K/V mismatch by argv order, so --kv-dtype must precede a
    -ctk/-ctv pair (probe: `--kv-dtype f32 -ctk q8_0` is fatal, the reverse
    order passes)."""
    keys = [p.key for p in K.PARAMS]
    assert keys.index("kv_dtype") < keys.index("cache_type_k")
    assert keys.index("cache_type_k") < keys.index("cache_type_v")


def test_alias_flags_collapse_into_single_controls():
    """CLI aliases share one control: they appear in `flags` (for --help
    indexing) but never as a second key."""
    assert K.PARAMS_BY_KEY["temperature"].flag == "--temperature"
    assert "--temp" in K.PARAMS_BY_KEY["temperature"].flags
    assert "--repetition-penalty" in K.PARAMS_BY_KEY["repeat_penalty"].flags
    assert "temp" not in K.PARAMS_BY_KEY          # llama's key name, not ours
    assert "presence_penalty" in K.PARAMS_BY_KEY
    assert K.PARAMS_BY_KEY["mmproj_offload"].emit == "bool_neg"
    assert set(K.PARAMS_BY_KEY["mmproj_offload"].flag) == {
        "--mmproj-offload", "--no-mmproj-offload"}


def test_unsupported_flags_have_no_control():
    """Multi-GPU / parallel / draft model / NVMe / jinja / logging flags must not
    be controllable at all (plan: 'help 里出现了' is not the criterion)."""
    schema_flags = set()
    for p in K.PARAMS:
        if isinstance(p.flag, str):
            schema_flags.add(p.flag)
        elif p.flag:
            schema_flags.update(p.flag)
        schema_flags.update(p.flags)
    for flag in K.REJECTED_FLAGS:
        assert flag not in schema_flags, f"{flag} is rejected by the parser"
    for flag in K.NO_CONTROL_KEYS:
        assert flag not in schema_flags, f"{flag} must stay control-free"
    assert not {"--kvmem-nvme-gb", "--kvmem-nvme-dir", "--kvmem-raw-k-nvme",
                "--jinja", "--parallel", "--model-draft", "-ub"} & set(schema_flags)


def test_harvest_v_is_a_real_control():
    """--kvmem-harvest-v harvests V into the host tier in this RAM-only build
    too, so it is an advanced switch — not something to hide or call inert."""
    p = K.PARAMS_BY_KEY["kvmem_harvest_v"]
    assert p.tab == "kvmem" and p.widget == "check" and p.emit == "bool_pos"
    text = I._EN[p.tooltip]
    assert "host" in text.lower() and "RAM" in text
    assert not re.search(r"(inert|no-op|without effect|没有作用|无意义)", text, re.I)


def test_chat_template_key_reuses_the_panel_special_case():
    """AdvancedPanel._build_param_row keys its dynamic-template combo on
    p.key == 'chat_template'; reusing that name is what keeps
    core/params_schema.py untouched."""
    assert K.PARAMS_BY_KEY["chat_template"].widget == "combo_edit"
    assert K.PARAMS_BY_KEY["chat_template"].wattr == "adv_chat_template"
    assert K.PARAMS_BY_KEY["chat_template_file"].emit == "diff_nonempty"


def test_thinking_tristate_uses_two_independent_emitters():
    tm = K.PARAMS_BY_KEY["thinking_mode"]
    assert tm.emit == "none" and tm.widget == "combo_index" and len(tm.items) == 3
    for key in ("enable_thinking", "no_think"):
        p = K.PARAMS_BY_KEY[key]
        assert p.wattr is None and p.emit == "bool_pos" and p.default is False


def test_reasoning_effort_items_are_the_levels_that_work():
    """`high` is llama.cpp's vocabulary; this build's ladder is different.

    Measured: the --help line reads "template effort; default uses template
    default, none disables thinking", and the shipped Qwen3.8 GSQ template
    raises for anything outside xhigh (its default) / medium / low. Before this
    list was fixed the schema offered `high` and the validator *rejected*
    `xhigh`, i.e. it blocked the only value that works.
    """
    p = K.PARAMS_BY_KEY["reasoning_effort"]
    assert list(p.items) == K.REASONING_EFFORT_ITEMS
    assert set(p.items) == {"none", "default", "low", "medium", "xhigh"}
    assert "high" not in p.items
    assert p.widget == "combo_edit"  # editable: the items suggest, they do not lock


@pytest.mark.parametrize("effort", ["default", "low", "medium", "xhigh",
                                    "minimal", "think_hard"])
def test_reasoning_effort_is_not_a_hard_whitelist(effort):
    """A custom chat template may define its own levels, and the parser accepts
    any string (`--reasoning-effort bogus` passes), so anything but the `none`
    collision has to reach the command line unchecked."""
    values = dict(K.fallback_defaults(), reasoning_effort=effort,
                  thinking_mode=K.THINKING_ON)
    assert "reasoning_effort" not in dict(K.validate_params(values))


def test_reasoning_effort_none_still_collides_with_thinking_on():
    """The one effort value the server acts on itself switches thinking off, so
    the pair stays a contradiction (rule 4) — that is the exception, not a
    whitelist."""
    values = dict(K.fallback_defaults(), reasoning_effort="none",
                  thinking_mode=K.THINKING_ON)
    assert "reasoning_effort" in dict(K.validate_params(values))
    for mode in (K.THINKING_TEMPLATE_DEFAULT, K.THINKING_OFF):
        values = dict(K.fallback_defaults(), reasoning_effort="none",
                      thinking_mode=mode)
        assert "reasoning_effort" not in dict(K.validate_params(values))


def test_schema_i18n_coverage():
    missing = [s for s in K.schema_i18n_strings()
               if _CJK.search(s) and s not in I._EN]
    assert not missing, f"missing _EN entries: {missing}"


def test_quick_keys_are_valid_for_the_shared_quick_pool():
    from ui.quick_params import quick_eligible, sanitize_quick_keys
    pool = {p.key for p in quick_eligible(K)}
    assert set(K.QUICK_DEFAULT_KEYS) <= pool
    # non-eligible and unknown keys are dropped; an emptied list falls back to
    # the engine's own default set (sanitize_quick_keys' documented contract).
    assert sanitize_quick_keys(["nope", ("x",), None], schema=K) == \
        list(K.QUICK_DEFAULT_KEYS)
    assert sanitize_quick_keys(["model"], schema=K) == list(K.QUICK_DEFAULT_KEYS)
    assert sanitize_quick_keys([], schema=K) == list(K.QUICK_DEFAULT_KEYS)
    # BASIC_OWNED_KEYS must name real kvmem keys, or the filter is a no-op
    assert set(K.BASIC_OWNED_KEYS) <= set(K.PARAMS_BY_KEY)
