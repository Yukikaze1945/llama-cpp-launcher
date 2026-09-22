# -*- coding: utf-8 -*-
"""Command-line emission for the kvmem engine (CommandBuilder + KVMEM_PARAMS).

Pins the two behaviours the plan calls out as acceptance criteria:
  * the official recipe really reaches argv (notably `--port 18200`, plan §5.1);
  * an untouched page emits nothing at all (plan §5.2), so no sampling flag can
    leak into the command line as a value the user never chose.
"""
import pytest

from core import kvmem_params_schema as K
from ui.command_builder import CommandBuilder

SAMPLING_FLAGS = ["--temperature", "--top-p", "--top-k", "--min-p",
                  "--presence-penalty", "--frequency-penalty",
                  "--repeat-penalty", "--seed"]


@pytest.fixture
def baseline():
    return K.fallback_defaults()


@pytest.fixture
def cb(baseline):
    return CommandBuilder(baseline, params=K.PARAMS)


def recipe_state(baseline):
    """What MainWindow.params looks like right after switching to kvmem: the
    binary baseline seeded with INITIAL_OVERRIDES plus a model path."""
    v = dict(baseline)
    v.update(K.INITIAL_OVERRIDES)
    v["model"] = "D:/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
    return v


def test_untouched_state_emits_nothing(cb, baseline):
    assert cb.build(dict(baseline)) == []


def test_untouched_state_emits_no_sampling_flag(cb, baseline):
    argv = cb.build(dict(baseline))
    assert not [f for f in SAMPLING_FLAGS if f in argv]


def test_recipe_emits_the_official_iq3_command_line(cb, baseline):
    argv = cb.build(recipe_state(baseline))
    model = argv[argv.index("-m") + 1]
    assert model.endswith("IQ3_S-mtp.gguf")
    # Order is part of the contract: --kv-dtype must precede -ctk/-ctv, and the
    # port must really be there (is_default() would drop a 18200 *default*).
    assert [a for a in argv if a.startswith("--port")] == ["--port"]
    assert argv[argv.index("--port") + 1] == "18200"
    assert argv[argv.index("-c") + 1] == "262144"
    assert argv[argv.index("-n") + 1] == "16384"
    assert argv[argv.index("--kvmem-budget") + 1] == "36864"
    assert argv[argv.index("--kvmem-gen-reserve") + 1] == "16384"
    assert argv[argv.index("--spec-type") + 1] == "draft-mtp"
    assert argv[argv.index("--reasoning-budget") + 1] == "4096"
    assert "--enable-thinking" in argv and "--no-think" not in argv
    assert not [f for f in SAMPLING_FLAGS if f in argv]


def test_recipe_golden_argv(cb, baseline):
    v = recipe_state(baseline)
    v["model"] = "M.gguf"
    assert cb.build(v) == [
        "-m", "M.gguf",
        "-c", "262144",
        "-n", "16384",
        "--kvmem-budget", "36864",
        "--kvmem-gen-reserve", "16384",
        "--spec-type", "draft-mtp",
        "--enable-thinking",
        "--reasoning-budget", "4096",
        "--port", "18200",
    ]


def test_every_recipe_value_differs_from_the_binary_default(baseline):
    """The mirror of the is_default() gate: a recipe entry that equalled the
    binary default would silently never be sent."""
    for key, val in K.INITIAL_OVERRIDES.items():
        assert baseline[key] != val, key


@pytest.mark.parametrize("key,flag", [
    ("temperature", "--temperature"), ("top_p", "--top-p"), ("top_k", "--top-k"),
    ("min_p", "--min-p"), ("presence_penalty", "--presence-penalty"),
    ("frequency_penalty", "--frequency-penalty"),
    ("repeat_penalty", "--repeat-penalty"), ("seed", "--seed"),
])
def test_setting_a_sampling_value_sends_it(cb, baseline, key, flag):
    v = dict(baseline)
    p = K.PARAMS_BY_KEY[key]
    v[key] = p.min if p.default != p.min else 1
    v[key] = 0.7 if p.fmt == "f2" else 7
    argv = cb.build(v)
    i = argv.index(flag)
    assert i >= 0 and argv[i + 1] == ("0.70" if p.fmt == "f2" else "7")
    assert not cb.is_default(key, v)


def test_harvest_v_only_emits_when_checked(cb, baseline):
    off = dict(baseline)
    assert "--kvmem-harvest-v" not in cb.build(off)
    on = dict(baseline, kvmem_harvest_v=True)
    assert cb.build(on) == ["--kvmem-harvest-v"]


def test_bool_neg_flags_emit_one_way_only(cb, baseline):
    # mmproj offload defaults to GPU: checked -> nothing, unchecked -> the negation
    assert cb.build(dict(baseline)) == []
    assert cb.build(dict(baseline, mmproj_offload=False)) == ["--no-mmproj-offload"]
    assert cb.build(dict(baseline, kvmem_enabled=False)) == ["--no-kvmem"]


def test_kv_dtype_precedes_explicit_cache_types(cb, baseline):
    # q8_0 is the binary default, so it must be left unsent; use a non-default
    # --kv-dtype to check the ordering the parser's conflict rule depends on.
    v = dict(baseline, kv_dtype="f32", cache_type_k="f32", cache_type_v="f32")
    argv = cb.build(v)
    assert argv.index("--kv-dtype") < argv.index("-ctk") < argv.index("-ctv")
    assert cb.build(dict(baseline, kv_dtype="q8_0")) == []
    # leaving them empty must not emit -ctk/-ctv at all
    assert "-ctk" not in cb.build(dict(baseline, kv_dtype="f32"))


def test_spec_type_none_is_skipped(cb, baseline):
    assert "--spec-type" not in cb.build(dict(baseline, spec_type="none"))
    assert cb.build(dict(baseline, spec_type="draft-mtp")) == ["--spec-type", "draft-mtp"]


def test_ui_dir_and_extra_args(cb, baseline):
    assert cb.build(dict(baseline, ui_dir="F:/ui")) == ["--ui-dir", "F:/ui"]
    assert cb.build(dict(baseline, extra_args="--no-ui\n--host 0.0.0.0")) == \
        ["--no-ui", "--host", "0.0.0.0"]


def test_llama_only_flags_cannot_appear_through_the_schema(cb, baseline):
    """Even with every kvmem control at a non-default value, no rejected llama.cpp
    flag is emitted (the schema is the only source of flags)."""
    v = {p.key: (p.min if p.widget in ("spin", "dspin") and p.min is not None
                 else p.default) for p in K.PARAMS}
    v.update({"model": "M.gguf", "mmproj": "P.gguf", "ui_dir": "U",
              "chat_template": "t", "chat_template_kwargs": "{}",
              "reasoning_effort": "none", "reasoning_budget_message": "m",
              "cache_type_k": "q8_0", "cache_type_v": "q8_0",
              "extra_args": "", "thinking_mode": 2, "enable_thinking": True,
              "no_think": True, "kvmem_harvest_v": True, "kvmem_enabled": False,
              "mmproj_offload": False, "no_ui": True, "spec_type": "draft-mtp"})
    argv = cb.build(v)
    for flag in K.REJECTED_FLAGS:
        assert flag not in argv, flag
    assert "-ub" not in argv and "--flash-attn" not in argv
