# -*- coding: utf-8 -*-
"""Stage 3: per-engine --help parsing and version-drift detection (kvmem).

The kvmem binary never writes `default:` (llama.cpp's token) — it writes
`(default 2048)`, sometimes with prose after it, sometimes on a wrapped
continuation line. Reusing llama.cpp's extractor unchanged would find zero
defaults and freeze the baseline at whatever our table says, so the drift check
would always answer "in sync". These tests pin the kvmem rules instead:

  * parsing the captured help of v0.16.0-rc2 reproduces the schema baseline
    exactly (the table and the binary agree — no drift on the current build);
  * a changed number/word in the help *is* picked up, per value kind, so a
    future build's new default surfaces instead of being silently ignored;
  * keys we refuse to adopt (NO_DRIFT_KEYS) keep the baseline even when the
    help offers a parseable value;
  * mid-parenthetical prose ("(llama.cpp name; default q8_0)") is not a default.
"""
import inspect
import os

import pytest

from core import defaults as LLAMA
from core import defaults_kvmem as D
from core import kvmem_params_schema as K

KVMEM_SERVER_EXE = os.environ.get("KVMEM_SERVER_EXE", "")
HELP = open(os.path.join(os.path.dirname(__file__), "data",
                         "kvmem_help_v0.16.0-rc2.txt"), encoding="utf-8").read()


def _help_with(*replacements):
    """(old, new, old, new, ...) edits on the captured fixture."""
    text = HELP
    for old, new in zip(replacements[0::2], replacements[1::2]):
        assert old in text, f"fixture no longer contains: {old!r}"
        text = text.replace(old, new)
    return text


def test_golden_help_reproduces_the_baseline():
    assert D._parse_help_to_defaults(HELP) == K.fallback_defaults()


def _mentions_default(key):
    """Does the help entry for this param say anything about a default?"""
    p = K.PARAMS_BY_KEY[key]
    if not p.flags:
        return False
    return any(f in line and "default" in line.lower() for line in HELP.split("\n")
               for f in p.flags)


def test_golden_help_actually_yields_defaults():
    """Guard against the silent failure this whole module exists to avoid: a
    regex that matches nothing also 'passes' the test above."""
    assert "default:" not in HELP
    parsed = D._parse_help_to_defaults(
        _help_with("(default 2048)", "(default 999999)"))
    assert parsed["ctx_size"] == 999999
    # --help really does talk about defaults for a large part of the table.
    offered = {k for k in K.PARAMS_BY_KEY if _mentions_default(k)}
    assert len(offered) >= 20, sorted(offered)


def test_drift_is_detected_per_value_kind():
    base = K.fallback_defaults()
    parsed = D._parse_help_to_defaults(_help_with(
        "(default 2048)", "(default 8192)",            # int  (ctx_size)
        "(default 0.50)", "(default 0.80)",            # float(kvmem_gpu_ratio)
        "(default 127.0.0.1)", "(default 0.0.0.0)",    # str  (host)
        "(default retrieval)", "(default recency)",    # str  (kvmem_method)
        "(default off; RAM until NVMe flush)", "(default on)",  # bool (harvest_v)
    ))
    assert parsed["ctx_size"] == 8192 and base["ctx_size"] == 2048
    assert parsed["kvmem_gpu_ratio"] == 0.80
    assert parsed["host"] == "0.0.0.0"
    assert parsed["kvmem_method"] == "recency"
    assert parsed["kvmem_harvest_v"] is True


def test_wrapped_default_on_the_continuation_line_is_joined():
    """--kvmem-query-max-tokens and --reasoning-budget wrap: the (default X)
    token sits on the continuation line, so a line-local parser misses it."""
    parsed = D._parse_help_to_defaults(_help_with(
        "from the end of the span (default 512; qw3-style)",
        "from the end of the span (default 1024)",
        "N>0 force </think> after N think tokens (default -1)",
        "N>0 force </think> after N think tokens (default 2048)",
    ))
    assert parsed["kvmem_query_max_tokens"] == 1024
    assert parsed["reasoning_budget"] == 2048


def test_dual_valued_default_is_not_adopted_and_falls_back():
    """`(default 512; qw3-style)` is not an int: the value must stay at the
    table's 512 rather than become garbage or raise."""
    parsed = D._parse_help_to_defaults(HELP)
    assert parsed["kvmem_query_max_tokens"] == 512


def test_no_drift_keys_are_never_adopted():
    parsed = D._parse_help_to_defaults(_help_with(
        "(default 8080)", "(default 19000)",           # port
        "(default random)", "(default 42)",             # seed
    ))
    assert parsed["port"] == 8080, "a moved baseline would stop us sending --port"
    assert parsed["seed"] == -1


def test_mid_parenthetical_prose_is_not_a_default():
    """`-ctk, --cache-type-k TYPE  GPU K cache type (llama.cpp name; default
    q8_0)` — the trailing token describes the naming, and our empty value means
    'follow --kv-dtype', which --help cannot express."""
    assert _mentions_default("cache_type_k")
    parsed = D._parse_help_to_defaults(HELP)
    assert parsed["cache_type_k"] == "" and parsed["cache_type_v"] == ""
    # And it stays "" even when a future build writes it in the parseable form,
    # because cache_type_k is in NO_DRIFT_KEYS.
    parsed2 = D._parse_help_to_defaults(_help_with(
        "(llama.cpp name; default q8_0)", "(default f16)"))
    assert parsed2["cache_type_k"] == ""


def test_help_wording_is_normalised_to_a_selectable_value():
    assert "(default replay with MTP)" in HELP
    parsed = D._parse_help_to_defaults(HELP)
    assert parsed["kvmem_mtp_state"] == "replay"
    items = list(K.PARAMS_BY_KEY["kvmem_mtp_state"].items)
    assert parsed["kvmem_mtp_state"] in items, "combo could not select the parsed value"
    for key, mapping in D._KVMEM_DEFAULT_NORMALIZE.items():
        assert key in K.PARAMS_BY_KEY, key
        assert mapping, key


def test_placeholder_wording_becomes_the_empty_string():
    assert "--reasoning-budget-message MSG  injected before forced </think> (default none)" in HELP
    parsed = D._parse_help_to_defaults(HELP)
    assert parsed["reasoning_budget_message"] == ""
    assert parsed["reasoning_budget_message"] == K.PARAMS_BY_KEY["reasoning_budget_message"].default


def test_empty_help_falls_back_without_raising():
    assert D._parse_help_to_defaults("") == K.fallback_defaults()


def test_get_default_params_never_probes_an_ungiven_path(monkeypatch):
    """The launcher's single configured path may point at llama-server.exe;
    kvmem must only ever run a binary it was explicitly handed."""
    calls = []

    def spy(server_path=None):
        calls.append(server_path)
        return ""

    monkeypatch.setattr(LLAMA, "fetch_help_text", spy)
    assert D.get_default_params() == K.fallback_defaults()
    assert calls == []
    text = _help_with("(default 2048)", "(default 32768)")
    assert D.get_default_params(help_text=text)["ctx_size"] == 32768
    assert calls == []
    assert D.get_default_params(server_path="F:/somewhere/llama-kvmem-server.exe") \
        == K.fallback_defaults()
    assert calls == ["F:/somewhere/llama-kvmem-server.exe"]


def test_unreachable_binary_falls_back():
    """A missing exe must not raise into the startup worker (llama-side
    behaviour: _run_server_command swallows FileNotFoundError)."""
    assert D.get_default_params(server_path="Z:/nope/llama-kvmem-server.exe") \
        == K.fallback_defaults()


def test_public_surface_matches_the_llama_engine():
    """Stage 4's engine registry calls either engine through the same names."""
    for name in ("_FALLBACK_DEFAULTS", "_parse_help_to_defaults",
                 "get_default_params", "get_chat_templates", "parse_device_list",
                 "USER_INPUT_PARAMS"):
        assert hasattr(D, name), name
    assert set(inspect.signature(D.get_default_params).parameters) == \
        set(inspect.signature(LLAMA.get_default_params).parameters)
    assert D.get_chat_templates(help_text=HELP) == []
    assert D.parse_device_list(HELP) == []
    assert set(D._FALLBACK_DEFAULTS) == set(K.PARAMS_BY_KEY)
    assert set(D.USER_INPUT_PARAMS) <= set(K.PARAMS_BY_KEY)


def test_llama_engine_parsing_is_untouched_by_the_shared_core():
    """The refactor moved the loop into parse_help_to_defaults(); llama.cpp's
    wrapper must still read its own `default:` token and nothing else — and
    kvmem's help must not be able to move llama.cpp's baseline."""
    assert LLAMA._parse_help_to_defaults(HELP) == dict(LLAMA._FALLBACK_DEFAULTS)
    text = "  -c, --ctx-size N   size of the prompt context (default: 4096)"
    assert LLAMA._parse_help_to_defaults(text)["ctx_size"] == 4096
    assert D._parse_help_to_defaults(text) == K.fallback_defaults()


@pytest.mark.skipif(not KVMEM_SERVER_EXE, reason="needs the kvmem binary")
def test_real_binary_help_reproduces_the_baseline():
    """End-to-end: the captured fixture is not the only thing we tested against.
    kvmem prints --help on stderr and exits 1, which fetch_help_text tolerates."""
    assert D.get_default_params(KVMEM_SERVER_EXE) == K.fallback_defaults()
