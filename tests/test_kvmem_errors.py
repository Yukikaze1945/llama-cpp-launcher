# -*- coding: utf-8 -*-
"""Stage 6: reading a kvmem server's own rejection (core/kvmem_errors.py).

Plan acceptance item 7 is the headline: put `--parallel 4` in 附加参数 and the
launcher must end up in a dialog that names `unknown flag: --parallel` — not a
"服务异常退出" the user has to decode.

Every input line below was captured from
``F:\\llama-kvmem-rc2\\bin\\llama-kvmem-server.exe`` (v0.16.0-rc2) by running it
with the offending argument and no model, e.g.

    $ llama-kvmem-server.exe --parallel 4      -> unknown flag: --parallel
    $ llama-kvmem-server.exe --port            -> missing value for --port
    $ llama-kvmem-server.exe -ctk q8_0 -ctv f16
        incompatible KV cache types: K=q8_0, V=f16; quantized K/V must match; ...

so these tests pin the classifier against what the shipped build really says,
not against a guess at its wording. The wording is also why the rules are
ordered: `unsupported MTP cache type` has to be tried before `unsupported cache
type` would otherwise be free to explain the wrong control.
"""
import re

import pytest

from core import engine as engine_mod
from core import i18n
from core import kvmem_errors as ke

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
#: An unfilled `t()` placeholder, e.g. "{flag}". The JSON-object copy contains
#: real braces, but those follow a quote rather than a bare identifier.
PLACEHOLDER_RE = re.compile(r"\{[a-z_]+\}")


@pytest.fixture(autouse=True)
def pin_zh():
    """The wording checks below quote Chinese fragments, so the module pins the
    language it runs its assertions in and gives whatever the session used back.
    """
    previous = i18n.get_language()
    i18n.set_language("zh")
    try:
        yield
    finally:
        i18n.set_language(previous)


def hit(text):
    """classify() a single line and insist that exactly one rule fires."""
    hits = ke.classify(text)
    assert len(hits) == 1, hits
    return hits[0]


# --------------------------------------------------------------------------
# the registry seam


def test_the_label_matches_the_engine_registry():
    """This module repeats KVMEM.display_name on purpose (it must stay Qt- and
    registry-free); the two are only allowed to agree by accident otherwise."""
    assert ke.ENGINE_LABEL == engine_mod.KVMEM.display_name
    assert "error_classification" in engine_mod.KVMEM.supports
    # llama.cpp never gets here: it has its own, older error story.
    assert "error_classification" not in engine_mod.LLAMA.supports


# --------------------------------------------------------------------------
# flag -> control


@pytest.mark.parametrize("flag,key", [
    ("-c", "ctx_size"),
    ("--ctx-size", "ctx_size"),
    ("--temp", "temperature"),
    ("-ctk", "cache_type_k"),
    ("--cache-type-k", "cache_type_k"),
    ("--spec-kv-dtype", "spec_kv_dtype"),
    ("--kvmem-recent-tokens", "kvmem_recent_tokens"),
    ("--no-think", "no_think"),
    ("--parallel", ""),          # llama-only
    ("--no-mmap", ""),           # llama-only
    ("--api-key", ""),           # neither engine's schema exposes it
    ("", ""),
    (None, ""),
])
def test_flag_to_key_resolves_every_alias(flag, key):
    assert ke.flag_to_key(flag) == key


def test_flag_to_key_covers_the_whole_schema():
    """A schema param whose flags were never registered would silently lose its
    jump target, so the table is checked against the table it comes from.

    Every param, not just the visible ones: a bool_neg param's `flag` is a
    [pos, neg] list and an emitter with no widget still reaches argv.
    """
    missing = []
    for param in engine_mod.KVMEM.schema.PARAMS:
        for flag in [param.flag, *(param.flags or ())]:
            for f in (flag if isinstance(flag, (list, tuple)) else [flag]):
                if isinstance(f, str) and f and ke.flag_to_key(f) != param.key:
                    missing.append((param.key, f))
    assert missing == []


# --------------------------------------------------------------------------
# acceptance 7: the llama flag in a kvmem launch


def test_unknown_llama_flag_names_the_flag_and_says_why():
    h = hit("unknown flag: --parallel")
    assert "--parallel" in h["message"]
    # The one mistake a dual-engine launcher makes constantly.
    assert "llama.cpp" in h["message"]
    assert ke.ENGINE_LABEL in h["message"]
    # Not a schema flag, so the only place it can have come from — and the
    # control the dialog then jumps to.
    assert h["param"] == "extra_args"
    assert h["line"] == "unknown flag: --parallel"


def test_the_acceptance_7_route_really_reaches_the_binary_verbatim():
    """The whole chain, without a server: 附加参数 is passed through unquoted, so
    what the user typed is exactly what the parser then rejects — and the report
    points back at that same box."""
    from ui.command_builder import CommandBuilder
    K = engine_mod.KVMEM.schema
    args = CommandBuilder(K.fallback_defaults(), params=K.PARAMS).build(
        dict(K.fallback_defaults(), extra_args="--parallel 4"))
    assert "--parallel" in args and "4" in args
    # This is the line the process writes for the argv above.
    h = hit("unknown flag: " + args[args.index("--parallel")])
    assert h["param"] == "extra_args"


@pytest.mark.parametrize("line", [
    "unknown flag: --frobnicate",
    "unknown flag: --kvmem-nvme-gb",
])
def test_an_unknown_flag_that_is_not_llamas_gets_no_wrong_hint(line):
    h = hit(line)
    assert "llama.cpp" not in h["message"]
    assert h["param"] == "extra_args"


# --------------------------------------------------------------------------
# one test per rule, each on the measured wording


@pytest.mark.parametrize("line,key,fragment", [
    ("missing value for --port", "port", "--port"),
    ("incompatible KV cache types: K=q8_0, V=f16; quantized K/V must match; "
     "set both -ctk and -ctv, or use --kv-dtype TYPE to set both",
     "cache_type_v", "q8_0"),
    ("unsupported MTP cache type (want f16|q8_0|q5_0|q4_0|f32)",
     "spec_kv_dtype", "--spec-kv-dtype"),
    ("unsupported cache type (want f16|f32|q8_0|q5_0|q4_0)",
     "cache_type_k", "bf16"),
    ("unsupported --spec-type draft-x (P7-0: draft-mtp|none)",
     "spec_type", "draft-x"),
    ("invalid --kvmem-recent-tokens (want >= 0)",
     "kvmem_recent_tokens", "--kvmem-recent-tokens"),
    ("invalid --kvmem-query-max-tokens (want > 0)",
     "kvmem_query_max_tokens", "--kvmem-query-max-tokens"),
    ("invalid --reasoning-budget: expected an integer >= -1",
     "reasoning_budget", "--reasoning-budget"),
    ("cannot enforce reasoning_budget_tokens without an end-of-thinking token",
     "reasoning_budget", "-1"),
    ("--chat-template-kwargs requires a JSON object",
     "chat_template_kwargs", "JSON"),
    ("--image-min-tokens requires a positive integer",
     "image_min_tokens", "正整数"),
    ("cannot read chat template: D:/nope.tmpl",
     "chat_template_file", "D:/nope.tmpl"),
    ("invalid chat template", "chat_template", "Jinja"),
    ("failed to load mmproj: D:/x.gguf", "mmproj", "D:/x.gguf"),
    ("failed to load model", "model", "GGUF"),
])
def test_each_rule_names_its_control(line, key, fragment):
    h = hit(line)
    assert h["param"] == key, h
    assert fragment in h["message"], h


def test_nvme_rejection_has_no_control_to_point_at():
    """Deliberately param-less: the plan forbids an NVMe control in this build,
    so the dialog explains and does not jump."""
    h = hit("NVMe offload is disabled in this build (KVMEM_ENABLE_NVME=OFF)")
    assert h["param"] == ""
    assert "NVMe" in h["message"]


def test_the_mtp_rule_wins_over_the_plain_cache_type_rule():
    """Both wordings contain "unsupported" and both name cache types; only the
    MTP one belongs to --spec-kv-dtype."""
    hits = ke.classify(
        "unsupported MTP cache type (want f16|q8_0|q5_0|q4_0|f32)\n"
        "unsupported cache type (want f16|f32|q8_0|q5_0|q4_0)")
    assert [h["param"] for h in hits] == ["spec_kv_dtype", "cache_type_k"]


def test_missing_path_or_value_still_produces_a_message():
    # `cannot read chat template:` with nothing after it, and an mmproj failure
    # that names no file, must not hand the user "…：" and nothing else.
    assert hit("cannot read chat template:")["message"]
    assert hit("failed to load mmproj")["message"]


# --------------------------------------------------------------------------
# classify() as a whole-blob reader


def test_lines_are_explained_in_stream_order():
    blob = "\n".join([
        "loading model from F:\\models\\qwen3.8-27b-iq3.gguf",
        "unknown flag: --parallel",
        "something the launcher has never seen",
        "missing value for --port",
    ])
    hits = ke.classify(blob)
    assert [h["param"] for h in hits] == ["extra_args", "port"]
    assert hits[0]["line"] == "unknown flag: --parallel"


def test_one_explanation_per_line_even_when_several_rules_fit():
    h = hit("unknown flag: --parallel; failed to load model")
    assert h["param"] == "extra_args"      # the most specific rule comes first


def test_a_repeated_line_is_explained_once():
    assert len(ke.classify("unknown flag: --parallel\n" * 4)) == 1
    # Different flags are different problems, even with the same wording.
    assert len(ke.classify("unknown flag: --parallel\nunknown flag: --ub")) == 2


def test_every_recognised_line_comes_back_shaped_the_same():
    blob = "\n".join([
        "unknown flag: --parallel", "missing value for --port",
        "unsupported cache type (want f16|f32|q8_0|q5_0|q4_0)",
        "invalid --kvmem-query-max-tokens (want > 0)",
        "--image-max-tokens requires a positive integer",
        "failed to load model",
    ])
    hits = ke.classify(blob)
    assert len(hits) == 6
    for h in hits:
        assert set(h) == {"param", "message", "line"}
        assert h["message"]
        assert h["line"]
        if h["param"]:
            assert h["param"] in engine_mod.KVMEM.schema.PARAMS_BY_KEY, h


def test_a_long_line_is_cut_but_still_explained():
    h = hit("cannot read chat template: " + "x" * 600)
    assert len(h["line"]) == 400
    assert h["param"] == "chat_template_file"


def test_nothing_recognised_means_no_report():
    """An empty list is the caller's "show the normal error path" signal."""
    for text in ("", "\n\n   \n", "0.0.2.084.780 I sdv", "batch_size = 512",
                 "error: failed to initialize backend",
                 "0.0.0.000.026 W load_backend: ggml-backend-..: failed to load "
                 "'cublas', reason 'The specified module could not be found'"):
        assert ke.classify(text) == []


@pytest.mark.parametrize("junk", [
    None, 0, 12345, 3.5, True, b"unknown flag: --parallel", ["--parallel"],
    {"a": "unknown flag"}, "\x00\x1f", object(),
    # U+2028/U+2029 and \x0b are *line* separators to str.splitlines() while
    # staying invisible in a log viewer: neither may take the explanation down.
    "\u2028unknown flag: --parallel\u2029", "\x0bmissing value for --port",
])
def test_classify_never_raises(junk):
    """A mis-parsed log must not become a second failure on top of the one it
    is explaining, so the contract is "returns a list", not "returns answers"."""
    out = ke.classify(junk)
    assert isinstance(out, list)


# --------------------------------------------------------------------------
# language


def test_english_mode_leaks_no_chinese():
    """The classifier builds its strings through t(), and a missing entry is
    invisible until an English user reads the dialog: check the whole table.

    Raw server tokens (flags, paths) are ASCII, so any CJK left in a message is
    untranslated copy rather than quoted output.
    """
    i18n.set_language("en")
    blob = "\n".join([
        "unknown flag: --parallel", "missing value for --port",
        "incompatible KV cache types: K=q8_0, V=f16; quantized K/V must match",
        "unsupported MTP cache type (want f16|q8_0|q5_0|q4_0|f32)",
        "unsupported cache type (want f16|f32|q8_0|q5_0|q4_0)",
        "unsupported --spec-type draft-x (P7-0: draft-mtp|none)",
        "invalid --kvmem-recent-tokens (want >= 0)",
        "invalid --kvmem-query-max-tokens (want > 0)",
        "invalid --reasoning-budget: expected an integer >= -1",
        "cannot enforce reasoning_budget_tokens without an end-of-thinking token",
        "--chat-template-kwargs requires a JSON object",
        "--image-min-tokens requires a positive integer",
        "cannot read chat template: D:/nope.tmpl",
        "cannot read chat template:",
        "invalid chat template",
        "NVMe offload is disabled in this build (KVMEM_ENABLE_NVME=OFF)",
        "failed to load mmproj:", "failed to load model",
    ])
    hits = ke.classify(blob)
    assert len(hits) == len(blob.splitlines()), hits
    leaked = [h["message"] for h in hits if CJK_RE.search(h["message"])]
    assert leaked == []
    # A t() call whose kwargs were forgotten: the braces survive into the dialog.
    half = [h["message"] for h in hits if PLACEHOLDER_RE.search(h["message"])]
    assert half == []


def test_chinese_mode_is_the_default_wording():
    i18n.set_language("zh")
    assert CJK_RE.search(hit("unknown flag: --parallel")["message"])
