# -*- coding: utf-8 -*-
"""Stage 6: reading a kvmem run's output (ui/log_parser.py + the level filter).

The kvmem server is not llama-server: it writes its own printf-style
diagnostics, so the runtime-info panel and the "ready" status — which are all
driven by patterns written against llama-server's `HH:MM:SS.mmm L srv …` lines —
would stay empty for a successful kvmem run. The fix is deliberately one rule,
on the one line that says the server is up and serving, taken verbatim from the
string table of the rc2 binary:

    llama-kvmem-server listening on http://%s:%d  model=%s kvmem=%d method=%s
        n_ctx=%d spec=%s n_max=%d think=%d rbudget=%d qmax=%d

Everything else about a kvmem run keeps the llama.cpp parser's behaviour, and
the last test here is the one that matters for the llama engine: an unrelated
line must still not be claimed by the new rule.
"""
from ui.log_parser import line_level, parse_log_line

# The format string above, filled in the way v0.16.0-rc2 would print it for the
# launcher's own recipe (port 18200, ctx 262144, MTP draft on).
KVMEM_LISTENING = ("llama-kvmem-server listening on http://127.0.0.1:18200  "
                   "model=qwen3.8-27b-q3k.gguf kvmem=1 method=retrieval "
                   "n_ctx=262144 spec=draft-mtp n_max=2 think=1 rbudget=-1 "
                   "qmax=512")

LLAMA_LISTENING = ("0.0.5.123.456 I srv  srv_start: With specified configs "
                   "server is listening on 127.0.0.1:8080")


def test_the_kvmem_ready_line_sets_status_and_context():
    info = {}
    assert parse_log_line(KVMEM_LISTENING, info) is True
    assert info["status"] == "✅ 服务就绪"
    # The same thousands separator the llama.cpp rules use (ui/log_parser.py:156).
    assert info["ctx_size"] == "262,144"


def test_the_kvmem_rule_survives_the_fast_rejection_set():
    """parse_log_line skips every pattern when no anchor matches; "listening"
    alone is an anchor already, but this line must reach the handler anyway."""
    info = {}
    lower = KVMEM_LISTENING.lower()
    assert "n_ctx=" in lower and "kvmem=" in lower
    assert parse_log_line(KVMEM_LISTENING, info) is True


def test_a_kvmem_line_has_no_level_so_it_stays_visible():
    """line_level() None = shown in every filter view — the reason the window
    hides the level selector instead of leaving a control that cannot narrow."""
    assert line_level(KVMEM_LISTENING) is None
    assert line_level("unknown flag: --parallel") is None
    # llama.cpp's own prefix still parses, i.e. the engine that has the
    # selector keeps getting real levels.
    assert line_level(LLAMA_LISTENING) == "I"


def test_the_llama_ready_line_still_works():
    info = {}
    assert parse_log_line(LLAMA_LISTENING, info) is True
    assert info["status"] == "✅ 服务就绪"


def test_the_new_rule_does_not_claim_llama_lines():
    """A llama context line carries `n_ctx` but never `kvmem=`; whatever it does
    or does not read, it must not be made to look like a ready server."""
    for line in ("0.0.9.876.543 I llama_context: n_ctx  = 131072 (131072)",
                 "0.0.1.100.000 I srv    operator(): n_ctx_slot = 4096"):
        info = {}
        parse_log_line(line, info)
        assert info.get("status") != "✅ 服务就绪", line


def test_the_port_does_not_become_the_context():
    """`http://127.0.0.1:18200` precedes n_ctx=; a greedy pattern would happily
    read the port as the context size."""
    info = {}
    parse_log_line(KVMEM_LISTENING, info)
    assert info["ctx_size"] != "18,200"


def test_an_unrelated_kvmem_line_is_not_invented():
    """The parse-time rejections are the error classifier's job, not the
    runtime-info panel's: nothing here may look like a loaded model."""
    info = {}
    for line in ("unsupported cache type (want f16|f32|q8_0|q5_0|q4_0)",
                 "NVMe offload is disabled in this build (KVMEM_ENABLE_NVME=OFF)",
                 "KVMEM_TRACE mtp_pool cells=64 target_cells=64 n_ctx=262144 "
                 "bytes=1048576 layers=48 block_tokens=32 type_k=q8_0 type_v=q8_0"):
        assert parse_log_line(line, info) is False, line
    assert info == {}
