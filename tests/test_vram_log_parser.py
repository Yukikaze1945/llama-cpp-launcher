# -*- coding: utf-8 -*-
"""Reading kvmem-llama.cpp's own allocation report, and handing it to layer 1.

`tests/test_kvmem_log_parsing.py` covers the readiness line; this file covers the
five memory lines. The format strings below are copied from the rc2 binary's own
output, so a pattern that stops matching means the engine changed its wording —
which is exactly the drift worth failing on, because the plan puts these numbers
above the launcher's arithmetic ("日志里的实际 allocation 优先用于下一次预测").

The second half tests `kvmem_actual_alloc()`, the only bridge into
core.vram_estimator: rates (row bytes, slot bytes, card size) may outlive the
run, totals (kv_bytes, mtp_bytes) describe the one budget that ran and must not.
"""
import pytest

from core import vram_estimator as ve
from ui.log_parser import kvmem_actual_alloc, parse_log_line

GIB = ve.GIB

KV_BYTES_LINE = (
    "0.10 I kvmem: KVMEM_KV_BYTES bytes=1853882368 cells=53248 slots=416 "
    "budget=36864 pool=53248 ratio=0.50 high=0.45 low=0.35 cap_blocks=1920 "
    "gpu_total=34359738368 block_bytes=4456448")
MTP_POOL_LINE = (
    "0.11 I kvmem: KVMEM_TRACE mtp_pool cells=53248 target_cells=53248 "
    "n_ctx=262144 bytes=218103808 layers=1 block_tokens=128 type_k=f16 "
    "type_v=f16 k_row_bytes=2048 v_row_bytes=2048 v_trans=1")
SLOT_POOL_LINE = (
    "0.12 I kv_unified: load: KVMem slot-pool cells=53248 slots=416 "
    "block_tokens=128 budget=36864 gen_reserve=16384 sink_blocks=1 "
    "method=retrieval harvest_v=0 type_k=q8_0 type_v=q8_0 n_embd_k=1024 "
    "attn_layers=16")
GDN_ALLOC_LINE = ("0.13 I kvmem: KVMEM_GDN_ALLOCATION mode=hybrid "
                  "recurrent_bytes=4194304 conv_bytes=1048576 rollback_bytes=524288")
GDN_MEM_LINE = ("0.14 I kvmem: KVMEM_GDN_MEMORY mode=replay layers=28 "
                "capacity=4096 state_bytes=117440512 record_bytes=8388608 "
                "descriptor_bytes=262144")

ALL_KVMEM = [KV_BYTES_LINE, MTP_POOL_LINE, SLOT_POOL_LINE, GDN_ALLOC_LINE, GDN_MEM_LINE]


def _parse(lines):
    info = {}
    for line in lines:
        parse_log_line(line, info)
    return info


def _alloc():
    return _parse(ALL_KVMEM)["kvmem_alloc"]


# --------------------------------------------------------------------------
# the five lines
# --------------------------------------------------------------------------

def test_kv_bytes_line_reports_the_main_pool_in_numbers_not_display_strings():
    a = _parse([KV_BYTES_LINE])["kvmem_alloc"]
    assert a["kv_bytes"] == 1853882368
    assert a["pool_cells"] == 53248 and a["line_cells"] == 53248
    assert a["slots"] == 416 and a["budget"] == 36864
    assert a["slot_bytes"] == 4456448
    assert a["gpu_total_bytes"] == 34359738368
    assert a["cap_slots"] == 1920
    assert (a["ratio"], a["high"], a["low"]) == (0.5, 0.45, 0.35)


def test_mtp_pool_line_carries_its_row_sizes_and_dtype():
    a = _parse([MTP_POOL_LINE])["kvmem_alloc"]
    assert a["mtp_bytes"] == 218103808
    assert (a["mtp_cells"], a["mtp_target_cells"]) == (53248, 53248)
    assert a["mtp_layers"] == 1 and a["mtp_block_tokens"] == 128
    assert (a["mtp_k_row_bytes"], a["mtp_v_row_bytes"]) == (2048, 2048)
    assert (a["mtp_type_k"], a["mtp_type_v"]) == ("f16", "f16")
    assert a["mtp_v_trans"] == 1


def test_slot_pool_line_is_where_attn_layers_and_the_reserve_come_from():
    a = _parse([SLOT_POOL_LINE])["kvmem_alloc"]
    assert a["pool_line_cells"] == 53248 and a["pool_line_slots"] == 416
    assert a["pool_block_tokens"] == 128
    assert a["pool_budget"] == 36864 and a["pool_gen_reserve"] == 16384
    assert a["pool_attn_layers"] == 16 and a["pool_n_embd_k"] == 1024
    assert (a["pool_type_k"], a["pool_type_v"]) == ("q8_0", "q8_0")
    assert a["pool_method"] == "retrieval" and a["pool_sink_slots"] == 1


def test_gdn_lines_are_captured_as_bytes():
    a = _parse([GDN_ALLOC_LINE, GDN_MEM_LINE])["kvmem_alloc"]
    assert a["gdn_mode"] == "hybrid"
    assert a["gdn_recurrent_bytes"] == 4194304
    assert a["gdn_conv_bytes"] == 1048576 and a["gdn_rollback_bytes"] == 524288
    assert a["gdn_layers"] == 28 and a["gdn_capacity"] == 4096
    assert a["gdn_state_bytes"] == 117440512
    assert a["gdn_record_bytes"] == 8388608
    assert a["gdn_descriptor_bytes"] == 262144


def test_all_five_lines_share_one_alloc_dict():
    assert set(_alloc()) >= {"kv_bytes", "mtp_bytes", "pool_block_tokens",
                             "gdn_state_bytes", "slot_bytes"}


def test_a_repeated_report_wins_over_the_earlier_one():
    # kvmem re-states the pool when it resizes; the newest number is the truth.
    resized = KV_BYTES_LINE.replace("bytes=1853882368", "bytes=999999") \
                          .replace("slots=416", "slots=222")
    a = _parse([KV_BYTES_LINE, resized])["kvmem_alloc"]
    assert a["kv_bytes"] == 999999 and a["slots"] == 222


def test_a_truncated_line_stores_what_it_matched_and_nothing_more():
    partial = "0.10 I kvmem: KVMEM_KV_BYTES bytes=1853882368 cells=53248"
    assert "kvmem_alloc" not in _parse([partial])   # the pattern needs the whole line
    assert partial not in _parse([partial])


def test_unrelated_lines_are_not_invented_into_an_alloc_report():
    for line in ("srv  load_model: n_ctx = 262144",
                 "ggml_backend_cuda0: graph pipeline error",
                 "kvmem: NVMe namespace not found"):
        assert "kvmem_alloc" not in _parse([line])


# --------------------------------------------------------------------------
# the llama.cpp engine's own lines must come out unchanged
# --------------------------------------------------------------------------

LLAMA_LINES = [
    "load_tensors:          CPU_Mapped model buffer size =  1024.50 MiB",
    "load_tensors:             CUDA0 model buffer size = 11545.60 MiB",
    "llama_kv_cache:         CUDA0 KV buffer size =  9579.50 MiB",
    "sched_reserve:          CUDA0 compute buffer size =   36.14 MiB",
    "sched_reserve:     CPU_Mapped compute buffer size =    8.00 MiB",
]


def test_the_kvmem_lines_do_not_touch_llamas_display_keys():
    alone = _parse(LLAMA_LINES)
    mixed = _parse(ALL_KVMEM + LLAMA_LINES)
    assert alone["model_bufs"] == mixed["model_bufs"] == {"CPU_Mapped": 1024.5,
                                                          "CUDA0": 11545.6}
    assert alone["kv_cache_total"] == mixed["kv_cache_total"] == 9579.5
    assert alone["compute_bufs"] == mixed["compute_bufs"] == {"CUDA0": 36.14,
                                                              "CPU_Mapped": 8.0}
    assert alone["model_vram"] == mixed["model_vram"] == "11545.60 MiB"


def test_llama_lines_never_produce_an_alloc_report():
    assert "kvmem_alloc" not in _parse(LLAMA_LINES)


# --------------------------------------------------------------------------
# kvmem_actual_alloc(): the bridge into the estimator
# --------------------------------------------------------------------------

# A real run logs the five kvmem reports *and* the buffer sizes above, so a test
# about the bridge reads a whole run — `model_vram_bytes` has no kvmem producer.
RUN_LINES = ALL_KVMEM + LLAMA_LINES


def test_rates_and_totals_are_handed_over_together_by_default():
    actual = kvmem_actual_alloc(_parse(RUN_LINES))
    assert actual["kv_bytes"] == 1853882368
    assert actual["mtp_bytes"] == 218103808
    assert actual["mtp_k_row_bytes"] == 2048
    assert actual["gpu_total_bytes"] == 34359738368
    assert actual["kv_per_token_bytes"] == 4456448 // 128 == 34816
    assert actual["model_vram_bytes"] == int(round(11545.60 * (1 << 20)))


def test_rate_only_drops_the_figures_that_belong_to_one_budget():
    actual = kvmem_actual_alloc(_parse(RUN_LINES), rate_only=True)
    assert "kv_bytes" not in actual and "mtp_bytes" not in actual
    assert actual["kv_per_token_bytes"] == 34816
    assert actual["mtp_v_row_bytes"] == 2048
    # mapped weights hold for any budget, so they are a rate and stay in
    assert actual["model_vram_bytes"] == int(round(11545.60 * (1 << 20)))


def test_a_slot_that_does_not_divide_by_the_block_is_refused_not_rounded():
    info = _parse(ALL_KVMEM)
    info["kvmem_alloc"]["slot_bytes"] = 4456449          # engine lines disagree
    assert "kv_per_token_bytes" not in kvmem_actual_alloc(info)
    info["kvmem_alloc"].pop("pool_block_tokens")
    assert "kv_per_token_bytes" not in kvmem_actual_alloc(info)


def test_cpu_only_weights_are_not_reported_as_vram():
    info = _parse(["load_tensors:          CPU_Mapped model buffer size = 1024.50 MiB"])
    assert "model_vram_bytes" not in kvmem_actual_alloc(info)
    assert kvmem_actual_alloc(info) == {}


def test_an_empty_or_broken_info_gives_an_empty_actual():
    assert kvmem_actual_alloc({}) == {}
    assert kvmem_actual_alloc({"kvmem_alloc": {"kv_bytes": 0, "slots": -3}}) == {}
    assert kvmem_actual_alloc({"kvmem_alloc": {"ratio": 0.5}}) == {}


def test_booleans_and_zeroes_never_count_as_measurements():
    info = {"kvmem_alloc": {"kv_bytes": True, "mtp_bytes": 0,
                            "mtp_k_row_bytes": 2048}}
    assert kvmem_actual_alloc(info) == {"mtp_k_row_bytes": 2048}


# --------------------------------------------------------------------------
# log truth through the whole chain
# --------------------------------------------------------------------------

def _geo():
    return ve.ModelGeometry(arch="qwen35", block_count=64, attn_layers=16,
                            n_k=1024, n_v=1024, n_embd=2048, mtp_layers=1,
                            n_k_mtp=1024, n_v_mtp=1024, has_nextn=True,
                            data_span_bytes=11_220_000_000)


@pytest.mark.parametrize("wrong_geometry", [
    dict(n_k=512, n_v=512),        # a head count read from the wrong key
    dict(attn_layers=64),          # every layer, not just the attention ones
])
def test_the_logged_slot_rate_overrules_a_misread_gguf(wrong_geometry):
    actual = kvmem_actual_alloc(_parse(RUN_LINES))
    est = ve.estimate(ve.VramInputs(
        geometry=_geo(), ctx_size=262144, budget=36864, gen_reserve=16384,
        block_tokens=128, type_k="q8_0", type_v="q8_0", spec_type="draft-mtp",
        spec_kv_dtype="f16", gpu_total_bytes=16 * GIB, actual=actual))
    assert est.kv_bytes == 1853882368          # the logged total, verbatim
    assert est.mtp_bytes == 218103808
    assert est.kv_per_token_bytes == 34816
    assert est.sources["kv_row_bytes"] == "log_slot_bytes"
    assert est.sources["weight"] == "log_model_buffer"
    assert est.weight_bytes == int(round(11545.60 * (1 << 20)))


def test_the_next_prediction_after_a_different_budget_keeps_the_rate_not_the_total():
    # A finished run's totals must not freeze the pool size for a new budget:
    # the sweep uses use_actual=False, and the *stored* rate carries on.
    alloc = _parse(RUN_LINES)
    rates = kvmem_actual_alloc(alloc, rate_only=True)
    inp = ve.VramInputs(geometry=_geo(), ctx_size=262144, budget=73728,
                        gen_reserve=16384, block_tokens=128, type_k="q8_0",
                        type_v="q8_0", spec_type="draft-mtp", spec_kv_dtype="f16",
                        gpu_total_bytes=16 * GIB, actual=rates)
    est = ve.estimate(inp)
    assert est.kv_bytes == (73728 + 16384) * 34816
    assert est.kv_per_token_bytes == 34816
    assert est.sources["kv_bytes"] == "model"
    assert est.sources["weight"] == "log_model_buffer"
