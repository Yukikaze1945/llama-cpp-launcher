# -*- coding: utf-8 -*-
"""Layer 1: the physical VRAM model, and the kvmem pool rules it stands on.

`compute_pool()` is a replica of kvmem-llama.cpp v0.16.0-rc2
`kvmem_compute_pool()`, so the numbers here are the numbers the engine prints —
a rule that drifts from the binary is a wrong suggestion, not a style problem.
The golden set comes from the plan's worked example (16 attention layers,
n_k = n_v = 1024, main KV q8_0, MTP KV f16, block 128), where the engine's own
report says the pool costs 1.7266 GiB and one F16 MTP layer 208 MiB.

Two pairs of numbers are deliberately separated and pinned below:
`engine_slot_bytes` (main KV only — what the gpu_ratio cap is charged against)
and `total_pool_slot_bytes` (KV + MTP — what VRAM actually costs).
"""
import math

import pytest

from core import vram_estimator as ve

GIB = ve.GIB
MIB = ve.MIB


# --------------------------------------------------------------------------
# fixtures: the plan's worked example
# --------------------------------------------------------------------------

def _geo(**over):
    base = dict(arch="qwen35", block_count=64, attn_layers=16, n_k=1024, n_v=1024,
                n_embd=2048, mtp_layers=1, n_k_mtp=1024, n_v_mtp=1024,
                has_nextn=True, data_span_bytes=10 * GIB,
                estimated_nbytes=int(10.7 * GIB))
    base.update(over)
    return ve.ModelGeometry(**base)


def _inp(**over):
    base = dict(geometry=_geo(), ctx_size=262144, batch_size=512, n_gpu_layers=-1,
                kvmem_enabled=True, budget=36864, gen_reserve=16384,
                block_tokens=128, gpu_ratio=0.50,
                gpu_total_bytes=16 * GIB, type_k="q8_0", type_v="q8_0",
                spec_type="draft-mtp", spec_kv_dtype="f16")
    base.update(over)
    return ve.VramInputs(**base)


def _pool(**over):
    base = dict(budget=36864, gen_reserve=16384, block_tokens=128, n_ctx=262144,
                gpu_ratio=0.50, gpu_total_bytes=None,
                kv_per_token_bytes=34816, mtp_per_token_bytes=4096)
    base.update(over)
    return ve.compute_pool(ve.PoolRequest(**base))


# --------------------------------------------------------------------------
# ggml row sizes
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name,n,expected", [
    ("f16", 1024, 2048),
    ("f32", 1024, 4096),
    ("q8_0", 1024, 1088),          # ceil(1024/32) * 34
    ("q8_0", 100, 136),            # a partial block still costs a whole block
    ("q5_0", 1024, 704),           # ceil(1024/32) * 22
    ("q4_0", 1024, 576),           # ceil(1024/32) * 18
])
def test_row_bytes_match_ggml_block_layout(name, n, expected):
    assert ve.ggml_row_bytes(name, n) == expected


@pytest.mark.parametrize("name,n", [("f16", 0), ("f16", -8), ("nosuchtype", 1024)])
def test_row_bytes_none_instead_of_guessing(name, n):
    assert ve.ggml_row_bytes(name, n) is None


def test_kv_per_token_is_layers_times_k_plus_v():
    assert ve.kv_per_token_bytes(16, "q8_0", 1024, "q8_0", 1024) == 34816
    assert ve.kv_per_token_bytes(0, "q8_0", 1024, "q8_0", 1024) == 0
    assert ve.kv_per_token_bytes(16, "q8_0", 0, "q8_0", 1024) == 0


# --------------------------------------------------------------------------
# kvmem_align_tokens()
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tokens,bt,expected", [
    (36864, 128, 36864),           # already whole
    (36000, 128, 35968),           # floor, not round and not ceil
    (36900, 128, 36864),
    (127, 128, 128),               # below one block -> one block, never zero
    (0, 128, 128),
    (1000, 2048, 2048),
])
def test_align_tokens_floors_with_a_one_block_floor(tokens, bt, expected):
    assert ve.align_tokens(tokens, bt) == expected


def test_align_tokens_with_a_broken_block_size_does_nothing():
    assert ve.align_tokens(1000, 0) == 1000
    assert ve.align_tokens(1000, -1) == 1000


# --------------------------------------------------------------------------
# compute_pool(): rc2's exact combination rule
# --------------------------------------------------------------------------

def test_golden_pool_and_its_two_slot_sizes():
    p = _pool()
    assert (p.pool_cells, p.slots) == (53248, 416)
    assert p.alloc_cells == 53248
    assert p.engine_slot_bytes == 4456448          # 128 * 34816, main KV only
    assert p.total_pool_slot_bytes == 4980736      # 128 * (34816 + 4096)
    assert not p.clamped


def test_budget_and_reserve_are_aligned_separately():
    # rc2 aligns each operand on its own: 36000 -> 35968, 16000 -> 16000.
    # Aligning the *sum* would give 51840 cells / 405 slots instead.
    p = _pool(budget=36000, gen_reserve=16000)
    assert (p.pool_cells, p.slots) == (51968, 406)


def test_reserve_of_zero_falls_back_to_the_binaries_default():
    p = _pool(budget=128, gen_reserve=0)
    assert p.pool_cells == 384 == 128 + ve.KVMEM_DEFAULT_GEN_RESERVE


def test_discriminator_reserve_alignment_is_not_ceiled():
    # 36064 -> 35968 and 200 -> 128: the pool is 36096, not 36064+200=36264
    # and not one align of the sum (36224 -> 36096 would agree here, which is
    # why the 200-reserve case has to be pinned next to the 16000 one above).
    p = _pool(budget=36064, gen_reserve=200)
    assert p.pool_cells == 36096
    assert p.slots == 282


def test_an_explicit_budget_is_not_capped_by_the_context():
    # rc2 sizes the pool from the budget, not from -c: `--kvmem-budget 40000
    # --kvmem-gen-reserve 256 -c 32768` logs budget=39936 cells=40192. Only the
    # budget==0 path clamps to n_ctx, so clamping here would under-charge the KV
    # pool by the whole difference.
    p = _pool(budget=40000, gen_reserve=256, n_ctx=32768)
    assert p.pool_cells == 40192 and p.slots == 314
    assert not p.clamped


def test_budget_zero_means_whole_context_and_rounds_up_to_slots():
    # The n_ctx clamp is what makes --kvmem-budget 0 mean "everything":
    # align(3000)+256 = 3200 would exceed the context, so cells land on 3000 —
    # a cell count that is NOT a multiple of block_tokens, hence ceil slots.
    p = _pool(budget=0, n_ctx=3000)
    assert p.pool_cells == 3000
    assert p.slots == 24                     # ceil(3000/128)
    assert p.alloc_cells == 3072
    assert p.pool_cells % p.block_tokens != 0


def test_budget_zero_without_n_ctx_reports_instead_of_dividing_by_zero():
    p = _pool(budget=0, n_ctx=0)
    assert (p.pool_cells, p.slots) == (0, 0)
    assert "no_n_ctx" in p.notes


def test_disabled_kvmem_pools_the_context_untouched():
    p = _pool(kvmem_enabled=False, budget=36864, n_ctx=3000)
    assert p.pool_cells == 3000
    assert p.notes == ("kvmem_disabled",)


def test_gpu_ratio_cap_is_counted_in_slots_and_charges_main_kv_only():
    # cap_slots = 0.50 * 16 GiB // 4456448 = 1927 — an MTP-inclusive divisor
    # (4980736) would give 1722, so this pins which basis rc2 uses.
    p = _pool(budget=1_000_000, gen_reserve=16384, n_ctx=262144,
              gpu_total_bytes=16 * GIB, gpu_ratio=0.50)
    assert p.cap_slots == 1927
    assert p.clamped and p.slots == 1927
    assert p.pool_cells == 1927 * 128
    assert p.requested_slots > p.slots


def test_no_cap_without_a_card_size_or_a_ratio():
    assert _pool(budget=1_000_000).cap_slots is None
    assert _pool(gpu_ratio=0.0, budget=1_000_000).cap_slots is None


def test_cap_not_triggered_leaves_clamped_false():
    p = _pool(gpu_total_bytes=16 * GIB, gpu_ratio=0.50)     # 416 << 1927
    assert p.slots == 416 and not p.clamped


# --------------------------------------------------------------------------
# compute_pool(): the same cases run against the real binary
#
# Probed on 2026-09-24 against F:\llama-kvmem-rc2\bin\llama-kvmem-server.exe
# (rc2, RTX 5060 Ti, gpu_total=17074421760) with
# Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf at -ctk/-ctv q8_0 — 16 attention layers of
# n_embd_k=1024 and one MTP layer of f16/1024, which is where the 34816 / 4096
# byte-per-token rates below come from. Every expected number is one the engine
# itself printed; these are the regression tests for "replicate rc2, do not
# redesign the pool rules".
# --------------------------------------------------------------------------

PROBE_GPU_TOTAL = 17074421760


def _probe(**over):
    base = dict(block_tokens=128, kv_per_token_bytes=34816,
                mtp_per_token_bytes=4096, gpu_ratio=0.50,
                gpu_total_bytes=PROBE_GPU_TOTAL, gen_reserve=256)
    base.update(over)
    return _pool(**base)


def test_measured_budget_zero_pools_the_context_and_no_more():
    # `--kvmem-budget 0 -c 32768` -> budget=32768 cells=32768 slots=256:
    # the reserve is inside the pool, not on top of it.
    p = _probe(budget=0, n_ctx=32768)
    assert (p.pool_cells, p.slots) == (32768, 256)
    assert p.pool_cells * 34816 == 1140850688          # KVMEM_KV_BYTES bytes=
    assert p.pool_cells * 4096 == 134217728            # mtp_pool bytes=


def test_measured_explicit_budget_pools_past_the_context():
    # `--kvmem-budget 40000 --kvmem-gen-reserve 256 -c 32768` ->
    # budget=39936 cells=40192 slots=314.
    p = _probe(budget=40000, n_ctx=32768)
    assert (p.pool_cells, p.slots) == (40192, 314)
    assert not p.clamped
    assert p.pool_cells * 34816 == 1399324672
    assert p.pool_cells * 4096 == 164626432


def test_measured_ratio_cap_bounds_the_pool_at_191_slots():
    # The same 40000 with `--kvmem-gpu-ratio 0.05` -> cap_blocks=191,
    # budget=24320 gen_reserve=128 cells=24448: the engine hands its own budget
    # back scaled down, and the cap counts main-KV slots (an MTP-inclusive
    # divisor would have said 171).
    p = _probe(budget=40000, n_ctx=32768, gpu_ratio=0.05)
    assert p.cap_slots == 191
    assert p.clamped and (p.pool_cells, p.slots) == (24448, 191)
    assert (p.requested_cells, p.requested_slots) == (40192, 314)
    assert p.pool_cells * 34816 == 851181568


def test_measured_cap_blocks_at_half_the_card():
    # Every probe line above reads cap_blocks=1915 at ratio 0.50 on this card —
    # the same 1915 that 0.50 * 17074421760 // 4456448 gives.
    assert _probe(budget=1_000_000, n_ctx=262144).cap_slots == 1915


def test_the_model_has_no_draft_token_count_to_size_the_mtp_pool():
    # `--spec-draft-n-max 1` and `3` logged an identical
    # mtp_pool cells=32768 bytes=134217728, so V_MTP follows the pool size and
    # nothing else. An input for it would be a bug waiting to double-count.
    assert "spec_draft_n_max" not in ve.VramInputs.__dataclass_fields__


# --------------------------------------------------------------------------
# estimate(): V_phys composition
# --------------------------------------------------------------------------

def test_golden_component_bytes():
    est = ve.estimate(_inp())
    assert est.kv_per_token_bytes == 34816
    assert est.mtp_per_token_bytes == 4096
    assert est.kv_bytes == 53248 * 34816 == 1853882368
    assert math.isclose(ve.gib(est.kv_bytes), 1.7266, abs_tol=1e-3)
    assert est.mtp_bytes == 53248 * 4096 == 218103808
    assert math.isclose(ve.gib(est.mtp_bytes), 0.203125, abs_tol=1e-6)   # 208 MiB
    assert est.weight_bytes == 10 * GIB
    assert est.phys_bytes == est.weight_bytes + est.kv_bytes + est.mtp_bytes


def test_mtp_head_is_free_when_spec_is_off():
    est = ve.estimate(_inp(spec_type="none"))
    assert est.mtp_bytes == 0
    assert est.sources["mtp_bytes"] == "disabled"
    assert est.phys_bytes == ve.estimate(_inp()).phys_bytes - 218103808


def test_mmproj_counts_only_when_offloaded():
    with_mm = ve.estimate(_inp(mmproj_geometry=_geo(data_span_bytes=800 * MIB),
                               mmproj_offload=True))
    assert with_mm.mmproj_bytes == 800 * MIB
    cpu_only = ve.estimate(_inp(mmproj_geometry=_geo(data_span_bytes=800 * MIB),
                                mmproj_offload=False))
    assert cpu_only.mmproj_bytes == 0
    assert "mmproj_cpu" in cpu_only.notes


def test_full_offload_and_ngl_zero_and_partial_are_three_different_answers():
    full = ve.estimate(_inp(n_gpu_layers=-1))
    cpu = ve.estimate(_inp(n_gpu_layers=0))
    partial = ve.estimate(_inp(n_gpu_layers=8))
    assert full.weight_bytes == 10 * GIB
    assert cpu.weight_bytes == 0 and "ngl0_weights_on_cpu" in cpu.notes
    # This launcher has no copy of split_layer(): report the whole model and
    # say it is degraded rather than invent a layer cutoff.
    assert partial.weight_bytes == 10 * GIB
    assert "partial_offload" in partial.degraded


def test_missing_gguf_degrades_instead_of_raising():
    est = ve.estimate(_inp(geometry=_geo(data_span_bytes=0)))
    assert est.weight_bytes == 0
    assert "no_gguf" in est.degraded


def test_unusable_row_sizes_are_reported_as_missing():
    est = ve.estimate(_inp(geometry=_geo(attn_layers=0)))
    assert est.kv_bytes == 0
    assert "kv_row_bytes" in est.missing and "attn_layers" in est.missing


# --------------------------------------------------------------------------
# estimate(): the log's own numbers outrank the model's arithmetic
# --------------------------------------------------------------------------

def test_logged_slot_bytes_outrank_the_gguf_geometry():
    # Read as n_k = 512 the same model would cost 16*(544+544) = 17408 B/token;
    # KVMEM_KV_BYTES' block_bytes says otherwise, and the engine is right.
    est = ve.estimate(_inp(geometry=_geo(n_k=512, n_v=512),
                           actual={"kv_per_token_bytes": 34816}))
    assert est.kv_per_token_bytes == 34816
    assert est.sources["kv_row_bytes"] == "log_slot_bytes"


def test_structural_arithmetic_is_used_when_no_log_line_has_landed():
    assert ve.estimate(_inp()).sources["kv_row_bytes"] == "structural"


def test_logged_totals_are_used_verbatim_but_not_for_a_different_budget():
    inp = _inp(actual={"kv_bytes": 123456789, "mtp_bytes": 987654321})
    assert ve.estimate(inp).kv_bytes == 123456789
    assert ve.estimate(inp).mtp_bytes == 987654321
    # use_actual=False is the safe-budget sweep: totals must scale per candidate
    swept = ve.estimate(ve.replace(_inp(actual={"kv_bytes": 123456789}),
                                   use_actual=False))
    assert swept.kv_bytes == 53248 * 34816


def test_logged_model_buffer_beats_the_gguf_span():
    est = ve.estimate(_inp(actual={"model_vram_bytes": 11 * GIB}))
    assert est.weight_bytes == 11 * GIB
    assert est.sources["weight"] == "log_model_buffer"


def test_logged_gpu_total_caps_even_without_a_sampler_sample():
    no_sample = _inp(gpu_total_bytes=None, budget=1_000_000,
                     actual={"gpu_total_bytes": 16 * GIB})
    est = ve.estimate(no_sample)
    assert est.pool.clamped and est.pool.cap_slots == 1927
    assert "no_gpu_total" not in est.notes
    assert "no_gpu_total" in ve.estimate(_inp(gpu_total_bytes=None)).notes


# --------------------------------------------------------------------------
# GGUF geometry derivation
# --------------------------------------------------------------------------

class _Tensor:
    def __init__(self, name, offset):
        self.name, self.offset = name, offset


class _Info:
    def __init__(self, metadata, tensors, file_size, offset, estimated=0):
        self.metadata, self.tensors = metadata, tensors
        self.file_size, self.tensor_data_offset = file_size, offset
        self.stats = type("S", (), {"total_estimated_bytes": estimated,
                                    "dominant_type_name": "IQ3_S"})()


def test_derive_geometry_separates_the_nextn_head_from_the_body():
    md = {"general.architecture": "qwen35", "qwen35.block_count": 17,
          "qwen35.embedding_length": 2048, "qwen35.attention.head_count_kv": 8,
          "qwen35.attention.key_length": 128, "qwen35.attention.value_length": 128,
          "qwen35.nextn_predict_layers": 1}
    tensors = []
    for layer in range(17):
        suffix = ".nextn" if layer == 16 else ""
        tensors.append(_Tensor(f"blk.{layer}{suffix}.attn_k.weight", 1000 + layer * 100))
        tensors.append(_Tensor(f"blk.{layer}{suffix}.attn_v.weight", 1050 + layer * 100))
    tensors.append(_Tensor("token_embd.weight", 999999))
    info = _Info(md, tensors, 1_000_000, 0)
    geo = ve.derive_model_geometry(info)
    assert geo.attn_layers == 16 and geo.mtp_layers == 1 and geo.has_nextn
    assert (geo.n_k, geo.n_v) == (1024, 1024)
    assert geo.arch == "qwen35" and geo.block_count == 17
    assert geo.unattributed_bytes > 0 and 16 in geo.layer_tensor_bytes


def test_derive_geometry_falls_back_to_head_count_and_interval():
    md = {"general.architecture": "qwen3next", "qwen3next.block_count": 48,
          "qwen3next.embedding_length": 2048, "qwen3next.attention.head_count": 16,
          "qwen3next.full_attention_interval": 4}
    geo = ve.derive_model_geometry(_Info(md, [], 10, 0))
    assert geo.n_k == 128 and "n_k_fallback" in geo.notes
    assert geo.attn_layers == math.ceil(48 / 4)
    assert "attn_layers_from_metadata" in geo.notes


def test_data_span_is_the_offset_table_not_an_estimate():
    tensors = [_Tensor("a", 0), _Tensor("b", 4096), _Tensor("c", 8192)]
    assert ve.data_span_bytes(tensors, 12288, 4096) == 12288 - 4096
    assert ve.data_span_bytes([], 12288, 4096) == 0


# --------------------------------------------------------------------------
# solve_safe_budget(): block-granular binary search
# --------------------------------------------------------------------------

class _FixedPredict:
    """A learning layer with no residual and no bound, so the roundtrip tests
    see the physics only."""

    def __init__(self, u_bytes=0):
        self.u_bytes = int(u_bytes)
        self.calls = 0

    def __call__(self, est):
        self.calls += 1
        return est.phys_bytes, self.u_bytes


def test_safe_budget_fits_and_one_more_block_does_not():
    inp = _inp()
    free = 13 * GIB
    adv = ve.solve_safe_budget(inp, free, _FixedPredict())
    assert adv.feasible and adv.budget_safe > 0
    assert adv.v_safe_bytes <= free
    assert not adv.at_upper_bound
    bigger = ve.estimate(ve.replace(inp, budget=adv.budget_safe + inp.block_tokens,
                                    use_actual=False))
    assert bigger.phys_bytes > free, "the search stopped one block too early"


def test_search_is_monotone_in_the_budget():
    inp = _inp()
    pred = _FixedPredict()
    budgets = [1024, 8192, 36864, 65536]
    peaks = [ve.estimate(ve.replace(inp, budget=b, use_actual=False)).phys_bytes
             for b in budgets]
    assert peaks == sorted(peaks) and len(set(peaks)) == len(peaks)
    advs = [ve.solve_safe_budget(inp, p, pred) for p in peaks]
    assert [a.budget_safe for a in advs] == budgets


def test_safety_bound_pushes_the_recommended_budget_down():
    inp = _inp()
    free = 13 * GIB
    plain = ve.solve_safe_budget(inp, free, _FixedPredict(0))
    guarded = ve.solve_safe_budget(inp, free, _FixedPredict(2 * GIB))
    assert guarded.budget_safe < plain.budget_safe
    assert guarded.v_safe_bytes <= free


def test_nothing_fits_recommends_no_budget():
    inp = _inp()
    adv = ve.solve_safe_budget(inp, inp.geometry.data_span_bytes // 2,
                               _FixedPredict(GIB))
    assert not adv.feasible and adv.budget_safe == 0
    assert adv.reason == "nothing_fits" and adv.at_lower_bound
    assert adv.headroom_bytes < 0


def test_a_card_that_fits_everything_saturates_at_the_context():
    inp = _inp(ctx_size=32768)
    adv = ve.solve_safe_budget(inp, 64 * GIB, _FixedPredict())
    assert adv.at_upper_bound and adv.budget_safe == 32768
    assert adv.reason == ""


def test_ratio_cap_is_reported_on_the_advice():
    inp = _inp(budget=0, gpu_total_bytes=16 * GIB, gpu_ratio=0.05)
    adv = ve.solve_safe_budget(inp, 15 * GIB, _FixedPredict())
    assert adv.clamped and adv.reason == "ratio_clamped"


def test_a_budget_limited_by_free_memory_is_not_reported_as_a_cap():
    # The search does probe the context-sized end, where the ratio clamps the
    # pool — but here free VRAM ran out far below that, so blaming
    # --kvmem-gpu-ratio would send the user to the wrong control.
    inp = _inp(gpu_total_bytes=16 * GIB, gpu_ratio=0.50)
    cap_slots = int(0.50 * 16 * GIB) // (128 * 34816)      # the engine's own basis
    adv = ve.solve_safe_budget(inp, 12 * GIB, _FixedPredict())
    assert adv.feasible and 0 < adv.slots < cap_slots
    assert not adv.clamped and adv.reason == ""


def test_advice_pool_numbers_come_from_the_same_compute_pool():
    inp = _inp()
    adv = ve.solve_safe_budget(inp, 15 * GIB, _FixedPredict())
    direct = ve.compute_pool(ve.PoolRequest(
        budget=adv.budget_safe, gen_reserve=inp.gen_reserve,
        block_tokens=inp.block_tokens, n_ctx=inp.ctx_size,
        gpu_ratio=inp.gpu_ratio, gpu_total_bytes=inp.gpu_total_bytes,
        kv_per_token_bytes=34816, mtp_per_token_bytes=4096))
    assert (adv.pool_cells, adv.slots) == (direct.pool_cells, direct.slots)
