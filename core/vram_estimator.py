"""Physical VRAM model for the kvmem-llama.cpp engine (Qt-free).

Layer 1 of the three-layer estimator: physics. core.vram_learning adds the
learned residual and the one-sided safety bound; the UI only shows what the two
agree on and never lets either one write a parameter.

The pool rules replicate kvmem-llama.cpp v0.16.0-rc2 `kvmem_compute_pool()`:
--kvmem-budget and --kvmem-gen-reserve are each put through kvmem_align_tokens()
(floor to --kvmem-block-tokens, never below one block) and then summed, a 0
reserve falls back to the binary's own default of 256, and only the budget==0
path clamps the pool to n_ctx — which is what makes --kvmem-budget 0 mean
"whole context", while an explicit budget is allowed to pool past it.
Because the two terms are aligned separately, the resulting cell count is not
generally a multiple of block_tokens, so slot counts round up.

Units: everything is bytes unless the name ends in _gib. pool_cells is a token
count; slots is a block count.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Callable, Iterable

MIB = 1 << 20
GIB = 1 << 30

# rc2's own defaults, needed when the user leaves a value at 0.
KVMEM_DEFAULT_GEN_RESERVE = 256
# --kvmem-budget spin maximum in core/kvmem_params_schema.py
KVMEM_BUDGET_MAX = 1048576

KV_TYPE_NAMES = ("f16", "f32", "q8_0", "q5_0", "q4_0")

_BLK_LAYER_RE = re.compile(r"^blk\.(\d+)\.")


# --------------------------------------------------------------------------
# ggml row size
# --------------------------------------------------------------------------

def ggml_row_bytes(type_name: str, n: int) -> int | None:
    """Bytes one token's K (or V) row occupies for dtype `type_name`.

    row(type, n) = ceil(n / block_size) * bytes_per_block. None when the dtype
    or the element count is unusable — the caller degrades instead of guessing.
    """
    from gguf.ggml_types import get_type_info, type_id_from_name

    if n is None or n <= 0:
        return None
    type_id = type_id_from_name(type_name)
    if type_id is None:
        return None
    info = get_type_info(type_id)
    if info is None:
        return None
    block_size, bytes_per_block = info
    if block_size <= 0:
        return None
    return -(-n // block_size) * bytes_per_block


def kv_per_token_bytes(layers: int, type_k: str, n_k: int,
                       type_v: str, n_v: int) -> int:
    """One cached token across `layers` attention layers (K and V)."""
    row_k = ggml_row_bytes(type_k, n_k)
    row_v = ggml_row_bytes(type_v, n_v)
    if not layers or row_k is None or row_v is None:
        return 0
    return layers * (row_k + row_v)


# --------------------------------------------------------------------------
# kvmem slot pool
# --------------------------------------------------------------------------

def align_tokens(tokens: int, block_tokens: int) -> int:
    """kvmem_align_tokens(): floor to a whole block, floor of one block."""
    if block_tokens <= 0:
        return max(0, int(tokens))
    return max(block_tokens, (int(tokens) // block_tokens) * block_tokens)


@dataclass(frozen=True)
class PoolRequest:
    budget: int = 0
    gen_reserve: int = 0
    block_tokens: int = 128
    n_ctx: int = 0
    gpu_ratio: float = 0.50
    gpu_total_bytes: int | None = None
    kv_per_token_bytes: int = 0
    mtp_per_token_bytes: int = 0
    kvmem_enabled: bool = True


@dataclass(frozen=True)
class PoolResult:
    pool_cells: int              # logical token slots, not necessarily block-aligned
    alloc_cells: int             # what the GPU actually backs: slots * block_tokens
    slots: int
    requested_cells: int     # as handed to the gpu_ratio cap — the budget==0
    requested_slots: int     # n_ctx clamp happens earlier, so these are post-clamp
    cap_slots: int | None
    engine_slot_bytes: int       # block_tokens * kv_per_token — the cap's basis
    total_pool_slot_bytes: int   # block_tokens * (kv + mtp) — the VRAM basis
    block_tokens: int
    clamped: bool
    notes: tuple[str, ...] = ()


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def compute_pool(req: PoolRequest) -> PoolResult:
    """Replicate rc2's slot-pool sizing, including the gpu_ratio cap.

    The cap is expressed in blocks (slots) and, as the engine reports it, is
    charged against the main KV only: `engine_slot_bytes` ignores the MTP head.
    The MTP head still costs real VRAM, so total memory uses
    `total_pool_slot_bytes` — the two must not be conflated.
    """
    bt = max(1, int(req.block_tokens))
    n_ctx = max(0, int(req.n_ctx))
    notes: list[str] = []

    if not req.kvmem_enabled:
        cells = n_ctx
        notes.append("kvmem_disabled")
    else:
        reserve = align_tokens(req.gen_reserve or KVMEM_DEFAULT_GEN_RESERVE, bt)
        if req.budget and req.budget > 0:
            # An explicit budget is *not* capped by the context: rc2 pools
            # 39936+256 = 40192 cells for a 32768-token context (measured), so
            # clamping here would under-charge the prediction.
            cells = align_tokens(req.budget, bt) + reserve
        else:
            # --kvmem-budget 0 == "the whole context"; adding the decode reserve
            # on top is what this branch's n_ctx clamp is there to undo.
            cells = min(align_tokens(n_ctx, bt) + reserve, n_ctx) if n_ctx else 0
            if not n_ctx:
                notes.append("no_n_ctx")

    engine_slot_bytes = bt * max(0, int(req.kv_per_token_bytes))
    total_pool_slot_bytes = bt * (max(0, int(req.kv_per_token_bytes))
                                  + max(0, int(req.mtp_per_token_bytes)))

    cap_slots: int | None = None
    if req.gpu_total_bytes and engine_slot_bytes > 0 and req.gpu_ratio > 0:
        cap_slots = int(req.gpu_total_bytes * float(req.gpu_ratio)) // engine_slot_bytes

    slots = _ceil_div(cells, bt) if cells else 0
    requested_cells = cells
    requested_slots = slots
    clamped = False
    if cap_slots is not None and slots > cap_slots:
        slots = max(0, cap_slots)
        cells = slots * bt
        clamped = True

    return PoolResult(
        pool_cells=cells,
        alloc_cells=slots * bt,
        slots=slots,
        requested_cells=requested_cells,
        requested_slots=requested_slots,
        cap_slots=cap_slots,
        engine_slot_bytes=engine_slot_bytes,
        total_pool_slot_bytes=total_pool_slot_bytes,
        block_tokens=bt,
        clamped=clamped,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------
# GGUF geometry
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelGeometry:
    arch: str = ""
    block_count: int = 0
    attn_layers: int = 0
    n_k: int = 0
    n_v: int = 0
    n_embd: int = 0
    mtp_layers: int = 0
    n_k_mtp: int = 0
    n_v_mtp: int = 0
    has_nextn: bool = False
    data_span_bytes: int = 0        # tensor bytes as laid out in the file
    estimated_nbytes: int = 0       # ggml_types approximation (kept for display)
    layer_tensor_bytes: dict[int, int] = field(default_factory=dict)
    unattributed_bytes: int = 0
    notes: tuple[str, ...] = ()


def data_span_bytes(tensors: Iterable, file_size: int, tensor_data_offset: int) -> int:
    """Bytes the tensor data occupies, derived from the GGUF offset table.

    `ggml_types.estimate_tensor_nbytes()` is approximate for the IQ*/K quants
    (measured: +7.3% on an IQ3_S 27B), so the offset span is what the weight
    term uses; the last tensor runs to the end of the file.
    """
    offsets = sorted(t.offset for t in tensors)
    if not offsets:
        return 0
    data_end = max(0, int(file_size) - int(tensor_data_offset))
    total = 0
    for i, off in enumerate(offsets):
        nxt = offsets[i + 1] if i + 1 < len(offsets) else data_end
        total += max(0, nxt - off)
    return total


def _metadata_int(md: dict, *keys: str) -> int:
    for k in keys:
        v = md.get(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, int) and v > 0:
            return v
    return 0


def _layer_of(name: str) -> int | None:
    m = _BLK_LAYER_RE.match(name)
    return int(m.group(1)) if m else None


def derive_model_geometry(info, gguf_path_hint: str = "") -> ModelGeometry:
    """Read the KV geometry out of a parsed GGUF (gguf.parser.parse_gguf)."""
    md = dict(getattr(info, "metadata", {}) or {})
    arch = md.get("general.architecture") or ""
    p = f"{arch}." if arch else ""
    notes: list[str] = []

    block_count = _metadata_int(md, f"{p}block_count")
    n_embd = _metadata_int(md, f"{p}embedding_length")
    heads_kv = _metadata_int(md, f"{p}attention.head_count_kv")
    head_k = _metadata_int(md, f"{p}attention.key_length")
    head_v = _metadata_int(md, f"{p}attention.value_length") or head_k
    n_k = heads_kv * head_k if heads_kv and head_k else 0
    n_v = heads_kv * head_v if heads_kv and head_v else 0
    if not n_k:
        heads = _metadata_int(md, f"{p}attention.head_count")
        n_k = head_k or (n_embd // heads if heads and n_embd else 0)
        notes.append("n_k_fallback")
    if not n_v:
        n_v = n_k

    nextn_layers: set[int] = set()
    kv_layers: set[int] = set()
    per_layer: dict[int, int] = {}
    unattributed = 0
    tensors = list(getattr(info, "tensors", []) or [])
    offsets = sorted(t.offset for t in tensors)
    data_end = max(0, int(getattr(info, "file_size", 0))
                   - int(getattr(info, "tensor_data_offset", 0)))
    for i, t in enumerate(tensors):
        span = (offsets[i + 1] if i + 1 < len(offsets) else data_end) - t.offset
        layer = _layer_of(t.name)
        if layer is None:
            unattributed += max(0, span)
        else:
            per_layer[layer] = per_layer.get(layer, 0) + max(0, span)
            if ".nextn." in t.name:
                nextn_layers.add(layer)
            elif t.name.endswith(".attn_k.weight") or t.name.endswith(".attn_v.weight"):
                kv_layers.add(layer)

    mtp_meta = _metadata_int(md, f"{p}nextn_predict_layers")
    mtp_layers = len(nextn_layers) or mtp_meta
    attn_layers = len(kv_layers - nextn_layers)
    if not attn_layers:
        interval = _metadata_int(md, f"{p}full_attention_interval")
        body = max(0, block_count - mtp_layers)
        attn_layers = _ceil_div(body, interval) if interval else body
        notes.append("attn_layers_from_metadata")

    return ModelGeometry(
        arch=arch,
        block_count=block_count,
        attn_layers=attn_layers,
        n_k=n_k,
        n_v=n_v,
        n_embd=n_embd,
        mtp_layers=mtp_layers,
        n_k_mtp=n_k,
        n_v_mtp=n_v,
        has_nextn=bool(nextn_layers or mtp_meta),
        data_span_bytes=data_span_bytes(tensors, getattr(info, "file_size", 0),
                                        getattr(info, "tensor_data_offset", 0)),
        estimated_nbytes=int(getattr(getattr(info, "stats", None),
                                     "total_estimated_bytes", 0) or 0),
        layer_tensor_bytes=per_layer,
        unattributed_bytes=unattributed,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------
# the physical estimate
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class VramInputs:
    geometry: ModelGeometry
    ctx_size: int = 2048
    batch_size: int = 512
    n_gpu_layers: int = 99
    kvmem_enabled: bool = True
    budget: int = 0
    gen_reserve: int = 0
    block_tokens: int = 128
    gpu_ratio: float = 0.50
    gpu_total_bytes: int | None = None
    type_k: str = "q8_0"
    type_v: str = "q8_0"
    spec_type: str = "none"
    spec_kv_dtype: str = "f16"
    mmproj_geometry: ModelGeometry | None = None
    mmproj_offload: bool = True
    # From ui/log_parser's info["kvmem_alloc"]: the engine's own numbers.
    actual: dict = field(default_factory=dict)
    # The log's *total* pool bytes describe one specific budget, so the
    # safe-budget sweep turns them off and lets the pool scale per candidate.
    use_actual: bool = True


@dataclass(frozen=True)
class VramEstimate:
    weight_bytes: int = 0
    kv_bytes: int = 0
    mtp_bytes: int = 0
    mmproj_bytes: int = 0
    phys_bytes: int = 0
    kv_per_token_bytes: int = 0
    mtp_per_token_bytes: int = 0
    pool: PoolResult | None = None
    sources: dict = field(default_factory=dict)
    degraded: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def pool_per_token_bytes(self) -> int:
        return self.kv_per_token_bytes + self.mtp_per_token_bytes


def _weight_gpu_bytes(geo: ModelGeometry, n_gpu_layers: int,
                      degraded: list, notes: list) -> int:
    total = int(geo.data_span_bytes or 0)
    if total <= 0:
        degraded.append("no_gguf")
        return 0
    body = int(geo.block_count or 0)
    if n_gpu_layers is None or n_gpu_layers < 0 or n_gpu_layers >= body:
        return total                       # full offload: the split rule is moot
    if n_gpu_layers == 0:
        notes.append("ngl0_weights_on_cpu")
        return 0
    # Reaching here means a partial offload, and this launcher has no copy of
    # the engine's split_layer() to follow — report the whole tensor set and
    # say so, rather than invent a layer cutoff.
    degraded.append("partial_offload")
    return total


def _actual_int(actual: dict, *keys: str) -> int:
    for k in keys:
        v = actual.get(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)) and v > 0:
            return int(v)
    return 0


def estimate(inp: VramInputs) -> VramEstimate:
    """V_phys = weights + main KV pool + MTP pool + mmproj, in bytes."""
    degraded: list[str] = []
    missing: list[str] = []
    notes: list[str] = []
    sources: dict = {}
    geo = inp.geometry
    actual = inp.actual or {}

    # --- per-token row bytes: the engine's own numbers win over arithmetic ---
    # KVMEM_KV_BYTES never states the K/V row size, so the log's truth for the
    # main pool arrives as a slot: block_bytes / block_tokens is the engine's
    # own per-token figure. (The MTP trace does carry row bytes; see below.)
    struct_row_k = ggml_row_bytes(inp.type_k, geo.n_k)
    struct_row_v = ggml_row_bytes(inp.type_v, geo.n_v)
    layers = geo.attn_layers
    logged_per_token = _actual_int(actual, "kv_per_token_bytes")
    if layers and logged_per_token:
        kv_per_token = logged_per_token
        sources["kv_row_bytes"] = "log_slot_bytes"
    elif struct_row_k and struct_row_v and layers:
        kv_per_token = layers * (struct_row_k + struct_row_v)
        sources["kv_row_bytes"] = "structural"
    else:
        kv_per_token = 0
        missing.append("kv_row_bytes")
    if not layers:
        missing.append("attn_layers")

    mtp_on = (inp.spec_type or "none") != "none"
    mtp_per_token = 0
    if mtp_on and geo.mtp_layers:
        # I_mtp == 1: --spec-draft-n-max changes the draft graph, not the pool.
        m_row_k = _actual_int(actual, "mtp_k_row_bytes")
        m_row_v = _actual_int(actual, "mtp_v_row_bytes")
        s_row_k = ggml_row_bytes(inp.spec_kv_dtype, geo.n_k_mtp)
        s_row_v = ggml_row_bytes(inp.spec_kv_dtype, geo.n_v_mtp)
        if m_row_k and m_row_v:
            mtp_per_token = geo.mtp_layers * (m_row_k + m_row_v)
            sources["mtp_row_bytes"] = "log"
        elif s_row_k and s_row_v:
            mtp_per_token = geo.mtp_layers * (s_row_k + s_row_v)
            sources["mtp_row_bytes"] = "structural"
        else:
            missing.append("mtp_row_bytes")

    # The engine's own reading of the card is the one its cap was computed from;
    # it stands in when no sampler sample has landed yet.
    gpu_total = inp.gpu_total_bytes or _actual_int(actual, "gpu_total_bytes")
    pool = compute_pool(PoolRequest(
        budget=inp.budget, gen_reserve=inp.gen_reserve, block_tokens=inp.block_tokens,
        n_ctx=inp.ctx_size, gpu_ratio=inp.gpu_ratio, gpu_total_bytes=gpu_total,
        kv_per_token_bytes=kv_per_token, mtp_per_token_bytes=mtp_per_token,
        kvmem_enabled=inp.kvmem_enabled,
    ))

    kv_actual = _actual_int(actual, "kv_bytes") if inp.use_actual else 0
    mtp_actual = _actual_int(actual, "mtp_bytes") if inp.use_actual else 0
    if kv_actual:
        kv_bytes = kv_actual
        sources["kv_bytes"] = "log"
    else:
        kv_bytes = pool.alloc_cells * kv_per_token
        sources["kv_bytes"] = "model"
    if mtp_actual:
        mtp_bytes = mtp_actual
        sources["mtp_bytes"] = "log"
    else:
        mtp_bytes = pool.alloc_cells * mtp_per_token
        sources["mtp_bytes"] = "model" if mtp_on else "disabled"

    weight_bytes = _actual_int(actual, "model_vram_bytes")
    if weight_bytes:
        sources["weight"] = "log_model_buffer"
    else:
        weight_bytes = _weight_gpu_bytes(geo, inp.n_gpu_layers, degraded, notes)
        sources["weight"] = "gguf_data_span"

    mmproj_bytes = 0
    if inp.mmproj_geometry is not None:
        if not inp.mmproj_offload:
            notes.append("mmproj_cpu")
        else:
            mmproj_bytes = int(inp.mmproj_geometry.data_span_bytes or 0)
            sources["mmproj"] = "gguf_data_span"

    if not gpu_total:
        notes.append("no_gpu_total")
    if pool.clamped:
        notes.append("pool_clamped_by_ratio")

    phys = weight_bytes + kv_bytes + mtp_bytes + mmproj_bytes
    return VramEstimate(
        weight_bytes=weight_bytes, kv_bytes=kv_bytes, mtp_bytes=mtp_bytes,
        mmproj_bytes=mmproj_bytes, phys_bytes=phys,
        kv_per_token_bytes=kv_per_token, mtp_per_token_bytes=mtp_per_token,
        pool=pool, sources=sources,
        degraded=tuple(degraded), missing=tuple(missing), notes=tuple(notes + list(pool.notes)),
    )


# --------------------------------------------------------------------------
# safe budget: block-granular binary search over the same compute_pool()
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BudgetAdvice:
    budget_safe: int
    pool_cells: int
    slots: int
    free_bytes: int
    v_hat_bytes: int
    u_bytes: int
    v_safe_bytes: int
    headroom_bytes: int
    feasible: bool
    clamped: bool
    at_upper_bound: bool
    at_lower_bound: bool
    evaluations: int = 0
    reason: str = ""


def solve_safe_budget(inp: VramInputs, free_bytes: int,
                      predict: Callable[[VramEstimate], tuple[int, int]],
                      ) -> BudgetAdvice:
    """Largest --kvmem-budget whose safe peak still fits in `free_bytes`.

    `predict(est)` returns (v_hat_bytes, u_bytes) for a candidate estimate, so
    the learning layer keeps ownership of the residual and the safety bound.
    V_safe grows monotonically with the budget, which makes a block-granular
    binary search enough — no inverse formula to keep in sync with compute_pool.

    Candidates are evaluated with `use_actual=False`: a logged allocation is a
    fact about the *current* pool size and would otherwise freeze the sweep.
    """
    bt = max(1, int(inp.block_tokens or 1))
    hi_slots = max(1, min(int(inp.ctx_size or bt), KVMEM_BUDGET_MAX) // bt)

    evals = 0

    def evaluate(budget: int):
        nonlocal evals
        est = estimate(replace(inp, budget=budget, use_actual=False))
        v_hat, u = predict(est)
        evals += 1
        return est, v_hat, u, v_hat + u

    def fits(budget: int) -> bool:
        return evaluate(budget)[3] <= free_bytes

    est_lo, v_hat, u, v_safe = evaluate(bt)
    if v_safe > free_bytes:
        # Even one block does not fit: there is no value to recommend, so the
        # advice carries 0 and the panel's button stays out of reach.
        return BudgetAdvice(
            budget_safe=0,
            pool_cells=est_lo.pool.pool_cells if est_lo.pool else 0,
            slots=est_lo.pool.slots if est_lo.pool else 0,
            free_bytes=free_bytes, v_hat_bytes=v_hat, u_bytes=u,
            v_safe_bytes=v_safe, headroom_bytes=free_bytes - v_safe,
            feasible=False, clamped=bool(est_lo.pool and est_lo.pool.clamped),
            at_upper_bound=False, at_lower_bound=True, evaluations=evals,
            reason="nothing_fits",
        )

    hi_budget = hi_slots * bt
    if hi_budget > bt and fits(hi_budget):
        best_slots, at_upper = hi_slots, True
    elif hi_budget > bt:
        lo, hi = 1, hi_slots            # lo fits, hi does not
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if fits(mid * bt):
                lo = mid
            else:
                hi = mid
        best_slots, at_upper = lo, False
    else:
        best_slots, at_upper = 1, True

    best = best_slots * bt
    est, v_hat, u, v_safe = evaluate(best)
    # Only the chosen value can be capped: a candidate the search probed above
    # the ratio limit says nothing about what limits the recommendation.
    clamped = bool(est.pool and est.pool.clamped)
    return BudgetAdvice(
        budget_safe=best,
        pool_cells=est.pool.pool_cells if est.pool else 0,
        slots=est.pool.slots if est.pool else 0,
        free_bytes=free_bytes, v_hat_bytes=v_hat, u_bytes=u,
        v_safe_bytes=v_safe, headroom_bytes=free_bytes - v_safe, feasible=True,
        clamped=clamped, at_upper_bound=at_upper,
        at_lower_bound=best_slots <= 1, evaluations=evals,
        reason="ratio_clamped" if clamped else "",
    )


def gib(b: int | float) -> float:
    return float(b) / GIB
