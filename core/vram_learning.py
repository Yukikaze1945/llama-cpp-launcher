"""Online calibration for the kvmem VRAM prediction (Qt-free).

Layer 2 is a recursive-least-squares fit of the *residual* between the
measured peak and the physical model (core.vram_estimator), never of the peak
itself; layer 3 is a one-sided empirical upper bound on that residual, which
is what makes the suggested budget safe rather than merely accurate.

Nothing here changes a parameter or starts a server — it takes numbers in and
gives numbers out, and persists them per (GPU, GPU size, engine build, model)
profile so a 4090's learning never leaks into a 5060 Ti's prediction.

State lives in CONFIG_DIR/vram_learning.json. The file holds arithmetic and
size statistics only: no prompt, no chat content, no model file name or path.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from core import config as _config
from core import vram_sampler as _sampler

MIB = 1 << 20
GIB = 1 << 30

SCHEMA_VERSION = 1
FEATURE_NAMES = ("bias", "weight", "kv", "mtp", "mmproj", "log2_batch")
N_FEATURES = len(FEATURE_NAMES)
FORGETTING_FACTOR = 0.995
P_INITIAL = 1e3
MAX_RESIDUALS = 64
MAX_PROFILES = 32

BAND_SMALL_SAMPLE_MIN = 5          # n >= 5 switches to the empirical P95
MIN_SAMPLES_FOR_P95 = 5
ONE_GIB = GIB
FIVE_HUNDRED_MIB = 512 * MIB
TWO_HUNDRED_FIFTY_SIX_MIB = 256 * MIB


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# profile identity
# --------------------------------------------------------------------------

def model_fingerprint(arch: str, block_count: int, tensor_count: int,
                      file_size: int, dominant_type: str = "",
                      has_nextn: bool = False) -> str:
    """Stable id for "which model" without reading the model's contents."""
    raw = "|".join(str(x) for x in (arch, block_count, tensor_count, file_size,
                                    dominant_type, int(bool(has_nextn))))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def profile_key(gpu_name: str, gpu_total_bytes: int, engine_build: str,
                model_fp: str) -> str:
    return "|".join((str(gpu_name or "unknown").strip(), str(int(gpu_total_bytes)),
                     str(engine_build or "unknown"), str(model_fp or "unknown")))


def feature_vector(est, batch_size: int) -> list[float]:
    """[1, V_weight, V_KV, V_MTP, V_mmproj, log2(batch/512)] in GiB."""
    batch = max(1, int(batch_size or 1))
    return [
        1.0,
        float(est.weight_bytes) / GIB,
        float(est.kv_bytes) / GIB,
        float(est.mtp_bytes) / GIB,
        float(est.mmproj_bytes) / GIB,
        math.log2(batch / 512.0),
    ]


# --------------------------------------------------------------------------
# RLS
# --------------------------------------------------------------------------

@dataclass
class RlsModel:
    """theta ~ E[residual | features]; K = P x / (lambda + x'Px)."""
    theta: list[float] = field(default_factory=lambda: [0.0] * N_FEATURES)
    p_matrix: list[list[float]] = field(default_factory=lambda: [
        [P_INITIAL if i == j else 0.0 for j in range(N_FEATURES)]
        for i in range(N_FEATURES)])
    lam: float = FORGETTING_FACTOR
    n: int = 0

    def predict_residual(self, x: Sequence[float]) -> float:
        return _dot(self.theta, x)

    def update(self, x: Sequence[float], residual: float) -> float:
        """One RLS step. Returns the innovation (residual - old prediction)."""
        px = _mat_vec(self.p_matrix, x)
        denom = self.lam + _dot(x, px)
        if denom <= 0 or not math.isfinite(denom):
            raise ValueError("RLS denominator is not usable")
        gain = [v / denom for v in px]
        innovation = residual - _dot(self.theta, x)
        self.theta = [t + gain[i] * innovation for i, t in enumerate(self.theta)]
        # x'P, kept as a row vector so the outer product with K is explicit.
        xt_p = [sum(x[i] * self.p_matrix[i][j] for i in range(N_FEATURES))
                for j in range(N_FEATURES)]
        self.p_matrix = [[(self.p_matrix[i][j] - gain[i] * xt_p[j]) / self.lam
                          for j in range(N_FEATURES)] for i in range(N_FEATURES)]
        self.p_matrix = [[(self.p_matrix[i][j] + self.p_matrix[j][i]) / 2.0
                          for j in range(N_FEATURES)] for i in range(N_FEATURES)]
        self.n += 1
        return innovation

    # -- serialisation ------------------------------------------------------
    def to_json(self) -> dict:
        return {"theta": [_r(v) for v in self.theta], "lam": self.lam, "n": self.n,
                "P": [[_r(v) for v in row] for row in self.p_matrix]}

    @classmethod
    def from_json(cls, data: dict) -> "RlsModel":
        fresh = cls()
        if not isinstance(data, dict):
            return fresh
        theta = data.get("theta")
        if (isinstance(theta, list) and len(theta) == N_FEATURES
                and all(isinstance(v, (int, float)) for v in theta)):
            fresh.theta = [float(v) for v in theta]
        pmat = data.get("P")
        if (isinstance(pmat, list) and len(pmat) == N_FEATURES
                and all(isinstance(row, list) and len(row) == N_FEATURES for row in pmat)):
            fresh.p_matrix = [[float(v) if isinstance(v, (int, float)) else 0.0
                               for v in row] for row in pmat]
        lam = data.get("lam")
        if isinstance(lam, (int, float)) and 0.0 < float(lam) <= 1.0:
            fresh.lam = float(lam)
        n = data.get("n")
        if isinstance(n, int) and n >= 0:
            fresh.n = n
        return fresh


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _mat_vec(m: Sequence[Sequence[float]], v: Sequence[float]) -> list[float]:
    return [_dot(row, v) for row in m]


def _r(value: float) -> float:
    return round(float(value), 6)


def p95(values: Sequence[float]) -> float:
    """One-sided empirical 95th percentile (nearest rank)."""
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return 0.0
    rank = max(1, math.ceil(0.95 * len(ordered)))
    return ordered[min(len(ordered), rank) - 1]


# --------------------------------------------------------------------------
# safety bound
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SafetyBound:
    u_bytes: int
    band: str          # "structural" | "small_sample" | "empirical"
    dominant: str      # which term set the size, for the tooltip


def safety_bound(phys_bytes: int, residuals_gib: Sequence[float],
                 gpu_total_bytes: int | None) -> SafetyBound:
    """U: how far wrong the calibrated estimate is allowed to be.

    Negative residuals never raise U — an over-estimate costs headroom, an
    under-estimate costs an OOM, and only the second one is dangerous.
    """
    n = len(residuals_gib)
    phys = max(0, int(phys_bytes))
    if n <= 0:
        u = max(ONE_GIB, int(round(0.08 * phys)))
        return SafetyBound(u, "structural", "floor_1gib" if u == ONE_GIB else "eight_percent")
    if n < MIN_SAMPLES_FOR_P95:
        worst = max(0.0, max(float(v) for v in residuals_gib))
        u = max(int(round(worst * GIB)), FIVE_HUNDRED_MIB)
        return SafetyBound(u, "small_sample",
                           "max_positive_error" if u == int(round(worst * GIB)) else "floor_512mib")
    candidates = [("p95", int(round(max(0.0, p95(residuals_gib)) * GIB))),
                  ("three_percent_gpu", int(round(0.03 * (gpu_total_bytes or 0)))),
                  ("floor_256mib", TWO_HUNDRED_FIFTY_SIX_MIB)]
    dominant, u = max(candidates, key=lambda c: c[1])
    return SafetyBound(int(u), "empirical", dominant)


CONFIDENCE_BY_BAND = {"structural": "low", "small_sample": "medium", "empirical": "high"}


# --------------------------------------------------------------------------
# run gating
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RunOutcome:
    """Whether this run may teach the model. Failures never count."""
    ready: bool = False
    clean_stop: bool = False
    oom: bool = False
    param_error: bool = False
    sample_valid: bool = False
    sample_reason: str = ""

    @property
    def learnable(self) -> bool:
        return bool(self.ready and self.clean_stop and not self.oom
                    and not self.param_error and self.sample_valid)

    @property
    def reject_reason(self) -> str:
        if self.oom:
            return "oom"
        if not self.ready:
            return "startup_failed"
        if self.param_error:
            return "param_error"
        if not self.sample_valid:
            return self.sample_reason or "sample_invalid"
        if not self.clean_stop:
            return "unclean_stop"
        return ""


# --------------------------------------------------------------------------
# whole-card peak bookkeeping
# --------------------------------------------------------------------------

#: The engine's own memory-failure wording (plus ggml's, which it forwards).
#: A run that died here only got as far as a *partial* allocation, so its peak
#: says nothing about what the configuration needs — see RunOutcome.oom.
OOM_PATTERNS = (
    re.compile(r"out of memory", re.I),
    re.compile(r"cannot allocate", re.I),
    re.compile(r"failed to allocate", re.I),
    re.compile(r"alloc\w* (?:of \S+ )?failed", re.I),
    re.compile(r"cudaErrorMemoryAllocation", re.I),
    re.compile(r"not enough (?:free )?(?:memory|vram)", re.I),
    # Either word order: "failed to reserve …" and "… allocation failed" are
    # both rc2-shaped, and a kvmem error about something else is not OOM.
    re.compile(r"kvmem:.*(?:(?:allocat|reserve).*?(?:fail|error)"
               r"|(?:fail|error).*?(?:allocat|reserve))", re.I),
)


def detect_oom(lines) -> bool:
    return any(p.search(line) for line in (lines or ()) for p in OOM_PATTERNS)


class PeakMeasure:
    """`used` samples for one run: y = peak - median(before it started).

    On this driver per-process VRAM is not attributable, so the card's total
    `used` is the only peak available; the pre-start median removes whatever was
    already resident, and the post-stop median proves nothing else moved in the
    meantime (`noisy`). Sample counts are small because a run's baseline is
    taken in the ~0.4 s before the process is spawned.
    """

    BASELINE_SAMPLES = 5

    def __init__(self, gpu_total_bytes: int = 0):
        self.gpu_total_bytes = int(gpu_total_bytes or 0)
        self.phase = "idle"            # idle | baseline | running | after | done
        self._base: list[int] = []
        self._after: list[int] = []
        self._peak = 0
        self._run_samples = 0

    def start_baseline(self):
        self._base, self._after = [], []
        self._peak = 0
        self._run_samples = 0
        self.phase = "baseline"

    def begin_run(self):
        if self.phase == "baseline":
            self.phase = "running"

    def end_run(self):
        """Called when the process is gone; the tail samples follow."""
        if self.phase in ("running", "baseline"):
            self.phase = "after"

    def add(self, used_bytes: int):
        u = int(used_bytes)
        if self.phase == "baseline":
            if len(self._base) < self.BASELINE_SAMPLES:
                self._base.append(u)
        elif self.phase == "running":
            self._run_samples += 1
            if u > self._peak:
                self._peak = u
        elif self.phase == "after":
            if len(self._after) < self.BASELINE_SAMPLES:
                self._after.append(u)

    @property
    def baseline_ready(self) -> bool:
        return len(self._base) >= self.BASELINE_SAMPLES

    @property
    def tail_ready(self) -> bool:
        return len(self._after) >= self.BASELINE_SAMPLES

    def baseline_bytes(self) -> int:
        return _median(self._base)

    def tail_bytes(self) -> int:
        return _median(self._after)

    def peak_bytes(self) -> int:
        return max(0, self._peak - self.baseline_bytes())

    def noisy(self) -> bool:
        # The sampler owns the max(256 MiB, 2 % of the card) rule so both the
        # measurement path and any future caller agree on one number.
        return _sampler.is_noisy(self.baseline_bytes(), self.tail_bytes(),
                                 self.gpu_total_bytes)

    def verdict(self) -> tuple[bool, str]:
        """(usable_for_learning, reason)."""
        if not self.baseline_ready:
            return False, "no_baseline"
        if not self._run_samples:
            return False, "no_samples"
        if not self.tail_ready:
            return False, "no_tail"
        if self.noisy():
            return False, "noisy"
        if self.peak_bytes() <= 0:
            return False, "no_growth"
        return True, ""


def _median(values: list[int]) -> int:
    return int(statistics.median(values)) if values else 0


# --------------------------------------------------------------------------
# stored profile + store
# --------------------------------------------------------------------------

@dataclass
class Profile:
    key: str
    gpu_name: str = ""
    gpu_total_bytes: int = 0
    engine_build: str = ""
    model_fp: str = ""
    rls: RlsModel = field(default_factory=RlsModel)
    residuals: list[float] = field(default_factory=list)
    samples: list[dict] = field(default_factory=list)
    updated_utc: str = ""

    @property
    def n(self) -> int:
        return len(self.residuals)

    def to_json(self) -> dict:
        return {
            "gpu_name": self.gpu_name,
            "gpu_total_bytes": int(self.gpu_total_bytes),
            "engine_build": self.engine_build,
            "model_fp": self.model_fp,
            "rls": self.rls.to_json(),
            "residuals": [_r(v) for v in self.residuals],
            "samples": self.samples[-MAX_RESIDUALS:],
            "updated_utc": self.updated_utc,
        }

    @classmethod
    def from_json(cls, key: str, data: dict) -> "Profile":
        data = data if isinstance(data, dict) else {}
        residuals = [float(v) for v in (data.get("residuals") or [])
                     if isinstance(v, (int, float)) and math.isfinite(float(v))]
        return cls(
            key=key,
            gpu_name=str(data.get("gpu_name") or ""),
            gpu_total_bytes=int(data.get("gpu_total_bytes") or 0),
            engine_build=str(data.get("engine_build") or ""),
            model_fp=str(data.get("model_fp") or ""),
            rls=RlsModel.from_json(data.get("rls")),
            residuals=residuals[-MAX_RESIDUALS:],
            samples=[s for s in (data.get("samples") or []) if isinstance(s, dict)],
            updated_utc=str(data.get("updated_utc") or ""),
        )


@dataclass(frozen=True)
class Prediction:
    v_hat_bytes: int
    u_bytes: int
    v_safe_bytes: int
    residual_hat_gib: float
    n: int
    band: str
    confidence: str
    dominant: str


@dataclass(frozen=True)
class LearnResult:
    learned: bool
    reason: str
    residual_gib: float
    innovation: float


class VramLearningStore:
    """The on-disk learning state: {profile_key: profile}."""

    def __init__(self, path=None):
        self._path_override = path
        self.profiles: dict[str, Profile] = {}
        self.loaded = False
        self._load_failed = False

    # -- io -----------------------------------------------------------
    @property
    def path(self):
        if self._path_override is not None:
            return Path(self._path_override)
        return _config.VRAM_LEARNING_FILE

    def load(self) -> "VramLearningStore":
        self.profiles = {}
        self.loaded = True
        self._load_failed = False
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return self
        except OSError:
            self._load_failed = True
            return self
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            self._load_failed = True
            return self
        if not isinstance(data, dict) or int(data.get("schema") or 0) != SCHEMA_VERSION:
            self._load_failed = True
            self._backup_incompatible()
            return self
        profiles = data.get("profiles")
        if isinstance(profiles, dict):
            for key, value in profiles.items():
                if isinstance(key, str):
                    self.profiles[key] = Profile.from_json(key, value)
        self._trim()
        return self

    def _backup_incompatible(self):
        try:
            target = self.path.with_name(self.path.name + f".v{SCHEMA_VERSION}.bak")
            if not target.exists():
                os.replace(self.path, target)
        except OSError:
            pass

    def save(self) -> bool:
        self._trim()
        payload = {
            "schema": SCHEMA_VERSION,
            "profiles": {k: p.to_json() for k, p in self.profiles.items()},
        }
        path = self.path
        tmp = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            return True
        except OSError:
            try:
                if tmp is not None and tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            return False

    def _trim(self):
        if len(self.profiles) <= MAX_PROFILES:
            return
        ordered = sorted(self.profiles.values(), key=lambda p: p.updated_utc, reverse=True)
        keep = {p.key for p in ordered[:MAX_PROFILES]}
        self.profiles = {k: v for k, v in self.profiles.items() if k in keep}

    # -- use ----------------------------------------------------------
    def profile(self, key: str, *, gpu_name: str = "", gpu_total_bytes: int = 0,
                engine_build: str = "", model_fp: str = "") -> Profile:
        prof = self.profiles.get(key)
        if prof is None:
            prof = Profile(key=key, gpu_name=gpu_name, gpu_total_bytes=gpu_total_bytes,
                           engine_build=engine_build, model_fp=model_fp)
            self.profiles[key] = prof
        return prof

    def predict(self, est, *, key: str, batch_size: int,
                gpu_total_bytes: int | None = None) -> Prediction:
        """V_hat = V_phys + r_hat, V_safe = V_hat + U."""
        prof = self.profiles.get(key)
        x = feature_vector(est, batch_size)
        r_hat_gib = prof.rls.predict_residual(x) if prof else 0.0
        residuals = prof.residuals if prof else []
        bound = safety_bound(est.phys_bytes, residuals, gpu_total_bytes)
        v_hat = max(0, int(round(est.phys_bytes + r_hat_gib * GIB)))
        u = int(bound.u_bytes)
        return Prediction(
            v_hat_bytes=v_hat, u_bytes=u, v_safe_bytes=v_hat + u,
            residual_hat_gib=r_hat_gib,
            n=len(residuals), band=bound.band, dominant=bound.dominant,
            confidence=CONFIDENCE_BY_BAND.get(bound.band, "low"),
        )

    def record(self, est, *, key: str, measured_peak_bytes: int, outcome: RunOutcome,
               batch_size: int, gpu_total_bytes: int | None = None,
               gpu_name: str = "", engine_build: str = "", model_fp: str = "",
               extra: dict | None = None) -> LearnResult:
        """Feed one finished run. Rejected runs change nothing at all."""
        if not outcome.learnable:
            return LearnResult(False, outcome.reject_reason or "rejected", 0.0, 0.0)
        phys = int(est.phys_bytes)
        if phys <= 0 or int(measured_peak_bytes) <= 0:
            return LearnResult(False, "no_measurement", 0.0, 0.0)
        residual_gib = (int(measured_peak_bytes) - phys) / GIB
        if not math.isfinite(residual_gib):
            return LearnResult(False, "non_finite", 0.0, 0.0)
        cap = (gpu_total_bytes or 0) / GIB
        if cap > 0 and abs(residual_gib) > 2.0 * cap:
            return LearnResult(False, "implausible", residual_gib, 0.0)

        prof = self.profile(key, gpu_name=gpu_name,
                            gpu_total_bytes=int(gpu_total_bytes or 0),
                            engine_build=engine_build, model_fp=model_fp)
        if prof.gpu_total_bytes and gpu_total_bytes and prof.gpu_total_bytes != int(gpu_total_bytes):
            # The key already carries the size; a mismatch means a stale caller,
            # and teaching one card from another's samples is never right.
            return LearnResult(False, "profile_mismatch", residual_gib, 0.0)
        x = feature_vector(est, batch_size)
        try:
            innovation = prof.rls.update(x, residual_gib)
        except ValueError:
            return LearnResult(False, "rls_unstable", residual_gib, 0.0)
        prof.residuals.append(_r(residual_gib))
        del prof.residuals[:-MAX_RESIDUALS]
        sample = {"utc": _utc_now(), "peak_gib": _r(measured_peak_bytes / GIB),
                  "phys_gib": _r(phys / GIB), "residual_gib": _r(residual_gib),
                  "batch": int(batch_size or 0)}
        if extra:
            sample.update({k: extra[k] for k in ("ctx", "budget", "reserve", "ratio")
                           if k in extra})
        prof.samples.append(sample)
        del prof.samples[:-MAX_RESIDUALS]
        prof.updated_utc = _utc_now()
        return LearnResult(True, "", residual_gib, innovation)

    def summary(self) -> dict:
        return {k: {"n": p.n, "updated": p.updated_utc, "gpu": p.gpu_name,
                    "total_mib": int(p.gpu_total_bytes / MIB)}
                for k, p in self.profiles.items()}


def read_state(path=None) -> VramLearningStore:
    return VramLearningStore(path).load()
