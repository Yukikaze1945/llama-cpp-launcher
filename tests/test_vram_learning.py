# -*- coding: utf-8 -*-
"""Layers 2 and 3: the RLS residual fit and the one-sided safety bound.

The contract these tests protect is narrower than "accurate": layer 1 predicts a
physical peak, layer 2 learns how that prediction was *wrong* (the residual,
never the peak itself), and layer 3 turns the recorded residuals into an upper
bound U that is allowed to be pessimistic but not optimistic — an
over-estimate costs headroom, an under-estimate costs an OOM.

Everything here is arithmetic on numbers; no Qt, no GPU, no parameter writes.
"""
import json
import math

import pytest

from core import vram_estimator as ve
from core import vram_learning as vl

GIB = ve.GIB
MIB = ve.MIB


class _Est:
    """The five attributes feature_vector() reads off a VramEstimate."""

    def __init__(self, weight=10.0, kv=1.0, mtp=0.0, mmproj=0.0, phys=None):
        self.weight_bytes = int(weight * GIB)
        self.kv_bytes = int(kv * GIB)
        self.mtp_bytes = int(mtp * GIB)
        self.mmproj_bytes = int(mmproj * GIB)
        self.phys_bytes = int((phys if phys is not None
                               else weight + kv + mtp + mmproj) * GIB)


def _store(tmp_path):
    return vl.VramLearningStore(tmp_path / "vram_learning.json").load()


# --------------------------------------------------------------------------
# identity and features
# --------------------------------------------------------------------------

def test_fingerprint_is_stable_and_sensitive_to_what_shapes_vram():
    args = ("qwen35", 64, 512, 10 * GIB, "IQ3_S", True)
    assert vl.model_fingerprint(*args) == vl.model_fingerprint(*args)
    assert len(vl.model_fingerprint(*args)) == 16
    for over in [("qwen3next", 64, 512, 10 * GIB, "IQ3_S", True),
                 ("qwen35", 65, 512, 10 * GIB, "IQ3_S", True),
                 ("qwen35", 64, 513, 10 * GIB, "IQ3_S", True),
                 ("qwen35", 64, 512, 11 * GIB, "IQ3_S", True),
                 ("qwen35", 64, 512, 10 * GIB, "Q8_0", True),
                 ("qwen35", 64, 512, 10 * GIB, "IQ3_S", False)]:
        assert vl.model_fingerprint(*over) != vl.model_fingerprint(*args)


def test_fingerprint_never_mentions_a_path():
    fp = vl.model_fingerprint("qwen35", 64, 512, 10 * GIB, "IQ3_S", True)
    assert "/" not in fp and "\\" not in fp and fp.isalnum()


def test_profile_key_isolates_gpu_card_size_build_and_model():
    base = vl.profile_key("RTX 4090", 24 * GIB, "v0.16.0-rc2 (4837d45)", "abc")
    assert base != vl.profile_key("RTX 5060 Ti", 24 * GIB, "v0.16.0-rc2 (4837d45)", "abc")
    assert base != vl.profile_key("RTX 4090", 16 * GIB, "v0.16.0-rc2 (4837d45)", "abc")
    assert base != vl.profile_key("RTX 4090", 24 * GIB, "v0.16.0-rc1", "abc")
    assert base != vl.profile_key("RTX 4090", 24 * GIB, "v0.16.0-rc2 (4837d45)", "def")


def test_feature_vector_is_in_gib_with_the_batch_term():
    x = vl.feature_vector(_Est(weight=10, kv=1.5, mtp=0.25, mmproj=0.125), 1024)
    assert x == pytest.approx([1.0, 10.0, 1.5, 0.25, 0.125, 1.0])
    assert vl.feature_vector(_Est(), 128)[5] == pytest.approx(-2.0)
    assert vl.feature_vector(_Est(), 0)[5] == pytest.approx(math.log2(1 / 512.0))


# --------------------------------------------------------------------------
# RLS: the spec's update, checked against the formula rather than itself
# --------------------------------------------------------------------------

def test_first_update_matches_the_textbook_rls_step():
    m = vl.RlsModel()
    x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    y = 2.0
    lam = vl.FORGETTING_FACTOR
    p0 = vl.P_INITIAL
    denom = lam + x[0] * p0 * x[0]
    gain = p0 / denom

    innovation = m.update(x, y)

    assert innovation == pytest.approx(y)                 # theta started at 0
    assert m.theta[0] == pytest.approx(gain * y)
    assert m.p_matrix[0][0] == pytest.approx((p0 - gain * p0) / lam)
    assert m.p_matrix[1][1] == pytest.approx(p0 / lam)   # untouched direction inflates
    assert m.p_matrix[0][1] == pytest.approx(0.0, abs=1e-9)
    assert m.n == 1


def test_update_returns_the_innovation_against_the_old_theta():
    m = vl.RlsModel()
    x = [1.0, 2.0, 0.0, 0.0, 0.0, 0.0]
    m.update(x, 3.0)
    predicted = m.predict_residual(x)
    assert m.update(x, 5.0) == pytest.approx(5.0 - predicted)


def test_repeated_identical_samples_converge_to_the_observed_residual():
    m = vl.RlsModel()
    x = [1.0, 8.0, 1.7, 0.2, 0.0, 0.0]
    for _ in range(300):
        m.update(x, 1.25)
    assert m.predict_residual(x) == pytest.approx(1.25, abs=1e-3)


def test_p_matrix_stays_symmetric():
    m = vl.RlsModel()
    for i in range(12):
        m.update([1.0, float(i), 0.5 * i, 0.1, 0.0, 0.25], 0.4 + 0.1 * i)
    for r in range(vl.N_FEATURES):
        for c in range(vl.N_FEATURES):
            assert m.p_matrix[r][c] == pytest.approx(m.p_matrix[c][r], abs=1e-9)


def test_an_unusable_denominator_raises_instead_of_dividing_by_zero():
    m = vl.RlsModel()
    m.lam = 0.0
    m.p_matrix = [[0.0] * vl.N_FEATURES for _ in range(vl.N_FEATURES)]
    with pytest.raises(ValueError):
        m.update([1.0] * vl.N_FEATURES, 1.0)


def test_rls_state_survives_json_roundtrip():
    m = vl.RlsModel()
    for i in range(5):
        m.update([1.0, 2.0, 0.0, 0.5 * i, 0.0, 0.0], 0.75 + i)
    back = vl.RlsModel.from_json(json.loads(json.dumps(m.to_json())))
    x = [1.0, 2.0, 0.0, 0.5 * 5, 0.0, 0.0]
    assert back.predict_residual(x) == pytest.approx(m.predict_residual(x), abs=1e-5)
    assert back.n == m.n == 5
    assert vl.RlsModel.from_json({"theta": "junk"}).theta == [0.0] * vl.N_FEATURES


# --------------------------------------------------------------------------
# p95 and the three safety bands
# --------------------------------------------------------------------------

def test_p95_is_the_nearest_rank_and_one_sided_in_use():
    assert vl.p95([]) == 0.0
    assert vl.p95([1.0]) == 1.0
    assert vl.p95([1.0, 2.0, 3.0, 4.0, 5.0]) == 5.0     # ceil(0.95*5)=5
    assert vl.p95(list(range(1, 21))) == 19.0           # ceil(0.95*20)=19
    assert vl.p95([-1.0, -0.5, 0.25]) == 0.25


def test_no_samples_uses_the_structural_floor():
    # n=0: U = max(1 GiB, 8 % of the physical model) — the 1 GiB floor is what
    # protects a small model, the percentage what protects a big one.
    b = vl.safety_bound(10 * GIB, [], 16 * GIB)
    assert (b.band, b.dominant, b.u_bytes) == ("structural", "floor_1gib", GIB)
    big = vl.safety_bound(20 * GIB, [], 16 * GIB)
    assert big.dominant == "eight_percent"
    assert big.u_bytes == int(round(0.08 * 20 * GIB))
    assert big.u_bytes > GIB


def test_one_to_four_samples_take_the_worst_positive_error():
    b = vl.safety_bound(10 * GIB, [0.5, -2.0, 1.5], 16 * GIB)
    assert b.band == "small_sample"
    assert b.u_bytes == int(round(1.5 * GIB)) and b.dominant == "max_positive_error"
    # 512 MiB minimum, and a run that only over-estimated still keeps a margin
    floor = vl.safety_bound(10 * GIB, [-2.0, -1.0], 16 * GIB)
    assert floor.u_bytes == 512 * MIB and floor.dominant == "floor_512mib"


def test_five_or_more_samples_take_the_maximum_of_p95_three_percent_and_256mib():
    residuals = [0.1] * 4 + [2.0]
    b = vl.safety_bound(10 * GIB, residuals, 16 * GIB)
    assert b.band == "empirical" and b.dominant == "p95"
    assert b.u_bytes == int(round(2.0 * GIB))
    small = vl.safety_bound(10 * GIB, [0.01] * 5, 64 * GIB)
    assert small.dominant == "three_percent_gpu" and small.u_bytes == int(round(0.03 * 64 * GIB))
    tiny = vl.safety_bound(10 * GIB, [0.01] * 5, 1 * GIB)
    assert tiny.dominant == "floor_256mib" and tiny.u_bytes == 256 * MIB


def test_negative_residuals_never_raise_the_bound():
    positive = vl.safety_bound(10 * GIB, [0.5, 0.4, 0.3, 0.2, 0.1], 16 * GIB)
    with_noise = vl.safety_bound(10 * GIB, [-9.0, 0.4, 0.3, 0.2, 0.1], 16 * GIB)
    assert positive.u_bytes > with_noise.u_bytes


# --------------------------------------------------------------------------
# predict()
# --------------------------------------------------------------------------

def test_prediction_adds_the_learned_residual_and_the_bound(tmp_path):
    store = _store(tmp_path)
    est = _Est(weight=10, kv=2)
    key = vl.profile_key("GPU A", 16 * GIB, "rc2", "fp")

    cold = store.predict(est, key=key, batch_size=512, gpu_total_bytes=16 * GIB)
    assert cold.n == 0 and cold.band == "structural" and cold.confidence == "low"
    assert cold.residual_hat_gib == 0.0
    assert cold.v_hat_bytes == est.phys_bytes
    assert cold.v_safe_bytes == cold.v_hat_bytes + cold.u_bytes

    outcome = vl.RunOutcome(ready=True, clean_stop=True, sample_valid=True)
    for _ in range(6):
        store.record(est, key=key, measured_peak_bytes=int(10.0 * GIB),
                     outcome=outcome, batch_size=512, gpu_total_bytes=16 * GIB,
                     gpu_name="GPU A", engine_build="rc2", model_fp="fp")
    warm = store.predict(est, key=key, batch_size=512, gpu_total_bytes=16 * GIB)
    assert warm.n == 6 and warm.band == "empirical" and warm.confidence == "high"
    # V_hat is what was measured, not what the model guessed
    assert warm.v_hat_bytes == pytest.approx(int(10.0 * GIB), rel=0.05)
    assert warm.v_safe_bytes > warm.v_hat_bytes


def test_an_unknown_profile_predicts_the_physical_model_unchanged(tmp_path):
    store = _store(tmp_path)
    est = _Est(weight=10, kv=2)
    p = store.predict(est, key="nobody|0|here|yet", batch_size=512)
    assert (p.v_hat_bytes, p.residual_hat_gib, p.n) == (est.phys_bytes, 0.0, 0)


# --------------------------------------------------------------------------
# record(): what may teach the model, and what must not touch it
# --------------------------------------------------------------------------

def _good_outcome():
    return vl.RunOutcome(ready=True, clean_stop=True, sample_valid=True)


def test_a_successful_run_learns_the_residual_not_the_peak(tmp_path):
    store = _store(tmp_path)
    est = _Est(weight=10, kv=2)                       # V_phys = 12 GiB
    key = "GPU|1|build|fp"
    res = store.record(est, key=key, measured_peak_bytes=int(13.5 * GIB),
                       outcome=_good_outcome(), batch_size=512, gpu_total_bytes=16 * GIB,
                       gpu_name="GPU", engine_build="build", model_fp="fp")
    assert res.learned and res.reason == ""
    assert res.residual_gib == pytest.approx(1.5)
    prof = store.profiles[key]
    assert prof.residuals == [pytest.approx(1.5)]
    assert prof.rls.n == 1
    # the fit moves toward the residual instead of memorising the peak
    assert 0.0 < prof.rls.predict_residual(vl.feature_vector(est, 512)) < 1.5
    assert prof.samples[-1]["peak_gib"] == pytest.approx(13.5)
    assert prof.samples[-1]["phys_gib"] == pytest.approx(12.0)


@pytest.mark.parametrize("over,reason", [
    (dict(oom=True), "oom"),
    (dict(ready=False), "startup_failed"),
    (dict(param_error=True), "param_error"),
    (dict(clean_stop=False), "unclean_stop"),
    (dict(sample_valid=False, sample_reason="noisy"), "noisy"),
    (dict(sample_valid=False), "sample_invalid"),
])
def test_a_rejected_run_changes_nothing_at_all(tmp_path, over, reason):
    store = _store(tmp_path)
    est = _Est(weight=10, kv=2)
    key = "GPU|1|build|fp"
    before = store.predict(est, key=key, batch_size=512)
    res = store.record(est, key=key, measured_peak_bytes=int(20 * GIB),
                       outcome=vl.RunOutcome(**{"ready": True, "clean_stop": True,
                                                "sample_valid": True, **over}),
                       batch_size=512, gpu_total_bytes=16 * GIB, gpu_name="GPU",
                       engine_build="build", model_fp="fp")
    assert not res.learned and res.reason == reason
    assert store.profiles == {} or store.profiles[key].n == 0
    after = store.predict(est, key=key, batch_size=512)
    assert (after.v_hat_bytes, after.u_bytes, after.n) == \
           (before.v_hat_bytes, before.u_bytes, before.n)


def test_an_oom_run_is_never_persisted_even_after_other_success(tmp_path):
    store = _store(tmp_path)
    est = _Est(weight=10, kv=2)
    key = "GPU|1|build|fp"
    store.record(est, key=key, measured_peak_bytes=int(13 * GIB),
                 outcome=_good_outcome(), batch_size=512, gpu_total_bytes=16 * GIB,
                 gpu_name="GPU", engine_build="build", model_fp="fp")
    assert store.save()
    path = store.path
    n, residuals = store.profiles[key].n, list(store.profiles[key].residuals)

    bad = store.record(est, key=key, measured_peak_bytes=int(15.9 * GIB),
                       outcome=vl.RunOutcome(ready=True, clean_stop=True, oom=True),
                       batch_size=512, gpu_total_bytes=16 * GIB, gpu_name="GPU",
                       engine_build="build", model_fp="fp")
    assert not bad.learned and bad.reason == "oom"
    assert store.profiles[key].n == n
    assert store.profiles[key].residuals == residuals
    assert path.read_text(encoding="utf-8").count("residual_gib") == 1


@pytest.mark.parametrize("peak,reason", [
    (0, "no_measurement"),
    (-5, "no_measurement"),
    (int(60 * GIB), "implausible"),          # twice a 16 GiB card's worth of residual
])
def test_silly_measurements_are_refused(tmp_path, peak, reason):
    store = _store(tmp_path)
    res = store.record(_Est(), key="k", measured_peak_bytes=peak,
                       outcome=_good_outcome(), batch_size=512,
                       gpu_total_bytes=16 * GIB)
    assert not res.learned and res.reason == reason
    assert store.profiles == {}


def test_a_card_size_that_disagrees_with_the_key_is_not_taught(tmp_path):
    store = _store(tmp_path)
    est = _Est()
    key = vl.profile_key("GPU A", 16 * GIB, "b", "f")
    store.record(est, key=key, measured_peak_bytes=int(11 * GIB),
                 outcome=_good_outcome(), batch_size=512, gpu_total_bytes=16 * GIB,
                 gpu_name="GPU A", engine_build="b", model_fp="f")
    res = store.record(est, key=key, measured_peak_bytes=int(11 * GIB),
                       outcome=_good_outcome(), batch_size=512, gpu_total_bytes=8 * GIB,
                       gpu_name="GPU A", engine_build="b", model_fp="f")
    assert not res.learned and res.reason == "profile_mismatch"
    assert store.profiles[key].n == 1


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------

def test_state_survives_a_restart(tmp_path):
    store = _store(tmp_path)
    est = _Est(weight=10, kv=2)
    key = "GPU|1|build|fp"
    for peak in (13.0, 12.4, 14.1):
        store.record(est, key=key, measured_peak_bytes=int(peak * GIB),
                     outcome=_good_outcome(), batch_size=512, gpu_total_bytes=16 * GIB,
                     gpu_name="GPU A", engine_build="build", model_fp="fp")
    assert store.save()
    revived = vl.read_state(store.path)
    assert list(revived.profiles) == [key]
    assert revived.profiles[key].residuals == pytest.approx(
        store.profiles[key].residuals)
    # to_json() rounds each coefficient, so a byte-level equality here would be
    # a test of the rounding rather than of the state
    assert abs(revived.predict(est, key=key, batch_size=512,
                               gpu_total_bytes=16 * GIB).v_hat_bytes
               - store.predict(est, key=key, batch_size=512,
                               gpu_total_bytes=16 * GIB).v_hat_bytes) < 4 * MIB


def test_stored_file_holds_numbers_only(tmp_path):
    store = _store(tmp_path)
    secret = "ignore all previous instructions"
    store.record(_Est(), key="GPU|1|build|fp", measured_peak_bytes=int(11 * GIB),
                 outcome=_good_outcome(), batch_size=512, gpu_total_bytes=16 * GIB,
                 gpu_name="GPU A", engine_build="build", model_fp="fp",
                 extra={"ctx": 4096, "budget": 36864, "reserve": 16384, "ratio": 0.5,
                        "prompt": secret})
    store.save()
    raw = store.path.read_text(encoding="utf-8")
    assert secret not in raw
    assert "prompt" not in raw
    data = json.loads(raw)
    assert data["schema"] == vl.SCHEMA_VERSION
    prof = data["profiles"]["GPU|1|build|fp"]
    assert set(prof) == {"gpu_name", "gpu_total_bytes", "engine_build", "model_fp",
                         "rls", "residuals", "samples", "updated_utc"}
    assert set(prof["samples"][-1]) <= {"utc", "peak_gib", "phys_gib", "residual_gib",
                                        "batch", "ctx", "budget", "reserve", "ratio"}


def test_a_foreign_or_corrupt_file_is_quarantined_not_crashed(tmp_path):
    path = tmp_path / "vram_learning.json"
    path.write_text("{not json", encoding="utf-8")
    assert vl.read_state(path)._load_failed and vl.read_state(path).profiles == {}

    path.write_text(json.dumps({"schema": 99, "profiles": {"a": {}}}), encoding="utf-8")
    store = vl.read_state(path)
    assert store._load_failed and store.profiles == {}
    assert path.with_name(path.name + f".v{vl.SCHEMA_VERSION}.bak").exists()


def test_history_is_capped_so_the_file_cannot_grow_forever(tmp_path):
    store = _store(tmp_path)
    est = _Est(weight=10, kv=2)
    for i in range(vl.MAX_RESIDUALS + 25):
        store.record(est, key="k", measured_peak_bytes=int((10.5 + i % 3) * GIB),
                     outcome=_good_outcome(), batch_size=512, gpu_total_bytes=64 * GIB)
    prof = store.profiles["k"]
    assert len(prof.residuals) == vl.MAX_RESIDUALS
    assert len(prof.samples) == vl.MAX_RESIDUALS

    for i in range(vl.MAX_PROFILES + 10):
        store.profile(f"extra-{i}")
    store.save()
    assert len(vl.read_state(store.path).profiles) == vl.MAX_PROFILES


def test_summary_reports_what_a_user_would_want_to_see(tmp_path):
    store = _store(tmp_path)
    store.record(_Est(), key="GPU|1|b|f", measured_peak_bytes=int(11 * GIB),
                 outcome=_good_outcome(), batch_size=512, gpu_total_bytes=16 * GIB,
                 gpu_name="GPU A", engine_build="b", model_fp="f")
    s = store.summary()["GPU|1|b|f"]
    assert (s["n"], s["gpu"], s["total_mib"]) == (1, "GPU A", 16384)


# --------------------------------------------------------------------------
# run gating and the measurement hygiene
# --------------------------------------------------------------------------

@pytest.mark.parametrize("line,expected", [
    ("ggml_backend_cuda0: error: failed to allocate 1024 MiB of device memory", True),
    ("llm_load_tensors: out of memory", True),
    ("CUDA error: cudaErrorMemoryAllocation", True),
    ("kvmem: failed to reserve KV pool", True),
    ("kvmem: KV pool allocation failed", True),
    ("kvmem: unsupported method name", False),
    ("srv  init: not enough free memory", True),
    (" registering model #0 is ready", False),
    ("srv  init: cannot allocate 0 threads", True),
])
def test_oom_detection(line, expected):
    assert vl.detect_oom([line]) is expected


def test_oom_detection_survives_noise_and_empties():
    assert vl.detect_oom(None) is False
    assert vl.detect_oom([]) is False
    lines = ["a"] * 500 + ["out of memory"]
    assert vl.detect_oom(lines) is True


def _measure(base=(3000, 3000, 3000, 3000, 3000), run=(4000, 9000, 5000),
             tail=(3000, 3000, 3000, 3000, 3000), gpu_total=16 * GIB,
             skip_tail=False, stop_at="done"):
    pm = vl.PeakMeasure(gpu_total)
    pm.start_baseline()
    for v in base:
        pm.add(v)
    if stop_at in ("baseline",):
        return pm
    pm.begin_run()
    for v in run:
        pm.add(v)
    if stop_at == "running":
        return pm
    pm.end_run()
    if stop_at == "after":
        return pm
    for v in ([] if skip_tail else tail):
        pm.add(v)
    return pm


def test_peak_is_the_maximum_above_the_median_baseline():
    pm = _measure()
    assert pm.baseline_bytes() == 3000
    assert pm.peak_bytes() == 9000 - 3000
    assert pm.tail_bytes() == 3000
    assert pm.verdict() == (True, "")


def test_an_even_baseline_median_averages_the_middle_two():
    pm = _measure(base=(1000, 2000, 3000, 4000, 5000))
    assert pm.baseline_bytes() == 3000
    pm2 = _measure(base=(1000, 2000, 3000, 5000, 4000))
    assert pm2.baseline_bytes() == pm.baseline_bytes()   # order-independent


@pytest.mark.parametrize("kwargs,reason", [
    (dict(base=(1, 2, 3)), "no_baseline"),
    (dict(run=()), "no_samples"),
    (dict(stop_at="running"), "no_tail"),
    (dict(stop_at="after"), "no_tail"),
    (dict(skip_tail=True), "no_tail"),
])
def test_the_verdict_ladder_names_the_first_thing_that_is_missing(kwargs, reason):
    pm = _measure(**kwargs)
    usable, got = pm.verdict()
    assert not usable and got == reason


def test_a_drifting_card_is_noisy_and_therefore_unusable():
    # drift above max(256 MiB, 2 % of the card) means someone else was working
    thr = vl._sampler.noisy_threshold_bytes(16 * GIB)
    quiet = _measure(tail=(3000,) * 5)
    assert not quiet.noisy()
    disturbed = _measure(tail=(3000 + thr + 1,) * 5)
    assert disturbed.noisy()
    assert disturbed.verdict() == (False, "noisy")
    # exactly at the threshold is still allowed: the rule is >, not >=
    assert not _measure(tail=(3000 + thr,) * 5).noisy()


def test_no_growth_is_rejected_because_it_means_the_samples_were_wrong():
    pm = _measure(run=(3000, 3000, 3000))
    assert pm.peak_bytes() == 0
    assert pm.verdict() == (False, "no_growth")


def test_baseline_count_is_respected_and_extra_samples_ignored():
    pm = vl.PeakMeasure(16 * GIB)
    pm.start_baseline()
    for v in range(1, 50):
        pm.add(v)
    assert pm.baseline_bytes() == 3           # only the first BASELINE_SAMPLES
    assert pm.baseline_ready

