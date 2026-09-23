# -*- coding: utf-8 -*-
"""Does the kvmem binary really accept every flag this schema can emit?

'"it appeared in --help" is NOT the criterion (the plan\'s hard requirement): the
authoritative sources are the argv parser and a live probe. Three tiers:

  Tier 1  every flag the schema can emit appears in the captured --help of the
          exact binary (tests/data/kvmem_help_v0.16.0-rc2.txt, usage line
          normalised so the fixture is not machine-specific).
  Tier 2  live oracle: run `<exe> <flag>` bare. kvmem answers
          `unknown flag: X` + usage + exit 1 for anything its if/else chain
          does not know, and a value flag answers `missing value for X` (both
          confirmed against v0.16.0-rc2). Skipped unless KVMEM_SERVER_EXE points
          at a binary — argument parsing happens before any model load, so the
          whole sweep costs ~10 s and is cheap enough to run per commit.
  Tier 3  the probed-rejected list stays out of the schema, so a future edit
          cannot add a control that makes the server exit 1 on start.
"""
import os
import subprocess

import pytest

from core import kvmem_params_schema as K

HELP_FIXTURE = os.path.join(os.path.dirname(__file__), "data",
                            "kvmem_help_v0.16.0-rc2.txt")
KVMEM_SERVER_EXE = os.environ.get("KVMEM_SERVER_EXE", "")


def _schema_flags():
    for p in K.PARAMS:
        flags = set()
        if isinstance(p.flag, str):
            flags.add(p.flag)
        elif p.flag:
            flags.update(p.flag)
        flags.update(p.flags)
        for f in sorted(flags):
            yield p, f


@pytest.fixture(scope="module")
def help_text():
    with open(HELP_FIXTURE, encoding="utf-8") as f:
        return f.read()


def test_help_fixture_is_the_captured_binary_help(help_text):
    assert help_text.startswith("usage: llama-kvmem-server.exe")
    assert "--kvmem-block-tokens" in help_text
    # --help carries no `default:` token at all; kvmem writes `(default X)`.
    assert "default:" not in help_text


def test_tier1_every_schema_flag_is_in_the_help(help_text):
    missing = [f"{p.key}:{f}" for p, f in _schema_flags() if f not in help_text]
    assert not missing, f"flags not present in kvmem --help: {missing}"


def test_tier3_rejected_flags_are_absent_from_the_schema(help_text):
    schema_flags = {f for _, f in _schema_flags()}
    for flag in K.REJECTED_FLAGS:
        assert flag not in schema_flags, flag
    # The help-visible-but-control-free params: listed by --help (so a reader
    # wonders why there is no control) yet never emitted by this schema.
    for flag, reason in K.NO_CONTROL_KEYS.items():
        assert flag not in schema_flags, flag
        assert reason and " " in reason


def test_tier3_the_two_lists_split_by_parser_behaviour():
    """`--jinja` used to be in both, and it is the accepted side that is right.

    REJECTED_FLAGS means "the argv parser refuses this" (`unknown flag` + exit
    1); NO_CONTROL_KEYS means "the parser takes it, this schema just gives it no
    widget". A flag cannot be both, and putting an accepted no-op in the first
    list mislabels what a future edit is allowed to emit.
    """
    assert not set(K.REJECTED_FLAGS) & set(K.NO_CONTROL_KEYS)
    assert "--no-jinja" in K.REJECTED_FLAGS   # refused
    assert "--jinja" in K.NO_CONTROL_KEYS     # accepted, and inert
    assert "--jinja" not in K.REJECTED_FLAGS
    # Help-visible is what makes NO_CONTROL_KEYS readable in the first place.
    with open(HELP_FIXTURE, encoding="utf-8") as f:
        help_text = f.read()
    for flag in K.NO_CONTROL_KEYS:
        assert flag in help_text, flag


def _probe(flag):
    """Lower-cased combined output of `<exe> <flag>`, or None when unrunnable."""
    try:
        proc = subprocess.run([KVMEM_SERVER_EXE, flag], capture_output=True,
                              text=True, timeout=60, encoding="utf-8",
                              errors="replace")
    except (OSError, subprocess.TimeoutExpired):
        return None
    return ((proc.stdout or "") + (proc.stderr or "")).lower()


@pytest.mark.skipif(not KVMEM_SERVER_EXE,
                    reason="set KVMEM_SERVER_EXE to probe the binary live")
def test_tier2_the_two_lists_behave_differently_in_the_parser():
    """Live version of the split above: every REJECTED_FLAGS entry answers
    `unknown flag`, and nothing in NO_CONTROL_KEYS does (a value flag answers
    `missing value` instead, which is still an accepted flag)."""
    problems = []
    for flag in K.REJECTED_FLAGS:
        out = _probe(flag)
        if out is None:
            problems.append(f"{flag}: cannot run {KVMEM_SERVER_EXE!r}")
        elif "unknown flag" not in out:
            problems.append(f"{flag}: expected `unknown flag`, got {out[:70]!r}")
    for flag in K.NO_CONTROL_KEYS:
        out = _probe(flag)
        if out is None:
            problems.append(f"{flag}: cannot run {KVMEM_SERVER_EXE!r}")
        elif "unknown flag" in out:
            problems.append(f"{flag}: listed as accepted, parser refuses it")
    assert not problems, "\n".join(problems)


@pytest.mark.skipif(not KVMEM_SERVER_EXE,
                    reason="set KVMEM_SERVER_EXE to probe the binary live")
def test_tier2_parser_accepts_every_schema_flag():
    problems = []
    for p, flag in _schema_flags():
        try:
            proc = subprocess.run([KVMEM_SERVER_EXE, flag],
                                  capture_output=True, text=True, timeout=60,
                                  encoding="utf-8", errors="replace")
        except (OSError, subprocess.TimeoutExpired) as e:
            problems.append(f"{flag}: cannot run ({e})")
            continue
        out = (proc.stdout or "") + (proc.stderr or "")
        low = out.lower()
        if "unknown flag" in low:
            problems.append(f"{p.key}: parser rejects {flag}")
        elif "missing value" in low or "requires a" in low:
            if p.parser == "bool":
                problems.append(f"{p.key}: {flag} takes a value but schema says bool")
        elif p.parser != "bool":
            problems.append(f"{p.key}: {flag} looks like a switch but schema says {p.parser}")
    assert not problems, "\n".join(problems)
