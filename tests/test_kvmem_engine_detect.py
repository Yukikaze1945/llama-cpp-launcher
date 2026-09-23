# -*- coding: utf-8 -*-
"""Stage 4: the engine registry and how a binary is matched to an engine.

Three behaviours are pinned here, because each one is a way the launcher could
silently drive the wrong server:

  * **file-name detection** — `llama-kvmem-server.exe` (and its version-suffixed
    variants) resolve to kvmem, everything else to llama.cpp, so an install that
    never heard of kvmem behaves exactly as it does today;
  * **--help fingerprinting** — a renamed binary is still recognised from its
    help text, and the golden v0.16.0-rc2 capture is what proves the
    fingerprint tokens are real rather than invented;
  * **probe legality** — kvmem rejects `--version` and `--list-devices` with
    exit 1, so the flags that gate those probes must stay false.
"""
import pathlib

import pytest

from core import engine as E

DATA = pathlib.Path(__file__).parent / "data"
KVMEM_HELP = (DATA / "kvmem_help_v0.16.0-rc2.txt").read_text(encoding="utf-8")
LLAMA_HELP = (DATA / "llama-server-help-b10825.txt").read_text(encoding="utf-8")


def test_registry_contents():
    assert set(E.ENGINES) == {"llama", "kvmem"}
    assert E.DEFAULT_ENGINE_ID == "llama"
    assert E.get_engine_by_id("kvmem") is E.KVMEM
    assert E.get_engine_by_id("llama") is E.LLAMA
    # An unknown/empty id in settings.json must not brick startup.
    assert E.get_engine_by_id("nonsense") is E.LLAMA
    assert E.get_engine_by_id("") is E.LLAMA
    assert E.get_engine_by_id(None) is E.LLAMA


@pytest.mark.parametrize("path,expected", [
    (r"F:\llama-kvmem-rc2\bin\llama-kvmem-server.exe", "kvmem"),
    (r"f:/llama-kvmem-rc2/bin/llama-kvmem-server.exe", "kvmem"),
    (r"D:\x\llama-kvmem-server-0.16.exe", "kvmem"),
    (r"C:\llama.cpp\build\bin\llama-server.exe", "llama"),
    (r"C:\llama.cpp\build\bin\llama-server-cuda.exe", "llama"),
    # The CLI sibling is not a server: pointing at it must not look valid.
    (r"D:\x\llama-kvmem-cli.exe", "llama"),
    (r"", "llama"),
    (None, "llama"),
])
def test_detect_engine_by_file_name(path, expected):
    assert E.detect_engine(path).id == expected


def test_help_fingerprints_match_the_captures():
    assert E.KVMEM.help_matches(KVMEM_HELP)
    assert not E.KVMEM.help_matches(LLAMA_HELP)
    assert E.LLAMA.help_matches(LLAMA_HELP)
    assert not E.LLAMA.help_matches(KVMEM_HELP)
    assert E.engine_from_help(KVMEM_HELP) is E.KVMEM
    assert E.engine_from_help(LLAMA_HELP) is E.LLAMA
    assert E.engine_from_help("") is None
    assert E.engine_from_help("usage: something-else") is None


def test_kvmem_probes_that_would_exit_1_are_switched_off():
    # Measured on v0.16.0-rc2: both flags answer `unknown flag` and exit 1.
    assert E.KVMEM.probe_version is False
    assert E.KVMEM.probe_devices is False
    assert E.LLAMA.probe_version is True
    assert E.LLAMA.probe_devices is True
    assert "identity_file" in E.KVMEM.supports
    assert "identity_file" not in E.LLAMA.supports
    assert E.LLAMA.read_identity is None
    assert E.LLAMA.identity(r"C:\x\llama-server.exe") is None


def test_engines_expose_their_own_schema_and_baseline():
    from core import params_schema, kvmem_params_schema
    assert E.LLAMA.schema is params_schema
    assert E.KVMEM.schema is kvmem_params_schema
    assert E.LLAMA.params is not E.KVMEM.params
    assert len(E.LLAMA.params) == 226          # unchanged from main
    # The baseline is the *binary's* default: 8080, not our recommended 18200.
    assert E.KVMEM.fallback_defaults()["port"] == 8080
    assert E.KVMEM.initial_overrides()["port"] == 18200
    assert E.LLAMA.initial_overrides() == {}
    assert "port" in E.KVMEM.no_drift_keys()
    assert E.KVMEM.no_drift_keys() <= frozenset(
        p.key for p in E.KVMEM.schema.PARAMS)


def test_bare_name_and_explicit_path_rule():
    assert E.KVMEM.bare_name() == "llama-kvmem-server"
    assert E.LLAMA.bare_name() == "llama-server"
    assert E.KVMEM.needs_explicit_path is True
    # shutil.which on PATH is llama.cpp's historical behaviour.
    assert E.LLAMA.needs_explicit_path is False


def test_validation_problems_only_kvmem_has_them():
    assert E.LLAMA.validation_problems({"temp": 99}) == []
    # The engine's own seeding state must be startable as-is.
    kvmem = E.KVMEM.fallback_defaults()
    kvmem.update(E.KVMEM.initial_overrides())
    assert E.KVMEM.validation_problems(kvmem) == []
    # An out-of-range temperature is the sentinel-leak case: kvmem refuses it.
    problems = dict(E.KVMEM.validation_problems(dict(kvmem, temperature=5.0)))
    assert "temperature" in problems
    # Never raises, whatever it is handed.
    assert E.KVMEM.validation_problems({}) is not None
    assert E.LLAMA.validation_problems({}) == []


def test_server_path_resolution_is_per_engine(monkeypatch, tmp_path):
    """kvmem uses its own key, and only adopts the legacy key for its own binary."""
    from core import config as CC
    fake = tmp_path / "llama-kvmem-server.exe"
    fake.write_bytes(b"MZ")
    other = tmp_path / "llama-server.exe"
    other.write_bytes(b"MZ")
    paths = {"llama": "", "kvmem": ""}

    def fake_configured(engine_id):
        return paths.get(engine_id, "")

    monkeypatch.setattr(CC, "get_configured_server_path", fake_configured)
    monkeypatch.setattr(CC, "get_server_path", lambda: str(other))

    assert E.LLAMA.server_path() == str(other)
    assert E.KVMEM.server_path() == ""          # nothing configured yet
    paths["kvmem"] = str(fake)
    assert E.KVMEM.server_path() == str(fake)
    paths["kvmem"] = str(tmp_path / "missing.exe")
    assert E.KVMEM.server_path() == ""          # configured but gone
    # Legacy key holding a kvmem binary: honoured, no settings rewrite needed.
    paths["kvmem"] = ""
    paths["llama"] = str(fake)
    assert E.KVMEM.server_path() == str(fake)
    paths["llama"] = str(other)
    assert E.KVMEM.server_path() == ""


def test_resolve_startup_engine(monkeypatch, tmp_path):
    """Preference wins; with nothing configured anywhere startup stays llama."""
    from core import config as CC
    monkeypatch.setattr(CC, "load_preferred_engine_id", lambda: "kvmem")
    assert E.resolve_startup_engine() is E.KVMEM
    # An explicit llama.cpp choice is a preference too, not "no preference".
    monkeypatch.setattr(CC, "load_preferred_engine_id", lambda: "llama")
    assert E.resolve_startup_engine() is E.LLAMA
    monkeypatch.setattr(CC, "load_preferred_engine_id", lambda: "")
    monkeypatch.setattr(CC, "get_configured_server_path", lambda engine_id: "")
    assert E.KVMEM.server_path() == ""
    assert E.resolve_startup_engine() is E.LLAMA
    # The sniffing case: only the legacy key is set and it names a kvmem
    # binary — that install gets the kvmem schema with no settings rewrite.
    fake = tmp_path / "llama-kvmem-server.exe"
    fake.write_bytes(b"MZ")
    monkeypatch.setattr(CC, "get_configured_server_path",
                        lambda engine_id: str(fake) if engine_id == "llama" else "")
    assert E.resolve_startup_engine() is E.KVMEM


def test_explicit_llama_choice_survives_a_restart(tmp_path, monkeypatch):
    """Regression: both binaries configured, the user picks llama.cpp, restarts.

    settings.json is the only thing a restart carries over, so this is the real
    sequence — both paths saved, then the 设置 → 引擎 choice, with no loader
    monkeypatched and the path resolution cache dropped in between (that cache
    is the one piece of in-process state a new process would not have).
    """
    from core import config as CC
    settings = tmp_path / "settings.json"
    monkeypatch.setattr(CC, "SETTINGS_FILE", settings)

    def restart():
        monkeypatch.setattr(CC, "_server_path_cache", None)

    restart()
    llama_exe = tmp_path / "llama-server.exe"
    llama_exe.write_bytes(b"MZ")
    kvmem_exe = tmp_path / "llama-kvmem-server.exe"
    kvmem_exe.write_bytes(b"MZ")
    CC.save_server_path(str(llama_exe), engine_id="llama")
    CC.save_server_path(str(kvmem_exe), engine_id="kvmem")
    # Setup check: this install really is the ambiguous one. Without a recorded
    # choice, sniffing finds the kvmem binary first.
    assert CC.load_preferred_engine_id() == ""
    restart()
    assert E.resolve_startup_engine() is E.KVMEM
    # 设置 → 引擎 → llama.cpp, then a restart.
    CC.save_preferred_engine_id("llama")
    restart()
    assert E.resolve_startup_engine() is E.LLAMA
    # Symmetrically, choosing kvmem sticks (and clearing the choice goes back
    # to sniffing, which is what the pre-kvmem install relied on).
    CC.save_preferred_engine_id("kvmem")
    restart()
    assert E.resolve_startup_engine() is E.KVMEM
    CC.save_preferred_engine_id("")
    restart()
    assert E.resolve_startup_engine() is E.KVMEM
