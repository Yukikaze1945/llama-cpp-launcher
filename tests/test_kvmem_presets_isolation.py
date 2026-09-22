# -*- coding: utf-8 -*-
"""Stage 5: config and preset isolation between the two engines.

The engines share `~/.llama-cpp-launcher` — a kvmem install should still see the
presets the user built for llama.cpp, and the whole point of the engine tag is
that it then *ignores* them. Three failure modes are pinned here, all of them
silent otherwise:

  * an engine-tagged preset loaded into the wrong engine injects flags the
    running binary rejects with exit 1 (`unknown flag`), which reads as a bug in
    that binary rather than as a mismatched file;
  * engine-scoped UI keys (`mode`, `quick_params`, …) written un-suffixed would
    make a kvmem window reopen llama.cpp's tab layout, and — worse for the
    official build — change the bytes of the `ui` dict it reads;
  * `server_paths` must not be consulted across engines: a kvmem path handed to
    the llama.cpp engine is a server that rejects most of its flags.
"""
import json

import pytest

import core.config as CC


@pytest.fixture
def env(tmp_path, monkeypatch):
    presets = tmp_path / "presets"
    presets.mkdir()
    monkeypatch.setattr(CC, "PRESETS_DIR", presets)
    monkeypatch.setattr(CC, "SETTINGS_FILE", tmp_path / "settings.json")
    return presets


@pytest.fixture(autouse=True)
def _no_server_path_cache_leak(monkeypatch):
    """Keep this file's stub exes out of the process-wide resolver cache.

    `get_server_path()` memoises into a module global that only
    `save_server_path()` clears, and this file both saves paths and resolves
    them while SETTINGS_FILE points at a tmp dir. Without the same reset
    tests/test_server_path.py performs, the 2-byte "MZ" stub written here would
    stay the session's resolved llama-server, and every MainWindow built by a
    later test would probe *that* from its startup thread instead of the real
    binary. (It presented as a native crash in tests/test_quick_params.py,
    several files and ~20 tests after this one.)
    """
    monkeypatch.setattr(CC, "_server_path_cache", None)
    yield


def _write_preset(presets, name, params, engine=None, version=None):
    data = {"name": name, "created": "2026-01-01T12:00:00", "params": params}
    if engine is not None:
        data["engine"] = engine
    if version is not None:
        data["version"] = version
    (presets / f"{name}.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _manager(engine_id="llama", defaults=None):
    from core import engine as E
    eng = E.get_engine_by_id(engine_id)
    return CC.ConfigManager(defaults=defaults or eng.fallback_defaults(),
                            schema=eng.schema, engine_id=eng.id)


def _names(items):
    return [p["name"] if isinstance(p, dict) else p for p in items]


def test_legacy_presets_are_llamas_and_are_still_listed(env):
    """Every pre-kvmem file has no `engine` field: that means llama.cpp."""
    _write_preset(env, "old", {"temp": 0.7})
    assert CC.ConfigManager.preset_engine_id({"params": {}}) == "llama"
    llama = _manager("llama")
    kvmem = _manager("kvmem")
    assert _names(llama.list_presets()) == ["old"]
    # The same file is invisible to kvmem rather than listed-and-refused.
    assert _names(kvmem.list_presets()) == []
    assert kvmem.load_preset("old") is False
    assert llama.load_preset("old")["temp"] == 0.7


def test_kvmem_presets_are_tagged_and_filtered(env):
    kvmem = _manager("kvmem")
    kvmem.save_preset("recipe", {"ctx_size": 262144, "kvmem_budget": 36864})
    data = json.loads((env / "recipe.json").read_text(encoding="utf-8"))
    assert data["engine"] == "kvmem"
    assert data["version"] == 2                 # llama.cpp keeps writing 1
    assert _names(_manager("llama").list_presets()) == []
    assert _names(kvmem.list_presets()) == ["recipe"]
    loaded = kvmem.load_preset("recipe")
    assert loaded["ctx_size"] == 262144 and loaded["kvmem_budget"] == 36864


def test_saving_for_llama_keeps_the_official_shape(env):
    _manager("llama").save_preset("plain", {"temp": 0.5})
    data = json.loads((env / "plain.json").read_text(encoding="utf-8"))
    # The official build reads this file unchanged: version 1 is what a legacy
    # preset has, and `engine: "llama"` is a key it ignores.
    assert data["version"] == 1
    assert data["engine"] == "llama"
    assert set(json.loads((env / "plain.json").read_text(encoding="utf-8"))) >= {
        "name", "version", "created", "params"}


def test_foreign_keys_are_dropped_both_ways(env):
    """A hand-copied preset must not inject the other engine's flags."""
    llama = _manager("llama")
    kvmem = _manager("kvmem")
    # llama-shaped params forced into a kvmem-tagged file
    _write_preset(env, "mixed", {
        "ctx_size": 8192,               # shared key, kvmem keeps it
        "flash_attn": True,             # llama only -> dropped
        "model_draft": "x.gguf",        # llama only -> dropped
        "temp": 0.6,                    # llama spelling of temperature
    }, engine="kvmem", version=2)
    loaded = kvmem.load_preset("mixed")
    assert loaded["ctx_size"] == 8192
    for key in ("flash_attn", "model_draft"):
        assert key not in loaded or loaded[key] == kvmem._defaults.get(key)
    assert "temp" not in kvmem.drop_foreign_keys({"temp": 0.6})
    assert kvmem.preset_stored_keys("mixed") == {"ctx_size"}
    # And the reverse direction, with a key kvmem does not define at all.
    _write_preset(env, "k2l", {"kvmem_budget": 4096, "temp": 0.3}, engine="llama")
    back = llama.drop_foreign_keys({"kvmem_budget": 4096, "temp": 0.3})
    assert "kvmem_budget" not in back and "temp" in back


def test_key_renames_survive_the_foreign_filter(env):
    """Migration runs before filtering, so a renamed key is not lost."""
    _write_preset(env, "old", {"checkpoint_every_n_tokens": 100,
                               "ctx_size_draft": 4096, "temp": 0.5})
    llama = _manager("llama")
    assert llama.preset_stored_keys("old") == {"checkpoint_min_step", "temp"}
    loaded = llama.load_preset("old")
    assert loaded["checkpoint_min_step"] == 100
    assert "ctx_size_draft" not in loaded


def test_server_paths_are_per_engine(tmp_path, monkeypatch):
    settings = tmp_path / "settings.json"
    monkeypatch.setattr(CC, "SETTINGS_FILE", settings)
    kvmem_exe = tmp_path / "llama-kvmem-server.exe"
    kvmem_exe.write_bytes(b"MZ")
    llama_exe = tmp_path / "llama-server.exe"
    llama_exe.write_bytes(b"MZ")

    CC.save_server_path(str(llama_exe), engine_id="llama")
    CC.save_server_path(str(kvmem_exe), engine_id="kvmem")
    assert CC.get_configured_server_path("llama") == str(llama_exe)
    assert CC.get_configured_server_path("kvmem") == str(kvmem_exe)
    stored = json.loads(settings.read_text(encoding="utf-8"))
    # Additive: the legacy key keeps its exact meaning, kvmem lives in its own.
    assert stored["server_path"] == str(llama_exe)
    assert stored["server_paths"] == {"kvmem": str(kvmem_exe)}
    assert CC.get_server_path() == str(llama_exe)
    # Writing the kvmem path must not disturb the llama.cpp resolution.
    CC.save_server_path("", engine_id="kvmem")
    assert CC.get_configured_server_path("llama") == str(llama_exe)


def test_legacy_server_path_honoured_only_for_its_own_binary(tmp_path, monkeypatch):
    """Today's real settings.json: `server_path` pointing at the kvmem binary."""
    settings = tmp_path / "settings.json"
    monkeypatch.setattr(CC, "SETTINGS_FILE", settings)
    from core import engine as E
    exe = tmp_path / "llama-kvmem-server.exe"
    exe.write_bytes(b"MZ")
    settings.write_text(json.dumps({"server_path": str(exe)}), encoding="utf-8")
    # kvmem picks it up with no settings rewrite...
    assert E.KVMEM.server_path() == str(exe)
    # ...and startup sniffing therefore selects the kvmem engine.
    assert E.resolve_startup_engine() is E.KVMEM
    # A llama-only path is never claimed by kvmem.
    other = tmp_path / "llama-server.exe"
    other.write_bytes(b"MZ")
    settings.write_text(json.dumps({"server_path": str(other)}), encoding="utf-8")
    assert E.KVMEM.server_path() == ""
    assert E.LLAMA.server_path() == str(other)


def test_preferred_engine_round_trip(tmp_path, monkeypatch):
    settings = tmp_path / "settings.json"
    monkeypatch.setattr(CC, "SETTINGS_FILE", settings)
    assert CC.load_preferred_engine_id() == ""       # absent = llama.cpp
    CC.save_preferred_engine_id("kvmem")
    assert CC.load_preferred_engine_id() == "kvmem"
    assert json.loads(settings.read_text(encoding="utf-8"))["engine"] == "kvmem"
    # Choosing llama.cpp removes the key rather than writing "llama", so the
    # settings file of someone who switches back is what the official build wrote.
    CC.save_preferred_engine_id("llama")
    assert CC.load_preferred_engine_id() == ""
    assert "engine" not in json.loads(settings.read_text(encoding="utf-8"))


def test_engine_scoped_ui_keys_keep_llama_bytes_identical(tmp_path, monkeypatch):
    settings = tmp_path / "settings.json"
    monkeypatch.setattr(CC, "SETTINGS_FILE", settings)
    CC.save_ui_prefs({"mode": "advanced", "adv_tab": "context",
                      "quick_params": ["temp"], "bottom_tab": "log",
                      "geometry": "AAA"}, engine_id="llama")
    original = json.loads(settings.read_text(encoding="utf-8"))["ui"]
    assert original == {"mode": "advanced", "adv_tab": "context",
                        "quick_params": ["temp"], "bottom_tab": "log",
                        "geometry": "AAA"}
    # A kvmem window now writes only suffixed keys...
    CC.save_ui_prefs({"mode": "simple", "adv_tab": "kvmem",
                      "quick_params": ["ctx_size"], "bottom_tab": "log",
                      "geometry": "BBB"}, engine_id="kvmem")
    after = json.loads(settings.read_text(encoding="utf-8"))["ui"]
    # The engine-scoped keys are what gets suffixed; the unsuffixed view of
    # them is byte-for-byte what llama.cpp wrote.
    assert {k: original[k] for k in CC.UI_PREF_ENGINE_KEYS if k in original} == \
           {k: after[k] for k in CC.UI_PREF_ENGINE_KEYS if k in after}
    assert after["mode#kvmem"] == "simple"
    assert after["adv_tab#kvmem"] == "kvmem"
    assert after["quick_params#kvmem"] == ["ctx_size"]
    # ...each engine reads back its own view, and geometry stays shared.
    assert CC.load_ui_prefs("llama")["mode"] == "advanced"
    assert CC.load_ui_prefs("kvmem")["mode"] == "simple"
    assert CC.load_ui_prefs("kvmem")["adv_tab"] == "kvmem"
    assert CC.load_ui_prefs("kvmem")["geometry"] == "BBB"
    assert CC.load_ui_prefs("llama")["geometry"] == "BBB"
    # A single-key write (quick params change) is scoped the same way.
    CC.save_ui_pref("quick_params", ["temperature"], engine_id="kvmem")
    assert CC.load_ui_prefs("llama")["quick_params"] == ["temp"]
    assert CC.load_ui_prefs("kvmem")["quick_params"] == ["temperature"]


def test_last_preset_is_per_engine(tmp_path, monkeypatch):
    settings = tmp_path / "settings.json"
    monkeypatch.setattr(CC, "SETTINGS_FILE", settings)
    CC.save_last_preset("llama-side", engine_id="llama")
    CC.save_last_preset("kvmem-side", engine_id="kvmem")
    assert CC.load_last_preset("llama") == "llama-side"
    assert CC.load_last_preset("kvmem") == "kvmem-side"
    # The official build's key name is unchanged.
    stored = json.loads(settings.read_text(encoding="utf-8"))
    assert stored["last_preset"] == "llama-side"
    assert stored["last_preset#kvmem"] == "kvmem-side"
    CC.save_last_preset("", engine_id="kvmem")
    assert CC.load_last_preset("kvmem") == ""
    assert CC.load_last_preset("llama") == "llama-side"


def test_import_preset_keeps_its_own_engine_tag(env):
    src = env.parent / "incoming.json"
    src.write_text(json.dumps({"name": "incoming", "engine": "kvmem",
                               "version": 2, "params": {"ctx_size": 1024}}),
                   encoding="utf-8")
    llama = _manager("llama")
    assert llama.import_preset(str(src)) is True
    stored = json.loads((env / "incoming.json").read_text(encoding="utf-8"))
    assert stored["engine"] == "kvmem"
    assert llama.list_presets() == []
    assert _names(_manager("kvmem").list_presets()) == ["incoming"]
    # A legacy file stays llama.cpp's even when imported by nothing in particular
    legacy = env.parent / "legacy.json"
    legacy.write_text(json.dumps({"name": "legacy", "params": {"temp": 0.2}}),
                      encoding="utf-8")
    assert llama.import_preset(str(legacy)) is True
    assert json.loads((env / "legacy.json").read_text(encoding="utf-8"))["engine"] == "llama"
