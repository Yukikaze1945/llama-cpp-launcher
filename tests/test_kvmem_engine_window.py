# -*- coding: utf-8 -*-
"""Stage 4/5: the kvmem engine's real window (MainWindow, not just its schema).

Everything here was broken by the pure-plumbing stages and no lower-level test
caught it, because only the window can:

  * the panels build their widgets from each Param's own default, so the engine
    recipe (port 18200, ctx 262144 …) seeded into `params` was read *back* off
    the untouched widgets by the first preview tick and reverted to the binary's
    defaults — acceptance item 5 ("--port 18200 really appears") failed silently;
  * `main_window._connect_signals` named `adv_webui`, a llama.cpp-only widget,
    so constructing the kvmem window raised AttributeError;
  * the basic panel is a fixed set of llama.cpp controls whose "0 = do not send"
    sentinels collide with kvmem's real value ranges, so kvmem's window must not
    show it (Engine.has_basic_mode).

One window per engine, built once for the module: a full MainWindow is the
most expensive object this suite creates, and the file used to build twelve of
them — enough Qt object churn to tip the suite's known PyQt6 teardown race
(see tests/conftest.py) into killing the interpreter in a *later* test file.
Most assertions below are reads; the stage-6 ones that push a log line through
a window restore the ring buffer they touched, so sharing stays safe. The
switch test is the one case that needs windows of its own.

The QApplication comes from conftest's session fixture; the pinned PyQt6 build
crashes when a test body constructs it through an extra Python frame.
"""
import os
import pathlib
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from core import engine as E
import ui.main_window as MW

SAMPLING_FLAGS = ("--temperature", "--top-p", "--top-k", "--min-p",
                  "--presence-penalty", "--frequency-penalty",
                  "--repeat-penalty", "--seed")


@pytest.fixture(scope="module")
def app():
    from PyQt6.QtWidgets import QApplication
    a = QApplication.instance() or QApplication([])
    yield a


@pytest.fixture(scope="module")
def isolated_config():
    """An empty settings.json and no presets for every window in this module.

    Hygiene first: a window writes its UI state when it closes, so sharing
    .cfg-sandbox would make these assertions depend on what an earlier run left
    behind (which mode a window opens in is stored there, and the engine
    preference is written by the switch test).

    It was also crashing the process. With the shared sandbox settings, a llama
    window followed by any other window died with 0xC0000409 (Windows
    fast-fail) — in whichever test ran next, with nothing on the traceback
    because the interpreter itself went down. An empty settings.json made the
    same sequence pass; whatever the C++-level cause, a window test that reads
    the developer's real config is not reproducible.

    Module scope because the windows are, too: core.config reads these module
    attributes at call time, and both must outlive the last window's closeEvent.
    """
    import core.config as CC
    root = pathlib.Path(tempfile.mkdtemp(prefix="kvmem-window-cfg-"))
    (root / "presets").mkdir()
    old_settings, old_presets = CC.SETTINGS_FILE, CC.PRESETS_DIR
    # get_server_path() memoises into a module global, and a window resolves it
    # at construction — so without this the temp dir's answer would be the
    # session's llama-server for every test after this module.
    old_path_cache = CC._server_path_cache
    CC.SETTINGS_FILE = root / "settings.json"
    CC.PRESETS_DIR = root / "presets"
    try:
        yield root
    finally:
        CC.SETTINGS_FILE, CC.PRESETS_DIR = old_settings, old_presets
        CC._server_path_cache = old_path_cache


@pytest.fixture(scope="module")
def no_server_probe(isolated_config):
    """No window here may probe a real binary at startup.

    MainWindow.__init__ finishes with _check_server_info(), which spawns a
    QThread running --version/--help against the configured server, so a test
    window is nondeterministic and ~1 s slower. Stubbing it on the *instance*
    does not work: the constructor has already started the thread by the time
    it returns, so the patch has to go on the class.

    Manual set/restore rather than the monkeypatch fixture, which is
    function-scoped and cannot be used from a module-scoped fixture. Depends on
    isolated_config only so that (module-scoped) settings patch is installed —
    and later undone — around this window's whole lifetime.
    """
    original = MW.MainWindow._check_server_info
    MW.MainWindow._check_server_info = lambda self: None
    try:
        yield
    finally:
        MW.MainWindow._check_server_info = original


def _make_window(engine):
    return MW.MainWindow(work_dir=None,
                         defaults=dict(engine.fallback_defaults()),
                         engine=engine)


@pytest.fixture(scope="module")
def kvmem_window(no_server_probe):
    win = _make_window(E.KVMEM)
    yield win
    win.close()


@pytest.fixture(scope="module")
def llama_window(no_server_probe):
    win = _make_window(E.LLAMA)
    yield win
    win.close()


def _argv(win):
    return " ".join(str(a) for a in win.cmd_builder.build(win.params))


def _shown(widget):
    """Would this widget be visible?

    isHidden() rather than isVisible(): these windows are never show()n from a
    test, and a widget inside an unshown window answers isVisible() False no
    matter what it was set to — the assertion would pass without checking
    anything.
    """
    return not widget.isHidden()


def test_kvmem_window_starts_from_the_recipe(kvmem_window):
    win = kvmem_window
    argv = _argv(win)
    assert "--port 18200" in argv, argv
    assert "-c 262144" in argv, argv
    assert "--kvmem-budget 36864" in argv, argv
    assert "--spec-type draft-mtp" in argv, argv
    # The widget, not just the dict: the preview reads the panel back.
    assert win.advanced_panel.get_values()["port"] == 18200
    # The baseline stays the binary's, so every recipe line is a real diff.
    assert win.defaults["port"] == 8080
    assert win.params["port"] == 18200


def test_kvmem_window_emits_no_untouched_sampling_flag(kvmem_window):
    argv = _argv(kvmem_window)
    assert [f for f in SAMPLING_FLAGS if f in argv] == [], argv
    values = kvmem_window.advanced_panel.get_values()
    for key in ("temperature", "top_p", "top_k", "min_p",
                "presence_penalty", "frequency_penalty", "repeat_penalty",
                "seed"):
        # The sentinel is reachable and unclamped in the widget itself.
        assert values[key] == E.KVMEM.schema.PARAMS_BY_KEY[key].default, key


def test_kvmem_window_is_advanced_only(kvmem_window):
    """The basic form is llama.cpp's control set — hidden, and never restored."""
    assert kvmem_window.is_advanced is True
    assert not _shown(kvmem_window.mode_combo)
    assert not _shown(kvmem_window.mode_label)
    assert not _shown(kvmem_window.basic_panel)
    assert _shown(kvmem_window.advanced_panel)
    assert E.KVMEM.has_basic_mode is False
    assert E.LLAMA.has_basic_mode is True


def test_llama_window_is_unchanged(llama_window):
    win = llama_window
    # Nothing saved a mode yet (isolated_config), so llama.cpp still opens in
    # basic mode with the selector shown — the point being that kvmem's window
    # is the one that forces advanced mode, not this one.
    assert win.is_advanced is False
    assert _shown(win.mode_combo)
    assert _shown(win.mode_label)
    assert _shown(win.basic_panel)
    assert _shown(win.advanced_panel) is False
    argv = _argv(win)
    # No recipe for llama.cpp, so the preview stays as bare as it always was.
    assert "--port" not in argv
    assert E.LLAMA.initial_overrides() == {}
    assert len(win._schema.PARAMS) == 226


def test_webui_button_follows_the_engines_own_flag(llama_window, kvmem_window):
    """llama.cpp: --webui (on by default). kvmem: --no-ui, inverted."""
    assert llama_window._webui_served({"webui": True}) is True
    assert llama_window._webui_served({"webui": False}) is False
    assert llama_window._webui_served({}) is True
    assert llama_window._ui_toggle_widget() is llama_window.advanced_panel.adv_webui

    # kvmem has no `webui` key at all: reading it there answered "no UI".
    assert kvmem_window._webui_served({"no_ui": False}) is True
    assert kvmem_window._webui_served({"no_ui": True}) is False
    assert kvmem_window._webui_served(kvmem_window.params) is True
    assert kvmem_window._ui_toggle_widget() is kvmem_window.advanced_panel.adv_no_ui


def test_kvmem_window_blocks_a_rejected_command_line(kvmem_window):
    """The pre-flight gate is the engine's, and it refuses instead of exiting 1.

    Values are passed as a copy: this window is shared by the module's tests,
    and a mutated `params` would change what the recipe test above asserts.
    """
    values = dict(kvmem_window.params, temperature=5.0, spec_type="ngram-3")
    assert kvmem_window._engine_validation_problems(kvmem_window.params) == []
    problems = dict(kvmem_window._engine_validation_problems(values))
    assert set(problems) == {"temperature", "spec_type"}
    assert "2.0" in problems["temperature"]
    assert "draft-mtp" in problems["spec_type"]


def test_llama_window_has_no_pre_flight_gate(llama_window):
    # Even a value llama.cpp would clamp: llama.cpp always used to start
    # without a check, and that stays true.
    assert llama_window._engine_validation_problems(
        dict(llama_window.params, temp=99.0)) == []


def test_title_and_menu_name_the_engine(llama_window, kvmem_window):
    assert "kvmem" not in llama_window.windowTitle()
    assert llama_window._server_path_menu_text().startswith("设置 llama-server")
    assert set(llama_window._engine_actions) == {"llama", "kvmem"}
    assert llama_window._engine_actions["llama"].isChecked()
    assert "kvmem-llama.cpp" in kvmem_window.windowTitle()
    assert "kvmem-llama.cpp" in kvmem_window._server_path_menu_text()
    assert kvmem_window._engine_actions["kvmem"].isChecked()
    assert not kvmem_window._engine_actions["llama"].isChecked()


def test_engine_switch_builds_the_other_window(app, no_server_probe):
    """Switching replaces the window, keeps it alive, and cleans up on close.

    Its own windows: the test drives _switch_engine, which closes the one it
    was called on, so the module-scoped pair cannot spare either of them.
    """
    MW._pending_engine_windows[:] = []
    llama = _make_window(E.LLAMA)
    try:
        llama._switch_engine("kvmem", confirm=False)
        # The replacement is the only window the list has to hold open: the old
        # one took itself out of it in closeEvent (it was never added, and a
        # second switch would otherwise leave its predecessor behind).
        assert [w.engine.id for w in MW._pending_engine_windows] == ["kvmem"]
        fresh = MW._pending_engine_windows[0]
        assert fresh is not llama
        assert fresh.is_advanced is True
        assert "--port 18200" in _argv(fresh)
        assert llama._switching_engine is True
        # Closing it must give the reference back, or switching back and forth
        # piles windows up for the rest of the session.
        fresh.close()
        assert list(MW._pending_engine_windows) == []
    finally:
        llama.close()
        MW._pending_engine_windows[:] = []


# --------------------------------------------------------------------------
# stage 6: what a kvmem run shows the user when it dies


REJECTED = "llama-kvmem-server: unknown flag: --parallel"


def _feed(win, line):
    """Push one line through the window's real log entry point.

    Returns what the window had in its recent-line ring before this test needed
    it, so the caller can hand the shared window back as it found it.
    """
    saved = list(win._recent_log_lines)
    win._recent_log_lines.clear()
    win._append_log_line(line)
    return saved


def _restore(win, saved):
    win._recent_log_lines.clear()
    win._recent_log_lines.extend(saved)


def test_the_level_filter_is_llama_cpps_only(kvmem_window, llama_window):
    """llama.cpp prefixes every line with a level token; rc2's server does not,
    so the selector would filter nothing while hiding the exit-1 lines."""
    assert [b.isHidden() for b in llama_window._log_level_boxes.values()] \
        == [False] * 4
    assert [b.isHidden() for b in kvmem_window._log_level_boxes.values()] \
        == [True] * 4


def test_kvmem_window_explains_its_own_rejection(kvmem_window):
    """The classification is reached through the window, not just the module:
    `_append_log_line` -> `_recent_log_lines` -> `_engine_error_report` is the
    whole chain acceptance item 7 rides on."""
    saved = _feed(kvmem_window, REJECTED)
    try:
        hits = kvmem_window._engine_error_report()
    finally:
        _restore(kvmem_window, saved)
    assert [h["param"] for h in hits] == ["extra_args"]
    assert "--parallel" in hits[0]["message"]
    assert hits[0]["line"] == REJECTED


def test_the_llama_window_still_has_no_post_mortem_report(llama_window):
    """It has never had one: llama.cpp's failure story is its own, and an error
    classification written against another binary's wording must not appear
    there just because a line happens to match."""
    saved = _feed(llama_window, REJECTED)
    try:
        assert llama_window._engine_error_report() == []
    finally:
        _restore(llama_window, saved)


def test_the_report_opens_the_control_that_can_fix_it(kvmem_window):
    """A dialog that says "extra args" without switching to the tab holding them
    leaves the user hunting; a key with no row must not fake a jump either."""
    panel = kvmem_window.advanced_panel
    assert kvmem_window._param_label("extra_args")
    assert kvmem_window._param_label("") == ""
    assert kvmem_window._param_label("not_a_param") == ""
    before = panel.tabs.currentIndex()
    try:
        assert kvmem_window._jump_to_param("extra_args") is True
        assert panel.tab_keys()[panel.tabs.currentIndex()] == "server"
        assert kvmem_window._jump_to_param("not_a_param") is False
    finally:
        panel.tabs.setCurrentIndex(before)
