# -*- coding: utf-8 -*-
"""Engine isolation: the VRAM feature must not reach the llama.cpp path.

The spec's binding constraint is "不要修改 kvmem-llama.cpp 源码，也不要改变
llama.cpp 引擎现有行为", so the assertions here are all about the *absence* of
the feature on a llama.cpp window:

  * the install gate is `"param_linkage" in engine.supports` — the one place
    the card is mounted, and `vram_prediction` is only ever a kvmem capability;
  * a llama.cpp window holds no predictor, no card widget, no panel hook;
  * the prediction never reaches a command line: the sole route from a number
    to argv is the click that writes the `--kvmem-budget` widget, and that
    widget does not exist in the llama.cpp schema;
  * a window whose card could not be mounted behaves exactly like a llama.cpp
    one — every entry point is inert, so install() returning False cannot leave
    a half-wired predictor behind.

One window per engine, module-scoped, for the reason recorded in
tests/test_kvmem_engine_window.py: a full MainWindow is the most expensive
object this suite builds, and the empty settings.json below is what keeps the
llama-then-anything sequence from fast-failing the interpreter.
"""
import os
import pathlib
import tempfile
from contextlib import ExitStack

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from core import engine as E
import core.config as CC
import ui.main_window as MW
import ui.vram_monitor as vm
from tests.conftest import silenced_qt_method
from ui.vram_panel import VramPanel

GIB = 1 << 30


@pytest.fixture(scope="module")
def app():
    from PyQt6.QtWidgets import QApplication
    a = QApplication.instance() or QApplication([])
    yield a


@pytest.fixture(scope="module")
def isolated_config():
    """Fresh settings, no presets, and a learning file nothing may create.

    `VRAM_LEARNING_FILE` joins the list because the last test below reads it
    back: `install()` loads the store, and a load of an absent file must stay
    absent. Module scope so both windows — and their closes — live inside it.
    """
    root = pathlib.Path(tempfile.mkdtemp(prefix="vram-isolation-cfg-"))
    (root / "presets").mkdir()
    old = (CC.SETTINGS_FILE, CC.PRESETS_DIR, CC.VRAM_LEARNING_FILE, CC._server_path_cache)
    CC.SETTINGS_FILE = root / "settings.json"
    CC.PRESETS_DIR = root / "presets"
    CC.VRAM_LEARNING_FILE = root / "vram_learning.json"
    CC._server_path_cache = {}
    try:
        yield root
    finally:
        (CC.SETTINGS_FILE, CC.PRESETS_DIR, CC.VRAM_LEARNING_FILE,
         CC._server_path_cache) = old


@pytest.fixture(scope="module")
def no_probe(isolated_config):
    """No binary probing and no GPU polling thread from these windows.

    Both patches have to go on the class, not the instance: `_check_server_info`
    and `VramMonitor.start` have already run by the time the constructor
    returns. With the monitor never started, the kvmem card's backend stays
    "none" — which is also the deterministic half of the launch-delay test
    below. `silenced_qt_method` rather than a plain assignment for `start`,
    which QThread owns in C++ (see tests/conftest.py).
    """
    with ExitStack() as stack:
        stack.enter_context(silenced_qt_method(MW.MainWindow, "_check_server_info"))
        stack.enter_context(silenced_qt_method(vm.VramMonitor, "start"))
        yield


def _make_window(engine):
    return MW.MainWindow(work_dir=None,
                         defaults=dict(engine.fallback_defaults()),
                         engine=engine)


@pytest.fixture(scope="module")
def llama_window(no_probe):
    win = _make_window(E.LLAMA)
    yield win
    win.close()


@pytest.fixture(scope="module")
def kvmem_window(no_probe):
    win = _make_window(E.KVMEM)
    yield win
    win.close()


def _argv(win):
    return " ".join(str(a) for a in win.cmd_builder.build(win.params))


def _cards(win):
    """Cards mounted on this window: only an inserted panel is a descendant of
    it, and `install()` inserts after its page lookup succeeds."""
    return list(win.findChildren(VramPanel))


# -- the capability table -------------------------------------------------

def test_vram_prediction_is_a_kvmem_capability_only():
    assert "vram_prediction" in E.KVMEM.supports
    assert "vram_prediction" not in E.LLAMA.supports
    # The gate itself: `param_linkage` is what mounts the card, so a llama.cpp
    # engine has to keep answering "no" for every VRAM-related capability.
    assert "param_linkage" not in E.LLAMA.supports
    assert E.LLAMA.supports == frozenset({
        "chat_templates", "list_devices", "version_flag",
        "log_level_selector", "gpu_layer_hint", "basic_mode"})


# -- what a llama.cpp window holds ---------------------------------------

def test_llama_window_has_no_predictor_and_no_card(llama_window):
    assert llama_window._vram is None
    assert _cards(llama_window) == []


def test_llama_panel_keeps_its_hook_lists_empty(llama_window, kvmem_window):
    """The panel is built by the same code for both engines; kvmem's linkage is
    the only difference, so an empty hook list is what "unchanged" means."""
    assert llama_window.advanced_panel._retranslate_extras == []
    assert kvmem_window.advanced_panel._retranslate_extras != []
    assert len(_cards(kvmem_window)) == 1
    assert len(_cards(llama_window)) == 0


def test_llama_schema_has_no_budget_widget_for_the_card_to_write(llama_window, kvmem_window):
    assert llama_window.advanced_panel.param_widget("kvmem_budget") is None
    assert kvmem_window.advanced_panel.param_widget("kvmem_budget") is not None


def test_llama_command_line_is_still_the_shared_schema_table(llama_window):
    """Nothing in the emission path was specialised for the new feature."""
    from core import params_schema as PS
    from ui.command_builder import CommandBuilder
    assert llama_window.cmd_builder.params is PS.PARAMS
    independent = CommandBuilder(llama_window.defaults)
    assert [str(a) for a in llama_window.cmd_builder.build(llama_window.params)] == \
        [str(a) for a in independent.build(llama_window.params)]
    argv = _argv(llama_window)
    assert "--kvmem" not in argv and "--spec-" not in argv, argv


# -- the hooks the window calls on every run ------------------------------

def test_launch_is_never_delayed_by_a_feature_with_no_numbers(llama_window, kvmem_window):
    """`_vram_arm()` returns 0 for both engines here, for different reasons:
    no predictor at all, and a predictor whose backend never reported."""
    assert llama_window._vram is None
    assert llama_window._vram_arm() == 0
    assert kvmem_window._vram is not None
    assert kvmem_window._vram_arm() == 0
    assert kvmem_window._pending_launch is None


def test_a_missing_card_leaves_no_half_wired_predictor(llama_window, isolated_config):
    """`install()` answers False when its page is not there — the same path a
    llama.cpp window would take if the gate ever changed. Every entry point
    after that has to be inert, because the window keeps using them."""
    predictor = vm.VramPredictor(llama_window, llama_window._get_current_values)
    assert predictor.install() is False
    assert predictor._panel is None
    assert _cards(llama_window) == []
    assert llama_window.advanced_panel._retranslate_extras == []
    assert predictor._monitor.isRunning() is False

    assert predictor.arm_launch() == 0
    predictor.begin_run()
    predictor.note_state("running")
    predictor.end_run({}, [])
    predictor.model_changed()
    predictor.refresh()
    predictor.retranslate()
    predictor.shutdown()
    assert predictor._monitor.isRunning() is False
    llama_window._vram = None


def test_kvmem_argv_moves_only_on_the_click(kvmem_window):
    """The whole point of the read-only card: N refreshes change no parameter,
    one click changes exactly one."""
    panel = _cards(kvmem_window)[0]
    predictor = kvmem_window._vram
    before = kvmem_window.params["kvmem_budget"]
    for _ in range(20):
        predictor.refresh()
    assert kvmem_window.params["kvmem_budget"] == before
    assert panel.apply_btn.isEnabled() is False

    panel.apply_btn.click()          # nothing was suggested, so nothing is written
    assert kvmem_window.params["kvmem_budget"] == before


# -- the on-disk state ----------------------------------------------------

def test_nothing_here_has_written_the_learning_file(isolated_config):
    """Loading is read-only, and a run has to end before anything is stored."""
    assert (isolated_config / "vram_learning.json").exists() is False
