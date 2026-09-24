# -*- coding: utf-8 -*-
"""The exit path must destroy the widget tree while the QApplication is alive.

Why this file exists: with the KVMem VRAM card mounted, 2-8 of 12 launches died
with a 0xc0000005 inside sip's wrapper cast while `~QApplication` ran the tree
down — the destructor race `tests/conftest.py` documents, reached from the real
exit path instead of from a test frame.

A subprocess run of `main.main()` is the only way to observe that fault, and it
is a bad guard: it needs a GPU plus a kvmem engine config, costs seconds, and
catches a 2-in-12 crash 2 times in 12. So what is pinned here is the mechanism
and the call site, both deterministic:

* `_teardown_widget_tree()` really deletes the widgets — `deleteLater()` alone
  leaves them alive, because nothing services the deferred-delete queue once
  `exec()` has returned.
* `main()` runs it between `app.exec()` and `sys.exit()`, so a tail that drops
  back to a bare `sys.exit(rc)` fails here rather than in the field.
"""
import inspect

import pytest
from PyQt6.QtCore import QEvent
from PyQt6.QtWidgets import QApplication, QWidget

import main


@pytest.fixture
def app():
    return QApplication.instance()


def test_the_teardown_deletes_the_tree_instead_of_only_hiding_it(app):
    victim = QWidget()
    assert victim in app.topLevelWidgets()

    main._teardown_widget_tree(app)

    assert victim not in app.topLevelWidgets()
    # Deleted, not hidden: the C++ half is gone while the app is still running,
    # which is the whole point of doing it here instead of in ~QApplication.
    with pytest.raises(RuntimeError):
        victim.objectName()


def test_deleteLater_without_the_flush_is_what_leaves_a_widget_alive(app):
    """The `sendPostedEvents()` in the teardown is not decoration.

    Qt only services QEvent::DeferredDelete when control returns to the event
    loop, and after exec() has returned nothing runs the loop — so a teardown
    that forgets the flush deletes nothing.
    """
    victim = QWidget()
    victim.deleteLater()
    app.processEvents()
    assert victim in app.topLevelWidgets()

    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert victim not in app.topLevelWidgets()


def test_main_teardowns_the_tree_on_the_way_out():
    """A source check, deliberately: the behaviour it protects needs a crash.

    The two tests above cover the helper; this one covers the wiring, which
    otherwise only shows up as an exe that exits 139 on some machines. It is the
    narrowest form available — one call site, asserted between the loop and the
    exit — so a rename of the helper fails here with the message below rather
    than silently unregistering the guard.
    """
    src = inspect.getsource(main.main)
    teardown = src.find("_teardown_widget_tree(app)")
    assert teardown != -1, "main() must call _teardown_widget_tree(app) on exit"
    assert src.find("app.exec()") < teardown < src.find("sys.exit("), \
        "the tree has to go after the loop ends and before the process does"
