# -*- coding: utf-8 -*-
"""kvmem-llama.cpp (--help) -> default values: the second engine's defaults module.

Same public surface as core/defaults.py (``_FALLBACK_DEFAULTS``,
``_parse_help_to_defaults``, ``get_default_params``, ``get_chat_templates``,
``USER_INPUT_PARAMS``) so stage 4's engine registry can hand either engine to
the launcher, but the *rules* are this binary's, verified against
v0.16.0-rc2:

  * llama.cpp writes ``-c, --ctx-size N   (default: 4096)``; kvmem never uses
    ``default:`` — it writes ``(default 2048)``, so the extraction regex and
    the "what counts as the default token" rule are different (asserted in
    tests/data/kvmem_help_v0.16.0-rc2.txt: the captured help contains no
    ``default:`` at all).  Reusing llama.cpp's ``_extract_default_from_text``
    unchanged would find nothing on every kvmem line and silently freeze the
    baseline at our table's values — the drift check would always report
    "matches", which is worse than not having it.
  * kvmem wording a widget cannot select is normalised
    (``--kvmem-mtp-state`` prints ``(default replay with MTP)``).
  * ``NO_DRIFT_KEYS`` values are never adopted, so a future binary cannot move
    the baseline out from under a value this engine always sends.
  * ``--help`` goes to stderr and the exit code is nonzero; core.defaults's
    ``_run_server_command``/``fetch_help_text`` already join both streams and
    ignore the return code, so the plumbing is shared.

This module is Qt-free and never imported by the llama.cpp engine.
"""
import logging
import re

from core import defaults as _base
from core import kvmem_params_schema as K

logger = logging.getLogger(__name__)

#: Baseline for CommandBuilder.is_default() and the drift comparison — the
#: binary's own defaults, NOT the IQ3 recipe (that only seeds the UI state;
#: see kvmem_params_schema.INITIAL_OVERRIDES).
_FALLBACK_DEFAULTS = K.fallback_defaults()

#: Reverse flag index built from the kvmem table (shared core with llama.cpp).
_FLAG_INDEX = _base.index_for_schema(K)

#: What the drift check / validator ignores as per-user input (kvmem keys).
USER_INPUT_PARAMS = K.USER_INPUT_PARAMS

# "(default 2048)", "(default 512; qw3-style)", "(default off; RAM until NVMe
# flush)". Anchored right after the opening parenthesis on purpose: kvmem also
# writes "-ctk, --cache-type-k TYPE  GPU K cache type (llama.cpp name; default
# q8_0)", where `default` sits mid-parenthetical. That trailing token is prose
# about the naming, and adopting q8_0 there would mis-report the empty
# "follow --kv-dtype" state as drift, so the tight anchor is a feature — pinned
# by tests/test_kvmem_help_parse.py.
_KVMEM_DEFAULT_RE = re.compile(r"\(\s*default\s+([^)]+?)\s*\)")

#: Help wording -> the value the combo/spin actually holds.
_KVMEM_DEFAULT_NORMALIZE = {
    "kvmem_mtp_state": {"replay with MTP": "replay"},
}

#: kvmem's "not set" wording (only reachable on keys we index).
_KVMEM_PLACEHOLDER_KEYS = ("reasoning_budget_message", "chat_template",
                           "chat_template_file", "ui_dir", "mmproj")


def _extract_kvmem_default(text, flags):
    """(combined help entry, that param's flags) -> raw default string | None."""
    m = _KVMEM_DEFAULT_RE.search(text)
    if not m:
        return None
    # `(default 512; qw3-style)` -> "512; qw3-style"; int()/float() then fail
    # and the fallback stays 512, which is what this binary actually does.
    return m.group(1).rstrip(" ,;)")


def _parse_help_to_defaults(help_text):
    """kvmem --help -> {key: value} for every kvmem param."""
    return _base.parse_help_to_defaults(
        help_text, _FLAG_INDEX, _FALLBACK_DEFAULTS,
        extract_default=_extract_kvmem_default,
        normalize=_KVMEM_DEFAULT_NORMALIZE,
        skip_keys=K.NO_DRIFT_KEYS,
        placeholder_keys=_KVMEM_PLACEHOLDER_KEYS,
    )


def get_default_params(server_path=None, help_text=None):
    """Defaults for one kvmem binary. Never probes a path we were not given:
    the launcher's single `server_path` may point at llama-server.exe, and
    running *that* with --help here would produce a kvmem baseline polluted by
    llama.cpp values."""
    if help_text is None:
        if not server_path:
            logger.warning("kvmem defaults: no server path given, using the built-in baseline")
            return dict(_FALLBACK_DEFAULTS)
        help_text = _base.fetch_help_text(server_path)
    if help_text and help_text.strip():
        return _parse_help_to_defaults(help_text)
    return dict(_FALLBACK_DEFAULTS)


def get_chat_templates(server_path=None, help_text=None):
    """kvmem ships no built-in template list (its --help has no
    `list of built-in templates:`), so the GUI's template combo stays a
    free-text editor. Returning [] is the truthful answer, not a failed probe."""
    return []


def parse_device_list(text):
    """kvmem rejects --list-devices, so there is nothing to parse (the GPU
    layer-count hint is therefore never shown for this engine)."""
    return []
