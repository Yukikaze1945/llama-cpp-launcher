# -*- coding: utf-8 -*-
"""kvmem-llama.cpp build identity — the only machine-readable version source.

kvmem-llama.cpp has **no** ``--version`` (the flag is rejected as unknown), so
the launcher cannot ask the binary who it is. The release layout does carry the
answer next to the executable:

    <install>/BUILD-INFO.json          version, base_commit, llama_commit,
                                       status, nvme_supported, build_options
    <install>/provenance/source-manifest.json   base_commit, llama_commit

Both are read here. Rules this module is built under:

  * **display only** — nothing may block or refuse a server start because an
    identity file is missing or malformed. Callers get ``None`` and fall back to
    a "version unknown" label; the server is still perfectly launchable.
  * stdlib only (json/pathlib) plus core.i18n — Qt-free, importable from a
    worker thread.
  * ``utf-8-sig``: the packaging script (PowerShell ``Out-File``) emits a BOM,
    and the same script pads every key with two spaces — ``json`` handles that,
    a hand-rolled parser would not.
"""
import json
import logging
from pathlib import Path

from core.i18n import t

logger = logging.getLogger(__name__)

#: How much of the commit hashes we show.
SHORT_COMMIT = 9


def _read_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except (OSError, IOError, json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.debug("kvmem identity: unreadable %s (%s)", path, e)
        return None
    return data if isinstance(data, dict) else None


def _candidate_files(exe_path: str):
    """BUILD-INFO.json next to the exe and one level up (the release layout has
    bin/llama-kvmem-server.exe and BUILD-INFO.json at the install root), then
    the provenance manifest."""
    root = Path(str(exe_path or "")).resolve().parent
    return [root / "BUILD-INFO.json",
            root.parent / "BUILD-INFO.json",
            root / "provenance" / "source-manifest.json",
            root.parent / "provenance" / "source-manifest.json"]


def read_identity(exe_path: str) -> dict | None:
    """{version, base_commit, llama_commit, status, nvme_supported, source}.

    None when the install ships no identity file (a hand-built tree) — the
    caller shows "版本未知" and starts the server anyway.
    """
    if not exe_path:
        return None
    for path in _candidate_files(exe_path):
        if not path.is_file():
            continue
        data = _read_json(path)
        if data is None:
            continue
        identity = {
            "version": str(data.get("version") or ""),
            "base_commit": str(data.get("base_commit") or ""),
            "llama_commit": str(data.get("llama_commit") or ""),
            "status": str(data.get("status") or ""),
            # Absent in the manifest: leave None (unknown) rather than False,
            # so the About box does not claim NVMe is compiled out when it
            # simply did not say.
            "nvme_supported": data.get("nvme_supported")
            if isinstance(data.get("nvme_supported"), bool) else None,
            "source": str(path),
        }
        if any(identity[k] for k in ("version", "base_commit", "llama_commit")):
            return identity
    return None


def _short(commit: str, n: int = SHORT_COMMIT) -> str:
    return commit[:n] if commit else ""


def describe(identity: dict | None) -> str:
    """One line for the version label: "v0.16.0-rc2 (4837d45be)"."""
    if not identity:
        return ""
    version, base = identity.get("version", ""), _short(identity.get("base_commit", ""))
    if version and base:
        return f"v{version} ({base})"
    if version:
        return f"v{version}"
    if base:
        return base
    return ""


def detail_lines(identity: dict | None) -> list:
    """Rows for the About dialog, already translated.

    Translated here rather than by the caller because each row is assembled
    from the identity file: `t()` needs the source template *and* the values,
    and a caller handed "版本: 0.16.0-rc2" could not look anything up.
    """
    if not identity:
        return []
    out = []
    if identity.get("version"):
        out.append(t("版本: {v}", v=identity["version"]))
    if identity.get("base_commit"):
        out.append(t("kvmem 提交: {c}", c=_short(identity["base_commit"], 12)))
    if identity.get("llama_commit"):
        out.append(t("llama.cpp 基线: {c}", c=_short(identity["llama_commit"], 12)))
    if identity.get("status"):
        out.append(t("构建状态: {s}", s=identity["status"]))
    nvme = identity.get("nvme_supported")
    if nvme is not None:
        out.append(t("NVMe 卸载: {v}", v=t("支持") if nvme else t("此构建已编译关闭")))
    return out
