# -*- coding: utf-8 -*-
"""Stage 4: kvmem build identity — the only machine-readable version source.

kvmem-llama.cpp has no ``--version`` (the flag is rejected), so the version
label and the About box read ``BUILD-INFO.json`` from the install tree. Two
properties are pinned here because both are load-bearing elsewhere:

  * **display only** — every failure mode (no file, no read permission, a
    non-dict, invalid JSON) returns None instead of raising, so a hand-built
    tree still starts a server;
  * **the real layout** — PowerShell's ``Out-File`` pads keys with two spaces
    and may add a BOM, and the file sits one level *above* ``bin/``, which is
    why the fixture below is written that way rather than as a neat JSON.
"""
import json

import pytest

from core import kvmem_identity as KI


def _write(path, data, bom=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=4).replace('":', '":  ')
    raw = ("\ufeff" + text) if bom else text
    path.write_bytes(raw.encode("utf-8"))


INFO = {
    "version": "0.16.0-rc2",
    "base_commit": "4837d45bedbe7d3443d7de97ec22010dca84d718",
    "llama_commit": "b81c99b479d4c24e5eeca10de99032ebd343ef8f",
    "status": "experimental-windows-build",
    "nvme_supported": False,
    "component": "Runtime",
}


def _install(tmp_path):
    """The release layout: <root>/BUILD-INFO.json + <root>/bin/exe."""
    exe = tmp_path / "bin" / "llama-kvmem-server.exe"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_bytes(b"MZ")
    return exe


def test_reads_build_info_one_level_above_the_binary(tmp_path):
    exe = _install(tmp_path)
    _write(tmp_path / "BUILD-INFO.json", INFO)
    identity = KI.read_identity(str(exe))
    assert identity["version"] == "0.16.0-rc2"
    assert identity["base_commit"].startswith("4837d45be")
    assert identity["nvme_supported"] is False
    assert identity["source"].endswith("BUILD-INFO.json")


def test_reads_build_info_next_to_the_binary(tmp_path):
    """A tree without the bin/ split (an unzipped build folder) still works."""
    exe = tmp_path / "llama-kvmem-server.exe"
    exe.write_bytes(b"MZ")
    _write(tmp_path / "BUILD-INFO.json", INFO, bom=True)
    assert KI.read_identity(str(exe))["version"] == "0.16.0-rc2"


def test_provenance_manifest_is_the_fallback(tmp_path):
    """No BUILD-INFO anywhere: the manifest answers commits but not NVMe."""
    exe = _install(tmp_path)
    _write(tmp_path / "provenance" / "source-manifest.json",
           {"base_commit": INFO["base_commit"], "llama_commit": INFO["llama_commit"]})
    identity = KI.read_identity(str(exe))
    assert identity["version"] == ""
    assert identity["base_commit"].startswith("4837d45be")
    # Absent, unlike `false`: the About box must not claim NVMe is compiled out.
    assert identity["nvme_supported"] is None


@pytest.mark.parametrize("content", [
    "",                     # empty file
    "not json at all",
    "[1, 2, 3]",            # valid JSON, wrong shape
    '{"version": ""}',      # readable but says nothing
])
def test_unusable_files_degrade_to_none(tmp_path, content):
    exe = _install(tmp_path)
    (tmp_path / "BUILD-INFO.json").write_text(content, encoding="utf-8")
    assert KI.read_identity(str(exe)) is None


def test_missing_file_and_missing_path_degrade_to_none(tmp_path):
    exe = _install(tmp_path)
    assert KI.read_identity(str(exe)) is None      # no identity file at all
    assert KI.read_identity("") is None
    assert KI.read_identity(None) is None
    assert KI.describe(None) == ""
    assert KI.detail_lines(None) == []


def test_describe_is_the_version_label_line():
    identity = {"version": "0.16.0-rc2", "base_commit": "4837d45bedbe7d34"}
    assert KI.describe(identity) == "v0.16.0-rc2 (4837d45be)"
    assert KI.describe({"version": "0.16.0", "base_commit": ""}) == "v0.16.0"
    assert KI.describe({"version": "", "base_commit": "abcdef123456"}) == "abcdef123"
    assert KI.describe({"version": "", "base_commit": ""}) == ""


def test_detail_lines_only_show_what_the_file_says():
    lines = KI.detail_lines(dict(INFO, nvme_supported=True))
    assert any("0.16.0-rc2" in ln for ln in lines)
    assert any("4837d45bedbe" in ln for ln in lines)     # 12 chars, not 9
    assert any("experimental-windows-build" in ln for ln in lines)
    assert len([ln for ln in lines if "NVMe" in ln]) == 1
    # nvme_supported None => no NVMe row at all, rather than a wrong one.
    assert not any("NVMe" in ln for ln in KI.detail_lines(
        {"version": "1", "nvme_supported": None}))
    # An empty identity contributes nothing rather than blank rows.
    assert KI.detail_lines({}) == []


def test_detail_lines_translate():
    import core.i18n as I
    original = I.get_language()
    try:
        I.set_language("en")
        en = KI.detail_lines(dict(INFO, nvme_supported=False))
        I.set_language("zh")
        zh = KI.detail_lines(dict(INFO, nvme_supported=False))
    finally:
        I.set_language(original)
    assert en != zh
    assert any("NVMe offload: compiled out of this build" in ln for ln in en)
    assert any("kvmem commit: 4837d45bedbe" in ln for ln in en)
    assert any("此构建已编译关闭" in ln for ln in zh)


def test_engine_wires_the_reader():
    from core import engine as E
    assert E.KVMEM.read_identity is KI.read_identity
    assert E.KVMEM.describe_identity is KI.describe
    assert "identity_file" in E.KVMEM.supports
