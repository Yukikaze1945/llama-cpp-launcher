# -*- coding: utf-8 -*-
"""Engine registry: one row = one server binary the launcher can drive.

The launcher was hard-wired to ``llama-server.exe`` in four places: which
parameter table the panels build, which baseline ``is_default()`` compares
against, how ``--help`` is parsed, and which ``--version``/``--list-devices``
probes are legal. An Engine bundles exactly those, so everything else in the app
stays engine-agnostic and keeps taking ``schema=`` / ``defaults=`` arguments
(stage 1's seam).

Two rules this module exists to enforce:

  * **A schema is not filtered by engine id.** ``llama`` and ``kvmem`` share
    only ~8 key names, and where they overlap the value ranges and even the
    meaning differ (``ctx_size`` 4096 vs 2048, ``spec_type`` including
    ``ngram-*`` vs only ``none|draft-mtp``). One merged table with an
    ``engines=`` field could not express two defaults for one key, so each
    engine owns its own module.
  * **Detection is by file name first, then by --help fingerprint.** kvmem
    rejects unknown flags with exit 1, so launching a llama-shaped command
    line at it is a hard failure that looks like kvmem's bug; the fingerprint
    is what lets a path the user set themselves be validated before saving.

Falling back to ``llama`` when nothing matches keeps the original behaviour
byte-for-byte for every existing installation.
"""
from dataclasses import dataclass
from types import ModuleType
from typing import Callable, Optional

from core import defaults as llama_defaults
from core import defaults_kvmem as kvmem_defaults
from core import kvmem_identity
from core import params_schema as llama_schema
from core import kvmem_params_schema as kvmem_schema

#: Engine id used when settings.json says nothing (every pre-kvmem install).
DEFAULT_ENGINE_ID = "llama"


@dataclass(frozen=True)
class Engine:
    id: str
    #: Shown in the version label / About / menus (a product name, not i18n).
    display_name: str
    #: Parameter table module: PARAMS / UI_PARAMS / TAB_TITLES / tab_params /
    #: fallback_defaults / help_flag_maps / schema_i18n_strings (+ optional
    #: QUICK_DEFAULT_KEYS, BASIC_OWNED_KEYS, USER_INPUT_PARAMS, NO_DRIFT_KEYS,
    #: INITIAL_OVERRIDES, migrate_preset_keys).
    schema: ModuleType
    #: Per-engine --help parsing: _FALLBACK_DEFAULTS / _parse_help_to_defaults /
    #: get_default_params / get_chat_templates / parse_device_list.
    defaults: ModuleType
    #: Binary names (without extension) that identify this engine.
    exe_names: tuple
    #: Substrings that must ALL appear in --help for this engine to claim it.
    help_fingerprint: tuple
    #: Example path shown in the server-path dialog's placeholder (a path, so
    #: not an i18n string — the sentence around it is).
    path_example: str = ""
    #: `--version` is a legal probe? kvmem rejects the flag (exit 1).
    probe_version: bool = True
    #: `--list-devices` is a legal probe? kvmem rejects it too, so the
    #: "recommended layer count" hint must not be computed for it.
    probe_devices: bool = True
    #: Feature names the UI asks about instead of testing schemas
    #: ("chat_templates", "log_level_selector", "identity_file",
    #: "param_linkage", "error_classification", ...).
    supports: frozenset = frozenset()
    #: Read the build identity off the install tree instead of the binary.
    read_identity: Optional[Callable] = None
    #: Turn a read_identity() result into the version-label line ("" unknown).
    #: Only set for engines with no `--version` (kvmem), where the label has
    #: nothing else to show.
    describe_identity: Optional[Callable] = None

    # -- parameter table -------------------------------------------------

    @property
    def params(self):
        return self.schema.PARAMS

    @property
    def ui_params(self):
        return self.schema.UI_PARAMS

    def fallback_defaults(self) -> dict:
        return dict(self.defaults._FALLBACK_DEFAULTS)

    def initial_overrides(self) -> dict:
        """Values the UI is seeded with (not the is_default baseline)."""
        return dict(getattr(self.schema, "INITIAL_OVERRIDES", {}))

    def user_input_keys(self) -> frozenset:
        return frozenset(getattr(self.defaults, "USER_INPUT_PARAMS", ()))

    def no_drift_keys(self) -> frozenset:
        return frozenset(getattr(self.schema, "NO_DRIFT_KEYS", ()))

    # -- --help / probes -------------------------------------------------

    def parse_help(self, help_text: str) -> dict:
        return self.defaults._parse_help_to_defaults(help_text)

    def get_default_params(self, server_path=None, help_text=None) -> dict:
        return self.defaults.get_default_params(
            server_path=server_path, help_text=help_text)

    def get_chat_templates(self, server_path=None, help_text=None) -> list:
        return self.defaults.get_chat_templates(
            server_path=server_path, help_text=help_text)

    def parse_device_list(self, text) -> list:
        return self.defaults.parse_device_list(text)

    def validation_problems(self, values: dict) -> list:
        """Pre-start parameter checks this engine needs: [(key, reason)].

        Lives on the schema module, next to the rules it encodes. llama.cpp has
        no such function — its server tolerates or clamps what the panel offers,
        and the launcher has always started it without a pre-flight gate — so
        the empty list here is that behaviour preserved, not a shortcut.
        """
        check = getattr(self.schema, "validate_params", None)
        if not check:
            return []
        try:
            return list(check(values) or [])
        except Exception:      # a validator must never be the reason start fails
            return []

    # -- identity --------------------------------------------------------

    def identity(self, server_path: str):
        """Build identity for the About box / version label (None = unknown)."""
        return self.read_identity(server_path) if self.read_identity else None

    # -- recognition -----------------------------------------------------

    def owns_exe_name(self, path: str) -> bool:
        """Filename-only test — cheap enough to run on every resolution.

        Matches the exact stem and version-suffixed variants
        (llama-kvmem-server-0.16.exe), and is case/extension agnostic.
        """
        from pathlib import Path
        name = Path(str(path or "")).name.lower()
        stem = name[:-4] if name.endswith(".exe") else name
        return any(stem == n or stem.startswith(n + "-") for n in self.exe_names)

    def help_matches(self, help_text: str) -> bool:
        """Does this --help output come from this engine's binary?

        Used when validating a path the user picked, and by the regression
        test that pins the fingerprints to the captured kvmem help.
        """
        if not help_text or not self.help_fingerprint:
            return False
        return all(tok in help_text for tok in self.help_fingerprint)

    # -- path ------------------------------------------------------------

    def server_path(self) -> str:
        """This engine's server binary, "" when it has none.

        llama keeps the historical resolution (settings > shutil.which > bare
        "llama-server"). kvmem needs an explicit path: `shutil.which` must not
        pick up some other build, because the flags this engine emits would be
        unknown to it. The one extra liberty taken for a non-default engine is
        reading the *legacy* `server_path` key when that path really is one of
        its own binaries — which is how an install that already points at
        llama-kvmem-server.exe keeps working without a settings rewrite.
        """
        from pathlib import Path
        from core.config import get_server_path, get_configured_server_path
        if self.id == DEFAULT_ENGINE_ID:
            return get_server_path()
        explicit = get_configured_server_path(self.id)
        if explicit and Path(explicit).is_file():
            return explicit
        legacy = get_configured_server_path(DEFAULT_ENGINE_ID)
        if legacy and self.owns_exe_name(legacy) and Path(legacy).is_file():
            return legacy
        return ""

    # -- display ---------------------------------------------------------

    @property
    def needs_explicit_path(self) -> bool:
        """True when `shutil.which` must not guess this engine's binary.

        Finding *some* `llama-server` on PATH is the launcher's historical
        behaviour and a reasonable guess. A kvmem build picked up by accident
        would be a different build with a different flag surface, so this
        engine only ever runs a path the user pointed at.
        """
        return self.id != DEFAULT_ENGINE_ID

    @property
    def has_basic_mode(self) -> bool:
        """Is the hand-picked 基础模式 form valid for this engine?

        The basic panel is a fixed set of llama.cpp controls with llama.cpp's
        own "not set" sentinels (0 = send nothing for --top-p, --ctx-size …).
        kvmem's sentinels are -1/-1.00 and its meaningful knobs live in the
        --kvmem-* table, so those widgets would write 0 into parameters where 0
        is a real value and emit sampling flags the user never touched. Until a
        kvmem-shaped basic form exists, the engine's window is advanced-only.
        """
        return self.id == DEFAULT_ENGINE_ID

    def bare_name(self) -> str:
        """Placeholder for "no binary configured" (preview / dialogs)."""
        return self.exe_names[0]


LLAMA = Engine(
    id="llama",
    display_name="llama.cpp",
    schema=llama_schema,
    defaults=llama_defaults,
    # "llama-server" and every llama-server-<variant> build name.
    exe_names=("llama-server",),
    help_fingerprint=("--flash-attn", "--ctx-size"),
    path_example=r"C:\llama.cpp\build\bin\llama-server.exe",
    supports=frozenset({"chat_templates", "list_devices", "version_flag",
                        "log_level_selector", "gpu_layer_hint", "basic_mode"}),
)

KVMEM = Engine(
    id="kvmem",
    display_name="kvmem-llama.cpp",
    schema=kvmem_schema,
    defaults=kvmem_defaults,
    # The CLI sibling (llama-kvmem-cli.exe) is deliberately absent: it is not a
    # server, and pointing the launcher at it must not look like a valid choice.
    exe_names=("llama-kvmem-server",),
    help_fingerprint=("Independent single-slot OpenAI-compatible server",
                      "--kvmem-budget", "--kvmem-block-tokens"),
    path_example=r"D:\llama-kvmem\bin\llama-kvmem-server.exe",
    probe_version=False,        # rejected: `unknown flag: --version`
    probe_devices=False,        # rejected: `unknown flag: --list-devices`
    supports=frozenset({"identity_file", "param_linkage",
                        "error_classification"}),
    read_identity=kvmem_identity.read_identity,
    describe_identity=kvmem_identity.describe,
)

ENGINES = {e.id: e for e in (LLAMA, KVMEM)}


def get_engine_by_id(engine_id: str) -> Engine:
    """The engine for a settings.json id; unknown/empty -> llama (today)."""
    return ENGINES.get(engine_id or "", LLAMA)


def detect_engine(server_path: str) -> Engine:
    """Engine for a binary path, by file name. Defaults to llama.

    A renamed binary is the one case file names cannot catch; that is what
    engine_from_help() is for (and the reason --help output is fingerprinted
    when a path is validated).
    """
    for engine in ENGINES.values():
        if engine.id != DEFAULT_ENGINE_ID and engine.owns_exe_name(server_path):
            return engine
    return LLAMA


def engine_from_help(help_text: str) -> Optional[Engine]:
    """Engine whose --help fingerprint matches the text (None = no match)."""
    for engine in ENGINES.values():
        if engine.help_matches(help_text):
            return engine
    return None


def server_path_for(engine_id: str) -> str:
    """Convenience: the binary an engine id would run ("" when unset)."""
    return get_engine_by_id(engine_id).server_path()


def resolve_startup_engine() -> Engine:
    """Engine for this session: explicit preference, else sniff the paths.

    Sniffing matters right now: a settings.json whose `server_path` already
    points at llama-kvmem-server.exe gets the kvmem schema with no config
    rewrite at all, instead of the current behaviour (llama flags sent to a
    kvmem binary -> `unknown flag` -> exit 1).
    """
    from core.config import load_preferred_engine_id
    engine_id = load_preferred_engine_id()
    if engine_id:
        return get_engine_by_id(engine_id)
    for engine in ENGINES.values():
        if engine.id != DEFAULT_ENGINE_ID and engine.server_path():
            return engine
    return LLAMA
