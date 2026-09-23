import json
import logging
import os
import re
import shutil
from pathlib import Path
from datetime import datetime

from core.defaults import _FALLBACK_DEFAULTS
from core import params_schema
from core.i18n import t

logger = logging.getLogger(__name__)


def _sanitize_preset_name(name: str) -> str:
    """Remove path separators and traversal sequences to prevent path traversal."""
    result = re.sub(r'[/\\:\x00]', '_', name).strip('. ')
    return result if result else "unnamed"


# Sandbox override: LLAMA_CPP_LAUNCHER_CONFIG_DIR points the whole app (and
# every test/smoke run) at an alternative config directory, so development of
# the second engine can never write into the real ~/.llama-cpp-launcher.
# Read once at import time — every path below is derived from it.
CONFIG_DIR = Path(os.environ.get("LLAMA_CPP_LAUNCHER_CONFIG_DIR", "").strip()
                 or (Path.home() / ".llama-cpp-launcher"))
PRESETS_DIR = CONFIG_DIR / "presets"
SETTINGS_FILE = CONFIG_DIR / "settings.json"
# E3: full (un-truncated) log of the most recent server run
LOGS_DIR = CONFIG_DIR / "logs"
LAST_RUN_LOG = LOGS_DIR / "last_run.log"

_dirs_initialized = False

def _ensure_dirs():
    global _dirs_initialized
    if not _dirs_initialized:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        PRESETS_DIR.mkdir(parents=True, exist_ok=True)
        _dirs_initialized = True

DEFAULT_PRESET = dict(_FALLBACK_DEFAULTS)

#: Engine id of the original llama.cpp engine. Defined here, not in
#: core.engine, because config keys are named after it (`last_preset#kvmem`)
#: and importing the registry would be circular — core.engine reads settings.
#: Anything else in an `engine` / preset / suffixed key is the *other* engine's
#: id, and an unknown one resolves back to llama.cpp (see get_engine_by_id).
LLAMA_ENGINE_ID = "llama"

# A9: preset keys renamed or removed across llama-server versions. Both the
# params-dict migration (load_preset) and the key-set migration
# (preset_stored_keys) read these tables, so the rule lives in one place.
PRESET_KEY_RENAMES = {"checkpoint_every_n_tokens": "checkpoint_min_step"}
PRESET_KEY_DROPS = ("ctx_size_draft",)


def migrate_preset_keys(params: dict) -> dict:
    """Apply the preset key renames/removals to a params dict (in place)."""
    for old, new in PRESET_KEY_RENAMES.items():
        if old in params and new not in params:
            params[new] = params.pop(old)
    for key in PRESET_KEY_DROPS:
        params.pop(key, None)
    return params


def migrate_preset_key_set(keys) -> set:
    """Same rule for a bare key set (the protected-keys view)."""
    keys = set(keys)
    for old, new in PRESET_KEY_RENAMES.items():
        if old in keys and new not in keys:
            keys.discard(old)
            keys.add(new)
    return keys - set(PRESET_KEY_DROPS)


def _load_settings() -> dict:
    if not SETTINGS_FILE.exists():
        return {}
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, IOError, json.JSONDecodeError) as e:
        logger.warning(t("加载设置失败: {e}", e=e))
        return {}


def _save_settings(settings: dict):
    _ensure_dirs()
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
    except (OSError, IOError) as e:
        logger.warning(t("保存设置失败: {e}", e=e))


def save_scan_path(path: str):
    settings = _load_settings()
    settings["scan_path"] = path
    _save_settings(settings)


def load_scan_path() -> str | None:
    return _load_settings().get("scan_path")


def save_language(lang: str):
    settings = _load_settings()
    settings["language"] = lang
    _save_settings(settings)


def load_language() -> str:
    return _load_settings().get("language", "zh")


#: UI keys that are per-engine (a kvmem advanced tab index is meaningless for
#: llama.cpp and vice versa). Everything else in `ui` — geometry, splitter,
#: theme, language — is genuinely shared between engines.
#:
#: an adv_tab index and the tab key that replaces it (main_window._restore_ui_state)
#: are both engine-specific: kvmem has 6 tabs, llama.cpp 9.
UI_PREF_ENGINE_KEYS = ("mode", "adv_tab", "adv_tab_key", "bottom_tab",
                       "quick_params")


def _suffix_engine_keys(prefs: dict, engine_id: str) -> dict:
    """Suffix only what is genuinely per-engine; shared keys stay shared.

    `load_ui_prefs`/`_merge_engine_keys` override just UI_PREF_ENGINE_KEYS, so
    suffixing everything would write a `geometry#kvmem` that nothing reads back
    and a kvmem window would silently keep restoring the other engine's size.
    """
    return {_engine_key(k, engine_id) if k in UI_PREF_ENGINE_KEYS else k: v
            for k, v in prefs.items()}


def _merge_engine_keys(ui: dict, engine_id: str) -> dict:
    """Read view of the `ui` dict for one engine.

    llama.cpp reads the unsuffixed keys (byte-identical to before); another
    engine reads only shared keys plus its own suffixed ones, so switching
    engines cannot carry a tab index or a quick-toggle list across.
    """
    if not engine_id or engine_id == LLAMA_ENGINE_ID:
        return {k: v for k, v in ui.items() if "#" not in k}
    out = {k: v for k, v in ui.items() if "#" not in k}
    suffix = f"#{engine_id}"
    for key in UI_PREF_ENGINE_KEYS:
        if key + suffix in ui:
            out[key] = ui[key + suffix]
    return out


def _owned_by_engine(ui: dict, engine_id: str) -> set:
    """Keys of the stored `ui` dict that belong to one engine's view."""
    if not engine_id or engine_id == LLAMA_ENGINE_ID:
        return {k for k in ui if "#" not in k}
    suffix = f"#{engine_id}"
    return {k for k in ui if k.endswith(suffix)}


def save_ui_prefs(prefs: dict, engine_id: str = LLAMA_ENGINE_ID):
    """E2: persist window geometry/mode/tabs/splitter ratio (small JSON-safe dict).

    Replaces *this engine's* view only: another engine's suffixed keys (and the
    llama.cpp unsuffixed ones) survive, which is what keeps the two engines' UI
    state from overwriting each other.
    """
    settings = _load_settings()
    stored = settings.get("ui")
    stored = dict(stored) if isinstance(stored, dict) else {}
    for key in _owned_by_engine(stored, engine_id):
        stored.pop(key, None)
    stored.update(_suffix_engine_keys(prefs, engine_id))
    settings["ui"] = stored
    _save_settings(settings)


def load_ui_prefs(engine_id: str = LLAMA_ENGINE_ID) -> dict:
    prefs = _load_settings().get("ui")
    prefs = prefs if isinstance(prefs, dict) else {}
    return _merge_engine_keys(prefs, engine_id)


def save_ui_pref(key: str, value, engine_id: str = LLAMA_ENGINE_ID):
    """E10: update a single 'ui' prefs entry without rewriting the rest
    (e.g. quick_params changes are persisted immediately, not on close)."""
    settings = _load_settings()
    prefs = settings.get("ui")
    if not isinstance(prefs, dict):
        prefs = {}
    prefs[_engine_key(key, engine_id)] = value
    settings["ui"] = prefs
    _save_settings(settings)


# E5: light/dark theme, persisted in settings.json

def load_theme() -> str:
    return "dark" if _load_settings().get("theme") == "dark" else "light"


def save_theme(theme: str):
    settings = _load_settings()
    settings["theme"] = "dark" if theme == "dark" else "light"
    _save_settings(settings)


# E1: configurable llama-server path.
# Resolution order: explicit path in settings.json (if the file still exists)
# > shutil.which("llama-server") (i.e. PATH, the previous behavior) > bare
# "llama-server" (last resort, so errors surface the same way as before).
#
# Second engine: settings stays *additive*. `server_path` keeps its exact
# meaning for the llama.cpp engine (so an install written by the official
# launcher reads identically and the reverse upgrade never loses a path), and
# every other engine lives in server_paths[<id>]. The same rule covers the two
# engine-dependent UI values: llama.cpp writes the original unsuffixed keys,
# other engines write <key>#<engine-id>.
_server_path_cache = None
_SERVER_PATHS_KEY = "server_paths"
_PREFERRED_ENGINE_KEY = "engine"


def _engine_key(key: str, engine_id: str) -> str:
    """llama.cpp keeps the historical unsuffixed key; others get `key#engine`."""
    if not engine_id or engine_id == LLAMA_ENGINE_ID:
        return key
    return f"{key}#{engine_id}"


def load_preferred_engine_id() -> str:
    """Which engine this install drives ("" = never chosen -> detect it).

    "" means the settings file has no `engine` key — an install written before
    a second engine existed, where sniffing the configured paths is the right
    guess. It is NOT the same as a stored "llama": that is an explicit choice
    and must survive a restart (see save_preferred_engine_id).

    No validation here: core.engine.get_engine_by_id() maps anything unknown
    back to llama.cpp, which is exactly the pre-kvmem behaviour, and importing
    the registry from config would be circular.
    """
    value = _load_settings().get(_PREFERRED_ENGINE_KEY)
    return value if isinstance(value, str) else ""


def save_preferred_engine_id(engine_id: str):
    """Record the engine the user picked — including llama.cpp itself.

    Storing "llama" is the point. Popping the key for llama.cpp, as the first
    version of this function did, made an explicit choice indistinguishable
    from "never chosen", so the next startup of an install that has both
    binaries configured fell back to resolve_startup_engine()'s path sniffing
    and could reopen in kvmem after the user had said llama.cpp. Passing "" is
    the one way to clear the choice back to "detect it".
    """
    settings = _load_settings()
    if engine_id:
        settings[_PREFERRED_ENGINE_KEY] = engine_id
    else:
        settings.pop(_PREFERRED_ENGINE_KEY, None)  # "" = not chosen
    _save_settings(settings)


def load_server_path(engine_id: str = LLAMA_ENGINE_ID) -> str:
    """The configured path for one engine, exactly as stored (no existence check)."""
    settings = _load_settings()
    if not engine_id or engine_id == LLAMA_ENGINE_ID:
        return settings.get("server_path", "") or ""
    paths = settings.get(_SERVER_PATHS_KEY)
    if not isinstance(paths, dict):
        return ""
    value = paths.get(engine_id)
    return value if isinstance(value, str) else ""


# Public alias used by core.engine (reads better at that call site).
get_configured_server_path = load_server_path


def save_server_path(path: str, engine_id: str = LLAMA_ENGINE_ID):
    global _server_path_cache
    settings = _load_settings()
    if not engine_id or engine_id == LLAMA_ENGINE_ID:
        settings["server_path"] = path
    else:
        paths = settings.get(_SERVER_PATHS_KEY)
        paths = dict(paths) if isinstance(paths, dict) else {}
        paths[engine_id] = path
        settings[_SERVER_PATHS_KEY] = paths
    _save_settings(settings)
    _server_path_cache = None  # invalidate so every caller sees the new path


# Last-used preset: the launcher restores on startup the preset the user
# last clicked 加载 for. The name lives in settings.json ("" = never loaded,
# i.e. startup keeps the defaults — nothing to restore). Suffixed per engine
# (see _engine_key) so switching engines cannot restore a preset belonging to
# the other parameter table.

def load_last_preset(engine_id: str = LLAMA_ENGINE_ID) -> str:
    name = _load_settings().get(_engine_key("last_preset", engine_id), "")
    return name if isinstance(name, str) else ""


def save_last_preset(name: str, engine_id: str = LLAMA_ENGINE_ID):
    settings = _load_settings()
    settings[_engine_key("last_preset", engine_id)] = name
    _save_settings(settings)


def get_server_path() -> str:
    """Resolve the llama-server executable path (E1), cached.

    Called on the command-preview tick (300ms) and from the startup worker,
    so the result is cached and only invalidated by save_server_path().
    """
    global _server_path_cache
    if _server_path_cache is not None:
        return _server_path_cache
    explicit = load_server_path()
    if explicit:
        if Path(explicit).is_file():
            _server_path_cache = explicit
            return explicit
        logger.warning("Configured llama-server path no longer exists: %s — falling back to PATH", explicit)
    found = shutil.which("llama-server")
    _server_path_cache = found if found else "llama-server"
    return _server_path_cache


def refresh_defaults(defaults):
    global DEFAULT_PRESET
    DEFAULT_PRESET = defaults


class ConfigManager:
    """Preset JSON IO (plan C3).

    Deliberately stateless about parameter values: MainWindow.params is the
    single source of truth. Presets are passed in / returned as plain dicts;
    the only state kept here is the defaults baseline (used to store just the
    diff on save and to merge against on load) and the engine this manager
    belongs to.

    Engine isolation, which is the reason the engine id lives here: the two
    tables share ~8 key names whose values are *not* interchangeable
    (ctx_size, batch_size, spec_type, chat_template*, temperature...). Loading
    a llama.cpp preset into the kvmem engine would hand it `spec_type:
    ngram-simple`, which kvmem answers `unsupported --spec-type` + exit 1 —
    indistinguishable from a kvmem bug. So every preset carries its engine,
    the list only shows the current one, and any key the current schema does
    not define is dropped on the way in.
    """

    #: Preset file-format version per engine (A9). Existing v1 files carry no
    #: `engine` field and are llama.cpp's — that inference is why the kvmem
    #: engine writes 2, so a v2 file can never be mistaken for a legacy one.
    PRESET_VERSION_BY_ENGINE = {LLAMA_ENGINE_ID: 1, "kvmem": 2}

    def __init__(self, defaults=None, schema=None, engine_id: str = LLAMA_ENGINE_ID):
        self._defaults = defaults or dict(DEFAULT_PRESET)
        # Engine seam: an engine schema module may override the preset key
        # migration; core.config's own tables are the llama.cpp rules and
        # stay the default so every existing call site is unchanged.
        self._schema = schema or params_schema
        self._engine_id = engine_id or LLAMA_ENGINE_ID

    @property
    def engine_id(self):
        return self._engine_id

    def _migrate(self, params):
        return getattr(self._schema, "migrate_preset_keys",
                       migrate_preset_keys)(params)

    def _migrate_keys(self, keys):
        return getattr(self._schema, "migrate_preset_key_set",
                       migrate_preset_key_set)(keys)

    def drop_foreign_keys(self, params: dict) -> dict:
        """Keep only keys this engine's schema defines (see the class docstring)."""
        known = getattr(self._schema, "PARAMS_BY_KEY", None)
        if not known:
            return params
        return {k: v for k, v in params.items() if k in known}

    def drop_foreign_key_set(self, keys) -> set:
        """Set-flavoured :meth:`drop_foreign_keys`."""
        known = getattr(self._schema, "PARAMS_BY_KEY", None)
        if not known:
            return set(keys)
        return {k for k in keys if k in known}

    @staticmethod
    def preset_engine_id(data: dict) -> str:
        """Engine a parsed preset file belongs to (legacy files: llama.cpp)."""
        engine = data.get("engine")
        if isinstance(engine, str) and engine:
            return engine
        return LLAMA_ENGINE_ID

    def preset_matches_this_engine(self, path) -> bool:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, IOError, json.JSONDecodeError):
            # Unreadable: only the default engine may claim it, so a broken
            # file cannot silently load into the other engine's parameter set.
            return self._engine_id == LLAMA_ENGINE_ID
        return (isinstance(data, dict)
                and self.preset_engine_id(data) == self._engine_id)

    @property
    def defaults(self):
        return self._defaults

    def set_defaults(self, defaults):
        """Replace the defaults baseline (live-parsed defaults arriving after startup, plan A10)."""
        self._defaults = dict(defaults)

    def save_preset(self, name, params):
        """Save a params dict as a preset (only the diff vs. defaults is stored)."""
        _ensure_dirs()
        name = _sanitize_preset_name(name)
        path = PRESETS_DIR / f"{name}.json"
        # E6: overwriting a preset keeps its original creation time
        created = datetime.now().isoformat()
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    old = json.load(f)
                if isinstance(old.get("created"), str) and old["created"]:
                    created = old["created"]
            except (OSError, IOError, json.JSONDecodeError):
                pass
        data = {
            "name": name,
            "version": self.PRESET_VERSION_BY_ENGINE.get(self._engine_id, 1),
            # A9: schema version for future preset migrations + the engine the
            # diff was taken against (the baseline is per-engine).
            "engine": self._engine_id,
            "created": created,
            "params": {k: v for k, v in params.items()
                       if v != self._defaults.get(k)},
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except (OSError, IOError) as e:
            logger.warning(t("保存预设失败: {e}", e=e))
            return False
        return True

    def load_preset(self, name):
        name = _sanitize_preset_name(name)
        path = PRESETS_DIR / f"{name}.json"
        if not path.exists():
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, IOError, json.JSONDecodeError) as e:
            logger.warning(t("加载预设失败: {e}", e=e))
            return False
        if not isinstance(data, dict):
            logger.warning(t("加载预设失败: params 字段不是字典"))
            return False
        if self.preset_engine_id(data) != self._engine_id:
            # Reachable only for a hand-copied file (list_presets filters), and
            # loading it is exactly the failure the engine tag exists to prevent.
            logger.warning("preset %s belongs to engine %s, not %s — refused",
                           name, self.preset_engine_id(data), self._engine_id)
            return False
        params = data.get("params", {})
        if not isinstance(params, dict):
            logger.warning(t("加载预设失败: params 字段不是字典"))
            return False
        # Migration first: a renamed legacy key is by definition not in the
        # schema yet, so filtering before renaming would silently discard the
        # user's explicit value instead of carrying it to the new key.
        params = self.drop_foreign_keys(self._migrate(dict(params)))
        # C3: return the merged params instead of mutating self.current —
        # MainWindow.params is the single source of truth
        merged = dict(self._defaults)
        merged.update(params)
        return merged

    def preset_stored_keys(self, name):
        """Keys the preset explicitly set (post-migration, foreign keys
        dropped), or None.

        None when the preset is missing/unreadable. Used at startup so a
        restored preset's explicit values survive the live --help defaults
        merge: keys the user deliberately set are protected, keys the preset
        leaves to the defaults are free to adopt the live default.
        """
        name = _sanitize_preset_name(name)
        path = PRESETS_DIR / f"{name}.json"
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, IOError, json.JSONDecodeError):
            return None
        params = data.get("params", {}) if isinstance(data, dict) else {}
        if not isinstance(params, dict):
            return set()
        if self.preset_engine_id(data) != self._engine_id:
            return set()
        # Mirror load_preset()'s filtering + cross-version migration so the
        # protected set matches the keys that actually end up in the merged
        # params (migration before filtering, for the same reason).
        return self.drop_foreign_key_set(self._migrate_keys(params))

    def delete_preset(self, name):
        name = _sanitize_preset_name(name)
        path = PRESETS_DIR / f"{name}.json"
        if path.exists():
            try:
                path.unlink()
                return True
            except OSError:
                return False
        return False

    def list_presets(self):
        """Presets of *this* engine, sorted by name.

        A preset written for the other engine is not merely inconvenient to
        load, it is a parameter dict whose shared keys carry values the running
        binary may reject outright — so it is not listed at all rather than
        listed-and-refused.
        """
        _ensure_dirs()
        presets = []
        for f in PRESETS_DIR.glob("*.json"):
            try:
                # E6: the JSON's own created field is the source of truth;
                # mtime is only a fallback for hand-edited/legacy files
                created = datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                try:
                    with open(f, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    if not isinstance(data, dict) or \
                            self.preset_engine_id(data) != self._engine_id:
                        continue
                    if isinstance(data.get("created"), str) and data["created"]:
                        created = data["created"]
                except (OSError, IOError, json.JSONDecodeError):
                    # Unreadable: keep it visible only for the default engine,
                    # whose legacy files it may be.
                    if self._engine_id != LLAMA_ENGINE_ID:
                        continue
            except (FileNotFoundError, OSError):
                continue
            presets.append({
                "name": f.stem,
                "created": created,
                "path": str(f),
            })
        return sorted(presets, key=lambda x: x["name"])

    def export_preset(self, name, dest_path):
        name = _sanitize_preset_name(name)
        src = PRESETS_DIR / f"{name}.json"
        if src.exists():
            try:
                shutil.copy2(src, dest_path)
                return True
            except (OSError, IOError):
                return False
        return False

    def import_preset(self, src_path):
        _ensure_dirs()
        try:
            with open(src_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict) or "params" not in data:
                logger.warning(t("导入预设失败: 文件格式无效 {src_path}", src_path=src_path))
                return False
            if not isinstance(data["params"], dict):
                logger.warning(t("导入预设失败: params 字段不是字典"))
                return False
            dest_name = _sanitize_preset_name(Path(src_path).stem)
            dest = PRESETS_DIR / f"{dest_name}.json"
            # Normalise the engine tag instead of guessing from the importing
            # session: a legacy (no `engine` field) file is llama.cpp's, and an
            # imported kvmem preset must not start claiming to be llama.cpp's
            # just because it arrived through a llama.cpp window. It then shows
            # up in the list only for its own engine — which is the whole point
            # of the tag (see the class docstring).
            data["engine"] = self.preset_engine_id(data)
            with open(dest, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except (OSError, IOError, json.JSONDecodeError) as e:
            logger.warning(t("导入预设失败: {e}", e=e))
            return False
