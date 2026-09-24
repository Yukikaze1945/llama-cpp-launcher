<div align="center">

<img src="assets/icon.png" alt="Llama CPP Launcher icon" width="88" />

# 🦙 Llama CPP Launcher

**A full-featured GUI launcher for `llama-server` (llama.cpp) — bring your own binary.**

All **226** `llama-server` CLI parameters in one panel · live default detection · presets · real-time log parsing · built-in GGUF inspector

[![Version](https://img.shields.io/github/v/release/Mars-Albert/llama-cpp-launcher?label=version)](https://github.com/Mars-Albert/llama-cpp-launcher/releases/latest)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776ab)](https://www.python.org/)
[![PyQt6](https://img.shields.io/badge/PyQt6-6.11-41cd52)](https://pypi.org/project/PyQt6/)
[![License: MIT](https://img.shields.io/badge/License-MIT-2ea44f)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20exe%20%7C%20any%20OS%20(source)-8a2be2)]()

**[Download Latest Release](https://github.com/Mars-Albert/llama-cpp-launcher/releases/latest)** · **[📖 中文文档](README_zh.md)** · [⭐ Star this repo](https://github.com/Mars-Albert/llama-cpp-launcher)

</div>

---

<details>
<summary><b>📑 Table of Contents</b></summary>

- [Why Llama CPP Launcher?](#why)
- [Screenshot](#screenshot)
- [Quick Start](#quick-start)
- [Features](#features)
  - [Two Modes](#two-modes)
  - [Two Server Engines (`-kvmem` build)](#two-engines)
  - [VRAM Prediction & Online Calibration (`-kvmem` only)](#vram-prediction)
  - [Advanced Mode — 9 Tabs](#advanced-tabs)
  - [Model Browser](#model-browser)
  - [Real-time Log Parsing](#log-parsing)
  - [Log Panel](#log-panel)
  - [Presets](#presets)
  - [Per-Parameter Help](#param-help)
  - [GGUF Inspector](#gguf-inspector)
  - [Server Lifecycle](#server-lifecycle)
  - [Window & UI](#window-ui)
  - [i18n & Themes](#i18n-themes)
- [Comparison](#comparison)
- [Development](#development)
- [License](#license)

</details>

---

<a id="why"></a>

## 💡 Why Llama CPP Launcher?

Most GUI tools (Ollama, LM Studio, …) **bundle a fixed llama.cpp version** and hide the command line. Llama CPP Launcher is the opposite: it drives the `llama-server` binary *you* installed, and everything it does is visible.

| | Llama CPP Launcher | Bundled-backend tools |
|---|---|---|
| Upgrade llama.cpp | ✅ Swap the binary, done | ❌ Wait for an app update |
| Custom builds (CUDA / ROCm / Vulkan / Metal / SYCL) | ✅ Use any build | ❌ Stuck with the bundled one |
| Bleeding-edge commits | ✅ Build today, run today | ❌ Weeks or months of waiting |
| Rollback | ✅ Put the old binary back | ❌ Hope they ship it |
| The exact command the app runs | ✅ Always visible, one-click copy | ❌ Opaque |

And it stays in sync automatically:

- **🔍 Live default detection** — runs `llama-server --help` on startup and parses the *real* defaults of your binary version. No stale hardcoded values, no drift (drift is surfaced via a ⚠️ indicator).
- **🗂️ Chat template auto-discovery** — templates shipped by your binary appear in the UI automatically.
- **🖥️ GPU detection** — probes `--list-devices` and shows e.g. `2× GPU: RTX 5090 (32GB) + RTX 2080 (8GB)` next to the offload controls (never auto-fills, always your call).

**🪶 Lightweight & private** — ~19,000 lines of Python, one dependency (PyQt6). No bundled backend, no accounts, no telemetry, no phone home. 100% local.

<a id="screenshot"></a>

## 📸 Screenshots

*Advanced mode (light theme, left) · Basic mode with a model running (dark theme, right) — runtime info parsed live from server logs.*

<table>
  <tr>
    <td width="50%"><img src="en_light.png" width="560" alt="Advanced mode, light theme" /></td>
    <td width="50%"><img src="en.png" width="560" alt="Basic mode with a model running, dark theme" /></td>
  </tr>
</table>

<a id="quick-start"></a>

## 🚀 Quick Start

### Option 1: Windows exe (recommended)

1. Download `LlamaCppLauncher.exe` from [Releases](https://github.com/Mars-Albert/llama-cpp-launcher/releases/latest) and double-click to run — no Python needed.
2. Install `llama-server` (e.g. from [ggml-org/llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases)).
3. In the app: **File → Set llama-server path...** and point it at the binary (or just add it to `PATH` — the launcher finds it automatically).
4. Pick a model, hit **Start**, open the WebUI. Done.

> ⚠️ Windows SmartScreen may warn about the unsigned executable — click **More info → Run anyway**.

### Option 2: From source (any OS)

```bash
git clone https://github.com/Mars-Albert/llama-cpp-launcher.git
cd llama-cpp-launcher

python -m venv venv
venv\Scripts\activate   # Windows   ·  source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt

python main.py          # or double-click run.bat on Windows
```

<a id="features"></a>

## ✨ Features

<a id="two-modes"></a>

### 🎭 Two Modes

| Basic Mode | Advanced Mode |
|---|---|
| Model & mmproj pickers backed by the model browser | **226 parameters** in 9 tabs |
| Sliders for temp / top-p / min-p / repeat penalty, top-k spin | LoRA adapters & scales, control vectors, image token limits |
| Context quick-picks: Default → 4K → 262K | Model sources: local file, **HF repo**, **URL**, **Docker repo** |
| GPU layers auto / all / manual, host/port/parallel | Speculative decoding (draft-mtp, ngram, lookup cache) |
| ⚡ Quick toggles — **user-configurable** (Settings → Customize): FlashAttn, reasoning, split mode, spec type, draft max… | Full server config: SSL, CORS, slots, embedding/rerank, MCP |
| *Get a model running in seconds* | *Fine-tune every detail* |

<a id="two-engines"></a>

### 🔌 Two Server Engines — the `-kvmem` build

The official `LlamaCppLauncher.exe` drives `llama-server` and nothing else. A
second product, **`LlamaCppLauncher-kvmem.exe`**, drives an additional binary
from the very same window: `llama-kvmem-server.exe` — an independent,
single-slot OpenAI-compatible server with retrieval-based KV-cache offload and
MTP speculative decoding. Pick the engine under **设置 / Settings → 引擎: …**; the
window is rebuilt around that engine's own parameter table.

That choice is remembered — `settings.json` gets `"engine": "llama"` or
`"engine": "kvmem"`, including the deliberate llama.cpp — and it wins over
everything else on the next start. Only a configuration that never picked an
engine is auto-detected: there, a `server_path` already pointing at
`llama-kvmem-server.exe` opens the kvmem engine without rewriting anything.

|  | llama.cpp engine | kvmem engine |
|---|---|---|
| Server binary | `llama-server.exe` (path → `PATH` → bare name) | `llama-kvmem-server.exe`, path configured per engine |
| Parameters | **226** in 9 tabs | **52** in 6 tabs — only what this parser accepts |
| Defaults | live-parsed from its own `--help` | live-parsed from its own `--help`, drift-checked per engine |
| Build identity | `--version` | read from the install tree (`BUILD-INFO.json`); the binary rejects `--version` |
| Presets | same folder, tagged `"engine": "llama"` — the 20 legacy files you already have need no migration | tagged `"engine": "kvmem"`, and only those are listed |
| Mode switch | Basic + Advanced | Advanced only — the basic form is llama.cpp's control set |
| Extras | device probe, GPU-layer hint, log-level filter | none of those: this build has no `--list-devices`, and its output carries no level tokens to filter on — a full startup against the real 12 GB model produced 2,417 lines of which exactly 1 was level-prefixed |

Two guards are what make the second engine usable, because `llama-kvmem-server`
is a hand-written argv parser that exits `1` on its first complaint and prints
nothing readable afterwards:

- **before starting** — every value is checked against the ranges, pairings and
  exclusions that binary really enforces (quantized `-ctk` requires an
  identical `-ctv`, `--chat-template` excludes `--chat-template-file`,
  `--chat-template-kwargs` must be a JSON object, …). A rejected value opens a
  dialog naming the parameter and jumps to its tab, instead of launching a
  process doomed to exit.
- **after an unexpected exit** — the run's own output is matched against the 17
  error strings measured out of the shipped build, so `unknown flag: --parallel`
  turns into *"That is a llama.cpp server flag; kvmem-llama.cpp has no
  equivalent feature."* pointing at the free-text box, rather than
  *"abnormal exit"*.

Everything else follows from the same rule: **a control exists only if the
server you point at really acts on it.** The rc2 build has no NVMe offload
compiled in, no multi-GPU, `--parallel` fixed at 1, no independent draft model
and no `auto` GPU layers — so none of those have a widget here, even though
`llama-server` has flags for all of them.

The "do not send" sentinels are explicit and visible: a sampling spin sitting at
its engine default reads `engine default (not sent)` and contributes **no flag**
to the command line, and each parameter page carries a *Restore engine defaults
(send nothing from this page)* button that returns it to that state in one click.

<a id="vram-prediction"></a>

### 🧮 VRAM Prediction & Online Calibration — the `-kvmem` build only

The kvmem engine's **KVMem page** carries a read-only card on top: it works out
how much memory *this* run will really take, then inverts that into the largest
`--kvmem-budget` that still fits. On the llama.cpp engine the card, its sampling
thread, its learning file and the launch delay it can introduce do not exist.

| Card field | What it is |
|---|---|
| Predicted peak | Weights + KV pool + MTP pool + mmproj from the structure, plus the residual learned on this machine |
| Safe peak (learned) | Predicted peak + safety margin |
| Currently free | Live free / total from NVML |
| Headroom | Free − safe peak |
| Suggested KVMem budget | The biggest budget that fits the free memory, found by a block-granular search |
| Learned samples · Confidence | How many runs taught it, and which tier the safety margin came from |

The numbers come from three layers:

1. **Physical model** — a copy of rc2's `kvmem_compute_pool()`: `budget` and
   `gen_reserve` are each aligned down to `--kvmem-block-tokens` separately (one
   block minimum), `gen_reserve` 0 means 256, a `budget` of 0 never pools more
   than `-c`, and `--kvmem-gpu-ratio` caps on the main-KV slot size alone. Bytes
   per token come out of the GGUF — layer counts, head sizes, row sizes and
   quantisation type — and once a run has logged `KVMEM_KV_BYTES`,
   `KVMem slot-pool` or `KVMEM_TRACE mtp_pool`, the engine's own figures win.
2. **Online residual learning** — each finished run turns its measured whole-card
   peak minus the physical estimate into one residual, fitted by RLS with a
   0.995 forgetting factor. It learns the bias of *this card, this build, this
   model*, not a universal constant.
3. **One-sided safety bound** — structural headroom with no samples, the largest
   positive error with 1–4, residual P95 from five onwards: the estimate can only
   ever be learned conservative, never optimistic.

**A prediction never touches a parameter.** Nothing on the card is editable, and
the only write path is the *Apply Suggested Budget* button — one click puts the
number into the `--kvmem-budget` control, and the command line is still generated
from the parameter table alone.

Learning state lives in `vram_learning.json` in the config directory, keyed by
GPU name + card size + engine build + model structure fingerprint, at most 64
residuals per profile. It stores numbers only: no prompts, no chat content, no
model content.

These runs are **not** allowed to teach the model, and the card says which one it
was: ended out of memory, never reached ready, parameters rejected by the engine,
killed uncleanly, another process moving the card's memory, no pre-start
baseline, or only coarse sampling — a 1 s `nvidia-smi` poll can step over a short
peak, and an under-measured peak would shrink the safety bound, the wrong
direction to err in.

Sampling prefers NVML through `ctypes` (no new dependency, ~75 ms per read) and
falls back to `nvidia-smi` only when it is unavailable. Before a launch the card
takes a short baseline window (~450 ms on NVML) to establish what the card looked
like before this run; with no GPU or driver it simply has fewer numbers to show,
and the launch waits not at all.

<a id="advanced-tabs"></a>

### 🎛️ Advanced Mode — 9 Tabs

| Tab | What's inside |
|---|---|
| 🧠 Model | Model file, alias, tags, HF/URL/Docker sources, LoRA, control vectors, mmproj (vision) |
| 📏 Context | Context size, prompt & KV cache, RoPE / YaRN scaling |
| 🎲 Sampling | Temperature, top-k/p, min-p, penalties, grammar & repetition |
| 🎮 GPU/Performance | Offload, memory, CPU threads, affinity, priority |
| ⚡ Speculative | Draft models (draft-mtp), ngram, lookup cache, draft tokens |
| 🌐 Server | Host/port, slots, parallel, endpoints, SSL, CORS, embedding/rerank, router |
| 🤖 Agent/Tools | Tool calling, MCP server, agent settings |
| 💬 Chat/Reasoning | Chat templates, reasoning mode, thinking budget |
| 🔧 Advanced | Text I/O, logging, `extra_args` escape hatch |

<a id="model-browser"></a>

### 📂 Model Browser

- 🔎 Background-thread scan of your model directory — the UI never freezes
- 🏷️ Auto-categorizes `.gguf` files into **Models** / **Multimodal (mmproj)** with sizes
- 📏 Instant model info: size, estimated parameters, quantization type
- 🧾 **GGUF quick-metadata row**: architecture + max context of the selected model, parsed from the file *header only* (a few MB, cached, off the GUI thread); the row turns **amber** when your context setting exceeds the model's limit (the server would clamp it at load)
- 🔗 Auto-matches the mmproj to your model by name
- 📁 Scan directory remembered across sessions; F5 to re-scan

<a id="log-parsing"></a>

### 📊 Real-time Log Parsing

Every line of `llama-server` output is parsed as it arrives (74 patterns, both old and v9174+ `srv`-prefixed formats) and distilled into the **Runtime Info** panel — 40+ data points across 8 categories:

| Category | What you see |
|---|---|
| 🖥️ Hardware | GPU name / compute capability / total & free VRAM per GPU, CPU |
| 📦 Model | File, model name, quant type, GGUF version |
| 🏗️ Architecture | Params, layers, embed/FFN dims, vocab, tensor precision split |
| ⚙️ Runtime | Train vs runtime ctx, batch/ubatch, slots, RoPE freq, thinking mode |
| 💾 VRAM | Offload layers, model/KV-cache/compute buffers, projected usage |
| ⚡ Performance | Flash attention, KV unified, graph nodes & splits |
| 🔧 System | Threads, OpenMP, repack |
| 👁️ Vision | Encoder status, mmproj, image resolution, min image tokens |

The panel fills in *live* — offload layers appear during loading, buffer sizes during init, and the status flips to **ready** the moment the server starts listening.

<a id="log-panel"></a>

### 📄 Log Panel

- 🔍 **Ctrl+F search** with match count and wrap-around
- **Level filter** — Debug / Info / Warn / Error (follows Error by default)
- 📤 Export the visible area or the **full run** (every run is also mirrored to `~/.llama-cpp-launcher/logs/last_run.log`)
- 🪟 **Per-level history windows** — each level (D/I/W/E) keeps its own 5,000-line window, so a burst of hidden lines (e.g. a debug prompt dump) can never evict the lines you're filtering for
- Auto-scroll toggle, clear button, colorized by level

<a id="presets"></a>

### 💾 Presets

- Save / load / delete named presets — only the diff vs. defaults is stored
- Import / export as JSON for sharing; created-time shown in the list
- Choose to **include or exclude machine-local paths** (model/mmproj) when saving
- The preset you last loaded is **restored automatically on next start**
- Param keys renamed/removed across llama.cpp versions are migrated on load

<a id="param-help"></a>

### ❓ Per-Parameter Help

Every parameter row has a **?** button that opens a floating card: the CLI flag, your binary's *live default* (from `--help`), the valid range, and a plain-language explanation. No more memorizing what `--no-kv-offload` does.

<a id="gguf-inspector"></a>

### 🔬 GGUF Inspector

A built-in binary inspector (no weights loaded, pure-stdlib parser):

- **7 tabs**: Overview · Statistics · Metadata · Tensors · Tokenizer · Filename · Diagnostics
- 📊 Visual breakdowns: quantization distribution, layer/module structure, parameter concentration
- 🩺 **Launcher-aware diagnostics**: context exceeds model limit, mmproj name mismatch, draft-mtp without sidecar, chat-template / RoPE / MoE / shard info
- 📤 Export to JSON / CSV / Markdown · background parsing with an in-memory cache

<a id="server-lifecycle"></a>

### 🚀 Server Lifecycle

- ▶️ One-click start/stop with color-coded status + runtime counter (MM:SS)
- ⚠️ Port-conflict check before launch
- 🌐 One-click open of the llama-server WebUI
- Graceful stop (non-blocking, force-kill fallback); auto-stops when you close the app
- ↩️ **Undo** — 800ms-debounced snapshots, up to 20 steps back
- 📝 **Command preview** — the exact `llama-server` command, updated on every change, one-click copy (paths with spaces are quoted correctly)

<a id="window-ui"></a>

### 🪟 Window & UI

- 🖼️ **Frameless rounded-card window** — the app paints its own card with a soft drop shadow: one integrated title row (icon · title · menu · Win11-style minimize/maximize/close), drag to move, double-click to maximize, edge-resize right at the visible card edge. Maximized, the card goes full-bleed with the system's own rounded corners.
- 🎴 **Themed dialogs & message boxes** — every dialog, confirm and error box belongs to the same frameless card family (dark/light aware), so nothing pops out of the look.
- 📐 **Comfortable by default** — first launch opens at 1600×940 (clamped to your screen) with a 400px model column; size, position, mode, tabs and splitter layout are restored on next start.
- 🧘 **Small windows behave** — the parameter area never gets a vertical scrollbar and no control is squashed or overlapped: quick toggles re-wrap into more rows and the window's minimum size grows instead; very narrow windows scroll the wide rows sideways.

<a id="i18n-themes"></a>

### 🌐 i18n & Themes

- 🈶/🈷 **Chinese ↔ English live switching** — no restart, preference persisted
- 🌙 **Dark / light theme** — one click in the **Settings** menu (File · Settings · Help), persisted; the log panel and inspector stay theme-aware
- Menu layout: **File** (scan path, llama-server path, refresh, exit) · **Settings** (customize quick toggles, language, theme) · **Help** (about)

<a id="comparison"></a>

## 🆚 Comparison

| Feature | Llama CPP Launcher | Ollama | LM Studio |
|---|:---:|:---:|:---:|
| Bring your own llama.cpp binary | ✅ | ❌ | ❌ |
| Run the latest llama.cpp the day it ships | ✅ | ❌ | ❌ |
| Any custom backend (CUDA/ROCm/Vulkan/Metal/SYCL) | ✅ | ❌ | ❌ |
| Full CLI parameter access (226) | ✅ | ❌ | Partial |
| The exact command is visible & copyable | ✅ | ❌ | ❌ |
| Presets with version-aware migration | ✅ | ❌ | ❌ |
| Built-in GGUF inspector | ✅ | ❌ | ❌ |
| No telemetry, 100% local | ✅ | ❌ | ✅ |
| Single lightweight dependency (PyQt6) | ✅ | — | ❌ |

<a id="development"></a>

## 🛠️ Development

- Python 3.11+, PyQt6 (pinned), pytest for tests
- ~19,000 lines of application code (+~11,000 lines of tests); the core schema (`core/params_schema.py`) is the single source of truth for the UI, CLI emission, get/set values, and i18n coverage — adding a parameter is one entry
- The schema is *injectable*: `core/engine.py` maps an engine id to its parameter module, defaults module, baseline and capability set, so a second server binary is a table plus a few hooks rather than a fork (`ui/advanced_panel.py`'s `_read_hooks` / `_write_hooks` / `_retranslate_extras` stay empty for llama.cpp, which is why its behaviour is bit-for-bit unchanged)
- `gguf/`, `ui/log_parser.py`, `ui/command_builder.py`, `core/kvmem_params_schema.py` and `core/kvmem_errors.py` are Qt-free and unit-testable headlessly

<details>
<summary><b>Project structure</b></summary>

```
llama-cpp-launcher/
├── main.py                  # Entry point (async startup, logging setup)
├── run.bat                  # Windows launcher (activates venv)
├── build_config.py          # App name / version (CI rewrites on tag)
├── llama_cpp_launcher.spec  # PyInstaller spec (icon embedded) — official exe
├── llama_cpp_launcher_kvmem.spec  # …-kvmem exe (same app, second engine enabled)
├── requirements.txt
├── core/
│   ├── params_schema.py     # ★ 226-param schema (Qt-free, single source of truth)
│   ├── kvmem_params_schema.py  # ★ 52-param kvmem schema + per-engine validator
│   ├── engine.py            # Engine registry: schema, defaults, baseline, caps
│   ├── kvmem_identity.py    # Build identity from the install tree
│   ├── kvmem_errors.py      # kvmem server output → "which parameter, where"
│   ├── params_help.py       # Per-parameter help texts (226 entries)
│   ├── defaults.py          # `--help` parsing, fallback defaults, GPU probe
│   ├── defaults_kvmem.py    # …for the kvmem binary (per-engine flag index)
│   ├── config.py            # Preset & settings IO (~/.llama-cpp-launcher)
│   ├── runner.py            # QProcess wrapper (start/stop/readiness)
│   ├── i18n.py              # zh/en translation (Chinese source language)
│   └── constants.py
├── gguf/                    # Pure-stdlib GGUF binary reader
│   ├── parser.py  models.py  ggml_types.py  filename.py  diagnostics.py
├── ui/
│   ├── main_window.py       # Window orchestration, theme, log panel
│   ├── frameless.py         # Frameless rounded-card window, title bar, edge-resize
│   ├── message_box.py       # Themed frameless message boxes & dialog base
│   ├── basic_panel.py       # Basic mode (incl. user-configurable quick toggles)
│   ├── advanced_panel.py    # Advanced mode (schema-driven, 9 tabs / 6 for kvmem)
│   ├── kvmem_linkage.py     # kvmem-only widget behaviour (sentinels, pairing, gating)
│   ├── quick_params.py      # Quick-toggles pool + widget factory
│   ├── quick_params_dialog.py  # Customize-quick-toggles dialog
│   ├── server_path_dialog.py   # llama-server path dialog
│   ├── param_help.py        # "?" button + floating help card
│   ├── model_browser.py     # GGUF scanner (background thread)
│   ├── gguf_inspector.py    # 7-tab inspector dialog
│   ├── log_parser.py        # Log line patterns → runtime info (Qt-free)
│   ├── command_builder.py   # Params → `llama-server` argv (Qt-free)
│   └── runtime_info.py      # Runtime info HTML (Qt-free)
├── tests/                   # 38 test modules (headless, in-memory fakes)
└── assets/icon.ico|png
```

</details>

**Releases**: pushing a tag `v*` runs CI on `windows-latest` — tests, then PyInstaller, then the exe is published as a GitHub Release automatically.

**The `-kvmem` build is not published by CI** — the workflow only packages
`llama_cpp_launcher.spec`. Build it from a clone:
`pyinstaller --noconfirm --workpath build_kvmem llama_cpp_launcher_kvmem.spec`
writes the onefile `dist/LlamaCppLauncher-kvmem.exe`. Separate name, separate
work directory, so it never touches the official exe.

<a id="license"></a>

## 📄 License

[MIT](LICENSE) — do whatever you want with it.
