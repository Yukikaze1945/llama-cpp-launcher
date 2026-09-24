<div align="center">

<img src="assets/icon.png" alt="Llama CPP Launcher 图标" width="88" />

# 🦙 Llama CPP Launcher / Llama 启动器

**功能完整的 `llama-server` (llama.cpp) GUI 启动器 — 自带你的 llama.cpp 二进制。**

**226** 个 `llama-server` CLI 参数一屏全出 · 动态默认值检测 · 预设管理 · 实时日志解析 · 内置 GGUF 检查器

[![Version](https://img.shields.io/github/v/release/Mars-Albert/llama-cpp-launcher?label=version)](https://github.com/Mars-Albert/llama-cpp-launcher/releases/latest)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776ab)](https://www.python.org/)
[![PyQt6](https://img.shields.io/badge/PyQt6-6.11-41cd52)](https://pypi.org/project/PyQt6/)
[![License: MIT](https://img.shields.io/badge/License-MIT-2ea44f)](LICENSE)
[![Platform](https://img.shields.io/badge/平台-Windows%20exe%20%7C%20任意系统%20(源码)-8a2be2)]()

**[⬇️ 下载最新版本](https://github.com/Mars-Albert/llama-cpp-launcher/releases/latest)** · **[📖 English README](README.md)** · [⭐ Star 本项目](https://github.com/Mars-Albert/llama-cpp-launcher)

</div>

---

<details>
<summary><b>📑 目录</b></summary>

- [为什么选择 Llama CPP Launcher？](#why)
- [软件截图](#screenshot)
- [快速开始](#quick-start)
- [功能详解](#features)
  - [双模式设计](#two-modes)
  - [双服务器引擎（`-kvmem` 版）](#two-engines)
  - [显存预测与在线校准（仅 `-kvmem` 版）](#vram-prediction)
  - [高级模式 — 9 个标签页](#advanced-tabs)
  - [模型浏览器](#model-browser)
  - [实时日志解析](#log-parsing)
  - [日志面板](#log-panel)
  - [预设管理](#presets)
  - [逐参数帮助](#param-help)
  - [GGUF 检查器](#gguf-inspector)
  - [服务生命周期](#server-lifecycle)
  - [窗口与界面](#window-ui)
  - [国际化与主题](#i18n-themes)
- [与其他工具对比](#comparison)
- [开发](#development)
- [许可证](#license)

</details>

---

<a id="why"></a>

## 💡 为什么选择 Llama CPP Launcher？

大多数 GUI 工具（Ollama、LM Studio 等）**内置固定版本的 llama.cpp**，且命令行对用户不可见。Llama CPP Launcher 走的是另一条路：它驱动的是*你自己*安装的 `llama-server`，一切行为可见可控。

| | Llama CPP Launcher | 内置后端的工具 |
|---|---|---|
| 升级 llama.cpp | ✅ 换个二进制文件，完事 | ❌ 等应用更新 |
| 自定义编译（CUDA / ROCm / Vulkan / Metal / SYCL） | ✅ 任意构建都能用 | ❌ 只能用内置版本 |
| 体验最新提交 | ✅ 今天编译，今天跑 | ❌ 等几周甚至几个月 |
| 回退版本 | ✅ 换回旧二进制文件 | ❌ 祈祷官方提供 |
| 应用实际执行的命令 | ✅ 始终可见，一键复制 | ❌ 黑盒 |

而且它会自动保持同步：

- **🔍 动态默认值检测** — 启动时运行 `llama-server --help`，解析*你的*二进制版本的真实默认值。不存在过时的硬编码，版本漂移还会通过 ⚠️ 指示器提示。
- **🗂️ 聊天模板自动发现** — 你的二进制内置的模板会自动出现在 UI 中。
- **🖥️ GPU 检测** — 通过 `--list-devices` 探测，在卸载层数控制旁显示如 `检测到 2× GPU：RTX 5090 (32GB) + RTX 2080 (8GB)`（只展示，不自动填写，永远由你决定）。

**🪶 轻量且私密** — 约 19,000 行 Python，唯一运行时依赖是 PyQt6。无内置后端、无账号、无遥测、不联网上报，100% 本地运行。

<a id="screenshot"></a>

## 📸 软件截图

*高级模式（浅色主题，左）· 基础模式模型运行中（深色主题，右）— 运行时信息从服务端日志实时解析。*

<table>
  <tr>
    <td width="50%"><img src="cn_light.png" width="560" alt="高级模式（浅色主题）" /></td>
    <td width="50%"><img src="cn.png" width="560" alt="基础模式运行中（深色主题）" /></td>
  </tr>
</table>

<a id="quick-start"></a>

## 🚀 快速开始

### 方式一：Windows exe（推荐）

1. 从 [Releases](https://github.com/Mars-Albert/llama-cpp-launcher/releases/latest) 下载 `LlamaCppLauncher.exe`，双击运行 — 无需安装 Python。
2. 安装 `llama-server`（如从 [ggml-org/llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) 下载）。
3. 在应用中：**文件 → 设置 llama-server 路径…**，指向该二进制文件（或把它加入 `PATH` — 启动器会自动找到）。
4. 选模型，点 **启动**，打开 WebUI。搞定。

> ⚠️ Windows SmartScreen 可能警告未签名的可执行文件 — 点击 **更多信息 → 仍要运行** 即可。

### 方式二：从源码运行（任意系统）

```bash
git clone https://github.com/Mars-Albert/llama-cpp-launcher.git
cd llama-cpp-launcher

python -m venv venv
venv\Scripts\activate   # Windows   ·  source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt

python main.py          # Windows 下也可双击 run.bat
```

<a id="features"></a>

## ✨ 功能详解

<a id="two-modes"></a>

### 🎭 双模式设计

| 基础模式 | 高级模式 |
|---|---|
| 模型 & mmproj 选择器（背后是模型浏览器） | **226 个参数**，9 个标签页 |
| 温度 / Top-P / Min-P / 重复惩罚滑块 + Top-K 数值框 | LoRA 适配器与缩放、控制向量、图像 Token 限制 |
| 上下文快捷按钮：Default → 4K → 262K | 模型来源：本地文件、**HF 仓库**、**URL**、**Docker 仓库** |
| GPU 层数 auto / all / 手动，host/port/并行数 | 投机解码（draft-mtp、ngram、lookup cache） |
| ⚡ 快捷开关（**可自定义**，设置 → 自定义快捷开关…）：FlashAttn、推理、分割模式、投机类型、草稿 Token 上限… | 完整服务端配置：SSL、CORS、slots、embedding/rerank、MCP |
| *几秒钟让模型跑起来* | *精细调节每一个细节* |

<a id="two-engines"></a>

### 🔌 双服务器引擎 — `-kvmem` 版

正式版 `LlamaCppLauncher.exe` 只驱动 `llama-server`。第二个产物
**`LlamaCppLauncher-kvmem.exe`** 在**同一个窗口**里多驱动一个二进制：
`llama-kvmem-server.exe` —— 独立的单槽位 OpenAI 兼容服务器，带检索式 KV
缓存卸载与 MTP 投机解码。在 **设置 → 引擎: …** 里选择，窗口会按该引擎的参数表
重建。

这个选择会被记住：`settings.json` 里写下 `"engine": "llama"` 或
`"engine": "kvmem"`（明确选 llama.cpp 也一样写），下次启动它优先于其它任何判断。
只有从没选过引擎的旧配置才自动检测——这时若 `server_path` 已经指向
`llama-kvmem-server.exe`，就直接按 kvmem 引擎打开，不改写任何配置。

|  | llama.cpp 引擎 | kvmem 引擎 |
|---|---|---|
| 服务器二进制 | `llama-server.exe`（配置路径 → `PATH` → 同名命令） | `llama-kvmem-server.exe`，路径按引擎各存一份 |
| 参数数量 | **226 个**，9 个标签页 | **52 个**，6 个标签页 —— 只收录该 parser 真接受的 |
| 默认值 | 解析它自己的 `--help` | 解析它自己的 `--help`，逐引擎做版本漂移检测 |
| 构建身份 | `--version` | 读安装目录的 `BUILD-INFO.json`（该二进制拒绝 `--version`） |
| 预设 | 同一目录，文件内标记 `"engine": "llama"`；你已有的 20 个旧预设无需迁移 | 标记 `"engine": "kvmem"`，列表里也只出现这些 |
| 模式 | 基础 + 高级 | 只有高级 —— 基础面板是 llama.cpp 的那套控件 |
| 附加能力 | 设备探测、显存推荐层数、日志级别筛选 | 都没有：这个构建没有 `--list-devices`，输出里也没有可筛选的级别前缀 —— 用真实 12 GB 模型完整启动一次，2417 行日志里只有 1 行带级别 |

真正让这个引擎可用的是两道拦截，因为 `llama-kvmem-server` 是手写的 argv
parser：**碰到第一个不满就 exit 1，之后什么可读的都不打**。

- **启动前** —— 每个取值都按这个二进制真正强制的范围、配对与互斥规则校验
  （量化 `-ctk` 必须配同值 `-ctv`、`--chat-template` 与 `--chat-template-file`
  互斥、`--chat-template-kwargs` 必须是 JSON object …）。不通过就弹窗点名参数
  并跳到它所在的标签页，而不是去启一个必死的进程。
- **异常退出后** —— 把这次运行的输出与从发布二进制里逐条量出来的 17 条报错文案
  比对，于是 `unknown flag: --parallel` 会变成
  「这是 llama.cpp 服务器的参数，kvmem-llama.cpp 没有对应功能。」并指向附加参数
  输入框，而不是只剩一句「服务异常退出」。

其余一切都遵循同一条规则：**只有你指向的那个服务器真会响应的参数，才配有控件。**
rc2 构建编译期关闭了 NVMe 卸载、不支持多 GPU、`--parallel` 固定为 1、没有独立
草稿模型、也没有 `auto` GPU 层数 —— 所以这些在 kvmem 引擎下一个控件都没有，尽管
`llama-server` 对每一项都有 flag。

「不发送」哨兵是显式且看得见的：停在引擎默认值上的采样框显示
`引擎默认（不发送）`，对命令行**不贡献任何 flag**；每个参数页顶部还有一个
「恢复引擎默认（不发送本页参数）」按钮，一键把整页退回这个状态。

<a id="vram-prediction"></a>

### 🧮 显存预测与在线校准 — 仅 `-kvmem` 版

kvmem 引擎的 **KVMem 参数页**顶部多出一张只读卡片：它算出这次运行*实际会占用多少*
显存，再反解成最大的一个不会 OOM 的 `--kvmem-budget` 建议值。llama.cpp 引擎里
这张卡片、它的采样线程、它的学习文件、它可能带来的启动延迟——全都不存在。

| 卡片字段 | 含义 |
|---|---|
| 预计峰值 | 权重 + KV 池 + MTP 池 + mmproj 的结构估算，再叠加这台机器学出来的残差 |
| 经验安全峰值 | 预计峰值 + 安全余量 |
| 当前空闲 | NVML 实时采到的空闲 / 总量 |
| 安全余量 | 空闲 − 经验安全峰值 |
| 建议 KVMem Budget | 空闲显存放得下的最大 budget，按 block 粒度二分搜出来 |
| 成功样本数 · 置信度 | 学过了几次，以及安全余量来自哪一档 |

数字来自三层叠加：

1. **物理模型** —— 复刻 rc2 的 `kvmem_compute_pool()`：`budget` 与 `gen_reserve`
   各自向下对齐到 `--kvmem-block-tokens`（不足一个 block 保底一个）、
   `gen_reserve` 为 0 时取 256、`budget` 为 0 时池子不超过 `-c`、
   `--kvmem-gpu-ratio` 的上限只看主 KV 的 slot 大小。每个 token 占多少字节由
   GGUF 里的层数、头数、行大小和量化类型决定，运行之后改以引擎自己打印的
   `KVMEM_KV_BYTES` / `KVMem slot-pool` / `KVMEM_TRACE mtp_pool` 为准。
2. **在线残差学习** —— 每次运行结束，用实测整卡峰值减去物理模型值得到一个残差，
   以 RLS（遗忘因子 0.995）拟合。学的是*这张卡、这个构建、这个模型*的偏差，
   不是一个通用的经验常数。
3. **单侧安全上界** —— 没有样本时按结构余量，1–4 个样本取最大正向误差，
   5 个以上取残差 P95：预测可以被学得保守，学不出乐观。

**预测永不改参数。** 卡片上没有一个是可编辑控件，唯一的写入路径是
「采用建议 Budget」按钮——点下去才把数字填进 `--kvmem-budget`，命令行照旧只由
参数表生成。

学习状态存在配置目录的 `vram_learning.json`，按 GPU 名称 + 显存总量 + 引擎构建 +
模型结构指纹分档，每档至多 64 条残差。里面只有数字：没有 prompt、没有聊天内容、
没有模型内容。

这些运行**不允许**教模型，卡片会写明是哪一条：显存不足退出、没到就绪、参数被引擎
拒绝、非正常结束、别的进程扰动了显存、启动前没采到基线，以及只有粗粒度采样——
`nvidia-smi` 降级时 1 秒一跳会跨过短峰值，量少了峰值就把安全余量学向了错误方向。

采样优先走 NVML（`ctypes` 直连，无需任何新依赖，约 75 ms 一跳）；只有它不可用时
才退回 `nvidia-smi`。启动前留一小段基线窗口（NVML 下约 450 ms）确认这张卡在本次
运行之前是什么状态；没有 GPU 或驱动时，卡片少几个数字，启动一秒都不多等。

<a id="advanced-tabs"></a>

### 🎛️ 高级模式 — 9 个标签页

| 标签页 | 内容 |
|---|---|
| 🧠 模型 | 模型文件、别名、标签、HF/URL/Docker 来源、LoRA、控制向量、mmproj（视觉） |
| 📏 上下文 | 上下文大小、prompt 与 KV 缓存、RoPE / YaRN 缩放 |
| 🎲 采样 | 温度、top-k/p、min-p、惩罚项、语法与重复控制 |
| 🎮 GPU-性能 | 卸载、显存、CPU 线程、亲和性、优先级 |
| ⚡ 投机解码 | 草稿模型（draft-mtp）、ngram、lookup cache、草稿 Token 数 |
| 🌐 服务 | host/port、slots、并行、端点、SSL、CORS、embedding/rerank、router |
| 🤖 Agent/工具 | 工具调用、MCP 服务、agent 设置 |
| 💬 聊天-推理 | 聊天模板、推理模式、thinking budget |
| 🔧 高级 | 文本 I/O、日志、`extra_args` 万能出口 |

<a id="model-browser"></a>

### 📂 模型浏览器

- 🔎 后台线程扫描模型目录 — 界面永不卡顿
- 🏷️ 自动把 `.gguf` 文件分类为 **模型** / **多模态 (mmproj)**，并显示大小
- 📏 即时模型信息：大小、估算参数量、量化类型
- 🧾 **GGUF 快速元数据行**：当前选中模型的架构 + 最大上下文，只解析文件*头部*（几 MB，有缓存，后台线程执行）；当上下文设置超过模型上限时该行变**琥珀色**提示（启动后会被截断）
- 🔗 按名称自动匹配 mmproj 到对应模型
- 📁 扫描目录跨会话记忆；F5 重新扫描

<a id="log-parsing"></a>

### 📊 实时日志解析

`llama-server` 的每一行输出都会在被打印的同时解析（74 条规则，兼容新旧两种日志格式，含 v9174+ `srv` 前缀格式），提炼进 **运行时信息** 面板 — 8 大类、40+ 数据点：

| 类别 | 你看到的信息 |
|---|---|
| 🖥️ 硬件 | GPU 名称 / 计算能力 / 每卡总显存与空闲显存、CPU |
| 📦 模型 | 文件、模型名、量化类型、GGUF 版本 |
| 🏗️ 架构 | 参数量、层数、embed/FFN 维度、词表、张量精度分布 |
| ⚙️ 运行参数 | 训练 vs 运行上下文、batch/ubatch、slots、RoPE 频率、thinking 模式 |
| 💾 显存 | 卸载层数、模型/KV 缓存/计算缓冲、预计显存占用 |
| ⚡ 性能 | Flash attention、KV 统一、图节点数与分割数 |
| 🔧 系统 | 线程、OpenMP、repack |
| 👁️ 视觉 | 编码器状态、mmproj、图像分辨率、最小图像 Token |

面板**实时刷新** — 卸载层数在加载过程中逐步出现，缓冲大小在初始化时填充，服务端开始监听的那一刻状态立即切换为 **就绪**。

<a id="log-panel"></a>

### 📄 日志面板

- 🔍 **Ctrl+F 搜索**，带匹配计数与循环查找
- **级别过滤** — 调试 / 信息 / 警告 / 错误（默认跟随"错误"级别）
- 📤 导出可见区域或**完整运行日志**（每次运行都会完整镜像到 `~/.llama-cpp-launcher/logs/last_run.log`）
- 🪟 **每级独立历史窗口** — 调试 / 信息 / 警告 / 错误各保留自己 5,000 行的窗口，隐藏级别的日志爆发（如 debug 提示词转储）不会把你正在过滤的级别挤掉
- 自动滚动开关、清空按钮、按级别着色

<a id="presets"></a>

### 💾 预设管理

- 保存 / 加载 / 删除命名预设 — 只存储与默认值的差异
- 以 JSON 文件导入 / 导出，方便分享；列表中显示创建时间
- 保存时可选择**包含或排除本机路径**（model/mmproj）
- 上次加载的预设会在**下次启动时自动恢复**
- 跨 llama.cpp 版本被重命名/删除的参数键会在加载时自动迁移

<a id="param-help"></a>

### ❓ 逐参数帮助

每个参数行都有 **?** 按钮，点开后弹出浮动帮助卡片：CLI 参数名、你二进制版本的*实时默认值*（来自 `--help`）、取值范围，以及通俗的解释。再也不用记 `--no-kv-offload` 是干什么的了。

<a id="gguf-inspector"></a>

### 🔬 GGUF 检查器

内置二进制检查器（不加载权重，纯标准库解析）：

- **7 个标签页**：概览 · 统计 · 元数据 · 张量 · 分词器 · 文件名 · 诊断
- 📊 可视化分析：量化分布、层/模块结构、参数集中度
- 🩺 **启动器感知诊断**：上下文超出模型上限、mmproj 名称不匹配、draft-mtp 缺少 sidecar、聊天模板 / RoPE / MoE / 分片信息
- 📤 导出 JSON / CSV / Markdown · 后台线程解析 + 内存缓存

<a id="server-lifecycle"></a>

### 🚀 服务生命周期

- ▶️ 一键启动/停止，彩色状态指示 + 运行计时器（MM:SS）
- ⚠️ 启动前端口冲突检测
- 🌐 一键在浏览器中打开 llama-server WebUI
- 优雅停止（非阻塞，超时强杀兜底）；关闭应用时自动停止服务
- ↩️ **撤销** — 800ms 防抖快照，最多回退 20 步
- 📝 **命令预览** — 精确的 `llama-server` 命令，每次修改即时更新，一键复制（含空格路径已正确加引号）

<a id="window-ui"></a>

### 🪟 窗口与界面

- 🖼️ **无边框圆角卡片窗口** — 应用自绘卡片与柔和投影：一体式标题行（图标 · 标题 · 菜单 · Win11 风格最小化/最大化/关闭），拖动移动、双击最大化，贴住可见卡片边缘即可边缘缩放；最大化时卡片铺满全屏并使用系统自身的圆角。
- 🎴 **主题化对话框与消息框** — 所有对话框、确认框、错误提示都属于同一无边框卡片家族（深色/浅色自适应），风格统一不突兀。
- 📐 **默认尺寸舒适** — 首次启动以 1600×940（按屏幕可用区钳制）打开，左侧模型栏 400px；窗口大小、位置、模式、当前标签页与分割栏布局在下次启动时自动恢复。
- 🧘 **小窗体不塌方** — 参数区永不出垂直滚动条，控件不会被压缩或重叠：快捷开关自动换行成更多行并增大窗口最小尺寸；极窄窗口下宽行横向滚动。

<a id="i18n-themes"></a>

### 🌐 国际化与主题

- 🈶/🈷 **中 ↔ 英实时切换** — 无需重启，偏好持久保存
- 🌙 **深色 / 浅色主题** — **设置**菜单一键切换（文件 · 设置 · 帮助），自动记忆；日志面板与检查器均随主题适配
- 菜单布局：**文件**（扫描路径、llama-server 路径、刷新、退出）· **设置**（自定义快捷开关、语言、主题）· **帮助**（关于）

<a id="comparison"></a>

## 🆚 与其他工具对比

| 功能 | Llama CPP Launcher | Ollama | LM Studio |
|---|:---:|:---:|:---:|
| 自带 llama.cpp 二进制 | ✅ | ❌ | ❌ |
| 当天使用最新 llama.cpp | ✅ | ❌ | ❌ |
| 任意自定义后端（CUDA/ROCm/Vulkan/Metal/SYCL） | ✅ | ❌ | ❌ |
| 完整 CLI 参数访问（226 个） | ✅ | ❌ | 部分 |
| 实际命令可见且可复制 | ✅ | ❌ | ❌ |
| 版本感知的预设迁移 | ✅ | ❌ | ❌ |
| 内置 GGUF 检查器 | ✅ | ❌ | ❌ |
| 无遥测，100% 本地 | ✅ | ❌ | ✅ |
| 单一轻量依赖（PyQt6） | ✅ | — | ❌ |

<a id="development"></a>

## 🛠️ 开发

- Python 3.11+，PyQt6（版本锁定），pytest 测试
- 约 19,000 行应用代码（另有约 11,000 行测试）；核心 schema（`core/params_schema.py`）是 UI、CLI 命令生成、get/set 值、i18n 覆盖率的唯一事实来源 — 新增一个参数只需一条记录
- schema 是**可注入的**：`core/engine.py` 把引擎 id 映射到它的参数模块、默认值模块、基线与能力集，所以接第二个服务器二进制是「加一张表 + 几个钩子」而不是 fork；`ui/advanced_panel.py` 的 `_read_hooks` / `_write_hooks` / `_retranslate_extras` 对 llama.cpp 永远为空，这正是它行为逐字节不变的原因
- `gguf/`、`ui/log_parser.py`、`ui/command_builder.py`、`core/kvmem_params_schema.py`、`core/kvmem_errors.py` 均不依赖 Qt，可无界面单元测试

<details>
<summary><b>项目结构</b></summary>

```
llama-cpp-launcher/
├── main.py                  # 入口（异步启动、日志配置）
├── run.bat                  # Windows 启动脚本（激活 venv）
├── build_config.py          # 应用名 / 版本（CI 按 tag 改写）
├── llama_cpp_launcher.spec  # PyInstaller 构建配置（内嵌图标）— 正式版 exe
├── llama_cpp_launcher_kvmem.spec  # …-kvmem 版（同一个程序，开放第二引擎）
├── requirements.txt
├── core/
│   ├── params_schema.py     # ★ 226 参数 schema（Qt-free，唯一事实来源）
│   ├── kvmem_params_schema.py  # ★ 52 参数 kvmem schema + 每引擎参数校验
│   ├── engine.py            # 引擎注册表：schema、默认值、基线、能力集
│   ├── kvmem_identity.py    # 从安装目录读取构建身份
│   ├── kvmem_errors.py      # kvmem 服务器输出 → 「哪个参数、去哪儿改」
│   ├── params_help.py       # 逐参数帮助文本（226 条）
│   ├── defaults.py          # `--help` 解析、回退默认值、GPU 探测
│   ├── defaults_kvmem.py    # kvmem 二进制的对应实现（逐引擎 flag 索引）
│   ├── config.py            # 预设与设置 IO（~/.llama-cpp-launcher）
│   ├── runner.py            # QProcess 封装（启动/停止/就绪检测）
│   ├── i18n.py              # 中/英翻译（中文为源语言）
│   └── constants.py
├── gguf/                    # 纯标准库 GGUF 二进制读取器
│   ├── parser.py  models.py  ggml_types.py  filename.py  diagnostics.py
├── ui/
│   ├── main_window.py       # 窗口编排、主题、日志面板
│   ├── frameless.py         # 无边框圆角卡片窗口、标题栏、边缘缩放
│   ├── message_box.py       # 主题化无边框消息框与对话框基类
│   ├── basic_panel.py       # 基础模式（含可自定义快捷开关）
│   ├── advanced_panel.py    # 高级模式（schema 驱动，llama 9 页 / kvmem 6 页）
│   ├── kvmem_linkage.py     # 仅 kvmem 的控件行为（哨兵文案、KV 配对、门控）
│   ├── quick_params.py      # 快捷开关候选池 + 控件工厂
│   ├── quick_params_dialog.py  # 自定义快捷开关对话框
│   ├── server_path_dialog.py   # llama-server 路径对话框
│   ├── param_help.py        # "?" 按钮 + 浮动帮助卡片
│   ├── model_browser.py     # GGUF 扫描器（后台线程）
│   ├── gguf_inspector.py    # 7 标签页检查器对话框
│   ├── log_parser.py        # 日志行模式 → 运行时信息（Qt-free）
│   ├── command_builder.py   # 参数 → `llama-server` argv（Qt-free）
│   └── runtime_info.py      # 运行时信息 HTML（Qt-free）
├── tests/                   # 38 个测试模块（无界面，内存假数据）
└── assets/icon.ico|png
```

</details>

**发布**：推送 `v*` tag 触发 CI（windows-latest）— 跑测试 → PyInstaller 打包 → 自动发布 exe 到 GitHub Release。

**`-kvmem` 版不由 CI 发布** — 工作流只打包 `llama_cpp_launcher.spec`。想要它请在本地
clone 里执行 `pyinstaller --noconfirm --workpath build_kvmem llama_cpp_launcher_kvmem.spec`，
产出单文件 `dist/LlamaCppLauncher-kvmem.exe`：名字独立、工作目录独立，绝不覆盖正式版 exe。

<a id="license"></a>

## 📄 许可证

[MIT](LICENSE) — 想怎么用就怎么用。
