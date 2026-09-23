# -*- coding: utf-8 -*-
"""kvmem-llama.cpp engine parameter schema (v0.16.0-rc2).

Second engine for the launcher: `llama-kvmem-server.exe`, an independent
single-slot OpenAI-compatible server (it does not share llama.cpp's
`common_params`, so its CLI surface is neither a subset nor a superset of the
llama-server one). This module is the kvmem counterpart of
`core.params_schema` and exposes the same public surface, so every consumer
(CommandBuilder, AdvancedPanel, BasicPanel, quick toggles, ConfigManager) can
be handed either module unchanged.

Authority for every entry below (in this order, NOT "it appeared in --help"):

  1. the real argv parser — `tools/llama-kvmem-server.cpp` at tag
     v0.16.0-rc2 is a hand-written if/else chain that matches flags exactly and
     exits 1 with `unknown flag: X` for anything else;
  2. `llama-kvmem-server --help` (printed to stderr, defaults written as
     `(default X)`), captured verbatim as
     tests/data/kvmem_help_v0.16.0-rc2.txt;
  3. live probing of this exact binary (accepted / rejected / silently
     accepted nonsense), because the parser validates very little: `-ngl auto`,
     `--kvmem-gpu-ratio 2` and `--reasoning-effort bogus` all pass, so the GUI
     ranges below are the only clamp that exists.

Hard rules encoded here:

  * `default` is the *binary's* real default, never "the value we recommend".
    `CommandBuilder.defaults` (the is_default / preset-diff baseline) is built
    from this column, so a param whose UI value equals the binary default emits
    nothing — which is exactly right, and why the recommended IQ3 recipe is
    applied through INITIAL_OVERRIDES instead (see below).
  * no CLI alias gets two controls (`--temp/--temperature`,
    `--repeat-penalty/--repetition-penalty`, `--kvmem/--no-kvmem`, ... are one
    param each; the alias list in `flags` only feeds the --help index).
  * features this build cannot do have NO control at all: NVMe
    (`--kvmem-nvme-gb/-dir`, `--kvmem-raw-k-nvme` exit 1 — NVMe is compiled
    out), `--jinja` (accepted but inert — native Jinja rendering is always on,
    and unticking a merged box would emit the rejected `--no-jinja`),
    `--parallel`/`--device`/`--tensor-split` (single slot),
    `--model-draft`/`-ngld`, `-ub`, `-fa`, `--lora`, `-t`, `--mlock/--no-mmap`,
    `--verbose`/`-v`/`--log-verbosity`/`-lv`, `--api-key`/CORS,
    `--list-devices`. The two lists split those by what the *parser* does:
    REJECTED_FLAGS holds the ones probed live as `unknown flag`, and
    NO_CONTROL_KEYS the ones it accepts (so --help lists them and a reader
    wonders where the widget is) that this schema deliberately leaves without
    one.

`Param` gains no field, `CommandBuilder._emit` gains no mode, and
`core/params_schema.py` is not touched: engine-specific behaviour is expressed
with the existing fields (wattr=None hidden emitters, `value` as a free
value-kind label, min/max/items/skip_values/parser).
"""
import json
from pathlib import Path

from .i18n import t
from .params_schema import Param, P  # P is the same frozen dataclass

ENGINE_ID = "kvmem"

# ---------------------------------------------------------------------------
# Item lists (English literals — t() passes them through). kvmem rejects
# anything outside these sets at parse time (`-ctk bf16` -> "unsupported cache
# type (want f16|f32|q8_0|q5_0|q4_0)"), so they are narrower than llama.cpp's.
# ---------------------------------------------------------------------------
KVMEM_CACHE_TYPE_ITEMS = ["f16", "f32", "q8_0", "q5_0", "q4_0"]
# "" = follow --kv-dtype (the control emits nothing when left empty)
KVMEM_CTK_TYPE_ITEMS = [""] + KVMEM_CACHE_TYPE_ITEMS
KVMEM_SPEC_TYPE_ITEMS = ["none", "draft-mtp"]
# --spec-kv-dtype reports its accepted set in this order
KVMEM_SPEC_KV_TYPE_ITEMS = ["f16", "q8_0", "q5_0", "q4_0", "f32"]

#: Suggestions for the editable --reasoning-effort combo — the one item list
#: here that is NOT a hard set. --help types the flag as `LEVEL`, and the server
#: special-cases only two of these strings ("default uses template default, none
#: disables thinking"), forwarding the rest to the model template. The shipped
#: Qwen3.8 GSQ template accepts xhigh (its default), medium and low and raises
#: for anything else, so those three plus none and default are the five levels
#: that actually work here; "high" is not one of them, which is why it is not
#: offered. A custom template may define its own levels, so the combo stays
#: editable and validate_params() rule 2 treats this list as suggestions.
REASONING_EFFORT_ITEMS = ["none", "default", "low", "medium", "xhigh"]

TAB_TITLES = (
    ("model", "模型"),
    ("context", "上下文"),
    ("sampling", "采样"),
    ("kvmem", "KVMem"),
    ("reasoning", "推理"),
    ("server", "服务/高级"),
)

# ---------------------------------------------------------------------------
# The table. Tuple order IS command-line emission order (CommandBuilder walks
# this tuple); tab+row only drive the UI order. Notably --kv-dtype must precede
# an explicit -ctk/-ctv pair: kvmem resolves a K/V mismatch by argv order
# (`--kv-dtype f32 -ctk q8_0` is fatal, `-ctk q8_0 --kv-dtype f32` passes).
# ---------------------------------------------------------------------------
PARAMS = (
    # ---------- model ----------
    P(key='model', tab='model', row=0, label='模型 (--model)', wattr='adv_model',
      widget='file', default='', emit='truthy', fmt='str', flag='-m',
      flags=('-m', '--model'), parser='str', placeholder="选择或输入模型文件路径",
      browse='file', filter_str='GGUF Files (*.gguf)', browse_title="选择模型",
      tooltip="kvmem 只支持单个 GGUF 模型文件"),
    P(key='mmproj', tab='model', row=1, label='视觉投影 (--mmproj)', wattr='adv_mmproj',
      widget='file', default='', emit='truthy', fmt='str', flag='--mmproj',
      flags=('--mmproj',), parser='str', placeholder="选择或输入视觉投影模型路径",
      browse='file', filter_str='GGUF Files (*.gguf)', browse_title="选择MMProj"),
    P(key='mmproj_offload', tab='model', row=2,
      label='视觉编码 GPU 卸载 (--mmproj-offload):', wattr='adv_mmproj_offload',
      widget='check', default=True, emit='bool_neg',
      flag=['--mmproj-offload', '--no-mmproj-offload'],
      flags=('--mmproj-offload', '--no-mmproj-offload'), parser='bool',
      tooltip="默认放在 GPU；IQ4 配方可取消勾选改到 CPU"),
    P(key='image_min_tokens', tab='model', row=3,
      label='图像最小 Token (--image-min-tokens):', wattr='adv_image_min_tokens',
      widget='spin', default=0, emit='diff_pos', fmt='int',
      flag='--image-min-tokens', flags=('--image-min-tokens',), parser='int',
      min=0, max=999999, tooltip="0 = 不发送，使用模型内置值（该参数只接受正整数）"),
    P(key='image_max_tokens', tab='model', row=4,
      label='图像最大 Token (--image-max-tokens):', wattr='adv_image_max_tokens',
      widget='spin', default=0, emit='diff_pos', fmt='int',
      flag='--image-max-tokens', flags=('--image-max-tokens',), parser='int',
      min=0, max=999999, tooltip="0 = 不发送，使用模型内置值（官方配方用 512）"),

    # ---------- context ----------
    P(key='ctx_size', tab='context', row=0,
      label='上下文大小 (--ctx-size):', wattr='adv_ctx_size', widget='spin',
      default=2048, emit='diff', fmt='int', flag='-c',
      flags=('-c', '--ctx-size'), parser='int', min=1, max=1048576,
      tooltip="二进制默认 2048；官方 IQ3/IQ4 配方用 262144"),
    P(key='n_predict', tab='context', row=1,
      label='默认生成长度 (--n-predict):', wattr='adv_n_predict', widget='spin',
      default=128, emit='diff', fmt='int', flag='-n', flags=('-n', '--n-predict'),
      parser='int', min=-1, max=1048576,
      tooltip="每个请求未指定 max_tokens 时的默认值，可被请求覆盖；-1 = 不限制"),
    P(key='batch_size', tab='context', row=2,
      label='逻辑批大小 (--batch-size):', wattr='adv_batch_size', widget='spin',
      default=512, emit='diff', fmt='int', flag='-b',
      flags=('-b', '--batch-size'), parser='int', min=1, max=65536,
      tooltip="kvmem 只有 --batch-size，没有物理批 -ub"),
    P(key='n_gpu_layers', tab='context', row=3,
      label='GPU 层数 (--n-gpu-layers):', wattr='adv_n_gpu_layers', widget='spin',
      default=99, emit='diff', fmt='int', flag='-ngl',
      flags=('-ngl', '--n-gpu-layers'), parser='int', min=0, max=999,
      tooltip="引擎用 atoi 解析：非数字（含 auto）一律当 0，所以这里只能是整数"),

    # ---------- sampling ----------
    # The binary default is model-dependent (Thinking / non-Thinking selected
    # per request), and --help writes it as "(1.0 / 0.7)" — not parseable. The
    # schema default is therefore the "do not send" sentinel, and these keys
    # are all in NO_DRIFT_KEYS. Spin minimum == sentinel on purpose: Qt never
    # clamps an unmodified value into the legal range.
    P(key='temperature', tab='sampling', row=0,
      label='温度 (--temperature):', wattr='adv_temperature', widget='dspin',
      default=-1.0, emit='diff', fmt='f2', flag='--temperature',
      flags=('--temp', '--temperature'), parser='float', min=-1.0, max=2.0,
      step=0.01, tooltip="-1.00 = 引擎默认（不发送）；有效区间 [0, 2]，0 = 贪心"),
    P(key='top_p', tab='sampling', row=1, label='Top-P (--top-p):',
      wattr='adv_top_p', widget='dspin', default=-1.0, emit='diff', fmt='f2',
      flag='--top-p', flags=('--top-p',), parser='float', min=-1.0, max=1.0,
      step=0.01, tooltip="-1.00 = 引擎默认（不发送）；有效区间 [0, 1]"),
    P(key='top_k', tab='sampling', row=2, label='Top-K (--top-k):',
      wattr='adv_top_k', widget='spin', default=-1, emit='diff', fmt='int',
      flag='--top-k', flags=('--top-k',), parser='int', min=-1, max=2147483647,
      tooltip="-1 = 引擎默认（不发送）；0 = 关闭 Top-K，有效值 >= 0"),
    P(key='min_p', tab='sampling', row=3, label='Min-P (--min-p):',
      wattr='adv_min_p', widget='dspin', default=-1.0, emit='diff', fmt='f2',
      flag='--min-p', flags=('--min-p',), parser='float', min=-1.0, max=1.0,
      step=0.01, tooltip="-1.00 = 引擎默认（不发送）；有效区间 [0, 1]"),
    P(key='presence_penalty', tab='sampling', row=4,
      label='存在惩罚 (--presence-penalty):', wattr='adv_presence_penalty',
      widget='dspin', default=-3.0, emit='diff', fmt='f2',
      flag='--presence-penalty', flags=('--presence-penalty',), parser='float',
      min=-3.0, max=2.0, step=0.01,
      tooltip="-3.00 = 引擎默认（不发送）；有效区间 [-2, 2]"),
    P(key='frequency_penalty', tab='sampling', row=5,
      label='频率惩罚 (--frequency-penalty):', wattr='adv_frequency_penalty',
      widget='dspin', default=-3.0, emit='diff', fmt='f2',
      flag='--frequency-penalty', flags=('--frequency-penalty',), parser='float',
      min=-3.0, max=2.0, step=0.01,
      tooltip="-3.00 = 引擎默认（不发送）；有效区间 [-2, 2]"),
    P(key='repeat_penalty', tab='sampling', row=6,
      label='重复惩罚 (--repeat-penalty):', wattr='adv_repeat_penalty',
      widget='dspin', default=-1.0, emit='diff', fmt='f2', flag='--repeat-penalty',
      flags=('--repeat-penalty', '--repetition-penalty'), parser='float',
      min=-1.0, max=10.0, step=0.01,
      tooltip="-1.00 = 引擎默认（不发送）；必须 > 0，--repetition-penalty 是同义别名"),
    P(key='seed', tab='sampling', row=7, label='随机种子 (--seed):',
      wattr='adv_seed', widget='spin', default=-1, emit='diff', fmt='int',
      flag='--seed', flags=('--seed',), parser='int', min=-1, max=2147483647,
      tooltip="-1 = 随机（不发送）；引擎接受 uint32，此处上限为 Qt 整数上限"),

    # ---------- kvmem ----------
    P(key='kvmem_enabled', tab='kvmem', row=0,
      label='启用 KVMem (--kvmem):', wattr='adv_kvmem_enabled', widget='check',
      default=True, emit='bool_neg', flag=['--kvmem', '--no-kvmem'],
      flags=('--kvmem', '--no-kvmem'), parser='bool',
      tooltip="关闭后 KV 全部留在常规缓存里，检索/快照功能不再生效"),
    P(key='kvmem_budget', tab='kvmem', row=1,
      label='GPU 工作集 Token (--kvmem-budget):', wattr='adv_kvmem_budget',
      widget='spin', default=0, emit='diff', fmt='int', flag='--kvmem-budget',
      flags=('--kvmem-budget',), parser='int', min=0, max=1048576,
      tooltip="0 = 等于上下文长度；官方 IQ3 配方 36864，显存不足时下调"),
    P(key='kvmem_block_tokens', tab='kvmem', row=2,
      label='块大小 (--kvmem-block-tokens):', wattr='adv_kvmem_block_tokens',
      widget='spin', default=128, emit='diff', fmt='int',
      flag='--kvmem-block-tokens', flags=('--kvmem-block-tokens',), parser='int',
      min=1, max=65536, tooltip="官方配方用默认值 128"),
    P(key='kvmem_gen_reserve', tab='kvmem', row=3,
      label='解码预留 Token (--kvmem-gen-reserve):', wattr='adv_kvmem_gen_reserve',
      widget='spin', default=256, emit='diff', fmt='int',
      flag='--kvmem-gen-reserve', flags=('--kvmem-gen-reserve',), parser='int',
      min=0, max=1048576, tooltip="官方 IQ3 配方 16384，应与 --n-predict 一致"),
    P(key='kvmem_recent_tokens', tab='kvmem', row=4,
      label='常驻最近 Token (--kvmem-recent-tokens):',
      wattr='adv_kvmem_recent_tokens', widget='spin', default=0, emit='diff',
      fmt='int', flag='--kvmem-recent-tokens',
      flags=('--kvmem-recent-tokens',), parser='int', min=0, max=1048576,
      tooltip="工作集里始终保留的最新后缀长度"),
    P(key='kvmem_method', tab='kvmem', row=5,
      label='工作集策略 (--kvmem-method):', wattr='adv_kvmem_method',
      widget='combo', default='retrieval', emit='diff', fmt='str',
      flag='--kvmem-method', flags=('--kvmem-method',), parser='str',
      items=("retrieval", "recency"), curtext="retrieval",
      tooltip="retrieval = 按检索命中换入换出；recency = 纯近因保留"),
    P(key='kvmem_query_last', tab='kvmem', row=6,
      label='回退查询尾长 (--kvmem-query-last):', wattr='adv_kvmem_query_last',
      widget='spin', default=64, emit='diff', fmt='int',
      flag='--kvmem-query-last', flags=('--kvmem-query-last',), parser='int',
      min=0, max=65536, tooltip="拿不到最后一条用户消息时，用结尾这么多 token 做查询"),
    P(key='kvmem_query_max_tokens', tab='kvmem', row=7,
      label='查询 Token 上限 (--kvmem-query-max-tokens):',
      wattr='adv_kvmem_query_max_tokens', widget='spin', default=512,
      emit='diff', fmt='int', flag='--kvmem-query-max-tokens',
      flags=('--kvmem-query-max-tokens',), parser='int', min=1, max=65536,
      tooltip="从末尾截取查询用的最大 token 数（qw3 风格）"),
    P(key='kvmem_query_replay', tab='kvmem', row=8,
      label='查询回放模式 (--kvmem-query-replay):', wattr='adv_kvmem_query_replay',
      widget='combo', default='auto', emit='diff', fmt='str',
      flag='--kvmem-query-replay', flags=('--kvmem-query-replay',), parser='str',
      items=("auto", "legacy"), curtext="auto"),
    P(key='kvmem_query_policy', tab='kvmem', row=9,
      label='查询选取策略 (--kvmem-query-policy):',
      wattr='adv_kvmem_query_policy', widget='combo', default='user',
      emit='diff', fmt='str', flag='--kvmem-query-policy',
      flags=('--kvmem-query-policy',), parser='str',
      items=("user", "legacy"), curtext="user",
      tooltip="官方配方用 user：只取最后一条用户消息做检索查询"),
    P(key='kvmem_mtp_state', tab='kvmem', row=10,
      label='MTP 状态快照 (--kvmem-mtp-state):', wattr='adv_kvmem_mtp_state',
      widget='combo', default='replay', emit='diff', fmt='str',
      flag='--kvmem-mtp-state', flags=('--kvmem-mtp-state',), parser='str',
      items=("snapshots", "auto", "replay"), curtext="replay",
      tooltip="help 写作 “default replay with MTP”，本表存解析后的 replay"),
    P(key='kvmem_gpu_ratio', tab='kvmem', row=11,
      label='显存占用比例上限 (--kvmem-gpu-ratio):', wattr='adv_kvmem_gpu_ratio',
      widget='dspin', default=0.50, emit='diff', fmt='f2',
      flag='--kvmem-gpu-ratio', flags=('--kvmem-gpu-ratio',), parser='float',
      min=0.0, max=1.0, step=0.05,
      tooltip="槽位池最多占这么多比例的显存；引擎不校验，超过 1 也照收"),
    P(key='kvmem_cpu_gb', tab='kvmem', row=12,
      label='CPU 溢出区 GiB (--kvmem-cpu-gb):', wattr='adv_kvmem_cpu_gb',
      widget='dspin', default=0.0, emit='diff', fmt='f2', flag='--kvmem-cpu-gb',
      flags=('--kvmem-cpu-gb',), parser='float', min=0.0, max=512.0, step=0.5,
      tooltip="0 = 不分配；本构建没有 NVMe，CPU 层就是最外层"),
    P(key='kvmem_harvest_v', tab='kvmem', row=13,
      label='回收 V 到 Host (--kvmem-harvest-v):', wattr='adv_kvmem_harvest_v',
      widget='check', default=False, emit='bool_pos', flag='--kvmem-harvest-v',
      flags=('--kvmem-harvest-v',), parser='bool',
      tooltip="prefill 时把 V 一起 D2H 收到 host 层；本构建不落 NVMe，所以它会常驻内存"
              "（检索命中后不必重算 V）。开启会占用额外 RAM。"),
    P(key='kv_dtype', tab='kvmem', row=14,
      label='KV 缓存类型 (--kv-dtype):', wattr='adv_kv_dtype', widget='combo',
      default='q8_0', emit='diff', fmt='str', flag='--kv-dtype',
      flags=('--kv-dtype',), parser='str', items=KVMEM_CACHE_TYPE_ITEMS,
      curtext="q8_0", tooltip="同时设定 K 与 V；量化类型时 K/V 必须相同"),
    P(key='cache_type_k', tab='kvmem', row=15,
      label='K 缓存类型 (-ctk):', wattr='adv_cache_type_k', widget='combo',
      default='', emit='diff_nonempty', fmt='str', flag='-ctk',
      flags=('-ctk', '--cache-type-k'), parser='str',
      items=KVMEM_CTK_TYPE_ITEMS, curtext="",
      tooltip="留空 = 跟随 --kv-dtype；量化类型时 V 会被同步为同值"),
    P(key='cache_type_v', tab='kvmem', row=16,
      label='V 缓存类型 (-ctv):', wattr='adv_cache_type_v', widget='combo',
      default='', emit='diff_nonempty', fmt='str', flag='-ctv',
      flags=('-ctv', '--cache-type-v'), parser='str', items=KVMEM_CTK_TYPE_ITEMS,
      curtext="", tooltip="留空 = 跟随 --kv-dtype"),
    P(key='spec_type', tab='kvmem', row=17,
      label='投机解码类型 (--spec-type):', wattr='adv_spec_type', widget='combo',
      default='none', emit='diff_skip', fmt='str', flag='--spec-type',
      skip_values=('none',), flags=('--spec-type',), parser='str',
      items=KVMEM_SPEC_TYPE_ITEMS, curtext="none",
      tooltip="draft-mtp 需要模型自带 nextn/MTP 头，否则引擎会拒绝"),
    P(key='spec_kv_dtype', tab='kvmem', row=18,
      label='MTP KV 类型 (--spec-kv-dtype):', wattr='adv_spec_kv_dtype',
      widget='combo', default='f16', emit='diff', fmt='str',
      flag='--spec-kv-dtype', flags=('--spec-kv-dtype',), parser='str',
      items=KVMEM_SPEC_KV_TYPE_ITEMS, curtext="f16"),
    P(key='spec_draft_n_max', tab='kvmem', row=19,
      label='草稿 Token 数 (--spec-draft-n-max):', wattr='adv_spec_draft_n_max',
      widget='spin', default=3, emit='diff', fmt='int',
      flag='--spec-draft-n-max', flags=('--spec-draft-n-max',), parser='int',
      min=1, max=16, tooltip="官方配方用 3"),
    P(key='spec_draft_p_min', tab='kvmem', row=20,
      label='草稿最低概率 (--spec-draft-p-min):', wattr='adv_spec_draft_p_min',
      widget='dspin', default=0.0, emit='diff', fmt='f2',
      flag='--spec-draft-p-min', flags=('--spec-draft-p-min',), parser='float',
      min=0.0, max=1.0, step=0.01),

    # ---------- reasoning ----------
    # Tri-state lives in ONE UI control (thinking_mode, emit=none); the two
    # real flags are separate booleans with no widget (wattr=None emitters),
    # because --enable-thinking and --no-think are independent flags in the
    # parser and --reasoning-effort none is a third path — merging them into a
    # bool_neg control would emit a flag the user never chose.
    P(key='thinking_mode', tab='reasoning', row=0,
      label='思考模式:', wattr='adv_thinking_mode', widget='combo_index',
      default=0, emit='none', items=("（模板默认）", "开启思考", "关闭思考"),
      curidx=0, tooltip="只影响 --enable-thinking / --no-think 的发送，本身不是参数"),
    P(key='enable_thinking', tab='reasoning', row=None, label=None, wattr=None,
      widget=None, default=False, emit='bool_pos', flag='--enable-thinking',
      flags=('--enable-thinking',), parser='bool'),
    P(key='no_think', tab='reasoning', row=None, label=None, wattr=None,
      widget=None, default=False, emit='bool_pos', flag='--no-think',
      flags=('--no-think',), parser='bool'),
    P(key='reasoning_effort', tab='reasoning', row=1,
      label='推理力度 (--reasoning-effort):', wattr='adv_reasoning_effort',
      widget='combo_edit', default='', emit='diff_nonempty', fmt='str',
      flag='--reasoning-effort', flags=('--reasoning-effort',), parser='str',
      items=REASONING_EFFORT_ITEMS, value='combo_edit',
      placeholder="留空 = 使用模板默认",
      tooltip="none 会关闭思考，与「开启思考」互斥；下拉项只是建议值，"
              "自定义模板的其他 effort 会原样传给模板解释"),
    P(key='reasoning_budget', tab='reasoning', row=2,
      label='思考 Token 预算 (--reasoning-budget):', wattr='adv_reasoning_budget',
      widget='spin', default=-1, emit='diff', fmt='int',
      flag='--reasoning-budget', flags=('--reasoning-budget',), parser='int',
      min=-1, max=1048576,
      tooltip="-1 = 不限制（引擎默认，不发送）；0 = 立即结束思考；N>0 强制在第 N 个 token 处收尾"),
    P(key='reasoning_budget_message', tab='reasoning', row=3,
      label='强制收尾提示 (--reasoning-budget-message):',
      wattr='adv_reasoning_budget_message', widget='text', default='',
      emit='diff_nonempty', fmt='str', flag='--reasoning-budget-message',
      flags=('--reasoning-budget-message',), parser='str',
      placeholder="留空 = 不注入", tooltip="在强制 </think> 前注入的一句话"),

    # ---------- server / advanced ----------
    P(key='host', tab='server', row=0, label='主机 (--host):', wattr='adv_host',
      widget='text', default='127.0.0.1', emit='diff', fmt='str', flag='--host',
      flags=('--host',), parser='str'),
    # Baseline is the binary's 8080, NOT the official 18200: is_default()
    # suppresses every value equal to the baseline, so writing 18200 here would
    # mean --port is never sent. 18200 comes from INITIAL_OVERRIDES below,
    # which only seeds the UI state — so the flag really reaches argv.
    P(key='port', tab='server', row=1, label='端口 (--port):', wattr='adv_port',
      widget='spin', default=8080, emit='diff', fmt='int', flag='--port',
      flags=('--port',), parser='int', min=1, max=65535),
    P(key='no_ui', tab='server', row=2, label='关闭内置 Web UI (--no-ui):',
      wattr='adv_no_ui', widget='check', default=False, emit='bool_pos',
      flag='--no-ui', flags=('--no-ui',), parser='bool'),
    P(key='ui_dir', tab='server', row=3, label='UI 目录 (--ui-dir):',
      wattr='adv_ui_dir', widget='dir', default='', emit='truthy', fmt='str',
      flag='--ui-dir', flags=('--ui-dir',), parser='str',
      browse='dir', browse_title="选择 UI 目录",
      tooltip="引擎按可执行文件位置查找内置资源，一般留空即可"),
    # Deliberately keyed chat_template: AdvancedPanel._build_param_row already
    # special-cases that key for the editable template combo (the kvmem chat
    # template list is empty, so it degrades to a free-text combo) — which is
    # why core/params_schema.py stays untouched.
    P(key='chat_template', tab='server', row=4,
      label='聊天模板 (--chat-template):', wattr='adv_chat_template',
      widget='combo_edit', default='', emit='diff_nonempty', fmt='str',
      flag='--chat-template', flags=('--chat-template',), parser='str',
      value='combo_edit', placeholder="留空 = 使用模型内置模板",
      tooltip="直接写 Jinja 文本；与模板文件互斥"),
    P(key='chat_template_file', tab='server', row=5,
      label='模板文件 (--chat-template-file):', wattr='adv_chat_template_file',
      widget='file', default='', emit='diff_nonempty', fmt='str',
      flag='--chat-template-file', flags=('--chat-template-file',), parser='str',
      browse='file', filter_str="Jinja Files (*.jinja *.jinja2 *.txt);;All Files (*)",
      browse_title="选择模板文件", tooltip="与 --chat-template 互斥，且必须是非空文件"),
    P(key='chat_template_kwargs', tab='server', row=6,
      label='模板参数 (--chat-template-kwargs):', wattr='adv_chat_kwargs',
      widget='mtext', default='', emit='diff_nonempty', fmt='str',
      flag='--chat-template-kwargs', flags=('--chat-template-kwargs',),
      parser='str', placeholder='{"key": "value"}',
      tooltip='必须是 JSON object，引擎在解析参数阶段就会拒绝其他形状'),
    P(key='extra_args', tab='server', row=7, label='额外参数:',
      wattr='adv_extra_args', widget='mtext', default='', emit='extra',
      parser='bool', placeholder="额外参数，每行一个", value='text',
      tooltip="原样追加；kvmem 不认识的 flag 会让服务器直接退出"),
)

# ---------------------------------------------------------------------------
# Accessors (same public surface as core.params_schema)
# ---------------------------------------------------------------------------
PARAMS_BY_KEY = {p.key: p for p in PARAMS}
UI_PARAMS = [p for p in PARAMS if p.wattr is not None]
TAB_ORDER = tuple(key for key, _ in TAB_TITLES)


def tab_params(tab: str) -> list:
    """Params of one tab, in UI (row) order."""
    return sorted((p for p in UI_PARAMS if p.tab == tab), key=lambda p: p.row)


def fallback_defaults() -> dict:
    """Every binary real default (key -> value): the CommandBuilder baseline."""
    return {p.key: p.default for p in PARAMS}


def help_flag_maps():
    """(--help) parsing indexes, built from this schema's alias lists.

    Params with an empty ``flags`` tuple are deliberately invisible to --help
    parsing (thinking_mode is UI-only, extra_args is appended verbatim).

    ``parser`` must say whether the flag takes a value: kvmem's parser answers
    ``missing value for X`` for every value flag, and a live probe of
    v0.16.0-rc2 confirmed model / mmproj / ui_dir are value flags. Getting this
    wrong makes the --help index treat a path flag as a switch.

    Note: kvmem wraps long help entries, and the ``(default X)`` token can fall
    on the *continuation* line (--kvmem-query-max-tokens, --reasoning-budget),
    so core/defaults_kvmem.py joins continuation lines before extracting.
    """
    value_map, flag_map, neg_map = {}, {}, {}
    for p in PARAMS:
        if not p.flags:
            continue
        if p.parser != "bool":
            value_map[p.key] = (p.flags, p.parser)
        elif p.emit == "bool_neg":
            neg_map[p.key] = p.flags
        else:
            flag_map[p.key] = p.flags
    return value_map, flag_map, neg_map


def schema_i18n_strings():
    """Every Chinese-source string this schema carries (i18n coverage test)."""
    out = [title for _, title in TAB_TITLES]
    for p in PARAMS:
        for s in (p.label, p.placeholder, p.tooltip, p.browse_title,
                  p.list_title, p.list_filter):
            if s:
                out.append(s)
        if p.items:
            out.extend(p.items)
    return out


# ---------------------------------------------------------------------------
# Engine policy
# ---------------------------------------------------------------------------

#: Baseline the kvmem UI starts from: the official IQ3 recipe. This seeds
#: self.params ONLY — never the is_default baseline, which stays
#: fallback_defaults() — so every one of these values is actually emitted and
#: visible in the command preview.
INITIAL_OVERRIDES = {
    "port": 18200,
    "ctx_size": 262144,
    "n_predict": 16384,
    "kvmem_budget": 36864,
    "kvmem_gen_reserve": 16384,
    "spec_type": "draft-mtp",
    # spec_draft_n_max is deliberately absent: the binary default already is 3,
    # so repeating it here would be a no-op the tests below reject.
    "enable_thinking": True,
    "thinking_mode": 1,
    "reasoning_budget": 4096,
}

#: Keys whose value never comes from --help, even when a future binary starts
#: printing a parseable default there. Three reasons, all the same failure mode
#: (baseline := binary value => the user's visible value stops being sent, or a
#: drift report cries wolf):
#:   * the 8 sampling keys, whose defaults are dual-valued ("(1.0 / 0.7)") or
#:     structured ("(default random)") and cannot be parsed into a scalar;
#:   * port, which this engine always sends explicitly (see INITIAL_OVERRIDES);
#:   * cache_type_k / cache_type_v, whose "empty = follow --kv-dtype" state has
#:     no equivalent in --help wording, so adopting q8_0 there would report the
#:     normal state as drift.
NO_DRIFT_KEYS = frozenset({
    "temperature", "top_p", "top_k", "min_p", "presence_penalty",
    "frequency_penalty", "repeat_penalty", "seed", "port",
    "cache_type_k", "cache_type_v",
})

#: Per-user inputs: a "default" for them is never version drift (same role as
#: core.defaults.USER_INPUT_PARAMS, C5, but this engine's own key names). Read
#: by the stage-3/5 drift check, which compares --help against the baseline —
#: and by the per-engine validator, which must not treat an empty path as drift.
#: Kept separate from NO_DRIFT_KEYS on purpose: those are values we refuse to
#: *adopt* from a future --help because adopting them would stop us sending what
#: the user sees; these are values a --help default would simply not mean.
USER_INPUT_PARAMS = frozenset({
    "model", "mmproj", "ui_dir",
    "chat_template", "chat_template_file", "chat_template_kwargs",
    "reasoning_budget_message", "extra_args",
})

#: What the *parser* accepts as a meaningful value, used only by
#: MainWindow._validate_params. Deliberately separate from Param.min/max, which
#: are Qt spin ranges and must reach down to the "do not send" sentinel without
#: Qt clamping it into the legal range.
LEGAL_RANGE = {
    "temperature": (0.0, 2.0),
    "top_p": (0.0, 1.0),
    "top_k": (0, None),
    "min_p": (0.0, 1.0),
    "presence_penalty": (-2.0, 2.0),
    "frequency_penalty": (-2.0, 2.0),
    "repeat_penalty": (0.0, None),
    "seed": (0, 4294967295),
    "reasoning_budget": (-1, None),
    "kvmem_gpu_ratio": (0.0, 1.0),
}

#: Quick toggles for the kvmem basic panel (all must be quick-eligible kinds
#: and must not be owned by another basic-panel group).
QUICK_DEFAULT_KEYS = ("kvmem_enabled", "spec_type", "kvmem_method",
                      "kv_dtype", "mmproj_offload")

#: Params the kvmem basic panel renders in its own groups — never offered as
#: quick toggles (one widget per key).
BASIC_OWNED_KEYS = frozenset({
    "model", "mmproj", "ctx_size", "n_predict", "n_gpu_layers",
    "temperature", "top_p", "top_k", "min_p", "repeat_penalty", "seed",
    "host", "port",
})

#: Flags the argv parser refuses in this build, each one probed live against
#: v0.16.0-rc2 as `unknown flag: X` + usage + exit 1. Pinned by
#: tests/test_kvmem_flags_accepted.py so a future schema edit cannot quietly add
#: a control for something that exits 1.
#:
#: "Refused" is the whole membership test, and it is deliberately narrower than
#: "cannot be used": a flag the parser *accepts* belongs in NO_CONTROL_KEYS even
#: when using it is fatal. `--jinja` (accepted, inert) and the three NVMe flags
#: (accepted, then "NVMe offload is disabled in this build") are exactly that —
#: the first version of this list had them here, which mislabels what the schema
#: is allowed to emit. `--no-jinja` on the other hand is a real rejection.
REJECTED_FLAGS = (
    "--version", "--verbose", "-v", "--parallel", "--list-devices",
    "--model-draft", "-ngld", "-ub", "--ubatch-size", "--flash-attn", "-fa",
    "--mlock", "--no-mmap", "--api-key", "--no-jinja",
    "--device", "--tensor-split", "--threads", "-t", "--log-verbosity",
    "-lv", "--no-webui", "--webui",
)

#: Help-visible flags with no control on purpose, and why. The Tier3 test
#: asserts none of them is in PARAMS_BY_KEY.
NO_CONTROL_KEYS = {
    "--kvmem-nvme-gb": "parser 接受，但本构建 NVMe 编译关闭，非零值直接 exit 1",
    "--kvmem-nvme-dir": "只在 nvme_bytes>0 时才被读取，而那是硬错误",
    "--kvmem-raw-k-nvme": "parser 接受，但需要 NVMe，exit 1",
    "--jinja": "parser 接受但没有任何作用（Jinja 渲染始终开启）；"
               "合并成复选框会在取消勾选时发出被拒的 --no-jinja",
}

#: Cache types whose K forces V to be identical (measured: `--kv-dtype f32
#: -ctk q8_0` -> "incompatible KV cache types", fatal; the reverse order
#: passes, and a lone `-ctk q8_0` passes only because V falls back to
#: --kv-dtype's own q8_0).
QUANT_CACHE_TYPES = ("q8_0", "q5_0", "q4_0")

#: thinking_mode combo indices (the order of its `items` above).
THINKING_TEMPLATE_DEFAULT, THINKING_ON, THINKING_OFF = 0, 1, 2


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_params(values: dict) -> list:
    """States this engine's argv parser would reject: [(param_key, reason)].

    The rules are the *measured* failure paths of v0.16.0-rc2, plus the range
    checks the binary skips entirely. It exists because a kvmem rejection
    happens before anything readable is logged — the exit-1 line is the last
    thing the launcher sees, so an unchecked value looks like a crash rather
    than like a bad argument, and `-temperature -1.00` (the "do not send"
    sentinel leaking into argv) is precisely that case.

    Contract with the caller (MainWindow._engine_validation_problems): it only
    runs this when the engine's schema module defines it, so llama.cpp keeps
    starting with no pre-flight check at all. Nothing here rewrites values or
    touches the CommandBuilder — a passing command line is emitted exactly as
    the schema says.
    """
    problems: list = []

    def bad(key: str, reason: str):
        # Callers pass already-translated text: reasons are user-facing copy and
        # English mode must not leak Chinese into the rejection dialog.
        problems.append((key, reason))

    def get(key, default=None):
        return values.get(key, default)

    # 1. ranges the parser itself does not enforce (measured: `--kvmem-gpu-ratio
    #    2` and `-ngl auto` are accepted silently). A value equal to the schema
    #    default is the "do not send" sentinel, so it is legal by definition.
    for key, (lo, hi) in LEGAL_RANGE.items():
        value = get(key)
        if not _is_number(value) or value == PARAMS_BY_KEY[key].default:
            continue
        if lo is not None and value < lo:
            if hi is not None:
                bad(key, t("取值必须在 {lo} 到 {hi} 之间（当前 {v}）", lo=lo, hi=hi, v=value))
            else:
                bad(key, t("取值不得小于 {lo}（当前 {v}）", lo=lo, v=value))
        elif hi is not None and value > hi:
            bad(key, t("取值不得大于 {hi}（当前 {v}）", hi=hi, v=value))

    # 2. list-valued flags. A non-editable combo already restricts the choices
    #    in the UI, but a preset imported from the other engine shares key names
    #    (spec_type, chat_*) whose values are not in this set — `unsupported
    #    --spec-type` is a parse-time exit, so it is checked here as well.
    #    combo_edit is skipped on purpose: an editable combo's items are
    #    suggestions, and --reasoning-effort is the case that proves it (the
    #    parser takes any string and the model template decides, so a custom
    #    template's own level must reach the command line).
    for param in UI_PARAMS:
        if not param.items or param.widget != "combo":
            continue
        value = get(param.key)
        if value is None or value == "":
            # "" is a real choice when it is one of the items (cache types:
            # "follow --kv-dtype") or when the emit mode skips it
            # (--reasoning-effort: empty = template default, nothing sent).
            if "" in param.items or param.emit == "diff_nonempty":
                continue
            bad(param.key, t("不能留空，只能是 {items} 之一", items=" / ".join(param.items)))
            continue
        if value not in param.items:
            bad(param.key, t("取值只能是 {items} 之一（当前 {v}）",
                             items=" / ".join(param.items), v=value))

    # 3. KV cache pairing (§5.3). Effective K = -ctk when set, else --kv-dtype;
    #    same for V. A quantized K with any other V is fatal, so the pair is
    #    checked rather than each side alone.
    kv_dtype = get("kv_dtype") or ""
    eff_k = get("cache_type_k") or kv_dtype
    eff_v = get("cache_type_v") or kv_dtype
    if eff_k in QUANT_CACHE_TYPES and eff_v != eff_k:
        bad("cache_type_v", t("K 缓存为量化类型 {k} 时 V 必须与之相同（当前 {v}）",
                              k=eff_k, v=eff_v or t("（空）")))

    # 4. thinking tri-state (§5.4): --reasoning-effort none switches thinking
    #    off by itself, so pairing it with "开启思考" is contradictory (and the
    #    flags are independent, so nothing later resolves it).
    if get("thinking_mode") == THINKING_ON and get("reasoning_effort") == "none":
        bad("reasoning_effort", t("none 会关闭思考，与「开启思考」互斥"))
    if get("enable_thinking") and get("no_think"):
        bad("thinking_mode", t("不能同时发送 --enable-thinking 与 --no-think"))

    # 5. chat template (§5.5): exclusive pair, and the kwargs flag must be a
    #    JSON object (parse-time rejection, measured).
    if get("chat_template") and get("chat_template_file"):
        bad("chat_template", t("与模板文件互斥，只能填一个"))
    kwargs_text = get("chat_template_kwargs") or ""
    if kwargs_text.strip():
        try:
            parsed = json.loads(kwargs_text)
        except (ValueError, TypeError):
            parsed = None
        if not isinstance(parsed, dict):
            # No kwargs: t() leaves literal braces alone (C4).
            bad("chat_template_kwargs", t("必须是 JSON object，例如 {\"key\": \"value\"}"))

    # 6. template file must exist (the engine checks it at parse time); only an
    #    absolute path is judged, since a relative one resolves against the
    #    server's working directory, not ours.
    template_file = get("chat_template_file") or ""
    if template_file and Path(template_file).is_absolute() \
            and not Path(template_file).is_file():
        bad("chat_template_file", t("文件不存在"))

    return problems
