# -*- coding: utf-8 -*-
"""Turn a kvmem server's own error output into "which parameter, where to fix it".

Why this exists instead of "just read the log": the kvmem server is a
hand-written argv parser that dies in the *first* thing it does. When it prints
``unknown flag: --parallel`` it exits 1 before a single model byte is loaded, so
the log panel shows two lines and the window says "服务异常退出" — which reads
like a crash rather than like "you ticked something this binary has never
heard of". Every string matched below was read out of
``F:\\llama-kvmem-rc2\\bin\\llama-kvmem-server.exe`` (v0.16.0-rc2), i.e. these
are the messages that build can actually emit, not guesses:

    unknown flag: %s                       missing value for %s
    incompatible KV cache types: K=%s, V=%s; quantized K/V must match; ...
    unsupported cache type (want f16|f32|q8_0|q5_0|q4_0)
    unsupported MTP cache type (want f16|q8_0|q5_0|q4_0|f32)
    unsupported --spec-type %s (P7-0: draft-mtp|none)
    invalid --kvmem-recent-tokens (want >= 0)
    invalid --kvmem-query-max-tokens (want > 0)
    invalid --reasoning-budget: %s         must be between -1 and 2147483647
    --chat-template-kwargs requires a JSON object
    --image-m[ai]n-tokens requires a positive integer
    cannot read chat template: %s          NVMe offload is disabled in this build
    failed to load model                   failed to load mmproj:
    cannot enforce reasoning_budget_tokens

``classify`` is display-only and never raises: a mis-parsed log must not become
a second failure path on top of the one it is explaining.
"""
import re

from . import params_schema as llama_schema
from . import kvmem_params_schema as kvmem_schema
from .i18n import t

#: Same string as `core.engine.KVMEM.display_name`, spelled out because this
#: module must not import the registry (which pulls in defaults + identity +
#: both schemas). tests/test_kvmem_errors.py pins the two against each other.
ENGINE_LABEL = "kvmem-llama.cpp"

#: `--flag` -> schema key, for both engines. kvmem's own table resolves a
#: rejected flag into the control to fix (an alias like `--temp` maps to
#: `temperature`); the llama.cpp table is only consulted to say *why* an
#: unknown flag is unknown — "that one is llama.cpp's".
_KVMEM_FLAGS = {f: p.key for p in kvmem_schema.PARAMS
                for f in ((p.flags or ()) + ((p.flag,) if isinstance(p.flag, str) else ()))
                if f}
_LLAMA_FLAGS = {f for p in llama_schema.PARAMS
                for f in ((p.flags or ()) + ((p.flag,) if isinstance(p.flag, str) else ()))
                if f}


def flag_to_key(flag: str) -> str:
    """Schema key this CLI flag belongs to ("" when this engine has no such flag)."""
    return _KVMEM_FLAGS.get((flag or "").strip(), "")


def _key_for_flag_in_line(line: str):
    """(param key, flag) of the first flag mentioned on the line."""
    for flag in re.findall(r"--?[A-Za-z][\w-]*", line):
        key = flag_to_key(flag)
        if key:
            return key, flag
    return "", ""


def _llama_only(flag: str) -> str:
    """Hint for the most common mistake in a dual-engine launcher."""
    if flag and flag.strip() in _LLAMA_FLAGS:
        return t("这是 llama.cpp 服务器的参数，{engine} 没有对应功能。",
                 engine=ENGINE_LABEL)
    return ""


#: (pattern, param key or "" to resolve from the line, message(text) -> str)
#: Order matters: `unsupported MTP cache type` must be tested before the plain
#: `unsupported cache type`, which its wording also contains.
_RULES = (
    # A flag the parser has never heard of can only have arrived through the
    # free-text box — every schema param emits a flag this module knows — so an
    # unresolved one jumps there rather than nowhere. (Version drift can also
    # produce this line for a schema flag; `line` still names the real culprit.)
    (re.compile(r"unknown flag:\s*(\S+)"),
     lambda m, line: flag_to_key(m.group(1)) or "extra_args",
     lambda m, line: t("服务器不认识参数 {flag}。{hint}",
                       flag=m.group(1),
                       hint=_llama_only(m.group(1)))),
    (re.compile(r"missing value for\s+(\S+)"),
     lambda m, line: flag_to_key(m.group(1)),
     lambda m, line: t("参数 {flag} 后面缺少取值。", flag=m.group(1))),
    (re.compile(r"incompatible KV cache types:\s*K=(\S+?),\s*V=(\S+)"),
     lambda m, line: "cache_type_v",
     lambda m, line: t("量化 K 必须配同值的 V：K={k} 时 V 也得是 {k}"
                       "（把 -ctv 改成同一种类型，或两边都留空跟随 --kv-dtype）。",
                       k=m.group(1).rstrip(';.'), v=m.group(2).rstrip(';.'))),
    (re.compile(r"unsupported MTP cache type"),
     lambda m, line: "spec_kv_dtype",
     lambda m, line: t("--spec-kv-dtype 只接受 f16 / q8_0 / q5_0 / q4_0 / f32。")),
    (re.compile(r"unsupported cache type"),
     lambda m, line: "cache_type_k",
     lambda m, line: t("KV 缓存类型只接受 f16 / f32 / q8_0 / q5_0 / q4_0（没有 bf16）。")),
    (re.compile(r"unsupported --spec-type\s+(\S+)"),
     lambda m, line: "spec_type",
     lambda m, line: t("--spec-type 只接受 none 或 draft-mtp，当前是 {v}。"
                       "（draft-mtp 还需要模型自带 nextn/MTP 头）",
                       v=m.group(1).rstrip(';.'))),
    (re.compile(r"invalid --kvmem-recent-tokens"),
     lambda m, line: "kvmem_recent_tokens",
     lambda m, line: t("--kvmem-recent-tokens 必须 >= 0。")),
    (re.compile(r"invalid --kvmem-query-max-tokens"),
     lambda m, line: "kvmem_query_max_tokens",
     lambda m, line: t("--kvmem-query-max-tokens 必须 > 0。")),
    (re.compile(r"invalid --reasoning-budget|reasoning_budget_tokens must be between"),
     lambda m, line: "reasoning_budget",
     lambda m, line: t("--reasoning-budget 必须是 -1 到 2147483647 之间的整数。")),
    (re.compile(r"cannot enforce reasoning_budget_tokens"),
     lambda m, line: "reasoning_budget",
     lambda m, line: t("这个模型的模板里没有思考结束标记，无法强制思考预算，"
                       "请把 --reasoning-budget 设回 -1。")),
    (re.compile(r"requires a JSON object"),
     lambda m, line: _key_for_flag_in_line(line)[0],
     lambda m, line: t("这里必须是 JSON object，例如 {\"key\": \"value\"}。")),
    (re.compile(r"requires a positive integer"),
     lambda m, line: _key_for_flag_in_line(line)[0],
     lambda m, line: t("该参数只接受正整数，0 和负数都会被拒绝。")),
    (re.compile(r"cannot read chat template:\s*(.*)"),
     lambda m, line: "chat_template_file",
     lambda m, line: t("读不到聊天模板文件：{p}", p=m.group(1).strip() or t("（无路径）"))),
    (re.compile(r"invalid chat template"),
     lambda m, line: "chat_template",
     lambda m, line: t("聊天模板无法编译（Jinja 语法或变量与模型不匹配）。")),
    (re.compile(r"NVMe offload is disabled in this build"),
     lambda m, line: "",
     lambda m, line: t("本构建在编译时关闭了 NVMe 卸载，任何 --kvmem-nvme-* 都是硬错误。")),
    (re.compile(r"failed to load mmproj[: ]*(.*)"),
     lambda m, line: "mmproj",
     lambda m, line: t("视觉投影模型加载失败：{p}", p=m.group(1).strip() or t("请检查文件是否为 mmproj GGUF。"))),
    (re.compile(r"failed to load model|missing target model/context"),
     lambda m, line: "model",
     lambda m, line: t("模型加载失败：路径不存在、不是本引擎支持的单文件 GGUF，"
                       "或者显存/内存不够。")),
)


def classify(text: str) -> list:
    """Match a server output blob against the rules above.

    Returns [{"param": schema key or "", "message": translated explanation,
    "line": the raw line}] in the order the lines appear, deduplicated by
    (param, message). Empty list means "nothing recognised" — the caller shows
    its normal error path then, which is what llama.cpp has always done.
    """
    out, seen = [], set()
    if not text:
        return out
    try:
        for raw in str(text).splitlines():
            line = raw.strip()
            if not line:
                continue
            for pattern, key_of, message_of in _RULES:
                m = pattern.search(line)
                if not m:
                    continue
                try:
                    key = key_of(m, line) or ""
                    msg = message_of(m, line)
                except Exception:
                    continue
                if (key, msg) in seen:
                    continue
                seen.add((key, msg))
                out.append({"param": key, "message": msg, "line": line[:400]})
                break      # one explanation per line, most specific rule first
    except Exception:
        return out
    return out
