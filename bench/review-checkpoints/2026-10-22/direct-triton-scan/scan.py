#!/usr/bin/env python3
"""Replay this scoped source-organization scan, including reviewed source routes.

This is a lexical drift check plus explicit human route decisions, not Lean
elaboration, a correctness audit, or automatic reachability analysis.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "bench"))
from check_main_entry_roles import without_comments

BENCH = "bench/tritonbench_g/"
REPAIR = "bench/repairs/tritonbench_g/"

# Only actual source selections are listed here. An import closure also contains
# legacy programs and unrelated shared proof helpers; those are not source uses.
ROUTES = {
    "attention_score": [REPAIR + "attention_score_tile_checked/SourceDefinitions.lean"],
    "batched_vecmat_mult": [BENCH + "batched_vecmat_mult/Kernels.lean"],
    "chunk_gated_attention": [REPAIR + "chunk_gated_attention/Kernels.lean"],
    "chunk_retention": [REPAIR + "chunk_retention/Whole001/Kernels.lean"],
    "chunk_retention_ops": [REPAIR + "chunk_retention_ops/Kernels.lean"],
    "chunked_cumsum_fwd": [],  # Ledger reference corrected against accepted delivery.
    "destindex_copy": [REPAIR + "destindex_copy_kv1/Kernels.lean"],
    "f8_conversion_utils": [BENCH + "f8_conversion_utils/Kernels.lean"],
    "fp4_to_bf16": [BENCH + "fp4_to_bf16/Kernels.lean"],
    "int8_matmul_quantization": [REPAIR + "int8_matmul_quantization_i32_safe/Source.lean"],
    "ksoftmax_triton": [REPAIR + "ksoftmax_triton/Kernels.lean"],
    "matmul_leakyrelu": [BENCH + "matmul_leakyrelu/Proofs/FPTypedSource.lean",
                         BENCH + "matmul_leakyrelu/Proofs/FPFusedSource.lean"],
    "max_reduction": [BENCH + "max_reduction/Kernels.lean"],
    "parallel_attention": [REPAIR + "parallel_attention/KernelOnly/" + f + ".lean"
                           for f in ("SourceSyntax", "QuerySyntax", "CandidateSyntax", "ShellSyntax")],
    "parallel_retention_attention": [REPAIR + "parallel_retention_attention/" + f + ".lean"
                                     for f in ("SourceSyntax", "QuerySyntax", "CandidateSyntax", "ShellSyntax")],
    "rotary_transform_ops": [REPAIR + "rotary_transform_ops/" + f + ".lean"
                             for f in ("KernelSource", "XorSource")],
    "sgmv_expand_slice": [REPAIR + "sgmv_expand_slice/Kernels.lean",
                          REPAIR + "sgmv_expand_slice/ByteSeparated/Candidate.lean"],
    "spinning_lock_reduction": [],  # Complete literal source in its local reader.
    "streamk_matmul": [],           # Complete helper/stage literals, same distinction.
    "triton_attention": [REPAIR + "triton_attention_scalar_abi_checked/" + f + ".lean"
                          for f in ("SourceKernels", "CandidateKernels")],
    "triton_linear_activation": [REPAIR + "triton_linear_activation/" + f + ".lean"
                                  for f in ("SourceKernels", "CandidateKernels")],
    "triton_matmul": [REPAIR + "triton_matmul/Kernels.lean",
                       REPAIR + "triton_matmul/KernelOnly/CandidateKernels.lean"],
}

LABELS = {
    "outer_kernel_reconstruction": "外层重建",
    "source_in_proofs": "源码位于 Proofs",
    "proof_decomposition_in_source": "分解助手归位",
}


def source(path):
    p = ROOT / path
    raw = p.read_text()
    return raw, without_comments(raw)


def site(path, offset):
    raw, masked = source(path)
    line = masked.count("\n", 0, offset) + 1
    return {"path": path, "line": line, "excerpt": raw.splitlines()[line - 1].strip()}


def find_site(path, pattern):
    match = re.search(pattern, source(path)[1], re.M)
    assert match, (path, pattern)
    return site(path, match.start())


def wrappers(path):
    masked = source(path)[1]
    ends = [m.start() for m in re.finditer(
        r"(?m)^\s*(?:(?:private|noncomputable|protected)\s+)*"
        r"(?:def|abbrev|theorem|lemma|instance|structure|inductive)\s", masked)]
    result = []
    for match in re.finditer(r"(?:\bComputeKernel\.mk|(?<![\w.])\.mk)\s*(?=\[)", masked):
        end = next((n for n in ends if n > match.start()), len(masked))
        body = masked[match.start():end]
        if ".surfaceBody" not in body:
            continue
        # qkv/qkvIndexCommuted are genuine three-stage source composition.
        if re.search(r"\.surfaceBody\s*\+\+", body):
            continue
        result.append(site(path, match.start()))
    return result


def scan():
    ledger = json.loads((ROOT / "bench/tritonbench_g/task_progress.json").read_text())
    rows = []
    files = set()
    for task in ledger["tasks"]:
        name = task["task"]
        selected = task["kernel_revision"]["kernel_source"]
        paths = list(dict.fromkeys([selected, *ROUTES.get(name, [])]))
        files.update(paths)
        direct = wrappers(selected)
        shared = [s for p in paths[1:] for s in wrappers(p)]
        findings = []
        if direct or shared:
            findings.append({"category": "outer_kernel_reconstruction",
                             "route": "selected_entry" if direct else "shared_source",
                             "sites": direct + shared})
        if name == "matmul_leakyrelu":
            findings.append({"category": "source_in_proofs", "sites": [
                find_site(p, r"^def typed\b") for p in paths[1:]]})
        if name in ("bmm_optimized", "matmul_kernel", "triton_matmul"):
            p = REPAIR + "triton_matmul/Kernels.lean" if name == "triton_matmul" else selected
            findings.append({"category": "proof_decomposition_in_source", "sites": [
                find_site(p, r"^def prelude\b")]})
        note = ""
        if name in ("spinning_lock_reduction", "streamk_matmul"):
            note = ("完整源码使用题级 reader 和 List Statement；并非隐藏整段源码的生成宏。"
                    "本轮未把它当作外层重建或判定该接口已经统一为标准 ComputeKernel DSL。")
        if name == "rms_matmul_rbe":
            note = "单 kernel 包装计入；qkv 和 qkvIndexCommuted 的真实三阶段组合排除。"
        if name == "chunk_gated_attention":
            note = "当前使用 repair/NativeSource；未把历史 TypedSource 的整段生成宏计入。"
        if name == "chunked_cumsum_fwd":
            note = ("已验收的题根 Kernels 包含两个完整普通源码；本轮纠正清单仍指向旧缓存的入口引用。"
                    "历史源码和完整相等证据保留，不把旧缓存中的前端重新算作当前问题。")
        if name in ("f8_conversion_utils", "max_reduction"):
            note = "宏仅为普通完整源码的兼容转发；包装计入，转发宏不记为整段源码生成。"
        rows.append({"task": name, "selected_kernel": selected,
                     "reviewed_source_files": paths, "findings": findings, "note": note,
                     "scan_status": "cleanup_pending" if findings else "no_new_finding"})
    categories = Counter(f["category"] for r in rows for f in r["findings"])
    routes = Counter(f["route"] for r in rows for f in r["findings"] if "route" in f)
    return {"schema": 1, "date": "2026-10-22", "scope": "selected_source_construction_and_reviewed_delegates",
            "method": "Mask comments/strings, inspect all selected entries, manually trace delegated source/candidates; lexical checks do not infer semantic reachability.",
            "task_count": len(rows), "tasks_with_findings": sum(bool(r["findings"]) for r in rows),
            "category_task_counts_nonexclusive": dict(categories), "wrapper_task_routes": dict(routes),
            "proof_completion_counts_unchanged": dict(Counter(t["kernel_revision"]["status"] for t in ledger["tasks"])),
            "kernel_files_changed": 0, "new_numerical_assumptions": 0,
            "lean_build_run": False, "independent_full_source_replay": False, "gpu_execution": False,
            "source_entry_reference_corrections": [t["kernel_revision"]["source_entry_reference_correction"]
                for t in ledger["tasks"] if "source_entry_reference_correction" in t["kernel_revision"]],
            "reviewed_file_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sorted(files)},
            "tasks": rows}


def link(s):
    target = os.path.relpath(ROOT / s["path"], HERE)
    return f'[{Path(s["path"]).name}:{s["line"]}]({target}#L{s["line"]})'


def report(audit):
    text = """# 183 题标准 Triton 源码入口复查

本轮检查唯一清单中全部 183 个 `kernel_revision.kernel_source`，并追踪转发入口实际选择的共享原程序及候选。保留实际选定的缓存版本；发现清单引用过期时，以已验收的源码、完整保持性证据和构建记录校正，不仅凭文件路径推断实际版本。注释、字符串中的 Python AST、未选用的历史宏及导入中的其他程序不计为当前源码。

确认 **89 题**存在 `ComputeKernel.mk inputs outputs (...).surfaceBody` 一类外层重建：**80 题在选定入口，9 题在实际共享源码**。此外，**1 题**的实际源码仍位于 `Proofs/`；**3 题**在源码文件保留了可归入 `Proofs/` 的程序分解助手。类别重叠后共 **92 题有本轮确认的整理项，91 题未发现本轮所查模式**。

此前“源码整理全部完成”的判断不准确，已经在唯一逐题清单中重新登记待整理；旧修复证据保留。本轮没有修改 kernel、证明、Python 输入契约或数值规则。整题证明计数保持 **182 已完成、1 带数值延期完成**；源码呈现问题本身不撤回已成立的证明。91 题未发现这些模式，也不等于所有源码和文件职责已获全面认证。

## 具体分类

- 外层直接包装，例如 `chunk_cumsum_kernel`、`kldiv_compute`：完整 `triton` 被取出程序体，再手工填端口。
- 模板实例重新包装，例如 `l2_norm_bwd`、`l2_norm_triton1`、`int_scaled_matmul`：调用已完整的 kernel 后，又用 `.mk` 重填端口。
- 共享源码中的包装，例如 `attention_score`、`chunk_retention`、`destindex_copy`：入口转发本身合理，整理实际共享定义即可，不复制源码。
- `matmul_leakyrelu`：原程序和已有候选分别经 `Whole.kernel`、`Whole.contractedKernel` 引用 `Proofs/FPTypedSource`、`Proofs/FPFusedSource` 的实际完整源码，需调整位置。
- `bmm_optimized`、`matmul_kernel`、`triton_matmul`：完整源码已经可读；`prelude/body/epilogue` 等分解及等式属于证明辅助。这里的切片并未用于偷换权威源码，不记作程序错误。

`chunked_cumsum_fwd` 的初筛曾命中旧缓存中位于 `Proofs/KernelSource` 的题级前端。继续核对已验收 delivery 和 combined-minimum generation 后，确认它早已移到题根 `Kernels.lean`，有 `source = previousSource` 的完整相等检查。本轮只纠正清单过期的 `kernel_source` 引用与对应修复摘要哈希，保留旧缓存作为历史证据；不重复报为源码未修复。依据为 `.veritile-cache/finish-pending-105-20261021/chunked_cumsum_fwd/{delivery,root-acceptance}.json` 和 `combined-minimum-welford-20261022/SourcePreservation.lean` 的既有通过记录。

## 后续整理的边界

外层包装目前承载输入/输出端口，不能批量删括号后只看编译结果。先核对共享 Metadata 对指针别名、外部指针重绑定、条件分支和端口次序的处理，再逐题证明新旧**完整 ComputeKernel 相等**，或提供明确语义连接，然后针对性构建主入口。上一轮 `matrix_transpose` 的完整相等证据保留，该题本轮无此包装。

真实多阶段组合（如 `rms_matmul_rbe.qkv`）、透明共享源码和兼容转发宏保留。`spinning_lock_reduction`、`streamk_matmul` 的完整源码仍采用题级 reader/`List Statement`；这与隐藏源码生成宏不同，本轮单列说明，既不算外层包装，也不宣称其接口已统一为标准 `ComputeKernel := triton`。不因本次扫描新增硬件、host、dtype 或数值实验任务。

本轮仅做源码结构检查和登记，没有运行 Lean 构建、全源码独立回放或 GPU 实测。机器可读的位置、源码哈希与逐题结论在 [audit.json](audit.json)；`python bench/review-checkpoints/2026-10-22/direct-triton-scan/scan.py --check` 可复查记录是否漂移。该脚本使用明确审阅过的转发路线，不是自动语义分类器。

## 逐题结果

| Task | 本轮发现 | 具体位置 |
| --- | --- | --- |
"""
    for row in audit["tasks"]:
        labels = "、".join(LABELS[f["category"]] for f in row["findings"]) or "未发现本轮模式"
        sites = [s for f in row["findings"] for s in f["sites"]]
        sites = list({(s["path"], s["line"]): s for s in sites}.values())
        if not sites:
            sites = [{"path": row["selected_kernel"], "line": 1}]
        locations = "；".join(link(s) for s in sites[:4])
        if len(sites) > 4:
            locations += f"；其余 {len(sites) - 4} 处见 audit.json"
        text += f'| `{row["task"]}` | {labels} | {locations} |\n'
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    audit = scan()
    outputs = {HERE / "audit.json": json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
               HERE / "README.md": report(audit)}
    for path, content in outputs.items():
        if args.check:
            assert path.read_text() == content, f"Scan record changed: {path}"
        else:
            path.write_text(content)
    print(json.dumps({k: audit[k] for k in ("task_count", "tasks_with_findings",
          "category_task_counts_nonexclusive", "wrapper_task_routes", "proof_completion_counts_unchanged")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
