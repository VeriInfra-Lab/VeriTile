#!/usr/bin/env python3
"""Record this scoped cleanup; do not turn a lexical scan into a proof audit."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CAMPAIGN = ROOT / '.veritile-cache/direct-triton-fix-20261022'
SCAN = HERE.parent / 'direct-triton-scan'
spec = importlib.util.spec_from_file_location('prior_source_scan', SCAN / 'scan.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path):
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha(path)}


EXTRAS = {
    'rms_norm_triton': ['bench/tritonbench_g/rms_norm_triton/Kernels.lean'],
    'triton_argmax': ['bench/repairs/tritonbench_g/triton_argmax_nan_first/CandidateKernels.lean'],
    'attn_fwd_triton': ['bench/repairs/tritonbench_g/attn_fwd_triton/Kernels.lean'],
    'matmul_leakyrelu': [
        'bench/tritonbench_g/matmul_leakyrelu/SourceDefinitions.lean',
        'bench/tritonbench_g/matmul_leakyrelu/CandidateSourceDefinitions.lean'],
}


def collect():
    initial = read(CAMPAIGN / 'initial-audit.json')
    index_path = CAMPAIGN / 'queue16/task-evidence-index.json'
    index = read(index_path)
    assert index['summary']['targeted_preservation_passed'] == 92, 'Independent review incomplete'
    assert index['summary']['targeted_build_passed'] == 92, 'Independent build review incomplete'
    independent = read(CAMPAIGN / 'queue16-result.json')
    assert independent['status'].startswith('passed'), 'Independent review is not final/passed'
    assert read(CAMPAIGN / 'source-union-result.json')['status'] == 'passed'
    assert read(CAMPAIGN / 'shared-result.json')['status'] == 'passed'
    by_task = {t['task']: t for t in index['tasks']}
    before = read(CAMPAIGN / 'task_progress.before.json')
    current = read(ROOT / 'bench/tritonbench_g/task_progress.json')
    expected = {t['task']: t['kernel_revision']['status'] for t in before['tasks']}
    assert expected == {t['task']: t['kernel_revision']['status'] for t in current['tasks']}
    rows, files = [], set()
    for original in initial['tasks']:
        name = original['task']
        paths = list(dict.fromkeys(original['reviewed_source_files'] + EXTRAS.get(name, [])))
        files.update(paths)
        remaining = [{'category': 'outer_kernel_reconstruction', 'site': s}
                     for path in paths for s in prior.wrappers(path)]
        if name == 'matmul_leakyrelu':
            for path in original['reviewed_source_files']:
                if '/Proofs/' in path and 'triton {' in prior.source(path)[1]:
                    remaining.append({'category': 'source_in_proofs', 'path': path})
        if name in ('bmm_optimized', 'matmul_kernel', 'triton_matmul'):
            import re
            for path in paths:
                if '/Proofs/' not in path and re.search(r'(?m)^def prelude\b', prior.source(path)[1]):
                    remaining.append({'category': 'proof_decomposition_in_source', 'path': path})
        changed = bool(original['findings'])
        evidence = by_task.get(name)
        if changed:
            assert evidence, name
            assert evidence['source_preservation_status'] == 'passed', (name, evidence['source_preservation_status'])
            assert evidence['build_status'] == 'passed', (name, evidence['build_status'])
            assert evidence['whole_program_coverage_reviewed'], name
        rows.append({'task': name, 'changed_in_this_pass': changed,
                     'reviewed_source_files': paths, 'remaining_findings': remaining,
                     'validation': evidence if changed else None,
                     'retained_note': original['note']})
    assert not any(t['remaining_findings'] for t in rows), 'Unresolved source findings'
    changed = [t for t in rows if t['changed_in_this_pass']]
    assert len(rows) == 183 and len(changed) == 92
    limited = [t['task'] for t in changed if t['validation']['fresh_main_status'] != 'passed']
    assert limited == ['chunk_gla_fwd', 'int_scaled_matmul'], ('Unreviewed fresh check failure/pending', limited)
    assert by_task['chunk_gla_fwd']['fresh_main_status'] == 'passed_with_audit_omission'
    return {
        'schema': 1, 'date': '2026-10-22',
        'scope': 'Selected source construction and previously reviewed delegates; complete program preservation for the 92 targeted tasks. No new theorem or input/numerical domain requirements.',
        'task_count': 183, 'fixed_in_this_pass': 92, 'remaining_targeted_source_findings': 0,
        'whole_selected_program_preservation_passed': 92, 'targeted_source_build_passed': 92,
        'fresh_main_passed_tasks': 92 - len(limited), 'fresh_main_limited_tasks': limited,
        'fresh_main_declarations_only_tasks': ['chunk_gla_fwd'],
        'fresh_main_existing_abi_limited_tasks': ['int_scaled_matmul'],
        'proof_completion_counts_unchanged': dict(Counter(expected.values())),
        'primary_misc_dependencies': 0,
        'primary_misc_evidence_reused': 'bench/review-checkpoints/2026-10-22/io-consolidation/reader-roles.json',
        'new_numerical_assumptions': 0, 'numerical_experiments_run': 0,
        'independent_full_source_replay': False, 'gpu_execution': False,
        'before_scan': ref(SCAN / 'audit.json'), 'shared_validation': ref(CAMPAIGN / 'shared-result.json'),
        'selected_source_import_union': ref(CAMPAIGN / 'source-union-result.json'),
        'independent_evidence_index': ref(index_path),
        'independent_review': ref(CAMPAIGN / 'queue16-result.json'),
        'source_sha256': {p: sha(ROOT / p) for p in sorted(files)}, 'tasks': rows,
    }


def report(audit):
    text = f'''# 标准 Triton 源码入口修复

本轮处理前次复查登记的 **92 题**：移除单个 kernel 外层 `.mk ... (...).surfaceBody` 重建，将 `matmul_leakyrelu` 的实际原程序和候选移出 `Proofs/`，将三题的程序分解助手归入 `Proofs/`。追踪实际调用时补齐 RMS、argmax 候选和 attn_fwd 的共享源码。复查全部 183 个选定入口及已确认的共享路线，**本轮所查的剩余源码组织项为 0**。

92 题的完整选定程序保持性证据和针对性源码构建通过；保真覆盖端口、全部语句、参数、精度、分支和实际原程序／候选。对被外层接口覆盖的内部模板，端口整理与完整选定程序相等分别记录，不把模板 body 相同冒充所有 metadata 相同。复用了适用的既有证明对象，没有逐题重证底层浮点数或 host。

**{audit['fresh_main_passed_tasks']} 题**完成当前 `Correct`／`FPEquiv` 两个主文件的 fresh 检查。另两题分别记录：

- `chunk_gla_fwd`：`Correct` 原文件检查通过；`FPEquiv` 的五个实际主声明、别名、假设及证明逐字保留的源码夹具检查通过，仅省略末尾审计命令。原文件完整检查在附带审计耗时过长后主动停止；不声称这套审计完成，既有审计证据按原适用范围保留。
- `int_scaled_matmul`：`Correct` fresh 检查、完整源码保持性和源码构建通过；`FPEquiv` fresh 检查受旧整数点积助手的 `Algebra.minimum` ABI 不匹配限制。适用的既有主证明证据及本次检查限制分别列出，不宣称该 fresh `FPEquiv` 检查通过。

92 个选定源码模块的一次联合导入检查通过；它不等于所有主证明模块的联合导入。整题正确性计数保持 **182 已完成、1 带数值延期完成**，延期题仍为 `matmul_leakyrelu`。本轮没有改变 Python 输入契约、数学目标、主定理陈述或数值规则，没有新增／执行数值实验，没有进行全库独立源码回放或 GPU 实测。既有跨题整体导入兼容性状态单独保留。

## 共享源码接口

优先使用普通完整 `triton {{ ... }}`。自动推断无法保留原端口顺序、重复端口或可选／未用端口时，在源码顶部写 `tl.ports inputs=[...], outputs=[...]`。这是共享 DSL 的有序接口元数据，不发出指令，不改变读写、运算精度或输入安全前提。`inputs`／`outputs` 使用非保留语法，声明内容不参加程序体精度推断。

共享端口、旧指针别名推断、运行标量、精度四项回归均在最终构建对象上检查通过。旧缓存解释器的私有特化冲突通过在新隔离目录刷新直接 DSL 调用者解决，未修改旧验收缓存。日志及初次失败记录保留。

## 范围与证据

真实多阶段组合、透明共享入口和合理兼容转发保留。`spinning_lock_reduction`、`streamk_matmul` 的明确题级 reader／`List Statement` 接口仍按前次扫描单列；本报告不声称它们已经统一为 `ComputeKernel := triton`。本轮零剩余仅指已明确检查的源码构造模式，不是所有 183 题、全部历史文件或硬件实现的无条件认证。

- [修复前扫描](../direct-triton-scan/README.md)保持原文，作为历史证据。
- [audit.json](audit.json)记录当前源码哈希、逐题完整程序证据和构建边界。
- `python bench/review-checkpoints/2026-10-22/direct-triton-fix/review.py --check` 检查本轮记录是否漂移；它不替代 Lean 检查或自动推断语义可达性。

## 本轮 92 题

| Task | 完整程序保持／源码构建 | 当前主源码检查 | 逐题证据 |
| --- | --- | --- | --- |
'''
    import os
    for task in audit['tasks']:
        if not task['changed_in_this_pass']:
            continue
        ev = task['validation']
        path = ROOT / ev['owner_results'][0]['path']
        link = os.path.relpath(path, HERE)
        main = ('通过' if ev['fresh_main_status'] == 'passed' else
                '主声明通过；附带审计未完成' if task['task'] == 'chunk_gla_fwd' else
                '既有代际兼容限制；复用证据另列')
        text += f"| `{task['task']}` | 通过 | {main} | [证据]({link}) |\n"
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    audit = collect()
    outputs = {HERE / 'audit.json': json.dumps(audit, ensure_ascii=False, indent=2) + '\n',
               HERE / 'README.md': report(audit)}
    for path, content in outputs.items():
        if args.check:
            assert path.read_text() == content, f'Review record changed: {path}'
        else:
            path.write_text(content)
    print(json.dumps({k: audit[k] for k in ('task_count', 'fixed_in_this_pass',
          'remaining_targeted_source_findings', 'fresh_main_passed_tasks', 'fresh_main_limited_tasks')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
