#!/usr/bin/env python3
"""Keep accepted input/configuration scope separate from numerical assumptions.

This checks an explicit, source-bound review. It does not infer numerical laws
from coverage labels, accept experimental results, or discharge Lean premises.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'bench'))
from spec_sheet import split_decls, split_statement

REVIEW = ROOT / 'bench/tritonbench_g/numerical_obligations_review.json'
REPORT = ROOT / 'bench/tritonbench_g/numerical_obligations.json'
MARKDOWN = ROOT / 'bench/tritonbench_g/numerical_obligations.md'
EXPERIMENTS = ROOT / 'bench/tritonbench_g/numeric_experiment_tasks.json'
STATUS_ORDER = ('no_extra_numeric_law', 'existing_numeric_evidence',
                'awaiting_numeric_validation', 'requires_numeric_model_revision')
STATUS_LABELS = {
    'no_extra_numeric_law': '未使用额外数值律；不代表无需实验',
    'existing_numeric_evidence': '使用已有实验报告中的数值律',
    'awaiting_numeric_validation': '需补数值实验或精确绑定证据',
    'requires_numeric_model_revision': '含不可直接推广到真实浮点的理想化假设',
    'missing_source': '缺源码',
}
EVIDENCE_STATUSES = {'reported_evidence_available', 'binding_validation_pending',
                     'experiment_pending', 'counterexample_to_unrestricted_form',
                     'shared_model_contract'}
KINDS = {'reported_atom', 'explicit_numeric_law', 'shared_model_requirement'}
FP_MODEL_KINDS = {
    'mathematical_contract': 'mathematical_value_or_io',
    'rounding_model_contract': 'rounding_model_value_or_io',
    'backend_equivalence': 'candidate_vs_original',
    'backend_value_contract': 'backend_per_operation_value_or_io',
    'bit_interpreter_contract': 'bit_interpreter_value_or_io',
}
FP_ADEQUACY_LABELS = {
    'math_contract_only': '仅数学模型契约：缺浮点/固定宽语义证明',
    'rounding_contract_only': '仅cast/store舍入模型：缺算术浮点证明',
    'requires_numeric_model_revision': '存在数值反例：需修订模型或适用域',
    'backend_equivalence_no_extra_law': '有后端参数化的程序相对等价证明',
    'backend_equivalence_existing_reports': '有依赖已有报告前提的程序相对等价证明',
    'backend_equivalence_needs_validation': '有条件相对等价证明：精确数值绑定待补',
    'bit_interpreter_contract_only': '含参数化位解释器值契约：逐阶段范围及硬件对应另列',
    'backend_value_contract_only': '有逐算子后端值契约：原语及硬件对应仍待完成',
    'missing_source': '缺源码',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def statement_map(rows, root=ROOT):
    cache, result = {}, {}
    for row in rows:
        file = row['file']
        if file not in cache:
            cache[file] = {d['name']: d for d in split_decls((root / file).read_text())}
        try:
            statement, _ = split_statement(cache[file][row['declaration']]['text'])
        except KeyError as e:
            raise ValueError(f'Unresolved numerical-review declaration: {file}::{row["declaration"]}') from e
        result[file, row['declaration']] = statement
    return result


def numerical_status(ids, obligations):
    statuses = {obligations[k]['evidence_status'] for k in ids}
    if 'counterexample_to_unrestricted_form' in statuses:
        return 'requires_numeric_model_revision'
    if statuses & {'binding_validation_pending', 'experiment_pending'}:
        return 'awaiting_numeric_validation'
    if statuses:
        return 'existing_numeric_evidence'
    return 'no_extra_numeric_law'


def aggregate_status(statuses):
    return max(statuses, key=STATUS_ORDER.index, default=STATUS_ORDER[0])


def validate_review(data, rows, root=ROOT):
    if data.get('schema') != 1 or data.get('policy') != {
        'normal_input_conditions': 'accepted',
        'configuration_and_stage_scope': 'accepted',
    }:
        raise ValueError('Invalid numerical-review schema or user scope policy')
    fingerprints = data.get('input_sha256', {})
    if not fingerprints:
        raise ValueError('Numerical review has no bound inputs')
    for name, expected in fingerprints.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError('Invalid numerical-review input: ' + name)
        if digest(path.read_bytes()) != expected:
            raise ValueError('Numerical review is stale: ' + name)
    def check_evidence(evidence, label='numerical evidence'):
        if not isinstance(evidence, dict) or evidence.get('path') not in fingerprints:
            raise ValueError('Unbound ' + label + ': ' + str(evidence))
        if ('sha256' in evidence
                and evidence['sha256'] != fingerprints[evidence['path']]):
            raise ValueError('Stale embedded evidence fingerprint: ' + evidence['path'])
    obligations = {}
    for item in data['obligations']:
        key = item['id']
        if key in obligations or item['kind'] not in KINDS:
            raise ValueError('Duplicate or invalid numerical obligation: ' + key)
        if item['evidence_status'] not in EVIDENCE_STATUSES:
            raise ValueError('Invalid numerical evidence status: ' + key)
        for field in ('formula', 'domain', 'profiles', 'experiment', 'evidence'):
            if not item.get(field):
                raise ValueError(f'Missing {field} for numerical obligation {key}')
        if not isinstance(item['evidence'], list):
            raise ValueError('Numerical evidence must be a list: ' + key)
        for evidence in item['evidence']:
            if not isinstance(evidence, (str, dict)):
                raise ValueError('Invalid numerical evidence entry: ' + key)
            if isinstance(evidence, dict):
                check_evidence(evidence)
        if item['kind'] == 'shared_model_requirement':
            if item['evidence_status'] != 'shared_model_contract':
                raise ValueError('Shared model requirement must remain a separate axis: ' + key)
        elif item['evidence_status'] == 'shared_model_contract':
            raise ValueError('Explicit law cannot be hidden as a shared model contract: ' + key)
        obligations[key] = item
    wanted = {(r['file'], r['declaration']): r for r in rows}
    if len(wanted) != len(rows):
        raise ValueError('Duplicate public record supplied to numerical review')
    statements = statement_map(rows, root)
    reviewed = {}
    for row in data['declarations']:
        key = row['file'], row['declaration']
        if key not in wanted or key in reviewed:
            raise ValueError('Unexpected or duplicate numerical review: ' + '::'.join(key))
        if row['file'] not in fingerprints:
            raise ValueError('Unbound public source: ' + row['file'])
        if row.get('statement_sha256') != digest(statements[key].encode()):
            raise ValueError('Changed numerical-review statement: ' + '::'.join(key))
        if row.get('numeric_model') != wanted[key]['numeric_model']:
            raise ValueError('Numerical model changed: ' + '::'.join(key))
        if row.get('review_status') != 'reviewed' or not row.get('review_basis'):
            raise ValueError('Numerical assumptions have not been reviewed: ' + '::'.join(key))
        model = row.get('fp_model_review', {})
        if model.get('kind') not in FP_MODEL_KINDS or not model.get('review_basis'):
            raise ValueError('Missing or invalid FP model review: ' + '::'.join(key))
        if model.get('role', 'parametric_value_contract') not in {
                'parametric_value_contract', 'fixed_nonempty_witness'}:
            raise ValueError('Invalid FP contract role: ' + '::'.join(key))
        if not isinstance(model.get('evidence'), list) or not model['evidence']:
            raise ValueError('Missing FP model review evidence: ' + '::'.join(key))
        for evidence in model['evidence']:
            check_evidence(evidence, 'FP model review evidence')
        if row['numeric_model'] == 'admitted_fp' and model['kind'] != 'backend_equivalence':
            raise ValueError('Admitted-FP claim must expose its relative equivalence: ' + '::'.join(key))
        expected_kind = {'backend_fp': 'backend_value_contract',
                         'bit_interpreter': 'bit_interpreter_contract'}.get(row['numeric_model'])
        if expected_kind and model['kind'] != expected_kind:
            raise ValueError('Typed value model label conflicts with reviewed subject: ' + '::'.join(key))
        if 'accepted_scope_coverage' in model:
            if (model['accepted_scope_coverage'] not in {'complete_in_declared_profile', 'partial'}
                    or not model.get('scope_basis')):
                raise ValueError('Invalid accepted-scope FP repair review: ' + '::'.join(key))
        for field, shared in [('extra_numeric_ids', False), ('shared_model_ids', True)]:
            ids = row.get(field)
            if not isinstance(ids, list) or len(ids) != len(set(ids)):
                raise ValueError('Missing or duplicate numerical IDs: ' + '::'.join(key))
            for name in ids:
                if name not in obligations:
                    raise ValueError('Unreviewed numerical assumption: ' + name)
                if (obligations[name]['kind'] == 'shared_model_requirement') != shared:
                    raise ValueError('Numerical assumption recorded on the wrong axis: ' + name)
        reviewed[key] = {**row, 'numerical_status': numerical_status(row['extra_numeric_ids'], obligations)}
    if set(reviewed) != set(wanted):
        raise ValueError('Missing numerical reviews: ' + repr(sorted(set(wanted) - set(reviewed))))
    used = {name for row in reviewed.values() for field in ('extra_numeric_ids', 'shared_model_ids')
            for name in row[field]}
    if set(obligations) != used:
        raise ValueError('Unused or missing numerical experiment entries: ' + repr(set(obligations) ^ used))
    task_ids = set()
    task_statuses = {}
    for task in data.get('experiment_tasks', []):
        if task['id'] in task_ids or task['obligation_id'] not in obligations:
            raise ValueError('Duplicate or unbound experiment task: ' + task['id'])
        task_ids.add(task['id'])
        if task['status'] not in {'pending_exact_binding', 'existing_report_optional_replay',
                                  'shared_model_check', 'not_an_experiment'}:
            raise ValueError('Invalid experiment task status: ' + task['id'])
        if any(not task.get(field) for field in ('formula', 'profile', 'domain')):
            raise ValueError('Incomplete exact experiment profile: ' + task['id'])
        status = obligations[task['obligation_id']]['evidence_status']
        allowed = {
            'counterexample_to_unrestricted_form': {'not_an_experiment'},
            'shared_model_contract': {'shared_model_check'},
            'reported_evidence_available': {'existing_report_optional_replay'},
            'binding_validation_pending': {'pending_exact_binding', 'existing_report_optional_replay'},
            'experiment_pending': {'pending_exact_binding', 'existing_report_optional_replay'},
        }[status]
        if task['status'] not in allowed:
            raise ValueError('Experiment task contradicts numerical evidence: ' + task['id'])
        task_statuses.setdefault(task['obligation_id'], set()).add(task['status'])
    if 'experiment_tasks' in data:
        if set(task_statuses) != set(obligations):
            raise ValueError('Missing experiment disposition for numerical obligation')
        for key, item in obligations.items():
            if item['evidence_status'] in {'binding_validation_pending', 'experiment_pending'}:
                if 'pending_exact_binding' not in task_statuses[key]:
                    raise ValueError('Missing pending exact binding for numerical obligation: ' + key)
    return reviewed, obligations


def load_review(rows, root=ROOT, path=REVIEW):
    data = json.loads(path.read_text())
    reviewed, obligations = validate_review(data, rows, root)
    return data, reviewed, obligations


def task_status(task, reviewed):
    rows = [reviewed[d['file'], d['declaration']] for d in task['public_declarations']]
    if not task['source_entry']:
        status = 'missing_source'
    elif not rows:
        raise ValueError('Imported task has no numerical-review entries: ' + task['task'])
    else:
        status = aggregate_status(r['numerical_status'] for r in rows)
    models = {r['fp_model_review']['kind'] for r in rows}
    parametric_models = {r['fp_model_review']['kind'] for r in rows
                         if r['fp_model_review'].get('role') != 'fixed_nonempty_witness'}
    scope_reviews = {r['fp_model_review']['accepted_scope_coverage'] for r in rows
                     if 'accepted_scope_coverage' in r['fp_model_review']}
    if len(scope_reviews) > 1:
        raise ValueError('Conflicting accepted-scope FP repair reviews: ' + task['task'])
    if not task['source_entry']:
        adequacy = 'missing_source'
    elif status == 'requires_numeric_model_revision':
        adequacy = 'requires_numeric_model_revision'
    elif 'backend_equivalence' in models:
        backend_status = aggregate_status(r['numerical_status'] for r in rows
                                          if r['fp_model_review']['kind'] == 'backend_equivalence')
        adequacy = {
            'no_extra_numeric_law': 'backend_equivalence_no_extra_law',
            'existing_numeric_evidence': 'backend_equivalence_existing_reports',
            'awaiting_numeric_validation': 'backend_equivalence_needs_validation',
        }[backend_status]
    elif 'bit_interpreter_contract' in parametric_models:
        adequacy = 'bit_interpreter_contract_only'
    elif 'backend_value_contract' in parametric_models:
        adequacy = 'backend_value_contract_only'
    elif 'rounding_model_contract' in models:
        adequacy = 'rounding_contract_only'
    else:
        adequacy = 'math_contract_only'
    return {
        'scope_policy': 'normal_inputs_and_documented_configuration_or_stage_scope_accepted',
        'fp_adequacy_status': adequacy,
        'fp_model_kinds': sorted(models),
        'accepted_scope_fp_coverage': next(iter(scope_reviews), 'not_reassessed_by_this_repair'),
        'claim_subjects': sorted(FP_MODEL_KINDS[k] for k in models),
        'parametric_fp_model_kinds': sorted(parametric_models),
        'fixed_witness_declarations': [
            {'file': r['file'], 'declaration': r['declaration']}
            for r in rows if r['fp_model_review'].get('role') == 'fixed_nonempty_witness'],
        'native_functional_correctness': 'not_established_by_this_review',
        'backend_equivalence_declarations': [
            {'file': r['file'], 'declaration': r['declaration']}
            for r in rows if r['fp_model_review']['kind'] == 'backend_equivalence'],
        'backend_value_declarations': [
            {'file': r['file'], 'declaration': r['declaration']}
            for r in rows if r['fp_model_review']['kind'] == 'backend_value_contract'],
        'bit_interpreter_declarations': [
            {'file': r['file'], 'declaration': r['declaration']}
            for r in rows if r['fp_model_review']['kind'] == 'bit_interpreter_contract'],
        'numerical_status': status,
        'extra_numeric_ids': sorted({n for r in rows for n in r['extra_numeric_ids']}),
        'shared_model_ids': sorted({n for r in rows for n in r['shared_model_ids']}),
        'source_translation_blocked': any(d['coverage_level'] == 'blocked_summary' for d in task['public_declarations']),
        'scope_note': 'Accepted within each displayed theorem/stage contract; omitted producers, profiles and stages are not thereby proved.',
        'hardware_or_independent_replay_accepted': False,
    }


def collect():
    from check_proof_gap_manifest import collect as public_rows
    from coverage_review import render_inventory
    rows = render_inventory(public_rows())['rows']
    data, reviewed, obligations = load_review(rows)
    ports = ROOT / 'bench/tritonbench_g'
    from port_layout import source_file
    tasks = []
    for directory in sorted(p for p in ports.iterdir() if p.is_dir() and not p.name.startswith(('_', '.'))):
        declarations = [r for r in rows if Path(r['file']).parent.name == directory.name]
        task = {'task': directory.name, 'source_entry': str(source_file(directory)) if source_file(directory) else None,
                'public_declarations': declarations}
        tasks.append({'task': directory.name, **task_status(task, reviewed)})
    expanded = []
    for name, obligation in obligations.items():
        affected = [r for r in reviewed.values() if name in r['extra_numeric_ids'] + r['shared_model_ids']]
        expanded.append({**obligation, 'affected_tasks': sorted({Path(r['file']).parent.name for r in affected}),
                         'affected_declarations': [{'file': r['file'], 'declaration': r['declaration']} for r in affected]})
    return {
        'schema': 1, 'reviewed_on': data['reviewed_on'], 'policy': data['policy'],
        'review_sha256': digest(REVIEW.read_bytes()),
        'summary': {
            'tasks': len(tasks), 'declarations': len(rows),
            'task_fp_adequacy_status': dict(Counter(t['fp_adequacy_status'] for t in tasks)),
            'declaration_fp_model_kinds': dict(Counter(r['fp_model_review']['kind'] for r in reviewed.values())),
            'task_numerical_status': dict(Counter(t['numerical_status'] for t in tasks)),
            'declaration_numerical_status': dict(Counter(r['numerical_status'] for r in reviewed.values())),
            'extra_numeric_obligations': sum(o['kind'] != 'shared_model_requirement' for o in obligations.values()),
            'shared_model_requirements': sum(o['kind'] == 'shared_model_requirement' for o in obligations.values()),
            'shared_model_tasks': sum(bool(t['shared_model_ids']) for t in tasks),
            'shared_model_declarations': sum(bool(r['shared_model_ids']) for r in reviewed.values()),
            'source_translation_blocked_tasks': [t['task'] for t in tasks if t['source_translation_blocked']],
        },
        'obligations': expanded, 'tasks': tasks,
        'experiment_tasks': data.get('experiment_tasks', []),
        'declarations': list(reviewed.values()),
        'native_boundaries': data['native_boundaries'],
        'native_value_validation': data.get('native_value_validation', {}),
        'interpretation': 'Accepted input/configuration/stage scope does not accept replacement of floating arithmetic by real arithmetic. FP model adequacy is the primary axis; absence of extra numerical laws is only a subordinate syntactic/evidence count. A backend-equivalence record proves candidate-vs-original behavior under its stated model, not absolute native functional correctness or coverage of every other public claim in that task. Existing reports remain scoped empirical premises. No experiment is executed or admitted by this inventory.',
    }


def cell(value):
    return str(value).replace('|', '\\|').replace('\n', '<br>')


def render(report):
    summary = report['summary']
    out = ['# 浮点证明层级、缺口与数值实验', '',
           '正常输入条件、明确的配置与阶段范围已获接受；把浮点算术换成实数算术没有获接受。先检查证明实际使用的模型，再列数值实验。', '',
           '**撤回把129项解释成“无需额外数值实验支持”的口径。** 这个旧计数混入只有数学/抽象舍入结果的任务；空数值假设列表不能补齐浮点证明。', '',
           '| 任务内已有的证明种类（不代表全任务硬件验收） | 任务数 |', '|---|---:|']
    for status, label in FP_ADEQUACY_LABELS.items():
        count = summary['task_fp_adequacy_status'].get(status, 0)
        if count:
            out.append(f'| {label} | {count} |')
    out += ['', '“有相对等价证明”表示任务内有明确列出的candidate-vs-original声明；不表示该任务全部值契约已获得硬件数值正确性证明。相同运算可以在两侧产生相同误差。位解释器结论和真实GPU指令的对应也单独保留。', '',
           '本轮逐任务、逐声明的修复范围见 [值语义修复表](fp_value_repairs.md)，剩余模型缺口见 [浮点证明缺口清单](fp_proof_gaps.md)。固定输入的位值例子单列，不提升参数化结论。已有五个待补绑定只覆盖条件等价证明的局部规则。', '',
           '## 从属统计：声明内额外数值律（不表示完成）', '',
           '| 数值证据状态 | 任务数 | 公开声明数 |', '|---|---:|---:|']
    for status in (*STATUS_ORDER, 'missing_source'):
        out.append(f"| {STATUS_LABELS[status]} | {summary['task_numerical_status'].get(status, 0)} | {summary['declaration_numerical_status'].get(status, 0)} |")
    out += ['', f"共 {summary['tasks']} 项、{summary['declarations']} 条公开记录。这里仅按声明中最强的额外数值前提计数；即使没有前提，也可能根本没有浮点证明。各规则的受影响任务集合可以重叠。", '',
            f"共享舍入模型要求另涉及 {summary['shared_model_tasks']} 项、{summary['shared_model_declarations']} 条声明，单列而不冒充新的算法数值律。", '',
            '## 交给实验的具体配置', '',
            '以下只列已建立条件等价证明的规则配置；完整字段见 [机器可读实验清单](numeric_experiment_tasks.json)。数学/舍入模型任务需先补具体浮点契约与运算图，不能凭任务名生成交换律或用自反等价填数。', '',
            '| 实验项 | 状态 | 关系 / dtype / 精度 |', '|---|---|---|']
    for task in report['experiment_tasks']:
        out.append(f"| `{task['id']}` | `{task['status']}` | {cell(task.get('formula', ''))}；{cell(task.get('profile', ''))} |")
    native = report.get('native_value_validation', {})
    if native:
        out += ['', '## 新值契约所需的原语与硬件对应', '',
                '这些与代数重写律是两类实验。软件位模型和逐算子后端契约已明确比较对象，但没有在本轮运行 GPU。', '',
                '[逐任务绑定清单](fp_value_native_tasks.json)保留具体合同、配置和待检查关系；[实验运行说明](../../experiments/floating_point/value_validation/README.md)提供 836 个标量位比较案例。归约树、dot、转换和超越函数仍需各自的原核配置。', '',
                '| 任务 | 原语配置 | 硬件对应状态 |', '|---|---|---|']
        for task in native.get('task_profiles', []):
            out.append(f"| `{task['task']}` | `{task['native_profile']}` | `{task['native_binding_status']}` |")
    out += ['', '## 待核对的数值假设', '']
    for item in report['obligations']:
        out += [f"### {item['id']}", '', f"**{item.get('title_zh', item['id'])}**", '',
                f"- 类型：`{item['kind']}`；证据状态：`{item['evidence_status']}`。",
                f"- 精确关系：{item['formula']}", f"- 适用域：{item['domain']}",
                '- dtype / 精度 / 配置：' + cell(item['profiles']),
                f"- 影响：{len(item['affected_tasks'])} 项、{len(item['affected_declarations'])} 条公开声明。", '',
                '实验或修复要求：', '', str(item['experiment']), '', '现有证据：', '']
        for evidence in item['evidence']:
            if isinstance(evidence, dict):
                path = evidence.get('path')
                label = evidence.get('description') or path or '证据'
                out.append(f"- [{label}](../../{path})" if path else '- ' + cell(evidence))
            else:
                out.append('- ' + str(evidence))
        out += ['', '受影响声明：', '']
        for row in item['affected_declarations']:
            out.append(f"- [{Path(row['file']).parent.name}::{row['declaration']}](../../{row['file']})")
        out.append('')
    out += ['## 真实后端对应关系', '']
    for boundary in report['native_boundaries']:
        if isinstance(boundary, str):
            out += ['- ' + boundary, '']
        else:
            out += [f"**{boundary['id']}**", '', str(boundary.get('formula', '')), '',
                    boundary.get('effect', ''), '']
            for evidence in boundary.get('evidence', []):
                path = evidence['path']
                label = evidence.get('description') or path
                out.append(f'- [{label}](../../{path})')
            out.append('')
    out += ['', '## 逐项重新统计', '',
            '| 任务 | 浮点证据层级 | 声明内数值前提状态 | 额外数值假设 | 共享模型要求 | 完整源码转换阻塞 |', '|---|---|---|---|---|---|']
    for t in report['tasks']:
        out.append(f"| `{t['task']}` | {FP_ADEQUACY_LABELS[t['fp_adequacy_status']]} | {STATUS_LABELS[t['numerical_status']]} | {', '.join(t['extra_numeric_ids']) or '无'} | {', '.join(t['shared_model_ids']) or '无'} | {'是；已证阶段仍保留' if t['source_translation_blocked'] else '无此标记'} |")
    out += ['', '## 复核', '', '```sh', 'python3 bench/numerical_obligations.py --check', '```', '',
            '检查包括全部公开记录的成员对应、实际声明文本、源文件和证据哈希。新增、删除或更改声明必须重新审阅，不能自动归入“无额外数值律”。实验状态不会因为配置/阶段范围已获接受而自动提升。', '']
    return '\n'.join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    report = collect()
    if args.check:
        if (json.loads(REPORT.read_text()) != report or MARKDOWN.read_text() != render(report)
                or json.loads(EXPERIMENTS.read_text()) != report['experiment_tasks']):
            raise ValueError('Numerical obligation inventory is stale; regenerate it')
    else:
        REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        MARKDOWN.write_text(render(report))
        EXPERIMENTS.write_text(json.dumps(report['experiment_tasks'], indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report['summary'], indent=2, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit('Numerical obligation review failed: ' + str(error))
