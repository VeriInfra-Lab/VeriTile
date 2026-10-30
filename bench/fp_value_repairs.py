#!/usr/bin/env python3
"""Render explicitly reviewed value contracts and their pending native bindings.

This does not discover proofs from names or accept a GPU experiment. The registry
is reviewed manually; membership, declarations, roles and source hashes are
checked against the separate numerical review before rendering.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench'))
from numerical_obligations import collect

BASE = ROOT / 'bench/tritonbench_g'
REGISTRY = BASE / 'fp_value_repairs.json'
REPORT = BASE / 'fp_value_repairs.md'
NATIVE = BASE / 'fp_value_native_tasks.json'

# Reviewed primitive families, not inferred algebraic laws. Task-specific dtype,
# branch, tree and domain restrictions remain in each bound public statement.
FAMILIES = {
    'storage': '原始load/store与mask/undef oracle保持指定存储位；数值cast单列。',
    'scalar': '实际逐步add/sub/mul/div/neg/abs与合同的软件值或指定后端回调对应；保留舍入、FTZ/DAZ、NaN、signed-zero和融合配置。普通除法与div_rn分别绑定。',
    'cast': '实际源/目标格式转换与软件cast或指定回调对应；记录tie、overflow、subnormal及NaN payload规则。',
    'constant_conversion': '仅核对合同实际到达的浮点字面量、padding/初值、±Inf及fromNat/fromInt站点：固定源整数宽度、目标precision、常数折叠与实际位编码；不能从算术探针推定这些转换正确。',
    'comparison': '实际浮点eq/lt/le/gt/ge比较与合同一致；特别核对NaN、signed-zero、Inf和比较结果控制的分支。',
    'order': 'maximum/argmax的比较、NaN策略、signed-zero与tie/index选择符合合同；不能用无条件compare+where替代maximum。',
    'reduction': '原核的每个归约站点使用合同指定的相同树/次序、初值、算术精度与mask；软件顺序fold不自动对应GPU树。',
    'transcendental': 'exp/exp2/log/log2/pow/sin/cos/rsqrt的实际指令与精确回调绑定；按各合同实际使用的原语和输入域记录位输出，不能假设实数精确函数。',
    'float_to_int': '实际源浮点到目标有符号/无符号位宽的转换，在active输出的有限可表示域上符合toward-zero合同；域外lane任意policy的独立性单列。llrint不在此规则内。',
    'integer': 'packed解码、signedness、int8窄化、移位和int32点积累加符合固定宽解释器；核对wrap或饱和模式。',
    'dot': '浮点dot的输入/累加精度、原始操作次序及allow_tf32等配置与显式dot回调对应；普通位例不证明任意dot回调正确。',
    'composition': '编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。',
}
PROFILES = {
    'raw_copy': ['storage'],
    'fp32_uint8_conversion': ['storage', 'float_to_int'],
    'unfused_fp32': ['storage', 'scalar', 'cast'],
    'unfused_fp32_and_raw_copy': ['storage', 'scalar', 'cast'],
    'reduction_fp32': ['storage', 'scalar', 'cast', 'reduction', 'transcendental'],
    'ordered_max': ['storage', 'cast', 'order', 'reduction'],
    'softmax_fp32': ['storage', 'scalar', 'cast', 'order', 'reduction', 'transcendental'],
    'mixed_transcendental': ['storage', 'scalar', 'cast', 'transcendental'],
    'fp32_int8_conversion': ['storage', 'scalar', 'cast', 'order', 'reduction', 'float_to_int'],
    'packed_int4_binary16': ['storage', 'integer', 'scalar', 'cast'],
    'packed_int4_binary16_typed_dot': ['storage', 'integer', 'scalar', 'cast', 'dot'],
    'fp32_binary16_max': ['storage', 'scalar', 'cast', 'order', 'reduction'],
    'fp32_typed_dot': ['storage', 'scalar', 'cast', 'dot', 'reduction'],
    'signed_int8_dot_int32_wrap': ['storage', 'integer'],
    'int8_dot_fp32_scale_fp16_store': ['storage', 'integer', 'scalar', 'cast'],
    'int8_quantizer_and_scaled_dot': ['storage', 'integer', 'scalar', 'cast', 'order', 'reduction', 'float_to_int'],
    'int8_dot_fp32_scale_int32_store': ['storage', 'integer', 'scalar', 'cast', 'float_to_int'],
    'sparse_attention_fp16_fp32': ['storage', 'scalar', 'cast', 'order', 'reduction', 'transcendental', 'dot'],
}
# Narrow the shared profile for each actual operation graph. These are inspected
# task lists, not keyword heuristics. A profile must not invent an unused law.
OVERRIDES = {
    'fused_recurrent_delta': ['storage', 'scalar', 'cast', 'reduction'],
    'fused_layernorm_triton': ['storage', 'scalar', 'cast', 'comparison', 'reduction', 'transcendental'],
    'fused_rwkv6_kernel': ['storage', 'scalar', 'cast', 'reduction', 'transcendental'],
    'matrix_vector_multip': ['storage', 'scalar', 'cast', 'reduction'],
    'rbe_triton_transform': ['storage', 'scalar', 'cast', 'transcendental'],
    'logsumexp_fwd': ['storage', 'scalar', 'cast', 'order', 'reduction', 'transcendental'],
    'parallel_retention_attention': ['storage', 'scalar', 'cast', 'transcendental', 'dot', 'reduction'],
}

# Explicitly inspected literal/integer-conversion sites; raw copies and the
# pure integer core do not acquire fictitious floating requirements.
CONSTANT_TASKS = {
    'fifth_order_sph_harmonics',
    'fused_layernorm_triton',
    'fused_recurrent_delta',
    'fused_recurrent_hgrn',
    'fused_rwkv6_kernel',
    'int4_matmul',
    'int8_dequant_matmul',
    'int8_matmul_quantization',
    'int8_quantization',
    'int_scaled_matmul',
    'kv_cache_filling',
    'layer_norm_welfold',
    'logsumexp_fwd',
    'matmul_dequant_int4',
    'matmul_dequantize',
    'matmul_dequantize_int4',
    'matrix_vector_multip',
    'mixed_sparse_attention',
    'parallel_attention',
    'parallel_retention_attention',
    'quant_transpose_kernel',
    'quantize_global',
    'quantize_kv_copy',
    'quantize_kv_transform',
    'rbe_triton_transform',
    'rope_backward_transform',
    'rope_embedding',
    'rope_transform',
    'rotary_emb_nopad',
    'rotary_transform',
    'rotary_transform_ops',
    'rowwise_quantization_triton',
    'token_softmax_bloom',
    'token_softmax_llama',
    'triton_argmax',
}


def collect_repairs():
    registry = json.loads(REGISTRY.read_text())
    numerical = collect()
    reviewed = {(r['file'], r['declaration']): r for r in numerical['declarations']}
    historical = json.loads((ROOT / 'bench/review-checkpoints/2026-10-14/fp-value-repair/before/bench/tritonbench_g/numerical_obligations.json').read_text())
    original = {t['task'] for t in historical['tasks'] if t['fp_adequacy_status'] == 'math_contract_only'}
    tasks = registry['tasks']
    if len({t['task'] for t in tasks}) != len(tasks) or {t['task'] for t in tasks} != original:
        raise ValueError('The repair registry must cover the exact original 41-task set')
    results = []
    for t in sorted(tasks, key=lambda t: t['task']):
        if t['accepted_scope_fp_coverage'] != 'complete_in_declared_profile':
            raise ValueError('Accepted stage still lacks a reviewed value contract: ' + t['task'])
        if t['native_binding_status'] != 'not_established':
            raise ValueError('Native status needs a new independent evidence review: ' + t['task'])
        if len({d['name'] for d in t['declarations']}) != len(t['declarations']):
            raise ValueError('Duplicate public declaration: ' + t['task'])
        declarations = []
        for d in t['declarations']:
            r = reviewed[t['file'], d['name']]
            m = r['fp_model_review']
            if m['kind'] != d['model_kind'] or m['role'] != d['role'] or r['numeric_model'] != d['numeric_model']:
                raise ValueError('Reviewed declaration/model/role mismatch: ' + d['name'])
            if (r['accepted_stage_conditions'] != [t['covered_scope'], t['remaining_scope']]
                    or m['scope_basis'] != t['covered_scope'] + ' ' + t['remaining_scope']
                    or m['accepted_scope_coverage'] != t['accepted_scope_fp_coverage']):
                raise ValueError('Reviewed declaration scope mismatch: ' + d['name'])
            declarations.append({**d, 'file': t['file'], 'line': r['line'],
                                 'statement_sha256': r['statement_sha256']})
        families = list(OVERRIDES.get(t['task'], PROFILES[t['native_profile']]))
        if t['task'] in CONSTANT_TASKS:
            families.append('constant_conversion')
        families.append('composition')
        results.append({**t, 'declarations': declarations,
                        'binding_relations': [{'id': f, 'relation': FAMILIES[f]} for f in families],
                        'input_domain': 'Exact ordinary input/configuration/domain premises in the bound declarations; no assumed final-output equality.',
                        'native_execution_performed': False,
                        'required_artifacts': ['exact input/output words', 'actual compiler/GPU versions',
                                               'dtype/shape/stride/constexpr/fusion/FTZ profile',
                                               'source and compiled IR/PTX hashes', 'mismatches including NaN and signed zero']})
    return {'schema': 1, 'status': 'model_contracts_reviewed_native_bindings_pending',
            'original_math_only_tasks': len(original),
            'new_public_declarations': sum(len(t['declarations']) for t in tasks),
            'native_execution_performed': False,
            'interpretation': 'The old mathematical declarations remain mathematical. New contracts cover the previously accepted stages under their explicit software or backend models. Fixed examples are not parametric or native proofs. Finite native tests can support a recorded profile or refute a proposed binding; they are not universal proofs.',
            'tasks': results}


def render(report):
    out = ['# 原41项数学模型任务的值语义修复', '',
           f"原41项均已为先前接受的配置/阶段补充逐算子后端或位解释器合同，共新增 {report['new_public_declarations']} 条公开记录。旧数学结论保留原标签。", '',
           '**这不是41项GPU验收通过。** 参数化后端合同仍需绑定原语；软件位合同仍需对应真实编译器和设备。固定输入的例子单列，只验证该实例。混合任务含位合同，也不代表所有阶段都已摆脱后端回调。', '',
           '每条合同检查实际执行成功及独立初始输入的结果/写回（辅助helper与固定例子按其实际声明），不把最终输出等式当前提。正常输入、配置、已接受的阶段范围继续保留。', '',
           '[机器可读绑定清单](fp_value_native_tasks.json)、[实验运行说明](../../experiments/floating_point/value_validation/README.md)、[全部184项状态](numerical_obligations.md)。本轮没有GPU执行记录。', '',
           '| 任务 | 参数化后端合同 | 参数化位合同 | 固定例子 | 已覆盖范围与保留边界 |', '|---|---:|---:|---:|---|']
    for t in report['tasks']:
        count = Counter(d['model_kind'] for d in t['declarations'] if d['role'] == 'parametric_value_contract')
        fixed = sum(d['role'] == 'fixed_nonempty_witness' for d in t['declarations'])
        scope = (t['covered_scope'] + ' ' + t['remaining_scope']).replace('|', '\\|')
        out.append(f"| [{t['task']}]({t['task']}/FPCorrect.lean) | {count['backend_value_contract']} | {count['bit_interpreter_contract']} | {fixed} | {scope} |")
    out += ['', '## 逐声明及原语绑定', '']
    for t in report['tasks']:
        out += [f"### {t['task']}", '', f"原语配置：`{t['native_profile']}`；硬件对应：`{t['native_binding_status']}`。", '']
        for d in t['declarations']:
            out.append(f"- [`{d['name']}`]({t['task']}/FPCorrect.lean)：`{d['model_kind']}`，`{d['role']}`。")
        out += ['', '待验证的对应关系：', '']
        for b in t['binding_relations']:
            out.append('- ' + b['relation'])
        out.append('')
    out += ['## 复核', '', '```sh', 'python3 bench/fp_value_repairs.py --check', '```', '',
            '检查精确的原41项成员集合、当前源哈希、公开声明文本与模型/角色审阅一致性；不由文件名自动判断证明强度。', '']
    return '\n'.join(out)


def remaining_gaps(report):
    """Retain the reviewed outstanding work; never infer closure from a file."""
    numerical = collect()
    historical = ROOT / 'bench/review-checkpoints/2026-10-14/fp-value-repair/before/bench/tritonbench_g/fp_proof_gaps.json'
    old = json.loads(historical.read_text())
    states = {t['task']: t['fp_adequacy_status'] for t in numerical['tasks']}
    math = {t for t, status in states.items() if status == 'math_contract_only'}
    casts = {t for t, status in states.items() if status == 'rounding_contract_only'}
    revisions = {t for t, status in states.items() if status == 'requires_numeric_model_revision'}
    # This repair covers exactly the original mathematical group. New work on
    # the other groups requires a fresh scope review, not silent list removal.
    if math or casts != {t['task'] for t in old['cast_store_only_tasks']}:
        raise ValueError('Outstanding model-gap membership requires a new review')
    if revisions != set(old['numeric_model_revision_tasks']):
        raise ValueError('Refuted-model membership requires a new review')
    missing = {t for t, status in states.items() if status == 'missing_source'}
    baseline = json.loads((historical.parent / 'numerical_obligations.json').read_text())
    if missing != {t['task'] for t in baseline['tasks'] if t['fp_adequacy_status'] == 'missing_source'}:
        raise ValueError('Missing-source membership requires a new review')
    declarations = {(r['file'], r['declaration']): r for r in numerical['declarations']}
    for task in old['cast_store_only_tasks']:
        for d in task['declarations']:
            current = declarations[d['file'], d['declaration']]
            if d['statement_sha256'] != current['statement_sha256']:
                raise ValueError('Outstanding cast/store statement changed')
            d['line'] = current['line']
    return {
        'schema': 2, 'policy': old['policy'], 'math_only_tasks': [],
        'cast_store_only_tasks': old['cast_store_only_tasks'],
        'numeric_model_revision_tasks': sorted(revisions),
        'missing_source_tasks': sorted(missing),
        'repaired_math_only_tasks': [
            {k: t[k] for k in ('task', 'file', 'covered_scope', 'remaining_scope',
                              'accepted_scope_fp_coverage', 'native_binding_status', 'declarations')}
            for t in report['tasks']],
        'summary': {'math_only': len(math), 'cast_store_only': len(casts),
                    'refuted_global_rounding_identity': len(revisions),
                    'original_math_group_repaired': len(report['tasks']),
                    'new_public_value_declarations': report['new_public_declarations'],
                    'native_hardware_bindings_established_by_this_repair': 0},
        'historical_baseline': str(historical.relative_to(ROOT)),
        'interpretation': 'Original mathematical statements retain their models. Supplemental contracts close the model gap only in the explicitly accepted stages and profiles. All native bindings remain pending. The unchanged cast/store and refuted-model groups remain outstanding.',
    }


def render_gaps(gaps):
    out = ['# 浮点与固定宽语义证明缺口', '',
           '**原41项的数学模型缺口已补充值语义合同；仍有23项只有cast/store舍入模型，3项含已有反例的数值假设，另6项缺源码。**', '',
           '41项的旧数学定理保留原标签。新增合同逐项保留已接受的输入、配置、阶段范围，并区分后端回调与软件位解释器。硬件对应仍未建立，不能把这次修复称为41项GPU验收通过。', '',
           '[41项修复范围及公开声明](fp_value_repairs.md)、[逐项硬件实验清单](fp_value_native_tasks.json)、[184项当前统计](numerical_obligations.md)。原始41项缺口保存在[历史基线](../review-checkpoints/2026-10-14/fp-value-repair/before/bench/tritonbench_g/fp_proof_gaps.md)。', '',
           '## 仍未补齐的23项cast/store模型', '',
           '这些契约允许cast/store舍入，但普通算术仍按实数执行。需要为已接受阶段补齐逐算子的精度、顺序和独立结果证明；仅提供round_idem不能关闭缺口。', '',
           '| 任务 | 已有cast/store声明 |', '|---|---|']
    for t in gaps['cast_store_only_tasks']:
        links = '<br>'.join(f"[{d['declaration']}]({Path(d['file']).relative_to('bench/tritonbench_g')})" for d in t['declarations'])
        out.append(f"| `{t['task']}` | {links} |")
    out += ['', '## 仍需修订的数值模型', '',
            '、'.join('`' + t + '`' for t in gaps['numeric_model_revision_tasks']) + ' 的全域 `round_fp16 = id` 假设已有反例。需修改参考模型，或证明实际中间值满足明确的精确适用域；增加抽样不能证明这个已被反驳的全域命题。', '',
            '## 缺少源码', '', '、'.join('`' + t + '`' for t in gaps['missing_source_tasks']) + '。', '',
            '## 实验边界', '',
            '[原有五个局部数值实验配置](numeric_experiment_handoff.md)继续保留；新增41项的原语/编译配置验证见上方逐项清单。有限样本可以反驳绑定或支持已记录的配置，不能代替普遍性证明。', '',
            '[机器可读缺口表](fp_proof_gaps.json)由当前声明审阅和保留的历史基线生成；`python3 bench/fp_value_repairs.py --check` 检查一致性。', '']
    return '\n'.join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    report = collect_repairs()
    native = json.dumps(report, indent=2, ensure_ascii=False) + '\n'
    markdown = render(report)
    gaps = remaining_gaps(report)
    artifacts = {NATIVE: native, REPORT: markdown,
                 BASE / 'fp_proof_gaps.json': json.dumps(gaps, indent=2, ensure_ascii=False) + '\n',
                 BASE / 'fp_proof_gaps.md': render_gaps(gaps)}
    if args.check:
        if any(path.read_text() != content for path, content in artifacts.items()):
            raise ValueError('FP value repair report is stale')
    else:
        for path, content in artifacts.items():
            path.write_text(content)
    print(json.dumps({k: report[k] for k in ('original_math_only_tasks', 'new_public_declarations', 'native_execution_performed')}))


if __name__ == '__main__':
    main()
