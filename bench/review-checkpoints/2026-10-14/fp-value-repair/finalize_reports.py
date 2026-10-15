"""Publish current review pages from the frozen acceptance result and inventories."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
CHECKPOINT = HERE.parent
sys.path.insert(0, str(ROOT / 'bench'))
from numerical_obligations import FP_ADEQUACY_LABELS
from acceptance_tritonbench_g import changed, digest, inputs


def read(path):
    return json.loads(path.read_text())


def main():
    bundle = CHECKPOINT / 'acceptance-after-value-repair'
    receipt = read(bundle / 'receipt.json')
    assert not receipt['changed_inputs'], 'Do not publish a drifting build snapshot'
    assert not changed(receipt['input_sha256'], inputs()), 'Workspace differs from the frozen receipt'
    for name, expected in receipt['artifact_sha256'].items():
        assert digest((bundle / name).read_bytes()) == expected, 'Changed receipt artifact: ' + name
    inventory = read(ROOT / 'bench/tritonbench_g/current_completion_inventory.json')
    numerical = read(ROOT / 'bench/tritonbench_g/numerical_obligations.json')
    repairs = read(ROOT / 'bench/tritonbench_g/fp_value_repairs.json')
    assert len(repairs['tasks']) == 41
    assert len(inventory['tasks']) == 184
    assert numerical['summary']['declarations'] == 747
    roles = Counter(d['role'] for t in repairs['tasks'] for d in t['declarations'])
    assert roles == {'parametric_value_contract': 107, 'fixed_nonempty_witness': 11}
    build = receipt['gates']['build']['status']
    rows = []
    for task in inventory['tasks']:
        rows.append({
            'task': task['task'], 'source_entry': task['source_entry'],
            'scope_category': task['scope_category'],
            'accepted_scope_review': task['accepted_scope_review'],
            'local_build_observation': build,
            'build_evidence': 'acceptance-after-value-repair/build.log',
            'public_files': sorted({d['file'] for d in task['public_declarations']}),
            'whole_task_accepted_by_this_bundle': False,
            'native_functional_correctness': 'not_established_by_this_review',
            'independent_replay': receipt['gates']['corpus_replay']['status'],
        })
    (CHECKPOINT / 'acceptance-tasks.json').write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + '\n')
    table = ['# 184项：实际证明层级与未完成项', '',
        '正常输入、配置和阶段范围继续保留。相对等价、后端值契约、软件位解释器和实际硬件对应分别记录；固定例子不扩大参数化证明范围。', '',
        '[当前收据](acceptance-after-value-repair/receipt.json)、[41项修复范围](../../tritonbench_g/fp_value_repairs.md)、[剩余缺口](../../tritonbench_g/fp_proof_gaps.md)。', '',
        '| 任务 | 实际证明层级 | 声明内额外数值律 | 声明模型 |', '|---|---|---|---|']
    for row in rows:
        r = row['accepted_scope_review']
        table.append(f"| `{row['task']}` | {FP_ADEQUACY_LABELS[r['fp_adequacy_status']]} | "
                     f"{', '.join(r['extra_numeric_ids']) or '无额外律；硬件对应另列'} | "
                     f"{', '.join(r['fp_model_kinds']) or '缺源'} |")
    (CHECKPOINT / 'acceptance-tasks.md').write_text('\n'.join(table) + '\n')
    lines = ['# 浮点值语义修复与当前验收结果', '',
        '**原41项数学模型任务已补充逐算子后端或软件位值契约；完整GPU验收仍未通过。** '
        '新增118条公开记录，其中107条参数化契约、11条固定实例。原629条声明的命题文本保持不变，总数为747。', '',
        '修复保留先前接受的输入、配置和阶段范围。旧数学定理仍标为数学模型；新增定理证明所选实际运算图的执行和独立值/写回结果。'
        '回调原语与软件位解释器均需单独对应具体编译器和设备。含位契约的任务也可能有其他阶段使用后端回调。', '',
        '| 当前证明层级 | 任务数 |', '|---|---:|']
    counts = numerical['summary']['task_fp_adequacy_status']
    for status in FP_ADEQUACY_LABELS:
        if counts.get(status):
            lines.append(f'| {FP_ADEQUACY_LABELS[status]} | {counts[status]} |')
    lines += ['', '## 可验收的修复内容', '',
        '- [逐任务范围与118条声明](../../tritonbench_g/fp_value_repairs.md)：每条保留模型、参数化/固定实例身份、适用范围和源文件链接。',
        '- [逐任务原语及硬件验证关系](../../tritonbench_g/fp_value_native_tasks.json)：逐项列出dtype、转换、常量、比较、归约、dot、超越函数、整数宽度及组合对应。',
        '- [实验运行说明](../../../experiments/floating_point/value_validation/README.md)：836个标量参考案例；普通除法与正确舍入除法分开。没有本轮GPU运行记录。',
        '- [原有五个待补数值绑定](../../tritonbench_g/numeric_experiment_handoff.md)：typed fp32乘法及四种BGMV加法配置仍待实验。BGMV按实际内层精度归并为五种配置，72个源码配置均有枚举检查。',
        '- [全部184项清单](acceptance-tasks.md)与[数值前提报告](../../tritonbench_g/numerical_obligations.md)：空额外数值律列表不作为完成证据。', '',
        '前端保留运算前的精度标记、真正负号与绝对值；旧源码形状和实数辅助证明同步修复。'
        '41个修复目录的全部证明模块已纳入八库构建，未用删除模块或加强公开前提消除编译失败。', '',
        '## 实际检查', '', '| 检查 | 结果 |', '|---|---|']
    for name in ('membership', 'source', 'inventory', 'contracts', 'numerical_obligations',
                 'build', 'fp_contract_surface', 'comparator_preflight', 'library_replay', 'corpus_replay'):
        g = receipt['gates'][name]
        lines.append(f"| `{name}` | `{g['status']}`" +
                     (f"，exit={g['exit_code']}" if 'exit_code' in g else '') + ' |')
    lines += ['', f"归档{len(receipt['input_sha256'])}个输入，零输入漂移；报告工具回归检查通过，"
        '[测试日志](fp-value-repair/report-tests.log)保留实际计数。'
        '[完整收据](acceptance-after-value-repair/receipt.json)与[归档一致性检查](acceptance-after-value-repair-check.json)保留实际结果。', '',
        '## 未完成事项', '',
        '**23项仍只有cast/store舍入模型，3项仍有全域fp16恒等反例，6项缺源码。** '
        '详见[剩余缺口](../../tritonbench_g/fp_proof_gaps.md)。新增41项的模型缺口补齐，不表示这些其他缺口消失。'
        '硬件原语、编译器组合对应及官方独立重放继续按实际证据验收；有限实验不能证明已被反例否定的全域命题。', '',
        '历史[模型审计及原报告](fp-value-repair/before/bench/review-checkpoints/2026-10-14/ACCEPTANCE.zh-CN.md)和'
        '[上轮收据](acceptance-after-model-review/receipt.json)保持原样。当前修改未提交或推送。', '',
        '```sh', 'python3 bench/fp_value_repairs.py --check',
        'python3 bench/numerical_obligations.py --check',
        'python3 bench/inventory_tritonbench_g_completion.py --check',
        'python3 bench/acceptance_tritonbench_g.py --check bench/review-checkpoints/2026-10-14/acceptance-after-value-repair',
        '```', '',
        '前三项检查当前报告与审阅一致性；最后一项即使归档一致仍会返回非零，因为全库验收条件尚未全部满足。', '']
    (CHECKPOINT / 'ACCEPTANCE.zh-CN.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
