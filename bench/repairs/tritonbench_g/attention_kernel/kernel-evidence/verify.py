"""Read-only verification of the frozen attention kernel delivery; no compiler/test rerun."""
from pathlib import Path
import hashlib
import json
import re

OUT = Path(__file__).resolve().parent
TASK = OUT.parent
ROOT = TASK.parents[3]
packet = json.loads((OUT / 'delivery.json').read_text())
checks = []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def visit(value):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            assert sha(value['path']) == value['sha256'], value['path']
            checks.append(value['path'])
        for item in value.values():
            visit(item)
    elif isinstance(value, list):
        for item in value:
            visit(item)


def closure(roots):
    seen = set()
    todo = list(map(Path, roots))
    while todo:
        source = todo.pop()
        if source in seen:
            continue
        seen.add(source)
        for line in source.read_text().splitlines():
            match = re.match(r'^(?:public\s+)?import\s+(.+)$', line)
            if match:
                for name in match[1].split():
                    dependency = ROOT / (name.replace('.', '/') + '.lean')
                    if re.fullmatch(r'[A-Za-z0-9_.]+', name) and dependency.is_file():
                        todo.append(dependency)
    return seen


visit(packet)
assert packet['task'] == 'attention_kernel' and packet['status'] == 'ready_for_independent_review'
assert packet['selected_source']['sha256'] == 'e7aaea2b35aff93f2f92b1a6d4e50a824259cbf38205346b731dfc8d0ae9e77a'
assert packet['source_and_guards_unchanged'] and not packet['source_defect_found']
assert packet['current_task_build_receipt_count'] == len(packet['successful_builds']) == 39
assert packet['new_Lean_modules'] == 7 and packet['original_Lean_modules'] == 32
assert packet['deferred_numerical_equalities'] == packet['deferred_numerical_experiments'] == []
actual_sources = {str(p) for p in TASK.rglob('*') if p.is_file() and OUT not in p.parents
                  and p.suffix in {'.lean', '.py', '.md', '.json'}}
assert actual_sources == {p['path'] for p in packet['task_sources']}
assert {p['path'] for p in packet['task_sources'] if p['path'].endswith('.lean')} == {b['source']['path'] for b in packet['successful_builds']}
for build in packet['successful_builds']:
    receipt = json.loads(Path(build['receipt']['path']).read_text())
    assert receipt['source_sha256'] == build['source']['sha256']
    assert str(ROOT / receipt['source']) == build['source']['path']
    assert receipt['exit_code'] == 0 and receipt['lean_path'] == build['lean_path']
    assert receipt['objects'] == {p['path']: p['sha256'] for p in build['artifacts']}
assert len(packet['preserved_originals']) == packet['original_file_count'] == 50
changed = set()
originals = json.loads((OUT / 'preserved-task-files.json').read_text())
assert originals['original_file_count'] == len(originals['files']) == 50
for entry in originals['files']:
    assert sha(ROOT / entry['snapshot']) == entry['sha256']
    if sha(ROOT / entry['path']) != entry['sha256']:
        changed.add(str(ROOT / entry['path']))
assert changed == set(packet['changed_existing_files']) == {str(TASK / n) for n in ['README.md', 'Correct.lean', 'FPEquiv.lean']}

data = json.loads(Path(packet['repository_source_import_closure']['path']).read_text())
visit(data)
assert closure(data['public_roots']) == set(map(Path, data['public_sources']))
assert closure([data['audit_root']]) == {Path(e['source']['path']) for e in data['entries']}
assert data['public_source_count'] == len(data['public_sources'])
assert data['audit_source_count'] == len(data['entries'])
assert all(b['lean_path'] == data['lean_path'] for b in packet['successful_builds'])
for entry in data['entries']:
    relative = Path(entry['source']['path']).relative_to(ROOT).with_suffix('.olean')
    effective = next(Path(p) / relative for p in data['lean_path'] if (Path(p) / relative).is_file())
    assert str(effective) == entry['effective_object']['path']
    assert entry['fresh_task_receipt'] == (str(ROOT / relative.with_suffix('.lean')) in {b['source']['path'] for b in packet['successful_builds']})

audit = (OUT / 'final-build.log').read_text()
assert not re.search(r'\berror(?:\([^\n]*\))?:', audit) and 'sorryAx' not in audit
for theorem in ['Correct.kernel_correct', 'FPEquiv.kernel_native', 'Correct.empty_kernel']:
    assert theorem + ': axiom footprint ⊆ standard base ✓' in audit
for target in ['mathematicalTarget', 'nativeReference', 'InputContract']:
    assert target + ': definition does not reference' in audit
assert 'total host controls: 55' in (OUT / 'host-checks.log').read_text()
assert 'complete address grids: 7' in (OUT / 'host-checks.log').read_text()
assert 'original addressing/tail counterexamples: 6' in (OUT / 'host-checks.log').read_text()
assert (OUT / 'precision-check.log').read_text().strip() == 'PASS 10 profiles'
binder = json.loads((OUT / 'binder-artifacts/report.json').read_text())
precision = json.loads((OUT / 'binder-artifacts/source-binding.json').read_text())
cpu = json.loads((OUT / 'interpreter-report.json').read_text())
applicability = json.loads((OUT / 'profile-applicability.json').read_text())
visit(binder)
visit(applicability)
assert packet['profile_applicability'] == applicability
assert binder['status'] == precision['status'] == 'PASS'
assert binder['actual_JIT_signature_binder'] and not binder['GPU_execution']
assert len(binder['results']) == len(precision['profiles']) == len(applicability['profiles']) == 10
artifacts = [a for r in binder['results'] for a in r['artifacts'].values()]
assert len(artifacts) == 40 and packet['compiler_artifacts'] == artifacts
for result, profile, app in zip(binder['results'], precision['profiles'], applicability['profiles']):
    assert result['source_sha256'] == packet['selected_source']['sha256']
    assert result['status'] == 'compiled' and result['actual_JIT_signature_binder']
    assert result['name'] == profile['profile'] == app['name']
    assert all(profile['checks'].values())
    assert sha(profile['ttir']) == profile['sha256']
    assert result['metadata']['shared'] == app['shared_bytes']
    assert app['within_sm80_per_block_shared_limit'] == (app['shared_bytes'] <= 166912)
    assert not app['runtime_success_claim']
assert applicability['maximum_shared_bytes_per_block'] == 166912
assert sum(p['within_sm80_per_block_shared_limit'] for p in applicability['profiles']) == 6
assert cpu['source_sha256'] == packet['selected_source']['sha256']
assert cpu['case_count'] == len(cpu['cases']) == 10
assert cpu['complete_FP16_kernel_cases'] == 4 and cpu['empty_calls'] == 6
assert not cpu['GPU_execution'] and not cpu['runtime_profile_conformance_claim']
assert sum(c['empty'] for c in cpu['cases']) == 6
assert all(c['full_backing_frame'] and c['input_frame'] and c['result_is_None'] for c in cpu['cases'])
assert all(c['empty'] or c['dtype'] == 'torch.float16' for c in cpu['cases'])
assert 'get_bf16' in (OUT / 'interpreter-probe-initial.log').read_text()
assert not packet['validation']['GPU_execution']
assert json.loads((TASK / 'numerical-obligations.json').read_text())['deferred_numerical_equalities'] == []
assert json.loads((OUT / 'run-receipts.json').read_text()) == packet['run_receipts']
print(json.dumps({'all_passed': True, 'delivery_sha256': sha(OUT / 'delivery.json'),
                  'pin_records': len(checks), 'unique_files': len(set(checks)),
                  'successful_build_receipts': 39, 'public_import_closure': data['public_source_count'],
                  'audit_import_closure': data['audit_source_count'], 'original_files_preserved': 50,
                  'changed_originals_with_preserved_snapshots': 3, 'host_controls': 55,
                  'compiler_profiles': 10, 'compiler_text_artifacts': 40, 'CPU_cases': 10,
                  'GPU_execution': False}, indent=2))
