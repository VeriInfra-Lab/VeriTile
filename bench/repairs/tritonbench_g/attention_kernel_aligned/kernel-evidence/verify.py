"""Read-only verification of the frozen aligned attention kernel delivery; no compiler/test rerun."""
from pathlib import Path
import hashlib
import json
import re
import sys
sys.dont_write_bytecode = True
from evidence_check import check_evidence

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
assert packet['task'] == 'attention_kernel_aligned' and packet['status'] == 'ready_for_independent_review'
assert packet['selected_source']['sha256'] == '365cda104f62655345ad002a7d2815fba0d3c812e20c5a569e9b0214c92c85ea'
assert packet['source_and_guards_unchanged'] and not packet['source_defect_found']
assert packet['current_task_build_receipt_count'] == len(packet['successful_builds']) == 72
assert packet['new_Lean_modules'] == 11 and packet['original_Lean_modules'] == 61
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
assert len(packet['preserved_originals']) == packet['original_file_count'] == 92
changed = set()
originals = json.loads((OUT / 'preserved-task-files.json').read_text())
assert originals['original_file_count'] == len(originals['files']) == 92
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
for theorem in ['Correct.kernel_correct', 'FPEquiv.kernel_native', 'Correct.empty_kernel', 'BF16.Correct.kernel_correct', 'BF16.FPEquiv.kernel_native', 'BF16.Correct.empty_kernel']:
    assert theorem + ': axiom footprint ⊆ standard base ✓' in audit
for target in ['mathematicalTarget', 'nativeReference', 'InputContract']:
    assert target + ': definition does not reference' in audit
assert packet['review_round'] == 2 and packet['review_correction']['finding'] == 'BF16_SERIAL_DOMAIN'
round2 = (OUT / 'round2-build.log').read_text()
assert not re.search(r'\berror(?:\([^\n]*\))?:', round2) and 'sorryAx' not in round2
assert 'BF16.KernelOnly.compilerPlan_original_domain: axiom footprint ⊆ standard base ✓' in round2
assert 'BF16.KernelOnly.CompilerPlan: definition does not reference' in round2
for module in ['KernelOnly/BF16Endpoints', 'Correct', 'FPEquiv', 'KernelOnly/Audit']:
    assert '"module": "' + str((TASK / (module + '.lean')).relative_to(ROOT)) + '", "exit_code": 0' in round2
archive = json.loads((OUT / 'round1/archive.json').read_text())
assert sha(OUT / 'round1/delivery.json') == archive['round1_delivery_sha256'] == '45862e2a08dedf66498540b1248c8820326a0aff2fe8121ece73ec45539f0ec1'
assert packet['review_correction']['superseded_packet']['sha256'] == archive['round1_delivery_sha256']
assert [(entry['original_path'], entry['snapshot']['path'], entry['snapshot']['sha256'])
        for entry in packet['review_correction']['preserved_round1_files']] == [
        (entry['original_path'], entry['snapshot_path'], entry['sha256']) for entry in archive['snapshots']]
checked = check_evidence()
visit(checked['binder'])
visit(checked['applicability'])
assert packet['compiler_artifacts'] == checked['artifacts']
assert packet['profile_applicability'] == checked['applicability']
assert not packet['validation']['GPU_execution']
assert not packet['validation']['BF16_CPU_numeric_execution']
assert packet['model_boundaries'] == checked['obligations']['model_boundaries']
assert json.loads((OUT / 'run-receipts.json').read_text()) == packet['run_receipts']
assert all(p['shared_bytes'] <= 166912 and not p['runtime_success_claim'] for p in checked['applicability']['profiles'])
print(json.dumps({'all_passed': True, 'delivery_sha256': sha(OUT / 'delivery.json'),
                  'pin_records': len(checks), 'unique_files': len(set(checks)),
                  'successful_build_receipts': 72, 'public_import_closure': data['public_source_count'],
                  'audit_import_closure': data['audit_source_count'], 'original_files_preserved': 92,
                  'changed_originals_with_preserved_snapshots': 3, 'host_admissions': 1042,
                  'host_rejections': 43, 'compiler_profiles': 8, 'compiler_text_artifacts': 32,
                  'CPU_FP16_complete_grids': 8, 'GPU_execution': False}, indent=2))
