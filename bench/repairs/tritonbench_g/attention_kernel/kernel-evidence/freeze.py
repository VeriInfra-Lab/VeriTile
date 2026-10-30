"""Freeze current attention kernel proofs and their existing evidence, without reruns."""
from pathlib import Path
import hashlib
import json
import re

OUT = Path(__file__).resolve().parent
TASK = OUT.parent
ROOT = TASK.parents[3]
CACHE = ROOT / '.veritile-cache/kernel-build/lib/lean'
DRIVER = ROOT / '.veritile-cache/build-tools/workspace_lean_local_first_v2.py'
EXPECTED_SOURCE = 'e7aaea2b35aff93f2f92b1a6d4e50a824259cbf38205346b731dfc8d0ae9e77a'
CHANGED = {str(TASK / name) for name in ['README.md', 'Correct.lean', 'FPEquiv.lean']}


def pin(path):
    path = Path(path).absolute()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def imports(path):
    result = []
    for line in path.read_text().splitlines():
        match = re.match(r'^(?:public\s+)?import\s+(.+)$', line)
        if match:
            result.extend(n for n in match[1].split() if re.fullmatch(r'[A-Za-z0-9_.]+', n))
    return result


def closure(roots):
    seen = set()
    todo = list(roots)
    while todo:
        source = todo.pop()
        if source in seen:
            continue
        seen.add(source)
        for name in imports(source):
            dependency = ROOT / (name.replace('.', '/') + '.lean')
            if dependency.is_file():
                todo.append(dependency)
    return seen


preservation = json.loads((OUT / 'preserved-task-files.json').read_text())
assert preservation['original_file_count'] == len(preservation['files']) == 50
preserved = []
changed = set()
for item in preservation['files']:
    source = pin(ROOT / item['path'])
    snapshot = pin(ROOT / item['snapshot'])
    assert snapshot['sha256'] == item['sha256'], snapshot
    if source['sha256'] != snapshot['sha256']:
        changed.add(source['path'])
    preserved.append({'before': snapshot, 'current': source, 'current_changed': source['sha256'] != snapshot['sha256']})
assert changed == CHANGED, changed
assert pin(TASK / 'attention_kernel_repaired.py')['sha256'] == EXPECTED_SOURCE

sources = sorted(p for p in TASK.rglob('*') if p.is_file() and OUT not in p.parents
                 and p.suffix in {'.lean', '.py', '.md', '.json'})
modules = [p for p in sources if p.suffix == '.lean']
assert len(modules) == 39
builds = []
for source in modules:
    receipt = CACHE / source.relative_to(ROOT).with_suffix('.workspace-build.json')
    data = json.loads(receipt.read_text())
    assert data['exit_code'] == 0 and data['source'] == str(source.relative_to(ROOT))
    assert data['source_sha256'] == pin(source)['sha256'], source
    objects = []
    for path, digest in data['objects'].items():
        artifact = pin(path)
        assert artifact['sha256'] == digest, path
        objects.append(artifact)
    builds.append({'source': pin(source), 'receipt': pin(receipt), 'artifacts': objects,
                   'lean_path': data['lean_path']})
paths = builds[0]['lean_path']
assert all(b['lean_path'] == paths for b in builds)
public = [TASK / n for n in ['Contract.lean', 'Correct.lean', 'FPEquiv.lean']]
audit_root = TASK / 'KernelOnly/Audit.lean'
public_sources = closure(public)
audit_sources = closure([audit_root])
assert public_sources <= audit_sources
entries = []
for source in sorted(audit_sources):
    relative = source.relative_to(ROOT).with_suffix('.olean')
    effective = next(Path(directory) / relative for directory in paths if (Path(directory) / relative).is_file())
    entries.append({'source': pin(source), 'effective_object': pin(effective),
                    'fresh_task_receipt': source in modules})
closure_data = {
    'scope': 'Exact repository source import closure and effective objects in receipt LEAN_PATH. Core/package caches reused; 39 listed task modules have source-current receipts.',
    'public_roots': [str(p) for p in public], 'audit_root': str(audit_root),
    'public_source_count': len(public_sources), 'audit_source_count': len(audit_sources),
    'public_sources': [str(p) for p in sorted(public_sources)], 'lean_path': paths, 'entries': entries}
(OUT / 'source-import-closure.json').write_text(json.dumps(closure_data, indent=2) + '\n')

audit = (OUT / 'final-build.log').read_text()
assert not re.search(r'\berror(?:\([^\n]*\))?:', audit) and 'sorryAx' not in audit
for theorem in ['Correct.kernel_correct', 'FPEquiv.kernel_native', 'Correct.empty_kernel']:
    assert theorem + ': axiom footprint ⊆ standard base ✓' in audit
for target in ['mathematicalTarget', 'nativeReference', 'InputContract']:
    assert target + ': definition does not reference' in audit
for module in ['Correct', 'FPEquiv', 'KernelOnly/Audit']:
    assert '"module": "' + str((TASK / (module + '.lean')).relative_to(ROOT)) + '", "exit_code": 0' in audit

binder = json.loads((OUT / 'binder-artifacts/report.json').read_text())
precision = json.loads((OUT / 'binder-artifacts/source-binding.json').read_text())
interpreter = json.loads((OUT / 'interpreter-report.json').read_text())
applicability = json.loads((OUT / 'profile-applicability.json').read_text())
obligations = json.loads((TASK / 'numerical-obligations.json').read_text())
assert binder['status'] == precision['status'] == 'PASS'
assert binder['actual_JIT_signature_binder'] and not binder['GPU_execution']
assert len(binder['results']) == len(precision['profiles']) == len(applicability['profiles']) == 10
artifacts = []
for result in binder['results']:
    assert result['status'] == 'compiled' and result['actual_JIT_signature_binder']
    assert result['source_sha256'] == EXPECTED_SOURCE
    assert set(result['artifacts']) == {'ttir', 'ttgir', 'llir', 'ptx'}
    for artifact in result['artifacts'].values():
        assert pin(artifact['path']) == artifact
        artifacts.append(artifact)
for profile in precision['profiles']:
    assert all(profile['checks'].values())
    assert pin(profile['ttir'])['sha256'] == profile['sha256']
assert interpreter['source_sha256'] == EXPECTED_SOURCE
assert interpreter['case_count'] == len(interpreter['cases']) == 10
assert interpreter['complete_FP16_kernel_cases'] == 4 and interpreter['empty_calls'] == 6
assert not interpreter['GPU_execution'] and not interpreter['runtime_profile_conformance_claim']
assert all(c['full_backing_frame'] and c['input_frame'] and c['result_is_None'] for c in interpreter['cases'])
assert all(c['empty'] or c['dtype'] == 'torch.float16' for c in interpreter['cases'])
assert applicability['maximum_shared_bytes_per_block'] == 166912
assert sum(p['within_sm80_per_block_shared_limit'] for p in applicability['profiles']) == 6
assert all(not p['runtime_success_claim'] for p in applicability['profiles'])
assert obligations['deferred_numerical_equalities'] == obligations['deferred_numerical_experiments'] == []
assert 'total host controls: 55' in (OUT / 'host-checks.log').read_text()
assert (OUT / 'precision-check.log').read_text().strip() == 'PASS 10 profiles'

evidence_names = [
    'preserved-task-files.json', 'canonical-selection.json', 'original-build-modules.txt',
    'original-build.log', 'kernel-input-build.log', 'kernel-build-2.log', 'kernel-build-3.log',
    'kernel-build-4.log', 'kernel-build-5.log', 'kernel-build-6.log', 'final-build-initial.log',
    'final-build-2.log', 'final-build.log', 'host-checks.log', 'precision-check.log',
    'binder_probe.py', 'binder-probe.log', 'binder-artifacts/report.json',
    'binder-artifacts/source-binding.json', 'interpreter_probe_initial.py',
    'interpreter-probe-initial.log', 'interpreter_probe.py', 'interpreter-probe.log',
    'interpreter-report.json', 'profile-applicability.json', 'official-sm80-shared-memory.html',
    'source-import-closure.json', 'run-receipts.json']
packet = {
    'task': 'attention_kernel', 'status': 'ready_for_independent_review',
    'scope': 'Complete kernel-only Contract, exact-Real Correct and one-model native FPEquiv with actual Tensor outputs and frames.',
    'task_directory': str(TASK), 'selected_source': pin(TASK / 'attention_kernel_repaired.py'),
    'kernel_source': pin(TASK / 'Kernels.lean'), 'source_and_guards_unchanged': True,
    'source_defect_found': False, 'public_entries': [pin(p) for p in public],
    'task_sources': [pin(p) for p in sources], 'preserved_originals': preserved,
    'original_file_count': 50, 'changed_existing_files': sorted(CHANGED),
    'successful_builds': builds, 'current_task_build_receipt_count': 39,
    'new_Lean_modules': 7, 'original_Lean_modules': 32,
    'repository_source_import_closure': pin(OUT / 'source-import-closure.json'),
    'proved_endpoints': {
        'KernelContract.accepted_iff': 'Python-aligned acceptance iff InputContract, with actual Tensor metadata, strides, original allocations and observed launch scale word.',
        'Correct.kernel_correct': 'Successful complete original quoted grid for any full-grid permutation; every logical Tensor output equals independent gathered Real softmax; complete tagged frame, physical exterior frame and preserved reads.',
        'FPEquiv.kernel_native': 'Successful complete original quoted grid and every FP16/BF16 Tensor output word equals independent native graph under one fixed instruction model; same frames for every compiler owner.',
        'Correct.empty_kernel': 'Any empty batch/head/context call gives empty work, identity source execution and unchanged caller words.',
        'KernelOnly.backing_exterior': 'Complete checked read allocations, including prefixes, gaps and tails, are outside all logical output stores.',
        'Closed': 'Existing full padded source integer and formed-address envelope applied to observed actual parameters.'},
    'independent_targets': ['KernelOnly.mathematicalTarget', 'KernelOnly.nativeReference'],
    'model_boundaries': [
        'Truthful ordinary two-byte Tensor API metadata, coordinate/capacity/pointer laws and Python type/scalar observations; no result or execution law.',
        'Caller memory stable between admission observations and execution; input aliases share one physical reader.',
        'One fixed scalar/conversion and seeded m16n8k16 instruction model; complete compiler-selected reductions, routing, fusion and accumulator seeds remain visible.',
        'Compiler/source/ISA correspondence and applicability, driver availability and device resources are external runtime conditions.'],
    'deferred_numerical_equalities': [], 'deferred_numerical_experiments': [],
    'new_numerical_transformations': [],
    'excluded_scope': ['host/allocator/copyback implementation', 'unified final byteheap', 'stronger Python guards', 'GPU runtime or resource proof'],
    'shared_grid_product_compatibility': 'Deferred outside this task as scoped; no implementation or guard added.',
    'validation': {
        'standard_base_axioms_only': True, 'independent_targets_non_circular': True,
        'current_Algebra_cache_rfl_probes': 'Passed in KernelOnly/Audit.lean',
        'host_controls': 55, 'address_grids': 7, 'historical_original_counterexamples': 6,
        'host_log_note': 'The unchanged legacy checker says floating instruction validation is deferred. That historical checker wording is superseded by the explicit current one-model theorem and numerical-obligations.json; no experiment is deferred.',
        'actual_binder_SM80_profiles': 10, 'saved_compiler_text_artifacts': 40,
        'precision_profiles': 10, 'CPU_FP16_complete_cases': 4, 'CPU_empty_cases': 6,
        'CPU_BF16_live': 'Unsupported by installed interpreter (get_bf16 absent); original failing script and log retained, no workaround.',
        'GPU_execution': False, 'universal_compiler_conformance_claim': False,
        'within_SM80_shared_memory_limit': 6, 'exceed_SM80_shared_memory_limit': 4,
        'SM80_shared_bytes_per_block': 166912,
        'resource_note': 'Four compiled D128/BM128/BN128/stages4 profiles require 231424/229632 bytes. Resource availability is external, not classified as source/math defect. Even the six within limit have no runtime success claim.'},
    'compiler_artifacts': artifacts, 'profile_applicability': applicability,
    'build_driver': pin(DRIVER), 'run_receipts': json.loads((OUT / 'run-receipts.json').read_text()),
    'evidence': [pin(OUT / name) for name in evidence_names],
    'freezer': pin(__file__), 'verifier': pin(OUT / 'verify.py'),
    'integration': 'Root owns independent review, canonical selection, ledger and Lake. This task edits none of those.'}
(OUT / 'delivery.json').write_text(json.dumps(packet, indent=2, sort_keys=True) + '\n')
print(json.dumps({'packet': pin(OUT / 'delivery.json'), 'task_sources': len(sources),
                  'successful_build_receipts': len(builds), 'public_import_closure': len(public_sources),
                  'audit_import_closure': len(audit_sources), 'preserved_originals': len(preserved)}, indent=2))
