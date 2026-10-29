"""Freeze current aligned attention kernel proofs and their existing evidence, without reruns."""
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
CACHE = ROOT / '.veritile-cache/kernel-build/lib/lean'
DRIVER = ROOT / '.veritile-cache/build-tools/workspace_lean_local_first_v2.py'
EXPECTED_SOURCE = '365cda104f62655345ad002a7d2815fba0d3c812e20c5a569e9b0214c92c85ea'
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
assert preservation['original_file_count'] == len(preservation['files']) == 92
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
assert pin(TASK / 'attention_kernel_aligned_repaired.py')['sha256'] == EXPECTED_SOURCE

sources = sorted(p for p in TASK.rglob('*') if p.is_file() and OUT not in p.parents
                 and p.suffix in {'.lean', '.py', '.md', '.json'})
modules = [p for p in sources if p.suffix == '.lean']
assert len(modules) == 72
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
    'scope': 'Exact repository source import closure and effective objects in receipt LEAN_PATH. Core/package caches reused; 72 listed task modules have source-current receipts.',
    'public_roots': [str(p) for p in public], 'audit_root': str(audit_root),
    'public_source_count': len(public_sources), 'audit_source_count': len(audit_sources),
    'public_sources': [str(p) for p in sorted(public_sources)], 'lean_path': paths, 'entries': entries}
(OUT / 'source-import-closure.json').write_text(json.dumps(closure_data, indent=2) + '\n')

audit = (OUT / 'final-build.log').read_text()
assert not re.search(r'\berror(?:\([^\n]*\))?:', audit) and 'sorryAx' not in audit
for theorem in ['Correct.kernel_correct', 'FPEquiv.kernel_native', 'Correct.empty_kernel', 'BF16.Correct.kernel_correct', 'BF16.FPEquiv.kernel_native', 'BF16.Correct.empty_kernel']:
    assert theorem + ': axiom footprint ⊆ standard base ✓' in audit
for target in ['mathematicalTarget', 'nativeReference', 'InputContract']:
    assert target + ': definition does not reference' in audit
for module in ['Correct', 'FPEquiv', 'KernelOnly/Audit']:
    assert '"module": "' + str((TASK / (module + '.lean')).relative_to(ROOT)) + '", "exit_code": 0' in audit

round2 = (OUT / 'round2-build.log').read_text()
assert not re.search(r'\berror(?:\([^\n]*\))?:', round2) and 'sorryAx' not in round2
assert 'BF16.KernelOnly.compilerPlan_original_domain: axiom footprint ⊆ standard base ✓' in round2
assert 'BF16.KernelOnly.CompilerPlan: definition does not reference' in round2
for module in ['KernelOnly/BF16Endpoints', 'Correct', 'FPEquiv', 'KernelOnly/Audit']:
    assert '"module": "' + str((TASK / (module + '.lean')).relative_to(ROOT)) + '", "exit_code": 0' in round2
archive = json.loads((OUT / 'round1/archive.json').read_text())
assert pin(OUT / 'round1/delivery.json')['sha256'] == archive['round1_delivery_sha256'] == '45862e2a08dedf66498540b1248c8820326a0aff2fe8121ece73ec45539f0ec1'
round1_files = []
for entry in archive['snapshots']:
    snapshot = pin(entry['snapshot_path'])
    assert snapshot['sha256'] == entry['sha256']
    round1_files.append({'original_path': entry['original_path'], 'snapshot': snapshot})

checked = check_evidence()
evidence_names = [
    'preserved-task-files.json', 'canonical-selection.json', 'original-build-modules.txt',
    'original-build.log', 'kernel-input-build.log', 'kernel-build-2.log', 'kernel-build-3.log',
    'kernel-build-4.log', 'kernel-build-5.log', 'kernel-build-6.log', 'kernel-build-7.log',
    'kernel-build-8.log', 'final-build.log', 'host-interpreter-checks.log', 'precision-check.log',
    'binder-probe.log', 'binder-artifacts/report.json', 'binder-artifacts/source-binding.json',
    'profile-applicability.json', 'source-import-closure.json', 'run-receipts.json',
    'round2-build.log', 'round2-build-sandbox.log', 'independent-review-round1.json']
packet = {
    'task': 'attention_kernel_aligned', 'status': 'ready_for_independent_review',
    'review_round': 2,
    'review_correction': {
        'finding': 'BF16_SERIAL_DOMAIN',
        'change': 'Restored original BF16 serialSource, denominatorSource and conditional serialRouting integer metadata; removed the added nativeSerial=false restriction.',
        'domain_coverage': 'CompilerPlan.ofOriginal constructs a new plan from every original BF16 CompilerBinding. compilerPlan_original_domain proves Nonempty CompilerPlan without restricting either nativeSerial value.',
        'rebuilt_modules': ['KernelOnly/BF16Endpoints', 'Correct', 'FPEquiv', 'KernelOnly/Audit'],
        'unaffected_CPU_and_compiler_evidence_reused': True,
        'superseded_packet': pin(OUT / 'round1/delivery.json'),
        'round1_archive': pin(OUT / 'round1/archive.json'),
        'preserved_round1_files': round1_files},
    'scope': 'Complete kernel-only Contract, independent exact-Real Correct and one-model native FPEquiv for both genuine source precision branches, with actual Tensor outputs and frames.',
    'task_directory': str(TASK), 'selected_source': pin(TASK / 'attention_kernel_aligned_repaired.py'),
    'kernel_source': pin(TASK / 'Kernels.lean'), 'source_and_guards_unchanged': True,
    'source_defect_found': False, 'public_entries': [pin(p) for p in public],
    'task_sources': [pin(p) for p in sources], 'preserved_originals': preserved,
    'original_file_count': 92, 'changed_existing_files': sorted(CHANGED),
    'successful_builds': builds, 'current_task_build_receipt_count': 72,
    'new_Lean_modules': 11, 'original_Lean_modules': 61,
    'repository_source_import_closure': pin(OUT / 'source-import-closure.json'),
    'proved_endpoints': {
        'KernelContract.accepted_iff': 'Acceptance iff complete observed Tensor/launch contract, retaining all original guards, scalar launch word and reversed sorted-axis scan.',
        'KernelContract.kernel_access': 'Actual Tensor coordinate/pointer read binding and finite storage capacities derive all active Q/K/V/bias/Out source access bounds. Original padded signed/unsigned address proofs remain available.',
        'Correct.kernel_correct and BF16.Correct.kernel_correct': 'Direct successful original quoted full-grid execution for any full-grid permutation; every caller output equals independent gathered exact-Real softmax; full tagged and physical word exterior frames with preserved reads.',
        'FPEquiv.kernel_native and BF16.FPEquiv.kernel_native': 'Direct successful original quoted full-grid execution; every output storage word equals the corresponding independent typed native recurrence under one instruction model. Both source QK precision paths remain distinct.',
        'BF16.KernelOnly.compilerPlan_original_domain': 'Every original BF16 compiler binding supplies the current kernel-only plan, including nativeSerial=true with the serial denominator route.',
        'Correct.empty_kernel and BF16.Correct.empty_kernel': 'Empty work, identity direct source execution and unchanged physical caller words for every empty dimension.',
        'KernelOnly.backing_exterior and BF16.KernelOnly.backing_exterior': 'Complete read allocations, including prefixes, gaps and tails, are outside every logical store.'},
    'independent_targets': ['KernelOnly.mathematicalTarget', 'KernelOnly.nativeReference',
                            'BF16.KernelOnly.mathematicalTarget', 'BF16.KernelOnly.nativeReference'],
    'model_boundaries': checked['obligations']['model_boundaries'],
    'deferred_numerical_equalities': [], 'deferred_numerical_experiments': [],
    'new_numerical_transformations': [],
    'excluded_scope': ['host/allocator/copyback implementation', 'unified final byteheap', 'stronger Python guards', 'GPU runtime or resource proof'],
    'shared_grid_product_compatibility': 'Deferred outside this task as scoped; no implementation or guard added.',
    'validation': {
        'standard_base_axioms_only': True, 'independent_targets_non_circular': True,
        'current_Algebra_cache_rfl_probes': 'Both native FP32 load probes passed in KernelOnly/Audit.lean',
        'host_admissions': 1042, 'host_rejections': 43, 'configuration_family_size': 1024,
        'admitted_configuration_pairs': 1004, 'CPU_FP16_complete_grids': 8,
        'historical_original_output_head_counterexample': 'Reproduced; current source passes',
        'actual_binder_SM80_profiles': 8, 'saved_compiler_text_artifacts': 32,
        'precision_profiles': 8, 'BF16_CPU_numeric_execution': False, 'GPU_execution': False,
        'universal_compiler_conformance_claim': False, 'within_SM80_shared_memory_limit': 8,
        'host_log_note': 'The unchanged historical checker prints deferred instruction validation. Current one-model theorems classify instruction/compiler correspondence as explicit model boundaries; no numerical experiment remains deferred.',
        'runtime_note': 'All eight compiled profiles are below 166912 shared bytes per block; this alone is not a runtime-success claim. The existing Python resource guard is unchanged.'},
    'compiler_artifacts': checked['artifacts'], 'profile_applicability': checked['applicability'],
    'build_driver': pin(DRIVER), 'run_receipts': json.loads((OUT / 'run-receipts.json').read_text()),
    'evidence': [pin(OUT / name) for name in evidence_names],
    'freezer': pin(__file__), 'verifier': pin(OUT / 'verify.py'), 'evidence_checker': pin(OUT / 'evidence_check.py'),
    'integration': 'Root owns independent review, canonical selection, ledger and Lake. No modification of these or the frozen attention_kernel task.'}
(OUT / 'delivery.json').write_text(json.dumps(packet, indent=2, sort_keys=True) + '\n')
print(json.dumps({'packet': pin(OUT / 'delivery.json'), 'task_sources': len(sources),
                  'successful_build_receipts': len(builds), 'public_import_closure': len(public_sources),
                  'audit_import_closure': len(audit_sources), 'preserved_originals': len(preserved)}, indent=2))
