"""Pin the completed kernel proof and existing regression/compiler evidence."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[4]
TASK = Path(__file__).resolve().parent
EVIDENCE = TASK / 'kernel-evidence'
OVERLAY = ROOT / '.veritile-cache/kernel-build/lib/lean'
BASE = ROOT / '.lake/build/lib/lean'
EXPECTED_SOURCE = 'c384a0bfe3d62579e9e4584a8e03d52bad3957c6790f55e30d493fb7f0350874'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pin(path):
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha(path)}


driver_evidence = ROOT / '.veritile-cache/build-tools/local-first-resolution.json'
driver_record = json.loads(driver_evidence.read_text())
assert sha(Path(driver_record['original_driver'])) == driver_record['original_sha256']
assert sha(ROOT / driver_record['new_driver']) == driver_record['new_sha256']
source = TASK / 'rotary_transform_ops_repaired.py'
assert sha(source) == EXPECTED_SOURCE
entries = [TASK / name for name in ('Correct.lean', 'FPEquiv.lean', 'Proofs/KernelAudit.lean')]
queue = entries.copy()
closure = {}
while queue:
    path = queue.pop()
    if path in closure:
        continue
    imports = []
    for line in path.read_text().splitlines():
        if not line.startswith('import '):
            continue
        for module in line[7:].split():
            if module.startswith('--'):
                break
            imported = ROOT / (module.replace('.', '/') + '.lean')
            if imported.is_file():
                imports.append(str(imported.relative_to(ROOT)))
                queue.append(imported)
    rel = path.relative_to(ROOT)
    objects = [base / rel.with_suffix('.olean') for base in (OVERLAY, BASE)]
    obj = next((obj for obj in objects if obj.is_file()), None)
    assert obj is not None, ('missing imported object', rel)
    receipt = OVERLAY / rel.with_suffix('.workspace-build.json')
    if receipt.is_file():
        record = json.loads(receipt.read_text())
        assert record['exit_code'] == 0 and record['source_sha256'] == sha(path), ('stale imported source', rel)
        assert record['objects'].get(str(obj)) == sha(obj), ('stale imported object', rel)
    closure[path] = {**pin(path), 'imports': imports,
                     'olean': str(obj), 'olean_sha256': sha(obj),
                     'olean_resolved': str(obj.resolve()),
                     'workspace_receipt': pin(receipt) if receipt.is_file() else None}
assert not any('/minimum/' in item['path'] for item in closure.values())
assert not any('/HistoricalPublic' in item['path'] for item in closure.values())

required = ['KernelSource', 'Correct', 'FPEquiv'] + ['Proofs/' + name for name in (
    'KernelAlgebra', 'ProgramModel', 'ProgramGeometry', 'ProgramScatter', 'ProgramResult',
    'ProgramCorrespondence', 'ProgramOrder', 'KernelTensor', 'KernelIndex', 'TypedSetup',
    'TypedLoads', 'TypedStores', 'TypedControl', 'TypedExecution', 'TypedResult', 'TypedGrid',
    'KernelEndpoints', 'KernelAudit')]
receipts = []
for name in required:
    path = TASK / (name + '.lean')
    receipt = OVERLAY / path.relative_to(ROOT).with_suffix('.workspace-build.json')
    data = json.loads(receipt.read_text())
    assert data['exit_code'] == 0 and data['source_sha256'] == sha(path), ('stale receipt', path)
    for object_path, object_hash in data['objects'].items():
        assert sha(Path(object_path)) == object_hash, ('object drift', object_path)
    receipts.append({**pin(receipt), 'source': str(path.relative_to(ROOT)),
                     'source_sha256': sha(path), 'lean_path': data['lean_path']})

profiles = json.loads((EVIDENCE / 'profiles.json').read_text())
assert profiles['gpu_execution'] is False and len(profiles['results']) == 16
for result in profiles['results']:
    assert result['status'] == 'compiled' and result['source_sha256'] == EXPECTED_SOURCE
    for artifact in result['artifacts']:
        assert sha(Path(artifact['path'])) == artifact['sha256']

manifest = {'entries': [str(path.relative_to(ROOT)) for path in entries],
            'workspace_module_count': len(closure),
            'files': sorted(closure.values(), key=lambda item: item['path'])}
(EVIDENCE / 'import-closure.json').write_text(json.dumps(manifest, indent=2) + '\n')
(EVIDENCE / 'build-receipts.json').write_text(json.dumps(receipts, indent=2) + '\n')
source_pins = [pin(path) for path in sorted(TASK.rglob('*'))
               if path.is_file() and EVIDENCE not in path.parents
               and path.suffix in ('.lean', '.py', '.md')]
(EVIDENCE / 'source-pins.json').write_text(json.dumps(source_pins, indent=2) + '\n')
packet = {
    'task': 'rotary_transform_ops',
    'status': 'kernel_proved',
    'canonical_python': pin(source),
    'canonical_python_and_guards_changed': False,
    'authoritative_quotation': pin(TASK / 'KernelSource.lean'),
    'public_entries': {
        'real_tensor': 'VeriTile.Bench.Repairs.RotaryTransformOps.rotary_transform_ops_correct',
        'real_arbitrary': 'VeriTile.Bench.Repairs.RotaryTransformOps.rotary_transform_ops_correct_real',
        'native_fp32': 'VeriTile.Bench.Repairs.RotaryTransformOps.rotary_transform_ops_fp32'},
    'closure': ['complete typed source success', 'independent matrix memory equality',
                'all logical output coordinates', 'full tagged-memory frame',
                'program-order permutation', 'empty and zero-rotary no-launch',
                'original signed metadata storage', 'actual Tensor input pointers'],
    'numerical_deferrals': 0,
    'model_boundaries': ['four total raw-word PTX primitive callbacks',
                         'fixed default compiler/FMA pattern supported by the recorded profiles',
                         'ordinary suffix-copy kernel-entry invariant'],
    'scope': 'Kernel proof only. No new Host/allocator/returned-object verification or unified final-byteheap gate.',
    'native_bit_patterns': 'All raw FP32 encodings, including exceptional words.',
    'compiler_evidence': pin(EVIDENCE / 'profiles.json'),
    'compiler_profiles': '16 actual wrapper/JIT-binder/default Triton 3.7.1 SM80 profiles; no GPU execution.',
    'regression_provenance': pin(EVIDENCE / 'regression-provenance.json'),
    'import_closure': pin(EVIDENCE / 'import-closure.json'),
    'build_receipts': pin(EVIDENCE / 'build-receipts.json'),
    'task_source_pins': pin(EVIDENCE / 'source-pins.json'),
    'required_fresh_receipts': len(receipts),
    'final_build_log': pin(EVIDENCE / 'final-build.log'),
    'build_warnings': 'Tensor binding and formed-pointer context arguments are unused by the tagged-memory execution theorem; their read/index laws are independently exposed. No final physical-byteheap closure is claimed.',
    'driver_resolution_evidence': pin(driver_evidence),
    'drivers': [pin(ROOT / '.veritile-cache/build-tools/workspace_lean_local_first.py'),
                {'path': '/tmp/veritile_workspace_lean.py', 'sha256': sha(Path('/tmp/veritile_workspace_lean.py'))}],
    'notes': ['Earlier numerical/host wrappers remain historical and are not the public endpoint.',
              'CPU replay provenance was carried from tool session 89261 without rerunning unchanged Python.',
              'Compiler profiles and numerical tests are not substituted for Lean proofs.']}
(EVIDENCE / 'delivery.json').write_text(json.dumps(packet, indent=2) + '\n')
print(json.dumps({'delivery': str(EVIDENCE / 'delivery.json'),
                  'sha256': sha(EVIDENCE / 'delivery.json'),
                  'workspace_modules': len(closure), 'fresh_receipts': len(receipts)}))
