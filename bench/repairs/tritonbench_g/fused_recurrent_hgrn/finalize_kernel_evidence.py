"""Freeze the HGRN kernel-only delivery after successful targeted checks."""
from pathlib import Path
import hashlib
import json

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[3]
OUT = TASK/'kernel-evidence'
OVERLAY = ROOT/'.veritile-cache/kernel-build/lib/lean'
BASE = ROOT/'.lake/build/lib/lean'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def pin(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path))

source = TASK/'fused_recurrent_hgrn_repaired.py'
assert sha(source) == 'f920993bd8b61bc71450ab7a1775ab8dc4c350435e81b83728834df1af922cde'
historical = json.loads((OUT/'historical-source-pins.json').read_text())
for row in historical['files']:
    path = ROOT/row['path']
    if path.name == 'README.md':
        continue
    if path.name in ('Correct.lean','FPEquiv.lean'):
        path = TASK/'Proofs'/('Historical'+path.name)
    assert sha(path) == row['sha256'], ('historical content changed',path)

entries = [TASK/n for n in ('Correct.lean','FPEquiv.lean','Proofs/KernelAudit.lean')]
queue, closure = entries.copy(), {}
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
            imported = ROOT/(module.replace('.','/')+'.lean')
            if imported.is_file():
                imports.append(str(imported.relative_to(ROOT)))
                queue.append(imported)
    rel = path.relative_to(ROOT)
    obj = next((p for p in (OVERLAY/rel.with_suffix('.olean'),BASE/rel.with_suffix('.olean')) if p.is_file()),None)
    assert obj, ('missing object',rel)
    receipt = OVERLAY/rel.with_suffix('.workspace-build.json')
    if receipt.is_file():
        record = json.loads(receipt.read_text())
        assert record['exit_code'] == 0 and record['source_sha256'] == sha(path), ('stale imported source',rel)
        assert record['objects'].get(str(obj)) == sha(obj), ('stale imported object',rel)
    closure[path] = {**pin(path), 'imports':imports, 'olean':str(obj), 'olean_sha256':sha(obj),
        'olean_resolved':str(obj.resolve()), 'workspace_receipt':pin(receipt) if receipt.is_file() else None}
assert not any('/minimum/' in row['path'] or '/Historical' in row['path'] for row in closure.values())
for forbidden in ('Host','NativeHost','Closed','Wrapper','StorageBinding','RealResult','FPResult','ComputedGradient'):
    assert TASK/'Proofs'/(forbidden+'.lean') not in closure

names = ['Contract','Kernels','Correct','FPEquiv']+['Proofs/'+n for n in (
    'Base','Reference','Execution','ForwardLoop','Forward','BackwardExecution','BackwardLoop',
    'Backward','Grid','WholeGrid','Whole','Gradient','Native','Scalar','Bounds','Physical',
    'Ownership','Safety','KernelTensor','KernelEndpoints','KernelAudit')]
required = [TASK/(n+'.lean') for n in names]+[
    ROOT/'bench/tritonbench_g/fused_recurrent_hgrn/FusedRecurrentHgrn.lean',
    ROOT/'bench/tritonbench_g/fused_recurrent_hgrn/Proofs/FPStages.lean']
receipts = []
for path in required:
    receipt = OVERLAY/path.relative_to(ROOT).with_suffix('.workspace-build.json')
    record = json.loads(receipt.read_text())
    assert record['exit_code'] == 0 and record['source_sha256'] == sha(path), ('stale receipt',path)
    for object_path, object_hash in record['objects'].items():
        assert sha(Path(object_path)) == object_hash, object_path
    receipts.append({**pin(receipt),'source':str(path.relative_to(ROOT)),
        'source_sha256':sha(path),'lean_path':record['lean_path']})
audit = json.loads((OUT/'source-and-compiler-audit.json').read_text())
assert audit['source_sha256'] == sha(source) and audit['quotation_sha256'] == sha(TASK/'Kernels.lean')
assert len(audit['compiler_checks']) == 64 and all(x['complete_body_equal'] for x in audit['complete_source_ast'])
for file, count in (('profiles.json',56),('loop-profiles.json',8)):
    profiles = json.loads((OUT/file).read_text())
    assert len(profiles['results']) == count and profiles['gpu_execution'] is False
    for row in profiles['results']:
        assert row['status'] == 'compiled' and row['source_sha256'] == sha(source)
        for artifact in row['artifacts']:
            assert sha(Path(artifact['path'])) == artifact['sha256']
assert '150 complete CPU paths' in (OUT/'cpu-regression.log').read_text()
assert '11 prelaunch rejections' in (OUT/'cpu-regression.log').read_text()
assert 'error:' not in (OUT/'final-build.log').read_text()

manifest = dict(entries=[str(p.relative_to(ROOT)) for p in entries], workspace_module_count=len(closure),
    files=sorted(closure.values(),key=lambda x:x['path']))
(OUT/'import-closure.json').write_text(json.dumps(manifest,indent=2)+'\n')
(OUT/'build-receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
sources = [pin(p) for p in sorted(TASK.rglob('*')) if p.is_file() and OUT not in p.parents
           and p.suffix in ('.lean','.py','.md')]
(OUT/'source-pins.json').write_text(json.dumps(sources,indent=2)+'\n')
driver = ROOT/'.veritile-cache/build-tools/workspace_lean_local_first_v2.py'
assert sha(driver) == '23464979d2a667dcebe68ab9ce1c958238ff1bd322c5a010b8989897a406da22'
packet = dict(task='fused_recurrent_hgrn',status='kernel_proved',canonical_python=pin(source),
    canonical_python_and_guards_changed=False,
    authoritative_quotation=pin(TASK/'Kernels.lean'),
    public_entries=dict(real='VeriTile.Bench.Repairs.FusedRecurrentHgrn.fused_recurrent_hgrn_correct',
                       native='VeriTile.Bench.Repairs.FusedRecurrentHgrn.fused_recurrent_hgrn_fp32'),
    closure=['complete forward/backward readable source and normalized AST correspondence',
        'independent arbitrary-Real recurrence and directional derivative with detached seed',
        'exact native add/multiply/FMA source graph with explicit software instruction laws',
        'actual FP32 Tensor pointers, nonzero-offset coordinates, input and result logical observations',
        'all grid programs, phase frames, coverage, unique writers and physical separation',
        'signed32 source arithmetic and u64 formed pointers, including padded/final endpoints',
        'T=0 and all original autotune BD/warp choices'],
    numerical_deferrals=0,
    model_boundaries=['three scalar raw-word instruction laws against concrete RNE FP32 software operations',
        'value-independent compiler schedule; observed contracted and looped families are recorded explicitly',
        'ordinary actual-Tensor metadata and coordinate laws',
        'arbitrary-Real semantics are independent from raw FP32 word interpretation'],
    scope='Kernel-only. No new Host, allocator, returned-object or unified final-byteheap gate. Grid-product runtime compatibility remains outside this scope.',
    native_bit_patterns='All FP32 encodings; no finite-input restriction.',
    historical_pins=pin(OUT/'historical-source-pins.json'),preflight=pin(OUT/'PREFLIGHT.md'),
    source_and_compiler_audit=pin(OUT/'source-and-compiler-audit.json'),
    compiler_profiles=[pin(OUT/'profiles.json'),pin(OUT/'loop-profiles.json')],compiler_profile_count=64,
    regression=pin(OUT/'cpu-regression.log'),gpu_execution=False,native_numerical_experiment=False,
    import_closure=pin(OUT/'import-closure.json'),build_receipts=pin(OUT/'build-receipts.json'),
    source_pins=pin(OUT/'source-pins.json'),required_fresh_receipts=len(receipts),
    shared_source_edited=False,driver=pin(driver),driver_adoption=pin(ROOT/'.veritile-cache/build-tools/v2-adoption.json'),
    final_build_log=pin(OUT/'final-build.log'),
    notes=['Two unchanged historical shared helper objects were refreshed after a stale-object declaration conflict.',
        'Historical Host endpoints remain preserved and outside the default public closure.',
        'T=3/BD=128/warps=1 retains the reverse loop; other audited short reverse candidates contract.',
        'No current canonical source defect was found. The historical ignored-final-cotangent defect remains reproduced.'])
(OUT/'delivery.json').write_text(json.dumps(packet,indent=2)+'\n')
print(json.dumps(dict(delivery=str(OUT/'delivery.json'),sha256=sha(OUT/'delivery.json'),
    workspace_modules=len(closure),fresh_receipts=len(receipts))))
