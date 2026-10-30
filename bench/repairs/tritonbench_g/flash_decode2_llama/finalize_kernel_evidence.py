"""Freeze source, public endpoints, resolved imports and targeted build receipts."""
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

source = TASK/'flash_decode2_llama_repaired.py'
assert sha(source) == '2d8a58c053404bcc7b8b58117aaea53cb4eed4742ddcadc90c8eaa6075e34c85'
entries = [TASK/n for n in ['Correct.lean','FPEquiv.lean','Proofs/KernelAudit.lean']]
queue, closure = entries.copy(), {}
while queue:
    path = queue.pop()
    if path in closure: continue
    imports = []
    for line in path.read_text().splitlines():
        if not line.startswith('import '): continue
        for module in line[7:].split():
            if module.startswith('--'): break
            imported = ROOT/(module.replace('.','/')+'.lean')
            if imported.is_file():
                imports.append(str(imported.relative_to(ROOT)))
                queue.append(imported)
    rel = path.relative_to(ROOT)
    obj = next((p for p in [OVERLAY/rel.with_suffix('.olean'),BASE/rel.with_suffix('.olean')] if p.is_file()),None)
    assert obj, ('missing imported object',rel)
    receipt = OVERLAY/rel.with_suffix('.workspace-build.json')
    if receipt.is_file():
        r = json.loads(receipt.read_text())
        assert r['exit_code']==0 and r['source_sha256']==sha(path), ('stale imported source',rel)
        assert r['objects'].get(str(obj))==sha(obj), ('stale imported object',rel)
    closure[path] = {**pin(path),'imports':imports,'olean':str(obj), 'olean_sha256':sha(obj),
        'olean_resolved':str(obj.resolve()),'workspace_receipt':pin(receipt) if receipt.is_file() else None}
assert not any('/minimum/' in row['path'] or '/Historical' in row['path'] for row in closure.values())
shared = ['Kernels']+['Proofs/'+n for n in ['SourceConnection','Execution','Grid','Geometry','Reference',
    'Native','Host','Coordinates','NativeHost','Admission','Physical','TensorBinding','Bounds','NativeSafety','Numerical']]
local = ['Contract','Kernels','Correct','FPEquiv']+['Proofs/'+n for n in ['KernelModel','KernelTensor',
    'KernelNative','KernelReal','KernelEndpoints','KernelSafety','KernelOrder','KernelAudit']]
required = [TASK/(n+'.lean') for n in local]+[TASK.parent/'flash_decode2_phi'/(n+'.lean') for n in shared]
receipts = []
for path in required:
    receipt = OVERLAY/path.relative_to(ROOT).with_suffix('.workspace-build.json')
    data = json.loads(receipt.read_text())
    assert data['exit_code']==0 and data['source_sha256']==sha(path), ('stale receipt',path)
    for object_path,object_hash in data['objects'].items(): assert sha(Path(object_path))==object_hash,object_path
    receipts.append({**pin(receipt),'source':str(path.relative_to(ROOT)),'source_sha256':sha(path),'lean_path':data['lean_path']})
profiles = json.loads((OUT/'profiles.json').read_text())
assert len(profiles['results'])==20 and profiles['gpu_execution'] is False
for row in profiles['results']:
    assert row['status']=='compiled'
    for a in row['artifacts']: assert sha(Path(a['path']))==a['sha256']
audit = json.loads((OUT/'source-and-compiler-audit.json').read_text())
assert audit['source_sha256']==sha(source) and len(audit['compiler_checks'])==20
assert '11 source/summary/tail/empty/alias cases and 30 rejections passed' in (OUT/'cpu-regression.log').read_text()
manifest = {'entries':[str(p.relative_to(ROOT)) for p in entries], 'workspace_module_count':len(closure),
            'files':sorted(closure.values(),key=lambda x:x['path'])}
(OUT/'import-closure.json').write_text(json.dumps(manifest,indent=2)+'\n')
(OUT/'build-receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
source_pins = [pin(p) for p in sorted(TASK.rglob('*')) if p.is_file() and OUT not in p.parents
               and p.suffix in ('.lean','.py','.md')]
(OUT/'source-pins.json').write_text(json.dumps(source_pins,indent=2)+'\n')
driver_record = ROOT/'.veritile-cache/build-tools/local-first-resolution.json'
driver = json.loads(driver_record.read_text())
assert sha(Path(driver['original_driver']))==driver['original_sha256']
assert sha(ROOT/driver['new_driver'])==driver['new_sha256']
packet = dict(task='flash_decode2_llama',status='kernel_proved',canonical_python=pin(source),
    canonical_python_and_guards_changed=False,
    authoritative_quotation={'file':pin(TASK.parent/'flash_decode2_phi/Kernels.lean'),'declaration':'VeriTile.Bench.Repairs.FlashDecode2.llama'},
    public_entries={'real':'VeriTile.Bench.Repairs.FlashDecode2Llama.flash_decode2_llama_correct',
                    'native':'VeriTile.Bench.Repairs.FlashDecode2Llama.flash_decode2_llama_fp32'},
    closure=['complete source quotation success','independent arbitrary-real weighted-summary formula',
             'independent exact native scalar recurrence','actual Tensor pointers and logical input observations',
             'every batch/head/lane output physical index','full tagged-memory frame','complete grid permutation',
             'empty grid','signed INT32 length and INT64 index arithmetic','active storage pointer bounds'],
    numerical_deferrals=0,
    model_boundaries=['six total raw-word scalar primitive callbacks',
        'two fixed compiler patterns selected by maximal sequence block; no full compiler-correctness theorem',
        'ideal-real observations are a separate mathematical interpretation with arbitrary real input values'],
    scope='Kernel-only. No new Host/allocator/returned-object verification or unified final-byteheap gate. Grid-product runtime compatibility remains deferred.',
    native_bit_patterns='All FP32 encodings; no finite-input restriction in the native theorem.',
    compiler_evidence=pin(OUT/'profiles.json'),source_and_compiler_audit=pin(OUT/'source-and-compiler-audit.json'),
    preflight=pin(OUT/'preflight.json'),
    compiler_profiles='20 actual wrapper/JIT-binder/default Triton 3.7.1 SM80 profiles; no GPU execution.',
    regression=pin(OUT/'cpu-regression.log'),gpu_execution=False,native_numerical_experiment=False,
    cpu_ast_regression=True,
    import_closure=pin(OUT/'import-closure.json'),build_receipts=pin(OUT/'build-receipts.json'),
    source_pins=pin(OUT/'source-pins.json'),required_fresh_receipts=len(receipts),
    shared_source_edited=False,local_import_compatibility='SquareMatrix.block import moved from Kernels to Contract; resolved only in Llama/Contract.',
    driver_resolution=pin(driver_record),final_build_log=pin(OUT/'final-build.log'),
    notes=['Historical public host wrappers are preserved outside the default public import closure.',
           'Unchanged Phi helper objects were refreshed to repair stale structural Algebra layouts.',
           'Primitive models, source AST replay and compiler observations are not GPU numerical validation.'])
(OUT/'delivery.json').write_text(json.dumps(packet,indent=2)+'\n')
print(json.dumps({'delivery':str(OUT/'delivery.json'),'sha256':sha(OUT/'delivery.json'),
                  'workspace_modules':len(closure),'fresh_receipts':len(receipts)}))
