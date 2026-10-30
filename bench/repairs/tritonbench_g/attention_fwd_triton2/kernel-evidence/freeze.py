"""Freeze task-local sources, proof receipts and completed controls, without reruns."""
from pathlib import Path
import hashlib
import json
import re

OUT=Path(__file__).resolve().parent
TASK=OUT.parent
ROOT=TASK.parents[3]
CACHE=ROOT/'.veritile-cache/kernel-build/lib/lean'
DRIVER=ROOT/'.veritile-cache/build-tools/workspace_lean_local_first_v2.py'
EXPECTED='eb7192d0aaab5db92b98eba026b05eb728774263b5f3afecbd7270d9d84bc005'
CHANGED={str(TASK/n) for n in ['Contract.lean','Correct.lean','FPEquiv.lean','README.md']}

def pin(path):
    p=Path(path).absolute()
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

def imports(path):
    names=[]
    for line in path.read_text().splitlines():
        match=re.match(r'^(?:public\s+)?import\s+(.+)$',line)
        if match:
            names.extend(n for n in match[1].split() if re.fullmatch(r'[A-Za-z0-9_.]+',n))
    return names

def closure(roots):
    seen=set();todo=list(roots)
    while todo:
        source=todo.pop()
        if source in seen:continue
        seen.add(source)
        for name in imports(source):
            dependency=ROOT/(name.replace('.','/')+'.lean')
            if dependency.is_file():todo.append(dependency)
    return seen

preservation=json.loads((OUT/'preserved-task-files.json').read_text())
assert preservation['original_file_count']==len(preservation['files'])==45
preserved=[];changed=set()
for item in preservation['files']:
    source=pin(ROOT/item['path']);snapshot=pin(ROOT/item['snapshot'])
    assert snapshot['sha256']==item['sha256']
    if source['sha256']!=snapshot['sha256']:changed.add(source['path'])
    preserved.append({'before':snapshot,'current':source,'current_changed':source['sha256']!=snapshot['sha256']})
assert changed==CHANGED,changed
original_contract=OUT/'preserved'/ (TASK/'Contract.lean').relative_to(ROOT)
assert (TASK/'KernelOnly/Admission.lean').read_bytes()==original_contract.read_bytes()
for name in ['Correct.lean','FPEquiv.lean']:
    old=(OUT/'preserved'/(TASK/name).relative_to(ROOT)).read_text()
    assert old in (TASK/name).read_text()
assert pin(TASK/'Whole001/attention_fwd_triton2_repaired.py')['sha256']==EXPECTED
sources=sorted(p for p in TASK.rglob('*') if p.is_file() and OUT not in p.parents and p.suffix in {'.lean','.py','.md','.json'})
modules=[p for p in sources if p.suffix=='.lean']
assert len(modules)==38
builds=[]
for source in modules:
    receipt=CACHE/source.relative_to(ROOT).with_suffix('.workspace-build.json')
    data=json.loads(receipt.read_text())
    assert data['exit_code']==0 and data['source']==str(source.relative_to(ROOT))
    assert data['source_sha256']==pin(source)['sha256'],source
    artifacts=[]
    for path,digest in data['objects'].items():
        artifact=pin(path);assert artifact['sha256']==digest,path;artifacts.append(artifact)
    builds.append({'source':pin(source),'receipt':pin(receipt),'artifacts':artifacts,'lean_path':data['lean_path']})
paths=builds[0]['lean_path'];assert all(b['lean_path']==paths for b in builds)
public=[TASK/n for n in ['Contract.lean','Correct.lean','FPEquiv.lean']]
audit_root=TASK/'KernelOnly/Audit.lean'
public_sources=closure(public);audit_sources=closure([audit_root]);assert public_sources<=audit_sources
entries=[]
for source in sorted(audit_sources):
    relative=source.relative_to(ROOT).with_suffix('.olean')
    effective=next(Path(directory)/relative for directory in paths if (Path(directory)/relative).is_file())
    ilean=effective.with_suffix('.ilean');idat=json.loads(ilean.read_text())
    declared=imports(source);actual=[entry[0] for entry in idat['directImports']]
    assert declared==actual,(source,declared,actual)
    assert idat['module']=='.'.join(source.relative_to(ROOT).with_suffix('').parts)
    entries.append({'source':pin(source),'effective_object':pin(effective),'effective_ilean':pin(ilean),
                    'source_direct_imports':declared,'ilean_direct_imports':actual,'fresh_task_receipt':source in modules})
closure_data={'scope':'Exact repository source import closure and effective olean/ilean objects under receipt LEAN_PATH; directImports checked against current source. Core/package caches reused; 38 task modules have current source-hash build receipts.',
    'public_roots':[str(p) for p in public],'audit_root':str(audit_root),'public_source_count':len(public_sources),
    'audit_source_count':len(audit_sources),'public_sources':[str(p) for p in sorted(public_sources)],'lean_path':paths,'entries':entries}
(OUT/'source-import-closure.json').write_text(json.dumps(closure_data,indent=2)+'\n')
audit=(OUT/'kernel-final-build-02.log').read_text()
assert not re.search(r'\berror(?:\([^\n]*\))?:',audit) and 'sorryAx' not in audit
for theorem in ['Correct.kernel_correct','FPEquiv.kernel_native','Correct.empty_kernel']:
    assert theorem+': axiom footprint ⊆ standard base ✓' in audit
for target in ['mathematicalTarget','nativeReference','InputContract']:
    assert target+': definition does not reference' in audit
for module in ['KernelOnly/Endpoints','Contract','Correct','FPEquiv','KernelOnly/Audit']:
    assert '"module": "'+str((TASK/(module+'.lean')).relative_to(ROOT))+'", "exit_code": 0' in audit
binder=json.loads((OUT/'binder-report.json').read_text())
precision=json.loads((OUT/'source-binding.json').read_text())
interpreter=json.loads((OUT/'interpreter-report.json').read_text())
obligations=json.loads((TASK/'numerical-obligations.json').read_text())
assert binder['source_sha256']==interpreter['source_sha256']==EXPECTED
assert precision['status']=='PASS' and len(precision['profiles'])==5
compiled=[r for r in binder['results'] if r['status']=='compiled']
assert len(compiled)==5 and sum(r['status']=='empty_skip' for r in binder['results'])==3
artifacts=[a for r in compiled for a in r['artifacts'].values()]
assert len(artifacts)==20
for artifact in artifacts:assert pin(artifact['path'])==artifact
assert not binder['GPU_execution'] and not interpreter['GPU_execution']
assert interpreter['complete_nonempty_cases']==4 and interpreter['empty_cases']==3 and len(interpreter['cases'])==7
assert all(c['actual_returned_tensor'] and c['all_input_backing_frames'] and c['unused_M_frame'] for c in interpreter['cases'])
assert obligations['deferred_numerical_equalities']==obligations['deferred_numerical_experiments']==[]
assert json.loads((OUT/'host-checks.log').read_text())=={'status':'PASS','controls':41,'source_structure':'PASS','GPU_execution':False}
evidence_names=['preserved-task-files.json','canonical-selection.json','original-build-modules.txt','run-receipts.json',
 'binder_probe.py','binder-report.json','interpreter_probe.py','interpreter-report.json','source_binding.py','source-binding.json','source-import-closure.json']
evidence_names+=sorted(p.name for p in OUT.glob('*.log'))
packet={'task':'attention_fwd_triton2','status':'ready_for_independent_review',
 'scope':'Complete kernel-only Contract, exact-Real Correct and one-model native FPEquiv with actual Tensor outputs and frames.',
 'task_directory':str(TASK),'selected_source':pin(TASK/'Whole001/attention_fwd_triton2_repaired.py'),
 'kernel_source':pin(TASK/'Whole001/Kernels.lean'),'source_and_guards_unchanged':True,'source_defect_found':False,
 'public_entries':[pin(p) for p in public],'task_sources':[pin(p) for p in sources],
 'preserved_originals':preserved,'original_file_count':45,'changed_existing_files':sorted(CHANGED),
 'old_admission_byte_identical':True,'original_public_theorem_bodies_preserved':True,
 'successful_builds':builds,'current_task_build_receipt_count':38,'original_Lean_modules':31,'new_Lean_modules':7,
 'repository_source_import_closure':pin(OUT/'source-import-closure.json'),
 'proved_endpoints':{
  'KernelContract.accepted_iff':'Python-aligned acceptance over actual Tensor observations; ordinary allocation/index laws remain separate.',
  'Correct.kernel_correct':'Complete quoted source grid succeeds; every logical output is the independent Real normalized base2 attention; source/physical/input frames, access safety and machine closure.',
  'FPEquiv.kernel_native':'Complete quoted source grid succeeds; every actual BF16 output word equals independent state-free graph under one fixed instruction model, with frames for every owner.',
  'Correct.empty_kernel':'Empty grid, identity source run and unchanged observations.',
  'KernelOnly.actual_address/scale_read':'Original Tensor storageOffset/physicalIndex to actual kernel pointers, including both packed 3D scales.',
  'KernelOnly.backing_exterior':'Full original read allocations, including prefix/tail cells, lie outside output stores.'},
 'independent_targets':['KernelOnly.mathematicalTarget','KernelOnly.nativeReference'],
 'model_boundaries':obligations['model_boundaries'],'deferred_numerical_equalities':[],'deferred_numerical_experiments':[],
 'new_numerical_transformations':[],'excluded_scope':obligations['excluded_scope'],
 'shared_grid_product_compatibility':'Deferred outside this task as scoped; no source/guard implementation change.',
 'validation':{'standard_base_axioms_only':True,'independent_targets_non_circular':True,'current_Algebra_cache_rfl_probes':True,
  'host_controls':41,'actual_wrapper_default_binder_compiled_SM80_profiles':5,'binder_empty_skips':3,
  'saved_compiler_text_artifacts':20,'CPU_complete_nonempty_cases':4,'CPU_empty_cases':3,
  'full_input_backing_frames':True,'unused_M_frame':True,'GPU_execution':False,
  'universal_compiler_conformance_claim':False,'compiler_shared_bytes':[r['metadata']['shared'] for r in compiled],
  'resource_note':'Metadata reports 98304 or 131076 shared bytes. Resource availability and runtime success remain external; no GPU execution performed.'},
 'compiler_artifacts':artifacts,'build_driver':pin(DRIVER),'run_receipts':json.loads((OUT/'run-receipts.json').read_text()),
 'evidence':[pin(OUT/n) for n in evidence_names],'freezer':pin(__file__),'verifier':pin(OUT/'verify.py'),
 'integration':'Root owns independent review and canonical ledger/Lake integration. This task edits none of those.'}
(OUT/'delivery.json').write_text(json.dumps(packet,indent=2,sort_keys=True)+'\n')
print(json.dumps({'packet':pin(OUT/'delivery.json'),'task_sources':len(sources),'successful_build_receipts':len(builds),
 'public_import_closure':len(public_sources),'audit_import_closure':len(audit_sources),'preserved_originals':len(preserved)},indent=2))
