"""Read-only verification of the complete attention_fwd_triton2 delivery."""
from pathlib import Path
import hashlib
import json
import re

OUT=Path(__file__).resolve().parent
TASK=OUT.parent
ROOT=TASK.parents[3]
packet=json.loads((OUT/'delivery.json').read_text())
checks=[]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def visit(value):
    if isinstance(value,dict):
        if isinstance(value.get('path'),str) and isinstance(value.get('sha256'),str):
            assert sha(value['path'])==value['sha256'],value['path'];checks.append(value['path'])
        for item in value.values():visit(item)
    elif isinstance(value,list):
        for item in value:visit(item)

def imports(path):
    names=[]
    for line in path.read_text().splitlines():
        match=re.match(r'^(?:public\s+)?import\s+(.+)$',line)
        if match:names.extend(n for n in match[1].split() if re.fullmatch(r'[A-Za-z0-9_.]+',n))
    return names

def closure(roots):
    seen=set();todo=list(map(Path,roots))
    while todo:
        source=todo.pop()
        if source in seen:continue
        seen.add(source)
        for name in imports(source):
            dependency=ROOT/(name.replace('.','/')+'.lean')
            if dependency.is_file():todo.append(dependency)
    return seen

visit(packet)
assert packet['task']=='attention_fwd_triton2' and packet['status']=='ready_for_independent_review'
assert packet['selected_source']['sha256']=='eb7192d0aaab5db92b98eba026b05eb728774263b5f3afecbd7270d9d84bc005'
assert packet['source_and_guards_unchanged'] and not packet['source_defect_found']
assert packet['current_task_build_receipt_count']==len(packet['successful_builds'])==38
assert packet['original_Lean_modules']==31 and packet['new_Lean_modules']==7
assert packet['deferred_numerical_equalities']==packet['deferred_numerical_experiments']==[]
sources={str(p) for p in TASK.rglob('*') if p.is_file() and OUT not in p.parents and p.suffix in {'.lean','.py','.md','.json'}}
assert sources=={p['path'] for p in packet['task_sources']}
module_sources={p['path'] for p in packet['task_sources'] if p['path'].endswith('.lean')}
assert module_sources=={b['source']['path'] for b in packet['successful_builds']}
for build in packet['successful_builds']:
    receipt=json.loads(Path(build['receipt']['path']).read_text())
    assert receipt['source_sha256']==build['source']['sha256']
    assert str(ROOT/receipt['source'])==build['source']['path']
    assert receipt['exit_code']==0 and receipt['lean_path']==build['lean_path']
    assert receipt['objects']=={p['path']:p['sha256'] for p in build['artifacts']}
preservation=json.loads((OUT/'preserved-task-files.json').read_text())
assert preservation['original_file_count']==len(preservation['files'])==len(packet['preserved_originals'])==45
changed=set()
for entry in preservation['files']:
    assert sha(ROOT/entry['snapshot'])==entry['sha256']
    if sha(ROOT/entry['path'])!=entry['sha256']:changed.add(str(ROOT/entry['path']))
assert changed==set(packet['changed_existing_files'])=={str(TASK/n) for n in ['Contract.lean','Correct.lean','FPEquiv.lean','README.md']}
assert (TASK/'KernelOnly/Admission.lean').read_bytes()==(OUT/'preserved'/(TASK/'Contract.lean').relative_to(ROOT)).read_bytes()
for name in ['Correct.lean','FPEquiv.lean']:
    assert (OUT/'preserved'/(TASK/name).relative_to(ROOT)).read_text() in (TASK/name).read_text()
data=json.loads(Path(packet['repository_source_import_closure']['path']).read_text());visit(data)
assert closure(data['public_roots'])==set(map(Path,data['public_sources']))
assert closure([data['audit_root']])=={Path(e['source']['path']) for e in data['entries']}
assert data['public_source_count']==len(data['public_sources']) and data['audit_source_count']==len(data['entries'])
assert all(b['lean_path']==data['lean_path'] for b in packet['successful_builds'])
for entry in data['entries']:
    source=Path(entry['source']['path']);relative=source.relative_to(ROOT).with_suffix('.olean')
    effective=next(Path(directory)/relative for directory in data['lean_path'] if (Path(directory)/relative).is_file())
    assert str(effective)==entry['effective_object']['path']
    assert str(effective.with_suffix('.ilean'))==entry['effective_ilean']['path']
    ilean=json.loads(effective.with_suffix('.ilean').read_text())
    assert imports(source)==[x[0] for x in ilean['directImports']]==entry['source_direct_imports']==entry['ilean_direct_imports']
    assert ilean['module']=='.'.join(source.relative_to(ROOT).with_suffix('').parts)
    assert entry['fresh_task_receipt']==(str(source) in module_sources)
    if str(source) in module_sources:
        build=next(b for b in packet['successful_builds'] if b['source']['path']==str(source))
        assert entry['effective_object'] in build['artifacts'] and entry['effective_ilean'] in build['artifacts']
audit=(OUT/'kernel-final-build-02.log').read_text()
assert not re.search(r'\berror(?:\([^\n]*\))?:',audit) and 'sorryAx' not in audit
for theorem in ['Correct.kernel_correct','FPEquiv.kernel_native','Correct.empty_kernel']:
    assert theorem+': axiom footprint ⊆ standard base ✓' in audit
for target in ['mathematicalTarget','nativeReference','InputContract']:
    assert target+': definition does not reference' in audit
binder=json.loads((OUT/'binder-report.json').read_text());visit(binder)
precision=json.loads((OUT/'source-binding.json').read_text());visit(precision)
cpu=json.loads((OUT/'interpreter-report.json').read_text())
assert binder['source_sha256']==cpu['source_sha256']==packet['selected_source']['sha256']
assert sha(OUT/'binder_probe.py')==binder['script_sha256']
assert len(binder['results'])==8 and len(precision['profiles'])==5 and precision['status']=='PASS'
compiled=[r for r in binder['results'] if r['status']=='compiled']
assert len(compiled)==5 and sum(r['status']=='empty_skip' for r in binder['results'])==3
assert packet['compiler_artifacts']==[a for r in compiled for a in r['artifacts'].values()]
for result,profile in zip(compiled,precision['profiles']):
    assert result['name']==profile['profile'] and result['artifacts']==profile['artifacts']
    assert all(profile['checks'].values()) and not profile['runtime_success_claim']
    assert result['metadata']['shared']==profile['shared_bytes']
assert not binder['GPU_execution'] and not binder['numerical_experiment']
assert not cpu['GPU_execution'] and not cpu['instruction_conformance_experiment']
assert cpu['complete_nonempty_cases']==4 and cpu['empty_cases']==3 and len(cpu['cases'])==7
assert all(c['actual_returned_tensor'] and c['all_input_backing_frames'] and c['unused_M_frame'] and c['output_dtype']=='bfloat16' for c in cpu['cases'])
assert json.loads((OUT/'host-checks.log').read_text())=={'status':'PASS','controls':41,'source_structure':'PASS','GPU_execution':False}
obligations=json.loads((TASK/'numerical-obligations.json').read_text())
assert obligations['deferred_numerical_equalities']==obligations['deferred_numerical_experiments']==[]
assert not obligations['native_equals_real_claim'] and not obligations['whole_lowered_result_hypothesis']
assert json.loads((OUT/'run-receipts.json').read_text())==packet['run_receipts']
print(json.dumps({'all_passed':True,'delivery_sha256':sha(OUT/'delivery.json'),'pin_records':len(checks),'unique_files':len(set(checks)),
 'successful_build_receipts':38,'public_import_closure':data['public_source_count'],'audit_import_closure':data['audit_source_count'],
 'source_ilean_imports_match':True,'original_files_preserved':45,'changed_originals_with_preserved_snapshots':4,
 'host_controls':41,'compiler_profiles':5,'compiler_text_artifacts':20,'CPU_cases':7,'GPU_execution':False},indent=2))
