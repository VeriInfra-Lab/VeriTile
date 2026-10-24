"""Freeze evidence after the final topological direct compile; never modifies W/R."""
import difflib,json,re,sys
from pathlib import Path
sys.dont_write_bytecode=True
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P))
from audit_util import sha,imports
W=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
R=Path('/tmp/veritile-next-value-integration-20261014')
TASK='bench/tritonbench_g/matmul_dequantize_int4'
def write(name,x): (P/name).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
records=json.loads((P/'last-build.json').read_text());assert len(records)==4
assert all(x['exit_code']==0 and x['changed_inputs']==[] for x in records)
roots=[Path(x) for x in records[0]['ordered_roots']]
external={}
def bind(p):external[str(p)]=sha(p);return external[str(p)]
for q in [R/'lean-path.txt',R/'lean-toolchain',R/'lake-manifest.json',R/'lakefile.toml',Path(records[0]['lean'])]:
 if q.is_file():bind(q)
for q in [Path('/tmp/veritile-final-workspace-validation-f3-20261015/integration-build-receipt.json')]:
 if q.is_file():bind(q)
old={}
for s in sorted((W/TASK).rglob('*')):
 if s.is_file() and s.suffix in ('.lean','.py','.md'):
  rel=str(s.relative_to(W));old[rel]=bind(s)
  oldR=R/rel
  if s.suffix=='.lean':assert oldR.is_file() and sha(oldR)==sha(s),(rel,'old W/R differs')
write('original-task-inputs.json',{'input_sha256':old,'old_lean_W_equals_frozen_R':True,
 'all_old_files_unchanged_by_this_candidate':True})
native=W/'bench/review-checkpoints/2026-10-15/dequant-int4-compiled-graph'
native_sha={str(s.relative_to(W)):bind(s) for s in sorted(native.rglob('*')) if s.is_file()}
cap=json.loads((native/'capture/receipt.json').read_text())
assert cap['compile_successes']==8 and cap['native_execution'] is False
write('native-evidence-inputs.json',{'root':str(native),'input_sha256':native_sha,'capture_receipt_sha256':sha(native/'capture/receipt.json'),
 'profiles':[{'label':x['label'],'target':x['target'],'constexprs':x['constexprs'],'requested_options':x['requested_options']} for x in cap['records']],
 'new_native_run':False,'compiler_machine_checked_refinement':False})
# Reconstruct transitive source imports for present-day environment binding.
# Only direct_objects_* above are compile-time before/after observations.
closure={};outside={};todo=[x['module'] for x in records]
while todo:
 m=todo.pop()
 if m in closure or m in outside:continue
 rel=Path(m.replace('.','/')+'.lean')
 src=next((base/rel for base in [P,Path('/tmp/veritile-dequant-int4-original-bits-20261015'),Path('/tmp/veritile-dequant-int4-dotacc-closure-20261015'),R] if (base/rel).is_file()),R/rel)
 obj=next((base/(m.replace('.','/')+'.olean') for base in roots if (base/(m.replace('.','/')+'.olean')).is_file()),None)
 assert obj is not None,m
 obs={str(x):sha(x) for x in [obj,Path(str(obj)+'.server'),Path(str(obj)+'.private')] if x.is_file()}
 for q,h in obs.items():
  if not Path(q).is_relative_to(P) or Path(q).is_symlink(): external[q]=h
 if src.is_file() and (m.startswith('VeriTile.') or m.startswith('bench.') or m=='VeriTile'):
  ds=imports(src.read_text());closure[m]={'source':str(src),'source_sha256':sha(src),'imports':ds,'objects':obs}
  if not src.is_relative_to(P):bind(src)
  todo.extend(ds)
 else:outside[m]=obs
write('project-closure.json',{'observation':'packaging-time reconstructed source closure, not per-row recorded compile-time transitive imports',
 'project_modules':closure,'external_direct_modules':outside,'implicit_Init_or_external_transitive_environment_not_claimed_exhaustive':True})
entry=TASK+'/Proofs/FPCharacterizationBounds.lean'
specs=re.findall(r'^specification\s+(\S+)',(P/entry).read_text(),re.M);assert len(specs)==0
axioms=sum(len(re.findall(r'^#axiomsClean\s+',Path(x['source']).read_text(),re.M)) for x in records);assert axioms==25
write('scope.json',{'task':'matmul_dequantize_int4','purpose':'CHARACTERIZATION_FIXED_LEGAL_CASE',
 'new_public_specifications':[],'ordinary_theorems_only':True,'case':'experiment-case.json',
 'scope':'M64 N128 K64, BM64 BN128 BK32 GROUP_M8, all8192outputs, twoKblocks, NO_GROUPS true/groupsize64 and false/groupsize32',
 'memory':'all logical memory; explicit word-addressed origin8 envelope, full C8208words and all input allocations',
 'byte_ABI_proved':False,'native_conformance_proved':False,'numeric_acceptance':False,
 'software_outputs':{'nogroups_false':{'separate':'0x5401','fused':'0x5400'},'nogroups_true':{'separate':'0x5402','fused':'0x5401'}},
 'discriminator':'scalar half fusion; source and unfused livecarry agree on this finite case only'})
parents={}
for label,base,expected in [
 ('original_source',Path('/tmp/veritile-dequant-int4-original-bits-20261015'),'1bbeb820ae43ec841b8be2a9937762f558680b2bcc1b5e40f371d81655feb621'),
 ('compiled_graphs',Path('/tmp/veritile-dequant-int4-dotacc-closure-20261015'),'f940faa3769207890ab65506c7f27231db20c1fe897ca0c78cc264c05f661a0c')]:
 assert sha(base/'receipt.json')==expected
 (P/'parents'/label).mkdir(parents=True,exist_ok=True)
 import shutil
 shutil.copyfile(base/'receipt.json',P/'parents'/label/'receipt.json')
 parents[label]={'path':str(base/'receipt.json'),'sha256':bind(base/'receipt.json')}
for name in ['receipt.json','input-binding.json','running-results.json','original.py','original_jit.py']:
 (P/'parents/native_capture').mkdir(parents=True,exist_ok=True)
 shutil.copyfile(native/'capture'/name,P/'parents/native_capture'/name)
parents['native_capture']={'path':str(native/'capture/receipt.json'),'sha256':bind(native/'capture/receipt.json')}
write('parent-bindings.json',parents)
# Every actual new source must be absent from W; package is an additive candidate.
installation=[]
for x in records:
 rel=str(Path(x['source']).relative_to(P));assert not (W/rel).exists()
 installation.append({'path':rel,'before_sha256':None,'after_sha256':sha(P/rel),'module':x['module']})
write('installation.json',{'status':'candidate_only_not_applied','files':installation,
 'registration':'not changed here; installation must independently check actual Lake recursive/explicit selectors',
 'metadata':'no metadata update or task upgrade performed'})
# Include every local regular file except immutable receipt itself and verifier output.
art={str(s.relative_to(P)):sha(s) for s in sorted(P.rglob('*'))
 if s.is_file() and not s.is_symlink() and s.name not in ('receipt.json','verify.log') and '__pycache__' not in s.parts}
write('receipt.json',{'schema':1,'status':'candidate_proofs_passed_native_binding_open',
 'root':str(P),'new_module_count':len(records),'named_axiom_checks':axioms,'public_entry':entry,'public_specifications':specs,
 'ordered_roots':records[0]['ordered_roots'],'artifact_sha256':art,'external_input_sha256':external,
 'compiled_records':'last-build.json','project_closure_count':len(closure),'external_direct_modules':len(outside),
 'new_native_runs':0,'workspace_or_baseline_modified':False,'all_old_failures_preserved':True})
print(json.dumps({'receipt':str(P/'receipt.json'),'receipt_sha256':sha(P/'receipt.json'),'modules':len(records),'axioms':axioms,
 'artifacts':len(art),'external_inputs':len(external),'closure':len(closure),'external_direct_modules':len(outside)},indent=2))
