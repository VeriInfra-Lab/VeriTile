"""Freeze selected-stage implementation, all attempts, and exact input bindings."""
from pathlib import Path
import ast, datetime, hashlib, json, re, sys
sys.dont_write_bytecode=True
root=Path(__file__).resolve().parent
repo=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sys.path.insert(0,str(repo/'bench'))
import adapter_common as adapter
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,value):
 p=root/name
 with p.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n')
 return {'path':str(p),'sha256':sha(p)}

build=root/'builds/20261015T190219599794Z/receipt.json'
native=root/'native-attempts/20261015T190220291606Z/receipt.json'
b=json.loads(build.read_text());n=json.loads(native.read_text())
assert b['status']==n['status']=='passed'
assert len(b['records'])==4
for rel,digest in b['source_sha256'].items():assert sha(root/'src'/rel)==digest
for r in b['records']:
 assert r['exit_code']==0 and r['dependency_hashes_stable']
 text=Path(r['log']).read_text()
 assert 'sorryAx' not in text and 'error:' not in text and 'warning:' not in text
 for p,h in r['dependency_olean_sha256'].items():assert sha(p)==h

modules={}
def walk(module):
 if module in modules:return
 local=root/'src'/(module.replace('.','/')+'.lean')
 source=local if local.exists() else repo/(module.replace('.','/')+'.lean')
 if not source.exists():return
 artifact=(build.parent/'src'/local.relative_to(root/'src')).with_suffix('.olean') if local.exists() else repo/'.lake/build/lib/lean'/(module.replace('.','/')+'.olean')
 modules[module]={'source':str(source),'source_sha256':sha(source),'olean':str(artifact),'olean_sha256':sha(artifact) if artifact.exists() else None,'candidate':local.exists()}
 for match in re.finditer(r'^\s*(?:public\s+)?import\s+([^\n]+)',source.read_text(),re.M):
  for dep in match.group(1).split():
   if re.fullmatch(r'[A-Za-z0-9_.]+',dep):walk(dep)
for module in b['requested_modules']:walk(module)
dependencies=write('dependency-bindings.json',{'scope':'Current local source closure plus actual existing dependency olean fingerprints. Candidate modules compiled fresh; unchanged library dependencies reused through lake env, not rebuilt by this package.','modules':modules})

public=[]
for p in sorted((root/'src').rglob('*.lean')):
 for name,d in adapter.declarations(p).items():
  if d['kind']=='specification':
   public.append({'file':str(p.relative_to(root/'src')),'declaration':name,'source_sha256':sha(p),'statement_sha256':d['statement_sha256'],'statement':adapter.split_statement(d['text'])[0]})
assert len(public)==2
originals={}
for task in ['quantize_global','quant_transpose_kernel']:
 for p in sorted((repo/'bench/tritonbench_g'/task).rglob('*')):
  if p.is_file() and p.suffix in {'.lean','.py'}:
   originals[str(p.relative_to(repo))]=sha(p)
for r in n['records']:
 assert r['status']=='compiled' and sha(r['original_source'])==r['original_sha256']

scope={
 'schema':1,'kind':'source_coupled_executing_quantization_stage','tasks':['quantize_global','quant_transpose_kernel'],
 'selected_signatures':{'input':'fp32','inverse_absmax':'fp32','llrint_result':'signed int64 / BitVec 64','output':'signed int8 / BitVec 8'},
 'implemented':['Masked source loads with unspecified inactive-lane values retained','Two configured software fp32 multiplies, exactly 127.0 * (input * inverse_absmax)','Actual existing ComputeExpr.llrint executes a supplied Llrint64 callback','Exact frontend output-register marker reads the preceding llrint result; unknown markers fail','Source-signature-bound int8 narrowing occurs at actual masked stores','Successful whole-stage execution, exact byte readback, and frame at every other region/offset'],
 'semantics_extension':{'existing_unsupported_node':'ComputeExpr.llrint','numeric_callback':'FP.Value .fp32 -> BitVec 64','output_marker':'runtime compute value `output`','integer_store':'BitVec.ofInt 8 result64.toInt, stored using its signed toInt representation','supported_client_fragment':'Straight-line source stages with static named region stores. Loops/branches are not extended; pointer-valued integer stores are not covered by these clients.','existing_library_semantics':'Reused for fp32 loads/multiplies, address/mask evaluation, typed memory/register cells, lane write order and frame. No reference result is substituted for execution.'},
 'source_correspondence':{'quantize_global':'Complete selected-signature kernel body, with source identifiers renamed to region parameters. Host wrapper remains separate.','quant_transpose_kernel':'Load/quantize/transposed-store tile stage at supplied independent tile coordinates, with pure pointer rematerialization inlined. Original grouped one-dimensional pid mapping and host/grid remain separate.','formal_checks':'SourceChecks pins exact llrint expression tree and precision, store dependency marker, statement counts and int8 source signature.'},
 'explicit_numerical_boundaries':['Native llrintf, including exceptional inputs and rounding behavior, must instantiate the explicit BitVec64 callback; no native equality is assumed silently.','Native floating multiplication/denormal/compiler correspondence to the chosen FP.Config remains explicit. The TTIR may commute the outer constant operand; this package proves the original source order, not native bitwise equivalence.','The fixed software int8 narrowing is exact modulo-256 signed interpretation. Native store-conversion correspondence is observed in source-bound TTIR and remains a separate native refinement obligation.'],
 'remaining_structural_scope':['Host abs/max/unsqueeze/reciprocal and relation of loaded inverse_absmax to input','Host allocations/assertions and returned absmax/output tuple','Complete launch/grid composition','quant_transpose original grouped-pid mapping, including last partial group','Other admitted input/scale dtype routes and exact admission/rejection contracts','Native physical memory/runtime/launch refinement'],
 'whole_original_program_accepted':0,'completion_increment':0,'numerical_deferrals_registered':0,
 'gpu_launches':0,'original_offline_signature_compilations':2,
 'existing_proofs':'All original proof sources preserved; no existing specification changed or discarded.',
 'public_declarations':public,'original_source_sha256':originals,
 'validation':{'lean_version':'4.29.0','passed_lean_modules':4,'axiom_checks_passed':17,'axioms':'Standard base only; no sorry/axiom introduced.','build':{'path':str(build),'sha256':sha(build)},'offline_admission':{'path':str(native),'sha256':sha(native)}}}
scope_ref=write('scope.json',scope)
installed={}
for p in [Path('/usr/local/lib/python3.12/site-packages/triton/language/extra/cuda/libdevice.py'),Path('/usr/local/lib/python3.12/site-packages/triton/language/semantic.py'),Path('/tmp/veritile-lean-toolchain/bin/lean'),Path('/tmp/veritile-lean-toolchain/bin/lake'),repo/'lean-toolchain',repo/'lakefile.toml']:
 installed[str(p)]=sha(p)
manifest=write('artifact-manifest.json',{'schema':1,'files':{str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()},'external_tooling_and_native_source_sha256':installed})
receipt=write('producer-receipt.json',{'schema':1,'kind':'frozen_quantization_integer_stage_producer','producer':'/root/numerical_adapter_finish','frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'compiled_selected_stage','scope':scope_ref,'dependencies':dependencies,'manifest':manifest,'selected_public_declarations':public,'latest_lean_build':{'path':str(build),'sha256':sha(build)},'offline_original_admission':{'path':str(native),'sha256':sha(native)},'retained_build_attempts':[{'path':str(p),'sha256':sha(p)} for p in sorted((root/'builds').glob('*/receipt.json'))],'whole_original_program_accepted':0,'completion_increment':0,'numerical_deferrals_registered':0,'repository_source_writes':0,'git_writes':0})
print(json.dumps({'receipt':receipt,'scope':scope_ref,'manifest':manifest,'dependency_modules':len(modules),'public_declarations':len(public)},sort_keys=True))
