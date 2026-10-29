from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import ast,copy,hashlib,json,re,sys
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');T=Path('/tmp/veritile-standardization-20261015');P=T/'relu-finite-acceptance-wip';O=T/'relu-finite-root-validation'
sys.path.insert(0,str(R/'bench'))
from adapter_common import canonical,declarations,hash_file,load,sha,write_json
from check_original_program_scope import declaration_scope,validate_case,validate_deferrals
from exact_scope_gate import validate_exact_case_links
checked={}
def exact(path,h):
 path=Path(path);assert hash_file(path)==h,str(path);checked[str(path)]=h
rh='1b18f21b18b1347799969bd5a037074759ee50a7193de3139b33933e9e002e42';exact(P/'evidence/receipt.json',rh);r=load(P/'evidence/receipt.json')
assert r['frozen'] and r['acceptance_increment']==0
for rel,h in {**r['modules'],**r['module_objects'],**r['artifacts']}.items():exact(P/rel,h)
assert r['checkpoint_module_source_paths']=={rel:rel+'.txt' for rel in r['modules']}
d=load(P/'evidence/dependencies-before.json');assert d==load(P/'evidence/dependencies-after.json')
for rel,row in d['project'].items():exact(row['source'],row['sha256']);exact(row['olean'],row['olean_sha256'])
for row in d['external_objects'].values():exact(row['olean'],row['sha256'])
for path,h in d['input_sha256'].items():exact(path,h)
history=load(P/'evidence/verification-attempts.json');assert len(history)==21 and sum(a['exit_code']!=0 for a in history)==2
for a in history:
 exact(P/a['source_snapshot'],a['snapshot_sha256']);exact(P/a['log'],a['log_sha256']);assert a['snapshot_sha256']==a['source_sha256_before']==a['source_sha256_after']
assert history[-5:]==r['final_clean_attempts']==load(P/'evidence/verification-results.json')
for a in history[-5:]:assert a['exit_code']==0 and not re.search(r'\berror:|sorryAx|DISALLOWED|warning:',(P/a['log']).read_text())
source='bench/tritonbench_g/relu_triton_kernel/relu_triton_kernel.py';tree=ast.parse((R/source).read_text());fs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
sc=load(P/'evidence/source-check.json');exact(R/source,sc['source_sha256'])
assert sc['host_statements']==[ast.unparse(n) for n in fs['relu'].body]
assert sc['kernel_statements']==[ast.unparse(n) for n in fs['relu_kernel'].body]
assert {n.func.id for n in ast.walk(fs['relu']) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}.isdisjoint({'test_relu'})
assert len([n for n in ast.walk(fs['test_relu']) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='relu'])==4
for sub,counts in [('frontend',(85,65)),('frontend-boundary',(68,52))]:
 receipt=load(P/'evidence'/sub/'receipt.json');exact(R/source,receipt['source_sha256'])
 fun=next(n for n in ast.parse((P/'evidence'/sub/'extracted_relu_kernel.py').read_text()).body if isinstance(n,ast.FunctionDef))
 assert ast.dump(fun,include_attributes=False)==ast.dump(fs['relu_kernel'],include_attributes=False)
 rows=load(P/'evidence'/sub/'frontend-profiles.json');assert len(rows)==counts[0] and sum(a['status']=='frontend_generated' for a in rows)==counts[1]
 for a in rows:
  if 'ttir' in a:
   exact(P/'evidence'/sub/a['ttir'],a['ttir_sha256']);tt=(P/'evidence'/sub/a['ttir']).read_text()
   assert tt.index('tt.load')<tt.index('arith.select')<tt.index('scf.if')<tt.index('tt.store')
  else:assert a['input'].startswith('float8_')
# Independent reruns use unchanged scripts in a separate directory, preserving frozen raw evidence.
old=load(P/'evidence/frontend-boundary/frontend-profiles.json');new=load(O/'boundary/frontend-profiles.json')
assert len(old)==len(new)==68
for a,b in zip(old,new):
 assert {k:v for k,v in a.items() if k not in {'ttir_sha256','error'}}=={k:v for k,v in b.items() if k not in {'ttir_sha256','error'}}
 if 'ttir' in a:
  original=(P/'evidence/frontend-boundary'/a['ttir']).read_text();replay=(O/'boundary'/b['ttir']).read_text()
  assert original.replace(str(P/'evidence/frontend-boundary'),'<PROBE>')==replay.replace(str(O/'boundary'),'<PROBE>')
 else:assert a['error'].replace(str(P/'evidence/frontend-boundary'),'<PROBE>')==b['error'].replace(str(O/'boundary'),'<PROBE>')
assert load(O/'metadata/metadata-observation.json')==load(P/'evidence/metadata-observation.json')
delta=load(P/'evidence/numerical-authority-delta.json');decl=load(P/'evidence/declaration-proposal.json');row=load(P/'evidence/scope-row-proposal.json');cases=load(P/'evidence/scope-cases.json')
ref={k:decl[k] for k in ['file','declaration','statement_sha256']};assert declarations(P/ref['file'])[ref['declaration']]['statement_sha256']==ref['statement_sha256']
assert decl['native_binding_status']=='not_established';declaration_scope(decl,row['source_inventory'])
assert len(cases)==340 and len({canonical(c) for c in cases})==340
assert Counter(c['configuration']['size_class'] for c in cases)=={k:85 for k in ['empty','one_block','multiple_blocks','unsigned_constexpr_boundary']}
for c in cases:validate_case(c)
assert len(delta['rules'])==12 and len(delta['provenance'])==84 and len(delta['requirements'])==1
assert delta['requirements'][0]['supported_cases_sha256']==sha(canonical(sorted(cases,key=canonical)))
validate_deferrals('relu_triton_kernel',row['deferred_numerical_assumptions'],{'rules':delta['rules']})
req=delta['requirements'][0];key=(ref['file'],ref['declaration']);exactreq={key:{canonical({k:c[k] for k in ['dtype_profile','configuration']}):c for c in req['cases']}}
links={callback:{(v['assumption_id'],v['scope_binding_id']) for v in values} for callback,values in row['callback_assumptions'].items()}
negative=0
for c,rq in exactreq[key].items():
 applicable={(v['assumption_id'],v['scope_binding_id']) for v in rq['required_binding_refs']}
 validate_exact_case_links(exactreq,key,c,applicable,links)
 if rq['configuration']['outcome']!='returned' or rq['configuration']['size_class']=='empty':assert not applicable
 for v in rq['required_binding_refs']:
  try:validate_exact_case_links(exactreq,key,c,applicable-{(v['assumption_id'],v['scope_binding_id'])},links)
  except ValueError:negative+=1
  else:raise AssertionError('Omitted numeric port accepted')
assert negative==84
expected_ports={'int8':2,'int16':2,'int32':2,'int64':2,'uint8':1,'uint16':1,'uint32':1,'uint64':1,'bool':1,'float16':4,'bfloat16':5,'float32':3,'float64':3}
assert Counter(v['port']['storage_dtype'] for v in delta['provenance'].values())=={k:3*v for k,v in expected_ports.items()}
for v in delta['provenance'].values():
 assert v['lean_source_sha256']==r['modules'];exact(R/v['original_source']['path'],v['original_source']['sha256']);exact(P/v['raw_ttir']['path'],v['raw_ttir']['sha256'])
 assert v['status']=='explicitly_deferred_unvalidated' and v['validation_evidence'] is None and not v['native_execution']
review={'schema':1,'kind':'relu_finite_candidate_root_review','reviewer':'root','status':'substantively_reviewed_for_finite_scope_integration','recorded_at_utc':datetime.now(timezone.utc).isoformat(),'producer_receipt_sha256':rh,'checked_file_sha256':checked,'checked_counts':{'files':len(checked),'project_dependencies':len(d['project']),'external_objects':len(d['external_objects']),'compile_attempts':21,'failed_compile_attempts_retained':2,'frontend_cases':153,'independent_boundary_replays':68,'independent_metadata_replays':10,'scope_cases':340,'numeric_families':12,'numeric_ports':84,'omitted_port_negative_checks':negative},'public_statement':ref,'semantic_review':'Root read the original host/kernel and all five final Lean modules, the inherited typed result materialization, producer scripts, exact source/frontend evidence, finite API review, schema2 coverage and numeric delta. The host composes read-only device/numel and fresh empty_like metadata/frames with actual existing typed and fp64 source-grid execution. All 17 canonical dtype profiles are retained, including four first-stage fp8 failures, all first primitive failures, empty launches and complete grids. The safe source-grounded CUDA sm90/Triton3.7.1 strided domain includes N=2^31: source i32 offsets remain nonnegative and mask comparison promotes to uint32; N=2^31+1 has an explicit overflow witness and is not mislabeled a frontend reject. Input storage offsets, byte extents, permissions, no-wrap and fresh physical separation derive source pointer applicability. Results are physical flat-prefix values; only pid0 stores and all other programs still execute pre-gate loads/selection. The arbitrary output tail remains unchanged for N>1024. Tensor metadata/allocation, frontend typing and memory/launch primitives are explicit trusted contracts; no final value or successful kernel launch is assumed. Non-JIT test initialization remains inventoried as a harness. The 12 source-selected numerical families bind 84 concrete ports over 39 nonempty successful cases; no float-to-integer UB or false real=floating equality is deferred. Independent original-AST frontend replay and ten CPU metadata cases reproduce evidence; these are not GPU correctness. Full acceptance requires the unchanged exact authority/queue, catalog, schema2 and integration checks after copying.','finite_acceptance_policy_sha256':hash_file(R/'bench/original_program_acceptance.md'),'native_execution':False,'numerical_conformance_validated':False,'acceptance_increment_before_integration':0,'root_replay_artifact_sha256':{p.relative_to(O).as_posix():hash_file(p) for p in O.rglob('*') if p.is_file() and p.name not in {'review.json','review.log'}}}
write_json(O/'review.json',review);print(json.dumps({'status':review['status'],'review_sha256':hash_file(O/'review.json'),'counts':review['checked_counts']},indent=2))
