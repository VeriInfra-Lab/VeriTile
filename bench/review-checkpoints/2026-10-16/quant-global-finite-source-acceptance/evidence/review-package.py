from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import ast,copy,json,re,sys
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');T=Path('/tmp/veritile-standardization-20261015');W=T/'quant-global-finite-acceptance-wip/frozen-v2';V=T/'quant-global-finite-root-validation'
sys.path.insert(0,str(R/'bench'))
from adapter_common import canonical,declarations,hash_file,imports,load,sha,write_json
import check_original_program_scope as gate
from exact_scope_gate import validate_exact_case_links
checked={}
def exact(path,h=None):
 path=Path(path);actual=hash_file(path)
 if h is not None:assert actual==h,str(path)
 checked[str(path)]=actual;return actual
receipt_hash='e1edc3543369fd52ed4d66dda58e6ea81778ea7312437adf7ebd2ed83c5e15e6'
exact(Path(__file__));exact(W/'evidence/receipt.json',receipt_hash);r=load(W/'evidence/receipt.json');assert r['frozen'] and r['task']=='quantize_global'
assert r['checkpoint_module_source_paths']=={p:p+'.txt' for p in r['modules']}
assert len(r['modules'])==11
for p,h in r['modules'].items():exact(W/'overlay'/p,h);exact(W/r['checkpoint_module_source_paths'][p],h)
for p,h in r['artifacts'].items():exact(W/p,h)
exact(r['source']['path'],r['source']['sha256']);exact(R/'bench/original_program_acceptance.md')
deps=load(W/r['dependencies']['path']);exact(W/r['dependencies']['path'],r['dependencies']['sha256']);assert len(deps)==5480
for dep in deps:
 exact(dep['source']['path'],dep['source']['sha256'])
 for obj in dep['artifacts']:exact(obj['path'],obj['sha256'])
project=load(W/'evidence/dependencies-project.json')['project'];assert len(project)==146
for p,e in project.items():exact(R/p,e['sha256'])
known={d['module'] for d in deps}|{p.removesuffix('.lean').replace('/','.') for p in r['modules']}
for p in r['modules']:
 for name in imports((W/'overlay'/p).read_text()):assert name in known,(p,name)
axioms=0;noncircular=0
for build in r['builds']:
 assert build['exit_code']==0 and build['source_unchanged'];p=build['module'].replace('.','/')+'.lean';assert r['modules'][p]==build['source_sha256']
 exact(W/build['saved_source'],build['source_sha256']);exact(W/build['log'],build['log_sha256'])
 for obj in build['compiled_artifacts']:exact(obj['path'],obj['sha256'])
 text=(W/build['log']).read_text();assert not re.search(r'(?m)^.*\.lean:\d+:\d+: error:',text)
 axioms+=text.count('axiom footprint ⊆ standard base');noncircular+=text.count('— non-circular ✓')
attempts=[]
for path in sorted((W/'evidence/producer-attempts/attempts').glob('*.json')):
 a=load(path);exact(path.with_suffix('.lean.txt'),a['source_sha256']);exact(path.with_suffix('.log'),a['log_sha256']);attempts.append(a)
public=[]
for p in r['modules']:
 for name,d in declarations(W/'overlay'/p).items():
  if d['kind']=='specification':public.append({'file':p,'declaration':name,**d})
assert len(public)==1 and public[0]['declaration']=='quantize_global_full_source_outcomes'
original=Path(r['source']['path']).read_text();nodes=lambda text:[ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name in ['_quantize_global','quantize_global']]
assert nodes(original)==nodes((W/'original/quantize_global.py').read_text())
for name in ['probe_native.py','probe_host_finite.py','probe_autotune.py','probe_dtypes.py']:
 exact(V/'replay'/name);assert (V/'replay'/name).read_bytes()==(W/name).read_bytes()
for relative in ['host-finite-probes/receipt.json','autotune-probes/receipt.json','dtype-probes/receipt.json']:
 assert load(V/'replay'/relative)==load(W/'evidence'/relative),relative
native_relative='evidence/native-attempts/20261015T204252377299Z/receipt.json'
producer_native=load(W/native_relative);replay_path=next((V/'replay/native-attempts').glob('*/receipt.json'));replay_native=load(replay_path)
assert producer_native['production_ast_unchanged'] and replay_native['production_ast_unchanged'];assert nodes(Path(replay_native['copied_prefix']).read_text())==nodes(original)
for field in ['torch_version','torch_commit','triton_version','source_sha256','gpu_launches']:assert producer_native[field]==replay_native[field]
assert len(producer_native['records'])==len(replay_native['records'])==52
for a,b in zip(producer_native['records'],replay_native['records']):
 for field in ['name','signature','constexprs','options','target','status']:assert a[field]==b[field],(field,a['name'])
 if a['status']=='compiled':
  assert a['arithmetic_ir']==b['arithmetic_ir']
  for stage in ['ttir','ttgir','llir','ptx']:
   ref=a['artifacts'][stage];exact(W/Path(ref['path']).relative_to(T/'quant-global-finite-acceptance-wip').as_posix().replace('native-attempts/','evidence/native-attempts/',1),ref['sha256'])
   exact(b['artifacts'][stage]['path'],b['artifacts'][stage]['sha256'])
 else:
  normalize=lambda text:re.sub(r'/tmp/[^\s\"\']+','<tmp-path>',text)
  assert normalize(a['error'])==normalize(b['error'])
assert Counter(c['status'] for c in producer_native['records'])=={'compiled':32,'failed':20}
for ref in load(W/'evidence/torch-source/receipt-escalated.json'):
 assert '/cf30153c4c131c8164ee7798e5022d810682e2cb/' in ref['url'];exact(W/'evidence/torch-source'/Path(ref['path']).name,ref['sha256'])
delta=load(W/'evidence/numerical-authority-delta.json');row=load(W/'evidence/scope-row-proposal.json');decl=load(W/'evidence/declaration-proposal.json')
current=next(t for t in load(R/'bench/original_program_scope.json')['tasks'] if t['task']=='quantize_global');assert row['source_inventory']==current['source_inventory'];assert decl['statement_sha256']==public[0]['statement_sha256']
enum=load(W/'evidence/dtype-enum-extension.json');exact(R/enum['checker'],enum['existing_checker_sha256']);assert set(enum['add_to_DTYPES'])=={'complex32','complex64','complex128','fp8e8m0fnu'}
# This root candidate check explicitly adds labels in memory; canonical schema
# integration remains a separate obligation. No validation predicate changes.
gate.DTYPES=gate.DTYPES|set(enum['add_to_DTYPES'])
gate.declaration_scope(decl,row['source_inventory']);gate.validate_deferrals('quantize_global',row['deferred_numerical_assumptions'],{'rules':delta['rules']})
assert len(delta['rules'])==16 and len(delta['provenance'])==420 and len(delta['requirements'])==1
table=delta['requirements'][0];cases=load(W/'evidence/scope-cases.json');assert len(cases)==len(table['cases'])==2400
assert table['supported_cases_sha256']==sha(canonical(sorted(cases,key=canonical)))
assert {canonical({k:c[k] for k in ['dtype_profile','configuration']}) for c in table['cases']}=={canonical(c) for c in cases}
for c in cases:gate.validate_case(c)
assert len({c['dtype_profile']['inputs']['x'] for c in cases})==22
assert sum(bool(c['required_binding_refs']) for c in table['cases'])==48
for port in delta['provenance'].values():
 ref=port['original_source'];exact(R/ref['path'],ref['sha256'])
 for p,h in port['lean_source_sha256'].items():exact(W/'overlay'/p,h)
 for field in ['raw_ttir','native_compile_receipt','host_wiring_receipt','torch_cuda_sources']:
  ref=port[field];exact(W/ref['path'],ref['sha256'])
 assert port['public_statement']==table['declaration']
key=(table['declaration']['file'],table['declaration']['declaration']);exactcases={key:{canonical({k:c[k] for k in ['dtype_profile','configuration']}):c for c in table['cases']}}
links={}
for c in table['cases']:
 for ref in c['required_binding_refs']:links.setdefault(ref['callback'],set()).add((ref['assumption_id'],ref['scope_binding_id']))
mutations=0
for case,c in exactcases[key].items():
 refs={(x['assumption_id'],x['scope_binding_id']) for x in c['required_binding_refs']};validate_exact_case_links(exactcases,key,case,refs,links)
 if c['configuration']['outcome']!='returned_or_undefined_conversion':assert not refs and not c['required_callbacks']
 for omitted in refs:
  try:validate_exact_case_links(exactcases,key,case,refs-{omitted},links)
  except ValueError:mutations+=1
  else:raise AssertionError('omitted port accepted')
assert mutations==420
review={'schema':1,'kind':'quant_global_finite_source_root_review','reviewer':'root','status':'substantively_reviewed_for_finite_scope_integration','recorded_at_utc':datetime.now(timezone.utc).isoformat(),'producer_receipt_sha256':receipt_hash,'checked_file_sha256':checked,'integration_sources':r['modules'],'public_contracts':[{k:public[0][k] for k in ['file','declaration','kind','statement_sha256']}],'checked_counts':{'files':len(checked),'dependency_records':len(deps),'project_source_dependencies':146,'final_modules':11,'producer_attempts':len(attempts),'failed_attempts_retained':sum(a['exit_code']!=0 for a in attempts),'named_axiom_checks':axioms,'independent_reference_checks':noncircular,'native_compile_profiles':52,'native_compile_successes':32,'native_compile_rejections':20,'host_outcome_probes':360,'autotuner_schedules':6,'cpu_dtype_probes':len(load(V/'replay/dtype-probes/receipt.json')['records']),'scope_cases':2400,'numerical_families':16,'exact_ports':420,'omitted_port_mutations':mutations},'semantic_review':'Root read original Python, all final finite source definitions/proofs and the retained real-erasure bridge. The exact original scalar parenthesization is127*(x*inverse), with source promotion followed by partial nearest-even llrint64 and int8 narrowing. Host abs traverses actual logical view offsets; maximum is an invocation-bound whole primitive on the complete abs list, never a false fixed left fold. The original linear data_ptr traversal and logical host reduction remain distinct for strided views. Runtime dtype, shape/rank, CUDA assertion, first errors, two original configs/defaults and arbitrary evolving-memory autotune trial sequences are explicit. All22 dtype cases retain actual first rejections; eight admitted storage types retain returned-or-undefined-conversion outcomes without Safe or success premises. N=2^31 preserves i64 runtime n with nonnegative i32 offsets; larger source offsets have an explicit sign-wrap hazard witness. Finite byte capacities, no-wrap, alignment, input/current output devices and launch permissions are explicit; fresh output/max/inverse are disjoint, input views retain actual offsets, and output initialization is arbitrary allocation payload. The actual host maximum/inverse word writes and every successful lane/program/grid/trial store are connected by a commuting memory representation and constructed PhysicalStores trace. Typed input and inverse loads relate to physical bytes; independent result and full logical/byte frames follow. Failed and undefined executions do not invent postfailure bytes, native trap or rollback. Ready is conditional ordinary data/metadata/primitive readiness, not launch success or a result oracle. Root independently replayed all52 original-AST compile profiles,360 host cases,6 real Autotuner schedules and actual CPU dtype probes, and checked all420 omitted numerical-port mutations. Numerical conformance is explicitly deferred at exact scalar/primitive operands and domains; no real=floating identity or undefined conversion is accepted as a deferral.','evidence_limitations':['Native compiler generation and CPU/host probes are not GPU correctness or numerical conformance. CUDA dtype admission is grounded in pinned installed-commit Torch sources.','The explicit source API is ordinary dense strided CPU/CUDA tensors with valid Torch metadata, default floating dtype float32, and inspected sm90/Triton3.7.1/Torch2.13 runtime. Generic primitive/compiler/allocator implementation proof is excluded by the acceptance policy.','Canonical integration must add only four exact rejected-dtype labels and pin all reviewed numerical authority inputs; candidate in-memory validation does not itself alter the canonical checker.','Raw IR locations change on independent replay; semantic signatures, options, arithmetic operations and success/error classes match. Historical frozen fp32 host/grid package remains unchanged.'],'root_replay_artifact_sha256':{p.relative_to(V).as_posix():hash_file(p) for p in V.rglob('*') if p.is_file() and p.name not in ['review.json','review.log'] and '__pycache__' not in p.parts and 'cache' not in p.parts},'proposed_scope_status':'verified_with_deferred_numerics','remaining_integration_obligations':['Additive canonical dtype enum and exact independently pinned numerical authority extension preserving the previous queue.','Current eight-target build, catalog, scope schema2 and unchanged global gates before acceptance increment.'],'native_execution':False,'numerical_validation':False,'acceptance_increment_before_integration':0}
assert not (V/'review.json').exists();write_json(V/'review.json',review);print(json.dumps({'status':review['status'],'review_sha256':hash_file(V/'review.json'),'checked_counts':review['checked_counts']},indent=2))
