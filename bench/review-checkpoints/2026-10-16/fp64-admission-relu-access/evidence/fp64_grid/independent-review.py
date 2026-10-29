"""Independent immutable-receipt, numerical-port and logical-contract review."""
import hashlib,json,re,shutil,subprocess,sys,tempfile
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
base=Path('/tmp/veritile-standardization-20261015');root=base/'vector-addition-fp64-grid-wip';core=base/'vector-addition-fp64-core-wip'
repo=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');ev=root/'evidence'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text())
assert sha(ev/'receipt.json')=='ae6a3ac04b77e40856757d7c0bde3c55e74fc8ea72e60e2986cceaae8159cab2'
assert sha(ev/'scope-review.json')=='4578e8c65acdd24d4f201809bd3cc0ca13dbe108f4db3e0640938e16ad492873'
r=read(ev/'receipt.json');scope=read(ev/'scope-review.json')
for rel,h in r['artifacts'].items():assert sha(root/rel)==h,rel
for rel,h in r['modules'].items():assert sha(root/rel)==h,rel
for rel,h in r['compiled_olean_sha256'].items():assert sha(root/rel)==h,rel
for field,path in [('receipt_sha256',core/'evidence/receipt.json'),('scope_review_sha256',core/'evidence/scope-review.json'),('independent_review_sha256',base/'vector-addition-fp64-core-independent-review.json')]:assert sha(path)==r['frozen_core'][field]
cr=read(core/'evidence/receipt.json')
for rel,h in cr['artifacts'].items():assert sha(core/rel)==h,rel
assert len(cr['artifacts'])==47
snaps=[read(ev/name) for name in ('final-dependencies-before.json','final-dependencies-after.json','seal-dependencies.json')]
assert snaps[0]==snaps[1]==snaps[2]
for rel,v in snaps[0]['files'].items():
 assert sha(Path(v['source_path']))==v['sha256'],rel
 if 'olean_path' in v:assert sha(Path(v['olean_path']))==v['olean_sha256'],rel
assert len(snaps[0]['files'])==15
ir=read(ev/'import-resolution.json');assert sha(ev/'import-resolution.json')==r['import_resolution_sha256']
assert ir['imported_project_oleans']==r['imported_project_oleans']
for rel,v in ir['imported_project_oleans'].items():
 p=Path(v['loaded_path']);assert p.is_symlink()==v['symlink'] and str(p.resolve())==v['resolved_path'] and sha(p)==v['olean_sha256']
 candidates=[Path(prefix)/Path(rel).with_suffix('.olean') for prefix in ir['lean_path'].split(':')]
 assert next(p for p in candidates if p.is_file())==Path(v['loaded_path'])
attempts=read(ev/'verification-attempts.json');final=read(ev/'verification-results.json')
assert len(attempts)==12 and len(final)==4 and attempts[-4:]==final
assert [i+1 for i,a in enumerate(attempts) if a['exit_code']!=0]==[1,2,4,6]
for a in attempts:
 assert sha(root/a['log'])==a['log_sha256']
 assert a['source_sha256_before']==a['source_sha256_after']==a['snapshot_sha256']==sha(root/a['source_snapshot'])
 if a['exit_code']!=0:assert 'error:' in (root/a['log']).read_text()
logs=[]
for a in final:
 assert a['exit_code']==0 and sha(root/a['module'])==a['source_sha256_before']
 assert a['lean_path']==ir['lean_path']
 assert sha(Path(a['command'][0]))==a['lean_executable_sha256']
 log=(root/a['log']).read_text();assert not re.search(r'\berror:|\bwarning:|sorryAx|DISALLOWED',log);logs.append(log)
assert sum(s.count('axiom footprint ⊆ standard base ✓') for s in logs)==33
assert sum(s.count('non-circular ✓') for s in logs)==5
assert sum((root/p).read_text().count('#auditModuleAxioms') for p in r['modules'])==4
assert sum((root/p).read_text().count('#axiomsClean ') for p in r['modules'])==33
sys.path.insert(0,str(repo/'bench'));from adapter_common import declarations
assert sha(repo/'bench/adapter_common.py')==r['statement_extractor_source_sha256']
parsed=declarations(root/'bench/tritonbench_g/vector_addition/FP64GridCorrect.lean')
assert len(parsed)==2
for p in r['public_contracts']:
 assert parsed[p['parser_name']]['statement_sha256']==p['statement_sha256']
 assert parsed[p['parser_name']]['kind']==p['kind']=='specification'
nb=read(ev/'numerical-bindings.json');families=read(core/'evidence/numerical-correspondence.json')
assert nb['families']==families['families'] and nb['public_contracts']==r['public_contracts']
assert sha(core/'evidence/numerical-correspondence.json')==nb['inherited_registry_sha256']
nsrc=core/'bench/tritonbench_g/vector_addition/Proofs/FP64TypedCore.lean'
structure=nsrc.read_text().split('structure Numerics where\n')[1].split('\ninductive Cast')[0]
callback_names=set(re.findall(r'^  (\w+) :',structure,re.M))
assert callback_names=={f['callback'] for f in nb['families']} and len(callback_names)==15
assert all(f['status']=='pending_explicitly_deferred' for f in nb['families'])
config=read(ev/'frontend-configuration.json');old=read(base/'vector-addition-dtype-extension-wip/evidence/extension-audit/receipt.json')
for k in ('environment','source_sha256','extracted_kernel_sha256'):assert config[k]==old[k]
for k in ('offline_target','constexprs','frontend_options'):assert config[k]==old['scope'][k]
assert sha(ev/'frontend-configuration.json')==r['frontend_configuration_sha256']==nb['configuration_sha256']
routebase=core/'evidence/source-routes';routes=read(routebase/'receipt.json')
assert sha(routebase/'receipt.json')==r['inherited_source_routes']['receipt_sha256']
route_by_pair={(p['x'],p['y']):p for p in routes['profiles']};assert len(route_by_pair)==29
floatnames={'float16','bfloat16','float32','float64','float8_e4m3fn','float8_e5m2'}
ints={'bool','int8','uint8','int16','uint16','int32','uint32','int64','uint64'}
assert set(route_by_pair)=={(x,y) for x in floatnames|ints for y in floatnames|ints if x=='float64' or y=='float64'}
by_pair={pair:[] for pair in route_by_pair}
for instance in nb['instances']:
 original=dict(instance);key=original.pop('instance_key')
 assert hashlib.sha256(json.dumps(original,sort_keys=True,separators=(',',':')).encode()).hexdigest()==key
 pair=(instance['dtype_profile']['x'],instance['dtype_profile']['y']);route=route_by_pair[pair]
 assert instance['dtype_profile']['output']==pair[0] and instance['dtype_profile']['compute']=='float64'
 assert instance['callback'] in callback_names and instance['callback_declaration'].endswith('Numerics.'+instance['callback'])
 assert instance['callback_source_sha256']==sha(nsrc)
 assert instance['configuration_sha256']==sha(root/instance['configuration_path'])
 assert instance['raw_ttir_path']==str(routebase/route['ttir_path'])
 assert instance['raw_ttir_sha256']==route['ttir_sha256']==sha(Path(instance['raw_ttir_path']))
 assert instance['original_source_sha256']==routes['source_sha256']==config['source_sha256']
 assert instance['declaration_bindings']==(r['public_contracts'] if pair[0] in floatnames else r['public_contracts'][:1])
 assert instance['status']=='explicitly_deferred_unvalidated' and instance['validation_evidence'] is None
 assert instance['queue_registration_status']=='concrete_candidate_ready_for_parent_registration; not an accepted numerical result'
 assert instance['structural_prerequisites_not_waived']
 family=next(f for f in nb['families'] if f['callback']==instance['callback'])
 assert instance['rule_id']==family['id'] and instance['numerical_correspondence']==family['required_correspondence']
 by_pair[pair].append(instance)
assert len(nb['instances'])==73 and len({i['instance_key'] for i in nb['instances']})==73
assert sum(len(i['declaration_bindings']) for i in nb['instances'])==120
# Trace raw value SSA from each load to the ordered add, then to the stored value.
# Numeric callback signatures must match every relevant node, with concrete bool
# normalization/uitofp kept outside the callback inventory.
traced=[]
for pair,route in route_by_pair.items():
 raw=(routebase/route['ttir_path']).read_text();defs={}
 for line in raw.splitlines():
  m=re.match(r'\s*(%[\w]+) = (.*)',line)
  if m:defs[m[1]]=m[2]
 add=defs[route['add_ssa']]
 assert add.startswith('arith.addf ') and re.findall(r'%[\w]+',add.split(':')[0])==route['ordered_add_operands']
 actual=[]
 def walk(end,stop,side):
  chain=[];current=end;seen=set()
  while current!=stop:
   assert current not in seen;seen.add(current)
   rhs=defs[current];op=rhs.split()[0];args=re.findall(r'%[\w]+',rhs.split(':')[0]);types=re.findall(r'tensor<1024x([^>]+)>',rhs)
   if op=='arith.cmpi':assert rhs.startswith('arith.cmpi ne,') and types==['i8']
   elif op=='arith.uitofp' and types==['i1','f64']:
    assert pair[0 if side=='x' else 1]=='bool'
   else:
    assert op in {'arith.sitofp','arith.uitofp','arith.extf','arith.fptosi','arith.fptoui','arith.truncf','tt.fp_to_fp'} and len(types)==2
    name=op+(' rounding=rtne' if 'rounding = rtne' in rhs else '')
    chain.append((side,name,(types[0],),types[1]))
   assert args;current=args[0]
  actual.extend(reversed(chain))
 for side,end in zip(('x','y'),route['ordered_add_operands']):walk(end,route['loads'][side],side)
 actual.append(('compute','arith.addf',('f64','f64'),'f64'))
 walk(route['stored_ssa'],route['add_ssa'],'output')
 stored=re.findall(r'tt.store ([^\n]+)',raw);assert len(stored)==1 and re.findall(r'%[\w]+',stored[0].split(':')[0])[1]==route['stored_ssa']
 expected=[(i['port']['side'],i['port']['operation'],tuple(i['port']['input_types']),i['port']['output_type']) for i in by_pair[pair]]
 assert actual==expected,(pair,actual,expected)
 traced.append({'x':pair[0],'y':pair[1],'callback_ports_checked':len(actual),'ttir_sha256':sha(routebase/route['ttir_path'])})
assert sum(p['callback_ports_checked'] for p in traced)==73
# Recheck actual staged blobs in a separate temporary index, including all
# immutable receipt artifacts; do not touch the repository or producer package.
staged=Path(tempfile.mkdtemp(prefix='veritile-fp64-grid-independent-stage-'))
for rel in r['artifacts']:
 target=staged/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/rel,target)
subprocess.run(['git','init','-q',str(staged)],check=True)
subprocess.run(['git','add','--','.'],cwd=staged,check=True)
for rel,h in r['artifacts'].items():
 blob=subprocess.check_output(['git','show',':'+rel],cwd=staged)
 assert hashlib.sha256(blob).hexdigest()==h
check=subprocess.run(['git','diff','--cached','--check'],cwd=staged,capture_output=True,text=True);assert check.returncode==0,check.stdout+check.stderr
report={'schema':1,'status':'accepted_partial_task_local_fp64_memory_grid_with_explicit_numerical_deferrals',
 'reviewer':'/root/comparator_restore','reviewed_at_utc':datetime.now(timezone.utc).isoformat(),
 'receipt_sha256':sha(ev/'receipt.json'),'scope_review_sha256':sha(ev/'scope-review.json'),'reviewed_source_sha256':r['modules'],
 'proof_defects_found':[],'whole_original_program_accepted':False,'original_program_completion_increment':0,
 'new_task_local_logical_grid_profile_increment':29,'shared_interpreter_profile_increment':0,
 'checker_path':str(Path(__file__).resolve()),'checker_sha256':sha(Path(__file__)),
 'verification':{'frozen_package_artifacts_checked':len(r['artifacts']),'frozen_core_artifacts_checked':47,
 'source_and_configuration_fingerprints_checked':15,'imported_project_oleans_checked':7,'own_oleans_checked':4,
 'preserved_compile_attempts':12,'preserved_failed_attempts':[1,2,4,6],'final_actual_lean_exits':[0,0,0,0],
 'named_axiom_checks':33,'noncircularity_checks':5,'module_axiom_audit_commands':4,'warnings':0,
 'independent_lean_rerun':False,'public_statement_hashes_checked':2,'numerical_families_checked':15,
 'numerical_port_instances_checked':73,'declaration_port_bindings_checked':120,'raw_ssa_profiles_retraced_for_numerical_ports':29,
 'independent_staged_blobs_checked':len(r['artifacts']),'independent_staged_whitespace_exit':0,'temporary_stage':str(staged)},
 'semantic_review':[
 'All four Lean sources read. Optional dependent tagged cells preserve full bit payload and refuse absent/wrong-tag active inputs; no default-value memory reads.',
 'InputWords is an initial-memory/tag premise. Equal input region names force equal full cells on active indices; distinct tags therefore cannot silently alias.',
 'runProgram takes a fixed program-entry read snapshot, computes/writes lanes sequentially to actual working memory, and returns an Except result. This is not the original load-all/compute-all/store-all instruction trace.',
 'lanes_success uses the previously reviewed actual task-local Expr source/runLane theorem to prove referenceIndices. The independent result/reference and OutputDefined avoid source/interpreter definitions in five checked noncircular commands.',
 'OutputDefined is automatic for all20 floating-output profiles under explicit total callbacks; only the9 integer-output profiles assume successful independent active conversion results. No inactive finiteness or missing-load premise is added.',
 'grid_success threads previous actual memory. Explicit output/input logical-region separation preserves input cells. Complete arbitrary program permutations cover exactly ceil(N/1024) whole programs; reference_frame covers all non-output and output-tail cells.',
 'Zero N, wholly masked/out-of-grid programs, absent input and undefined conversion failure propagation are explicit. Error outcomes contain no final memory and establish no native partial-store visibility, poison, trap or rollback.',
 'The new29 fp64 profiles live in a separate task-local model with15 unvalidated numerical callback families. Combined inventory160+29=189 spans distinct models and does not establish one interpreter, native conformance or complete original-program correctness.',
 'All73 dtype/port callback instances were compared to ordered raw SSA conversion/add/store dataflow, the exact frozen target/options, source/core hashes and applicable public statements. These remain queue candidates, with no accepted numeric experiment.'
 ],'numerical_bindings_sha256':sha(ev/'numerical-bindings.json'),'numerical_port_trace_checks':traced,
 'numerical_callbacks_remain_unvalidated':True,'remaining_structural_obligations':r['structural_gaps']}
p=base/'vector-addition-fp64-grid-independent-review.json';assert not p.exists();p.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'review_path':str(p),'review_sha256':sha(p),'verification':report['verification']}))
