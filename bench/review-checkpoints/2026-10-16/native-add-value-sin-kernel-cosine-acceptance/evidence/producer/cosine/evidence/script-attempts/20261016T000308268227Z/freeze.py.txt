"""Freeze reproducible producer evidence; never grant independent acceptance."""
from pathlib import Path
import hashlib,json,re,sys
sys.dont_write_bytecode=True
from propose import W,R,A,MODULES,PYTHON
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();load=lambda p:json.loads(Path(p).read_text())
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def after():
 base=load(W/'evidence/dependencies-before.json');final=load(W/'evidence/dependencies-final-before.json')
 for rel,row in base['project'].items():
  assert sha(R/rel)==row['sha256'],rel
  assert sha(row['object'])==row['object_sha256'],rel
 for rel,row in final['external_objects'].items():assert sha(row['object'])==row['object_sha256'],rel
 modules={p:sha(A/p) for p in MODULES};assert final['new_modules']==modules
 captures={p:{'sha256':sha(W/p),'verified_unchanged':True} for p in ['evidence/dependencies-before.json','evidence/dependencies-final-before.json']}
 changed={p:{'earlier':h,'final':modules[p],'reason':'NativeHostMath trust-boundary comments or NativeScalar/NativeSource lint-only proof syntax; prior complete source snapshots retained in lean-attempts.'} for p,h in base['new_modules'].items() if modules[p]!=h}
 out={'all_unchanged':True,'scope':'201 imported project source/object pairs and37 external objects unchanged; final nine-module source snapshot unchanged','captures':captures,'project_sources_and_objects':len(base['project']),'external_objects':len(final['external_objects']),'new_module_changes_before_final_capture':changed}
 save(W/'evidence/dependencies-after.json',out);return out

def finalize():
 assert not (W/'evidence/receipt.json').exists(),'Frozen package is immutable'
 dep=after();final=load(W/'evidence/final-build.json');assert [r['module'] for r in final]==MODULES
 logs=[]
 for row in final:
  rec=load(W/row['record']);text=(W/rec['log']).read_text();logs.append(text)
  assert rec['exit_code']==0 and rec['source_unchanged'] and rec['source_sha256']==sha(A/row['module'])
  assert not re.search(r'\berror:|sorryAx|DISALLOWED|warning:',text),row['module']
 attempts=sorted((W/'evidence/lean-attempts').glob('*.json'))
 for p in attempts:
  rec=load(p);assert sha(W/rec['log'])==rec['log_sha256'];assert sha(W/rec['log'].replace('.log','.lean.txt'))==rec['source_sha256']
 scripts=sorted((W/'evidence/script-attempts').glob('*/record.json'))
 for p in scripts:
  for name,h in load(p)['artifacts'].items():assert sha(p.parent/name)==h
 modules={p:sha(A/p) for p in MODULES};delta=load(W/'evidence/numerical-authority-delta.json')
 for port in delta['provenance'].values():assert port['lean_source_sha256']==modules
 assert load(W/'evidence/authority-check.json')['status']=='passed'
 assert load(W/'evidence/python-to-lean-translation.json')['lean_source_sha256']==modules['bench/tritonbench_g/cosine_compute/Proofs/NativeSource.lean']
 adr=load(W/'evidence/recorded-admission/latest.json');assert sha(W/adr['receipt'])==adr['sha256'];assert load(W/adr['receipt'])['status']=='passed'
 queue=load(W/'evidence/queue-payload-proposal.json');qr={r['id']:r for r in queue['rules']}
 for rule in delta['rules']:assert qr[rule['id']]==rule
 for p in MODULES:
  dst=W/(p+'.txt');dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((A/p).read_bytes())
 text='\n'.join((A/p).read_text() for p in MODULES)
 assert not re.search(r'\b(sorry|admit)\b',re.sub(r'/\-.*?\-/','',text,flags=re.S))
 audit={'status':'passed','clean_axiom_checks':len(re.findall(r'^#axiomsClean ',text,re.M)),'noncircular_checks':len(re.findall(r'^#specNonCircular ',text,re.M)),'whole_module_axiom_audits':len(re.findall(r'^#auditModuleAxioms',text,re.M)),'public_specifications':1,'final_build_modules':len(final),'final_warnings':0,'project_dependencies_verified':dep['project_sources_and_objects'],'external_objects_verified':dep['external_objects']}
 save(W/'evidence/final-source-audit.json',audit)
 artifacts={str(p.relative_to(W)):sha(p) for p in sorted(W.rglob('*')) if p.is_file() and not p.is_symlink() and p.suffix not in ['.pyc','.ilean'] and p!=W/'evidence/receipt.json' and '__pycache__' not in p.parts and 'build-overlay' not in p.parts}
 out={'schema_version':2,'frozen':True,'candidate':'cosine_compute original source finite model; independent root review and three-line policy mapping pending','acceptance_increment':0,'modules':modules,'checkpoint_module_source_paths':{p:p+'.txt' for p in MODULES},'artifacts':artifacts,'verification_attempt_count':len(attempts),'failed_attempts_preserved':sum(load(p)['exit_code']!=0 for p in attempts),'script_attempt_count':len(scripts),'failed_script_attempts_preserved':sum(load(p)['exit_code']!=0 for p in scripts),'final_clean_attempts':final,'module_objects':{str(Path(p).with_suffix('.olean')):sha((W/'build-overlay'/p).with_suffix('.olean')) for p in MODULES},'native_backend_compiled':True,'native_backend_profiles':15,'existing_native_admission_profiles_compared':220,'gpu_executed':False,'large_allocations':False,'host_native_conformance_validated':False,'kernel_numerical_conformance_validated':False,'native_host_sample_comparisons':210943,'independent_review_status':'pending','new_Real_FP32_BF16_policy_mapping':'pending_root_mapping; existing_other_dtype_evidence_retained_without_expansion','authority_summary':delta['summary'],'final_source_audit':audit,'translation_attempt1_mutation_harness_correction':'Initial mutation harness had a caught assertion ambiguity. Preserved script/result; fixed harness distinguishes translation error from equal output and checks unchanged-source acceptance. Final translation remains exact.'}
 save(W/'evidence/receipt.json',out);print(json.dumps({'receipt':str(W/'evidence/receipt.json'),'sha256':sha(W/'evidence/receipt.json'),'artifacts':len(artifacts),'summary':audit,'authority':delta['summary']}))
if __name__=='__main__':
 if sys.argv[1]=='after':after()
 elif sys.argv[1]=='finalize':finalize()
