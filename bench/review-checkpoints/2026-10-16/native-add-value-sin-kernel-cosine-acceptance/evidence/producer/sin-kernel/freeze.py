"""Immutable producer receipt; acceptance and independent review remain pending."""
from pathlib import Path
import hashlib,json,re,sys
sys.dont_write_bytecode=True
from propose import W,R,A,MODULES,PYTHON
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();load=lambda p:json.loads(Path(p).read_text())
def save(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def after():
 captures={};count=0;ext=0
 for p in sorted((W/'evidence').glob('dependencies*before*.json')):
  data=load(p)
  for module,row in data['project'].items():
   assert sha(row['source'])==row['sha256'],module
   assert sha(row['olean'])==row['olean_sha256'],module
   count+=1
  for module,row in data['external'].items():assert sha(row['olean'])==row['sha256'],module;ext+=1
  captures[str(p.relative_to(W))]={'sha256':sha(p),'verified_unchanged':True}
 save(W/'evidence/dependencies-after.json',{'all_unchanged':True,'captures':captures,'project_sources_and_objects':count,'external_objects':ext})
 return captures
def finalize():
 assert not (W/'evidence/receipt.json').exists(),'Frozen packages are immutable'
 after();final=load(W/'evidence/final-build.json');assert [r['module'] for r in final]==MODULES
 for r in final:
  a=load(W/r['record']);assert a['exit_code']==0 and a['source_unchanged'] and a['source_sha256']==sha(A/r['module'])
  assert not re.search(r'\berror:|sorryAx|DISALLOWED|warning:',(W/a['log']).read_text())
 attempts=sorted((W/'evidence/lean-attempts').glob('*.json'))
 for p in attempts:
  a=load(p);assert sha(W/a['log'])==a['log_sha256'];assert sha(W/a['log'].replace('.log','.lean.txt'))==a['source_sha256']
 scripts=sorted((W/'evidence/script-attempts').glob('*/record.json'))
 for p in scripts:
  for name,h in load(p)['artifacts'].items():assert sha(p.parent/name)==h
 modules={p:sha(A/p) for p in MODULES};delta=load(W/'evidence/numerical-authority-delta.json')
 for p in delta['provenance'].values():assert p['lean_source_sha256']==modules
 assert load(W/'evidence/adapter-check.json')['status']=='passed'
 queue=load(W/'evidence/queue-payload-proposal.json');queue_rules={r['id']:r for r in queue['rules']}
 for rule in delta['rules']:assert queue_rules[rule['id']]==rule
 # Retain canonical source snapshots for root installation and receipt binding.
 for p in MODULES:
  dst=W/(p+'.txt');dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((A/p).read_bytes())
 artifacts={str(p.relative_to(W)):sha(p) for p in sorted(W.rglob('*')) if p.is_file() and not p.is_symlink() and p.suffix not in ['.olean','.ilean','.pyc'] and p!=W/'evidence/receipt.json' and '__pycache__' not in p.parts and 'build-overlay' not in p.parts}
 out={'schema_version':2,'frozen':True,'acceptance_increment':0,'candidate':'Full original-source sin_kernel; independent root review/integration pending','modules':modules,'checkpoint_module_source_paths':{p:p+'.txt' for p in MODULES},'artifacts':artifacts,'verification_attempt_count':len(attempts),'failed_attempts_preserved':sum(load(p)['exit_code']!=0 for p in attempts),'final_clean_attempts':final,'module_objects':{str(Path(f).with_suffix('.olean')):sha((W/'build-overlay'/f).with_suffix('.olean')) for f in MODULES},'native_backend_compiled':True,'gpu_executed':False,'numerical_conformance_validated':False,'independent_review_status':'pending'}
 out.update(script_attempt_count=len(scripts),failed_script_attempts_preserved=sum(load(p)['exit_code']!=0 for p in scripts),initial_dependency_capture_failure_preserved='evidence/prepare-attempt001.log')
 save(W/'evidence/receipt.json',out);print(json.dumps({'receipt':str(W/'evidence/receipt.json'),'sha256':sha(W/'evidence/receipt.json'),'modules':len(MODULES),'attempts':len(attempts),'failures':out['failed_attempts_preserved'],'script_attempts':len(scripts),'script_failures':out['failed_script_attempts_preserved']}))
if __name__=='__main__':
 if sys.argv[1]=='after':after()
 elif sys.argv[1]=='finalize':finalize()
