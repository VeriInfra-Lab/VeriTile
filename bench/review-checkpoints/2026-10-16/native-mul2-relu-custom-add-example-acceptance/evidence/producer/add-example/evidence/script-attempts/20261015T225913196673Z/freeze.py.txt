"""Finalize producer evidence only; root owns review and acceptance."""
from pathlib import Path
import hashlib,json,re,sys
W=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
from propose import MODULES,PYTHON,R
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
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

def manifest():
 return {str(p.relative_to(W)):sha(p) for p in sorted(W.rglob('*')) if p.is_file() and not p.is_symlink() and p.suffix not in ['.olean','.ilean','.pyc'] and str(p.relative_to(W)) not in MODULES and p!=W/'evidence/receipt.json' and '__pycache__' not in p.parts}
def draft():
 return {'schema_version':2,'frozen':True,'acceptance_increment':0,'candidate':'Full original-source add_example; independent root review/integration pending','modules':{p:sha(W/p) for p in MODULES},'checkpoint_module_source_paths':{p:p+'.txt' for p in MODULES},'artifacts':manifest(),'native_backend_compiled':True,'gpu_executed':False,'numerical_conformance_validated':False,'independent_review_status':'pending'}
def finalize():
 assert not (W/'evidence/receipt.json').exists(),'Never mutate frozen packages'
 after()
 final=load(W/'evidence/final-build.json');assert [r['module'] for r in final]==MODULES
 for r in final:
  a=load(W/r['record']);assert a['exit_code']==0 and a['source_unchanged'] and a['source_sha256']==sha(W/r['module'])
  assert not re.search(r'\berror:|sorryAx|DISALLOWED|warning:',(W/a['log']).read_text())
 attempts=sorted((W/'evidence/attempts').glob('*.json'))
 for p in attempts:
  a=load(p);assert sha(W/a['log'])==a['log_sha256'];snap=W/a['log'].replace('.log','.lean.txt');assert sha(snap)==a['source_sha256']
 delta=load(W/'evidence/numerical-authority-delta.json')
 for p in delta['provenance'].values():assert p['lean_source_sha256']=={f:sha(W/f) for f in MODULES}
 assert load(W/'evidence/adapter-check.json')['status']=='passed'
 out=draft();out.update(verification_attempt_count=len(attempts),failed_attempts_preserved=sum(load(p)['exit_code']!=0 for p in attempts),final_clean_attempts=final,module_objects={str(Path(f).with_suffix('.olean')):sha((W/f).with_suffix('.olean')) for f in MODULES})
 save(W/'evidence/receipt.json',out)
 print(json.dumps({'receipt':str(W/'evidence/receipt.json'),'sha256':sha(W/'evidence/receipt.json'),'modules':len(MODULES),'attempts':len(attempts),'failures':out['failed_attempts_preserved']}))
if __name__=='__main__':
 if sys.argv[1]=='after':after()
 elif sys.argv[1]=='finalize':finalize()
