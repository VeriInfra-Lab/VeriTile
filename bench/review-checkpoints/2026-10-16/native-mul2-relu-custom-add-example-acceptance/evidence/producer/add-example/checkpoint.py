from pathlib import Path
import datetime,hashlib,json
W=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
d=json.loads((W/'evidence/dependencies-before.json').read_text());checked=0
for mod,row in d['project'].items():
 assert sha(row['source'])==row['sha256'],mod
 assert sha(row['olean'])==row['olean_sha256'],mod
 checked+=1
for mod,row in d['external'].items():assert sha(row['olean'])==row['sha256'],mod
modules=[]
for p in sorted((W/'bench/tritonbench_g/add_example/Proofs').glob('*.lean')):
 latest=json.loads((W/'evidence'/(p.stem+'-latest.json')).read_text());assert latest['exit_code']==0 and latest['source_sha256']==sha(p)
 modules.append({'path':str(p.relative_to(W)),'sha256':sha(p),'olean_sha256':sha(p.with_suffix('.olean')),'latest':latest})
attempts=[json.loads(p.read_text()) for p in sorted((W/'evidence/attempts').glob('*.json'))]
record={'kind':'reviewable intermediate checkpoint; not acceptance/freeze','timestamp':stamp,'complete_acceptance':False,'acceptance_increment':0,'completed_modules':modules,'attempts':len(attempts),'failed_attempts':sum(r['exit_code']!=0 for r in attempts),'dependencies_unchanged':{'project':checked,'external':len(d['external']),'before_snapshot_sha256':sha(W/'evidence/dependencies-before.json')},'native_receipts':{str(p.relative_to(W)):sha(p) for p in (W/'evidence').glob('**/receipt.json')},'remaining':['Initial-memory full-grid plan/execution/frame correspondence with input/output allocation separation','Exact zeros_like(x) host metadata, source order and full22 binder/frontend/backend outcomes','Full output zero-initialization and untouched-cell return theorem including rank0, empty and N=2^31','Independent source scope/schema2 numerical proposal, double-pin adapter, negative integrity checks','Final ordered recompile, after dependencies and immutable acceptance receipt; root review']}
p=W/'evidence'/('checkpoint-'+stamp+'.json');p.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'checkpoint':str(p),'sha256':sha(p),'modules':len(modules),'attempts':len(attempts),'failed':record['failed_attempts']}))
