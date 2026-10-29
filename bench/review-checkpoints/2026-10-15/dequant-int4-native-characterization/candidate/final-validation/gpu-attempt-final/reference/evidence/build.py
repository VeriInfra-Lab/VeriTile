import os,json,hashlib,re,subprocess,sys,time
sys.dont_write_bytecode=True
from pathlib import Path
O=Path(__file__).resolve().parent
R=Path('/tmp/veritile-next-value-integration-20261014')
L=Path('/tmp/lean-4.29.0-linux/bin/lean')
objects=O/'.lake/build/lib/lean';roots=[objects,Path('/tmp/veritile-dequant-int4-original-bits-20261015/.lake/build/lib/lean'),Path('/tmp/veritile-dequant-int4-dotacc-closure-20261015/.lake/build/lib/lean')]+[Path(s) for s in (R/'lean-path.txt').read_text().strip().split(':')]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def deps(path):
 out={}
 for line in path.read_text().splitlines():
  if line.startswith('import '):
   for m in line[7:].split():
    q=next((p/(m.replace('.','/')+'.olean') for p in roots if (p/(m.replace('.','/')+'.olean')).is_file()),None)
    if q is None:raise Exception('missing import '+m)
    out[m]={str(x):sha(x) for x in [q,Path(str(q)+'.server'),Path(str(q)+'.private')] if x.is_file()}
 return out
records=[]
for mod in sys.argv[1:]:
 rel=mod.replace('.','/')
 s=O/(rel+'.lean');target=objects/(rel+'.olean');target.parent.mkdir(parents=True,exist_ok=True)
 logdir=O/'logs';logdir.mkdir(exist_ok=True)
 stamp=time.strftime('%Y%m%dT%H%M%S')
 log=logdir/(stamp+'-'+mod+'.log')
 before=deps(s);sh=sha(s)
 command=[str(L),'--root='+str(O),'-o',str(target),str(s)]
 env=dict(os.environ,LEAN_PATH=':'.join(map(str,roots)))
 with log.open('w') as f:r=subprocess.run(command,cwd=O,env=env,stdout=f,stderr=subprocess.STDOUT)
 after=deps(s)
 record={'module':mod,'command':command,'cwd':str(O),'source':str(s),'source_before_sha256':sh,'source_after_sha256':sha(s),'direct_objects_before':before,'direct_objects_after':after,'log':str(log),'log_sha256':sha(log),'exit_code':r.returncode,'object':str(target),'object_sha256':sha(target) if r.returncode==0 else None,'lean':str(L),'lean_sha256':sha(L),'ordered_roots':list(map(str,roots)),'changed_inputs':[] if sh==sha(s) and before==after else ['source_or_dependency_drift']}
 out=log.with_suffix('.json');out.write_text(json.dumps(record,indent=2)+'\n');records.append(record)
 print(json.dumps({'module':mod,'exit_code':r.returncode,'log':str(log),'record':str(out)}),flush=True)
 if r.returncode:sys.exit(r.returncode)
(O/'last-build.json').write_text(json.dumps(records,indent=2)+'\n')
