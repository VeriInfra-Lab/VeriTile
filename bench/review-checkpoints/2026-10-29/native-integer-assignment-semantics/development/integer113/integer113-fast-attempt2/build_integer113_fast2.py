from pathlib import Path
import subprocess,json,os,time,hashlib
s=Path('/tmp/integer113-fast-frontend');plan=json.loads(Path('/tmp/integer113-fast-plan.json').read_text());out=Path('/tmp/integer113-fast2-build.json');env=dict(os.environ,LEAN_PATH=Path('/tmp/integer113-fast-lean-path.txt').read_text());lean='/home/argustest/.elan/toolchains/leanprover--lean4---v4.29.0/bin/lean';r={'status':'RUNNING','start':time.time(),'results':[]};out.write_text(json.dumps(r,indent=2)+'\n')
for m in plan['compile']:
 p=s/(m.replace('.','/')+'.lean');h=hashlib.sha256(p.read_bytes()).hexdigest();log=Path('/tmp/integer113-fast2-'+m.replace('.','__')+'.log');t=time.time()
 with log.open('w') as f:v=subprocess.run([lean,'-R',str(s),'-o',str(p.with_suffix('.olean')),str(p)],cwd=s,env=env,stdout=f,stderr=subprocess.STDOUT)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==h
 r['results'].append({'module':m,'source_sha256':h,'status':'PASS' if v.returncode==0 else 'FAIL','exit_code':v.returncode,'start':t,'end':time.time(),'log':str(log)});out.write_text(json.dumps(r,indent=2)+'\n')
 if v.returncode:break
r.update(status='PASS' if len(r['results'])==len(plan['compile']) and all(x['status']=='PASS' for x in r['results']) else 'FAIL',end=time.time());out.write_text(json.dumps(r,indent=2)+'\n');print(r['status'],len(r['results']),flush=True)
raise SystemExit(0 if r['status']=='PASS' else 1)
