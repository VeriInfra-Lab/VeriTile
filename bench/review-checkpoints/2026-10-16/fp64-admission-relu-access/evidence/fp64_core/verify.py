import hashlib, json, os, subprocess, sys, time
from pathlib import Path
root=Path(__file__).resolve().parent
env=dict(os.environ); env['LEAN_PATH']=str(root)+':'+env.get('LEAN_PATH','')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
results=[]
for rel in sys.argv[1:]:
 p=root/rel; log=root/'evidence'/(p.stem+'.compile.log')
 cmd=['/tmp/veritile-lean-toolchain/bin/lean','--root='+str(root),str(p),'-o',str(p.with_suffix('.olean'))]
 start=time.monotonic(); before=sha(p)
 with log.open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 row={'module':rel,'command':cmd,'exit_code':r.returncode,'seconds':time.monotonic()-start,'source_sha256_before':before,'source_sha256_after':sha(p),'log':str(log.relative_to(root)),'log_sha256':sha(log)}
 results.append(row); (root/'evidence/verification-results.json').write_text(json.dumps(results,indent=2)+'\n')
 print(json.dumps(row),flush=True);print(log.read_text(),flush=True)
 assert before==sha(p)
 if r.returncode:sys.exit(r.returncode)
