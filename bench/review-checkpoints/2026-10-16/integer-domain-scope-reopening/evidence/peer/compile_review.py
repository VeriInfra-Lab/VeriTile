import sys;sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json,os,subprocess,time
W=Path(__file__).resolve().parent;R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
attempts=W/'lean-attempts';attempts.mkdir(exist_ok=True);number=len(list(attempts.glob('*.json')))+1;tag=f'{number:04d}-Narrowing';src=W/'Narrowing.lean';snapshot=attempts/(tag+'.lean.txt');snapshot.write_bytes(src.read_bytes());log=attempts/(tag+'.log')
base=subprocess.check_output(['/tmp/veritile-lean-toolchain/bin/lake','env','printenv','LEAN_PATH'],cwd=R,text=True).strip();env=dict(os.environ,LEAN_PATH=base);cmd=['/tmp/veritile-lean-toolchain/bin/lean','--root='+str(W),str(src),'-o',str(src.with_suffix('.olean'))];start=time.monotonic()
with log.open('w') as stream:r=subprocess.run(cmd,cwd=R,env=env,stdout=stream,stderr=subprocess.STDOUT)
result={'source':str(src),'source_sha256':sha(snapshot),'source_unchanged':sha(src)==sha(snapshot),'command':cmd,'LEAN_PATH':base,'exit_code':r.returncode,'seconds':round(time.monotonic()-start,3),'log':str(log.relative_to(W)),'log_sha256':sha(log)}
(attempts/(tag+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(log.read_text());print(json.dumps(result));raise SystemExit(r.returncode)
