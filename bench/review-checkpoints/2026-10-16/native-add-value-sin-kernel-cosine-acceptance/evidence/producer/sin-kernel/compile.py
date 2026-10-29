import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json,os,subprocess,time
W=Path(__file__).resolve().parent;R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
env=dict(os.environ,PATH='/tmp/veritile-lean-toolchain/bin:'+os.environ['PATH'],LAKE_HOME='/tmp/veritile-lean-toolchain')
base=subprocess.check_output(['/tmp/veritile-lean-toolchain/bin/lake','env','printenv','LEAN_PATH'],cwd=R,env=env,text=True).strip()
overlay=W/'build-overlay';overlay.mkdir(exist_ok=True)
for prefix in base.split(':'):
 root=Path(prefix)
 if not root.is_absolute():root=R/root
 for group in ['bench','VeriTile']:
  if (root/group).is_dir():
   for p in (root/group).rglob('*.olean'):
    relative=p.relative_to(root);out=overlay/relative
    if not out.exists():out.parent.mkdir(parents=True,exist_ok=True);out.symlink_to(p)
env['LEAN_PATH']=str(overlay)+':'+base
src=W/'after'/sys.argv[1];out=(overlay/sys.argv[1]).with_suffix('.olean')
if out.is_symlink():out.unlink()
out.parent.mkdir(parents=True,exist_ok=True)
attempt=W/'evidence/lean-attempts';attempt.mkdir(exist_ok=True)
tag=f'{len(list(attempt.glob("*.json")))+1:04d}-{src.stem}'
snapshot=attempt/(tag+'.lean.txt');snapshot.write_bytes(src.read_bytes());log=attempt/(tag+'.log')
cmd=['/tmp/veritile-lean-toolchain/bin/lean','--root='+str(W/'after'),str(src),'-o',str(out)]
start=time.monotonic()
with log.open('w') as stream:r=subprocess.run(cmd,cwd=R,env=env,stdout=stream,stderr=subprocess.STDOUT)
record={'source':str(src.relative_to(W)),'source_sha256':sha(snapshot),'source_unchanged':sha(src)==sha(snapshot),'command':cmd,'LEAN_PATH':env['LEAN_PATH'],'exit_code':r.returncode,'seconds':round(time.monotonic()-start,3),'log':str(log.relative_to(W)),'log_sha256':sha(log)}
(attempt/(tag+'.json')).write_text(json.dumps(record,indent=2)+'\n');print(log.read_text());print(json.dumps(record));raise SystemExit(r.returncode)
