import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json,os,subprocess,time
W=Path(__file__).resolve().parent;R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
packages=['Cli','batteries','Qq','aesop','proofwidgets','importGraph','LeanSearchClient','plausible','mathlib']
base=':'.join(str(Path('/tmp/veritile-tritonbench-lake/packages')/p/'.lake/build/lib/lean') for p in packages)+':'+str(R/'.lake/build/lib/lean')+':/tmp/veritile-lean-toolchain/lib/lean'
env=dict(os.environ,LEAN_PATH=str(W)+':'+base);p=W/sys.argv[1];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=W/'evidence/attempts';a.mkdir(exist_ok=True);seq=len(list(a.glob('*.json')))+1;stem=f'{seq:04d}-{p.stem}';log=a/(stem+'.log')
(a/(stem+'.lean.txt')).write_bytes(p.read_bytes());before=sha(p)
cmd=['/tmp/veritile-lean-toolchain/bin/lean','--root='+str(W),str(p),'-o',str(p.with_suffix('.olean'))]
assert not p.with_suffix('.olean').is_symlink()
start=time.monotonic()
with log.open('w') as stream:r=subprocess.run(cmd,cwd=W,env=env,stdout=stream,stderr=subprocess.STDOUT)
record={'source':str(p.relative_to(W)),'source_sha256':before,'source_unchanged':sha(p)==before,'command':cmd,'LEAN_PATH':env['LEAN_PATH'],'exit_code':r.returncode,'seconds':time.monotonic()-start,'log':str(log.relative_to(W)),'log_sha256':sha(log)}
(a/(stem+'.json')).write_text(json.dumps(record,indent=2)+'\n');print(log.read_text());print(json.dumps(record));raise SystemExit(r.returncode)
