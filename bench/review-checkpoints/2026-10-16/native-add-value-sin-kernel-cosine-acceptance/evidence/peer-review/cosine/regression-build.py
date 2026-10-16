from pathlib import Path
import hashlib,json,os,subprocess,sys,time
W=Path(__file__).resolve().parent;S=W/'isolated';src=W/sys.argv[1];tag=src.stem
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
record=json.loads((W/'independent-build-receipt.json').read_text())['modules'][-1]
copy=S/src.name;copy.write_bytes(src.read_bytes());(W/(tag+'.lean.txt')).write_bytes(src.read_bytes())
env=dict(os.environ,LEAN_PATH=record['LEAN_PATH']);cmd=['/tmp/veritile-lean-toolchain/bin/lean','--root='+str(S),str(copy),'-o',str(copy.with_suffix('.olean'))];start=time.monotonic()
with (W/(tag+'.log')).open('w') as out:r=subprocess.run(cmd,cwd=S,env=env,stdout=out,stderr=subprocess.STDOUT)
(W/(tag+'.json')).write_text(json.dumps({'exit_code':r.returncode,'source_sha256':sha(src),'log_sha256':sha(W/(tag+'.log')),'command':cmd,'seconds':time.monotonic()-start},indent=2)+'\n');print((W/(tag+'.log')).read_text());sys.exit(r.returncode)
