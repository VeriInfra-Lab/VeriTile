from pathlib import Path
import os,sys,json,hashlib,subprocess
W=Path(__file__).resolve().parent;src=W/'isolated/IndependentRegression.lean';seq=sys.argv[1];stem='regression-'+seq
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(W/(stem+'.lean.txt')).write_bytes(src.read_bytes())
build=json.loads((W/'independent-build-receipt.json').read_text());env=dict(os.environ,LEAN_PATH=build['modules'][0]['LEAN_PATH'])
cmd=['/tmp/veritile-lean-toolchain/bin/lean','--root='+str(W/'isolated'),str(src),'-o',str(src.with_suffix('.olean'))]
log=W/(stem+'.log')
with log.open('w') as out:r=subprocess.run(cmd,cwd=W/'isolated',env=env,stdout=out,stderr=subprocess.STDOUT)
(W/(stem+'.json')).write_text(json.dumps({'command':cmd,'source_sha256':sha(src),'exit_code':r.returncode,'log_sha256':sha(log),'log':log.name,'LEAN_PATH':env['LEAN_PATH']},indent=2)+'\n');print(r.returncode,log)
