import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
env=dict(os.environ);env['LEAN_PATH']=str(root)+':/tmp/veritile-tritonbench-lake/build/lib/lean:'+env.get('LEAN_PATH','')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(root/'evidence/build-environment.json').write_text(json.dumps({'LEAN_PATH':env['LEAN_PATH'],'working_directory':str(Path.cwd()),'lean_binary_sha256':sha(Path('/tmp/veritile-lean-toolchain/bin/lean'))},indent=2)+'\n')
history_path=root/'evidence/verification-attempts.json'
history=json.loads(history_path.read_text()) if history_path.exists() else []
results=[]
for relative in sys.argv[1:]:
    source=root/relative
    stem=f'{len(history)+1:04d}-{source.stem}'
    log=root/'evidence/attempts'/(stem+'.log')
    snapshot=root/'evidence/attempts'/(stem+'.lean.txt')
    assert not log.exists() and not snapshot.exists()
    before=sha(source);snapshot.write_bytes(source.read_bytes())
    command=['/tmp/veritile-lean-toolchain/bin/lean','--root='+str(root),str(source),'-o',str(source.with_suffix('.olean'))]
    started=time.monotonic()
    with log.open('w') as out:
        process=subprocess.run(command,env=env,stdout=out,stderr=subprocess.STDOUT)
    row={'module':relative,'command':command,'exit_code':process.returncode,'seconds':time.monotonic()-started,
         'source_sha256_before':before,'source_sha256_after':sha(source),
         'log':str(log.relative_to(root)),'log_sha256':sha(log),
         'source_snapshot':str(snapshot.relative_to(root)),'snapshot_sha256':sha(snapshot)}
    results.append(row);history.append(row)
    history_path.write_text(json.dumps(history,indent=2)+'\n')
    (root/'evidence/verification-results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(row),flush=True);print(log.read_text(),flush=True)
    if process.returncode:raise SystemExit(process.returncode)
