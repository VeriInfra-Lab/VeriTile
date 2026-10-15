import hashlib,json,os,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
repo=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=root/'evidence/emit-model-rows.lean';out=root/'evidence/model-rows.tsv';err=root/'evidence/model-rows.stderr'
assert not out.exists() and not err.exists()
env=dict(os.environ);env['LEAN_PATH']=str(root)+':'+subprocess.check_output(['/tmp/veritile-lean-toolchain/bin/lake','env','printenv','LEAN_PATH'],cwd=repo,text=True).strip()
cmd=['/tmp/veritile-lean-toolchain/bin/lean','--root='+str(root),'--run',str(p)]
with out.open('w') as o,err.open('w') as e:r=subprocess.run(cmd,stdout=o,stderr=e,env=env)
info={'command':cmd,'exit_code':r.returncode,'source_sha256':sha(p),'output_sha256':sha(out),'stderr_sha256':sha(err),'row_count':len(out.read_text().splitlines()),'interpretation':'Executable finite-model row export for audit comparison; not a Python/compiler semantics theorem.'}
(root/'evidence/model-rows-result.json').write_text(json.dumps(info,indent=2)+'\n')
print(json.dumps(info));assert r.returncode==0 and info['row_count']==289 and not err.read_text()
