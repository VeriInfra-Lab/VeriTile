from pathlib import Path
import hashlib,json,os,subprocess,sys,time
ROOT=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
WORK=Path('/tmp/veritile-standardization-20261015/softmax-flaggems-host-grid-wip')
env=dict(os.environ,PATH='/tmp/veritile-lean-toolchain/bin:'+os.environ['PATH'],LAKE_HOME='/tmp/veritile-lean-toolchain')
base=subprocess.check_output(['/tmp/veritile-lean-toolchain/bin/lake','env','printenv','LEAN_PATH'],cwd=ROOT,env=env,text=True).strip()
env['LEAN_PATH']=str(WORK/'build-overlay')+':'+base
path=WORK/'after'/sys.argv[1]
seq=len(list((WORK/'evidence/attempts').glob('*.json')))+1
prefix=WORK/'evidence/attempts'/f'{seq:03d}-staged-{path.stem}'
prefix.with_suffix('.lean.txt').write_bytes(path.read_bytes())
command=['/tmp/veritile-lean-toolchain/bin/lean','--root='+str(WORK/'after'),str(path),'-o',str((WORK/'build-overlay'/sys.argv[1]).with_suffix('.olean'))]
start=time.monotonic()
with prefix.with_suffix('.log').open('w') as stream:
 result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
record={'module':'after/'+sys.argv[1],'command':command,'exit_code':result.returncode,'seconds':time.monotonic()-start,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'log_sha256':hashlib.sha256(prefix.with_suffix('.log').read_bytes()).hexdigest(),'LEAN_PATH':env['LEAN_PATH']}
prefix.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
print(prefix.with_suffix('.log').read_text())
print(json.dumps(record,indent=2))
sys.exit(result.returncode)
