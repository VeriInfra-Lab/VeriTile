from pathlib import Path
import hashlib,json,os,subprocess,sys,time
ROOT=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
WORK=Path(__file__).resolve().parent
LEAN='/tmp/veritile-lean-toolchain/bin/lean'
LAKE='/tmp/veritile-lean-toolchain/bin/lake'
base=subprocess.check_output([LAKE,'env','printenv','LEAN_PATH'],cwd=ROOT,text=True).strip()
env=dict(os.environ,LEAN_PATH=str(WORK)+':'+base)
path=WORK/sys.argv[1]
evidence=WORK/'evidence'; evidence.mkdir(exist_ok=True)
attempts=evidence/'attempts'; attempts.mkdir(exist_ok=True)
counter=len(list(attempts.glob('*.json')))+1
tag=f'{counter:04d}-{path.stem}'
log=attempts/(tag+'.log')
source=hashlib.sha256(path.read_bytes()).hexdigest()
(attempts/(tag+'.lean.txt')).write_bytes(path.read_bytes())
cmd=[LEAN,'--root='+str(WORK),str(path),'-o',str(path.with_suffix('.olean'))]
start=time.monotonic()
with log.open('w') as stream:
 result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
record={'source':str(path),'source_sha256':source,'source_unchanged':source==hashlib.sha256(path.read_bytes()).hexdigest(),'command':cmd,'LEAN_PATH':env['LEAN_PATH'],'exit_code':result.returncode,'seconds':round(time.monotonic()-start,3),'log':str(log.relative_to(WORK)),'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()}
(attempts/(tag+'.json')).write_text(json.dumps(record,indent=2)+'\n')
(evidence/(path.stem+'-latest.json')).write_text(json.dumps(record,indent=2)+'\n')
print(log.read_text()); print(json.dumps(record,indent=2))
sys.exit(result.returncode if record['source_unchanged'] else 2)
