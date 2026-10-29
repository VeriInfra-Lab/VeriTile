from pathlib import Path
import sys, hashlib, json, subprocess, os, time, shutil
sys.dont_write_bytecode=True
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
P=Path('/tmp/veritile-standardization-20261015/vector-custom-finite-acceptance-revision1-wip')
W=Path(__file__).resolve().parent
S=W/'isolated'
S.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt=P/'evidence/receipt.json'
assert sha(receipt)=='261d2be00ae6152fbbad9fc6c540ae96d0ff653198e9ae3ea7533878be7de8e1'
r=json.loads(receipt.read_text())
for path,h in r['artifacts'].items(): assert sha(P/path)==h,path
for path,h in r['modules'].items(): assert sha(P/'after'/path)==h,path
sys.path.insert(0,str(R/'bench'))
from adapter_common import imports
packages=['Cli','batteries','Qq','aesop','proofwidgets','importGraph','LeanSearchClient','plausible','mathlib']
roots=[Path('/tmp/veritile-tritonbench-lake/packages')/p/'.lake/build/lib/lean' for p in packages]+[Path('/tmp/veritile-lean-toolchain/lib/lean')]
module_names={str(Path(p).with_suffix('')).replace('/','.') for p in r['modules']}
project={};external={};seen=set()
def visit(name):
 if name in seen:return
 seen.add(name)
 rel=Path(*name.split('.'))
 if name in module_names:
  src=P/'after'/rel.with_suffix('.lean')
  for child in imports(src.read_text()):visit(child)
 elif (R/rel.with_suffix('.lean')).exists():
  src=R/rel.with_suffix('.lean');obj=R/'.lake/build/lib/lean'/rel.with_suffix('.olean')
  project[name]={'source':str(src),'sha256':sha(src),'olean':str(obj),'olean_sha256':sha(obj)}
  for child in imports(src.read_text()):visit(child)
 else:
  obj=next((root/rel.with_suffix('.olean') for root in roots if (root/rel.with_suffix('.olean')).is_file()),None)
  assert obj,name
  external[name]={'olean':str(obj),'sha256':sha(obj)}
for name in sorted(module_names):visit(name)
snapshot={'project':project,'external_objects':external,'modules':r['modules'],'producer_receipt_sha256':sha(receipt)}
before=W/'dependencies-before.json';assert not before.exists();before.write_text(json.dumps(snapshot,indent=2)+'\n')
old=json.loads((P/'evidence/dependencies-before.json').read_text())
oldproject={p.removesuffix('.lean').replace('/','.'):d for p,d in old['project'].items()}
oldexternal={p.removesuffix('.lean').replace('/','.') for p in old['external_imports']}
assert set(project)==set(oldproject),(set(project)-set(oldproject),set(oldproject)-set(project))
assert set(external)==oldexternal,(set(external)-oldexternal,oldexternal-set(external))
for name,d in project.items():
 assert d['sha256']==oldproject[name]['sha256'] and d['olean_sha256']==oldproject[name]['object_sha256'],name
 target=S/Path(*name.split('.')).with_suffix('.olean');target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(d['olean'])
for rel in r['modules']:
 target=S/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(P/'after'/rel,target)
env=dict(os.environ,LEAN_PATH=':'.join(map(str,[S,*roots,R/'.lake/build/lib/lean'])))
results=[]
order=['Proofs/NativeArithmetic.lean','Proofs/NativeScalar.lean','Proofs/NativeMemory.lean','Proofs/NativeProgram.lean','Proofs/NativeGrid.lean','Proofs/NativeAdmission.lean','NativeCorrect.lean']
for i,suffix in enumerate(order,1):
 rel='bench/tritonbench_g/vector_addition_custom/'+suffix
 src=S/rel;obj=src.with_suffix('.olean');log=W/f'build-{i:02d}-{src.stem}.log'
 cmd=['/tmp/veritile-lean-toolchain/bin/lean','--root='+str(S),str(src),'-o',str(obj)]
 start=time.monotonic()
 with log.open('w') as out: p=subprocess.run(cmd,cwd=S,env=env,stdout=out,stderr=subprocess.STDOUT)
 record={'source':rel,'sha256':sha(src),'command':cmd,'LEAN_PATH':env['LEAN_PATH'],'log':str(log.relative_to(W)),'log_sha256':sha(log),'exit_code':p.returncode,'elapsed_seconds':time.monotonic()-start,'object_sha256':sha(obj) if obj.exists() else None}
 results.append(record);(W/f'build-{i:02d}-{src.stem}.json').write_text(json.dumps(record,indent=2)+'\n')
 print(rel,p.returncode,flush=True)
 assert p.returncode==0,log
for d in project.values(): assert sha(Path(d['source']))==d['sha256'] and sha(Path(d['olean']))==d['olean_sha256']
for d in external.values(): assert sha(Path(d['olean']))==d['sha256']
(W/'dependencies-after.json').write_text(json.dumps(snapshot,indent=2)+'\n')
(W/'independent-build-receipt.json').write_text(json.dumps({'status':'passed','candidate':str(P),'receipt_sha256':sha(receipt),'artifacts_verified':len(r['artifacts']),'project_dependencies':len(project),'external_direct_imports':len(external),'dependency_snapshot_before':sha(before),'dependencies_unchanged':True,'modules':results},indent=2)+'\n')
print('BUILD PASSED',len(project),len(external),flush=True)
