from pathlib import Path
from datetime import datetime, timezone
import os, json, sys, shutil, subprocess, time
sys.dont_write_bytecode=True
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
T=Path('/tmp/veritile-standardization-20261015')
P=T/'vector-finite-acceptance-revision2-wip'
V=T/'vector-native-revision2-root-validation'
W=V/'build-attempt001'
sys.path.insert(0,str(R/'bench'))
from adapter_common import hash_file, load, write_json, imports
assert not W.exists();W.mkdir(parents=True)
receipt=load(P/'evidence/receipt.json')
rh=hash_file(P/'evidence/receipt.json')
assert rh=='8679242cdbdbe9d8972fe59c011fc89f51ba723de0b233bbe8af07ec32312cc4'
for relative,h in receipt['artifacts'].items(): assert hash_file(P/relative)==h,relative
for relative,h in receipt['modules'].items(): assert hash_file(P/'after'/relative)==h,relative
dep=load(P/'evidence/dependencies-before.json');project=dep['project']
final=load(P/'evidence/current-final-rebuild.json')
attempt=load(P/final[0]['attempt'])
search=[Path(p) for p in attempt['LEAN_PATH'].split(':')[1:]]
external={}
for relative in dep['external_imports']:
 object_relative=Path(relative).with_suffix('.olean')
 object_path=next(p/object_relative for p in search if (p/object_relative).is_file())
 external[relative]={'object':str(object_path),'sha256':hash_file(object_path)}
known=set(project)|set(dep['external_imports'])|set(receipt['modules'])
for relative in set(project)|set(receipt['modules']):
 source=(P/'after'/relative) if relative in receipt['modules'] else R/relative
 for mod in imports(source.read_text()):
  assert mod.replace('.','/')+'.lean' in known,(relative,mod)
ancillary={}
for relative,entry in project.items():
 assert hash_file(R/relative)==entry['sha256'],relative
 assert hash_file(Path(entry['object']))==entry['object_sha256'],relative
 for suffix in ['.olean','.olean.private','.olean.server','.ir','.ilean']:
  original=Path(entry['object']).with_suffix(suffix)
  if original.exists():
   dest=(W/relative).with_suffix(suffix);dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(original)
   ancillary[str(original)]=hash_file(original)
snapshot={'recorded_before_first_root_compile':datetime.now(timezone.utc).isoformat(),
 'producer_receipt_sha256':rh,'project':project,'external_objects':external,
 'linked_objects':ancillary,
 'historical_limit':'Producer snapshots name 37 direct external imports without historical object hashes. These are new root pre-build captures; no historical producer object equality is claimed.'}
write_json(V/'dependencies-before-root.json',snapshot)
for relative,h in receipt['modules'].items():
 destination=W/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(P/'after'/relative,destination)
ordered=[];active=set()
def visit(relative):
 if relative in ordered:return
 assert relative not in active;active.add(relative)
 for mod in imports((W/relative).read_text()):
  child=mod.replace('.','/')+'.lean'
  if child in receipt['modules']:visit(child)
 active.remove(relative);ordered.append(relative)
for relative in receipt['modules']:visit(relative)
env=os.environ.copy();env['LEAN_PATH']=':'.join([str(W),*[str(p) for p in search]])
records=[]
for i,relative in enumerate(ordered,1):
 source=W/relative;obj=source.with_suffix('.olean');log=V/f'build-{i:02d}-{source.stem}.log'
 command=['/tmp/veritile-lean-toolchain/bin/lean','--root='+str(W),str(source),'-o',str(obj)]
 start=time.monotonic()
 with log.open('w') as stream:result=subprocess.run(command,cwd=W,env=env,stdout=stream,stderr=subprocess.STDOUT)
 output=log.read_text()
 row={'module':relative,'source_sha256':hash_file(source),'command':command,'LEAN_PATH':env['LEAN_PATH'],
  'exit_code':result.returncode,'seconds':round(time.monotonic()-start,3),
  'log':log.name,'log_sha256':hash_file(log),
  'object_sha256':hash_file(obj) if obj.exists() else None,
  'named_clean_axiom_checks':output.count('axiom footprint ⊆ standard base ✓'),
  'noncircular_checks':output.count('non-circular ✓')}
 records.append(row);write_json(V/f'build-{i:02d}-{source.stem}.json',row)
 print(relative,result.returncode,flush=True)
 if result.returncode:break
unchanged=all(hash_file(R/p)==v['sha256'] and hash_file(Path(v['object']))==v['object_sha256'] for p,v in project.items())
unchanged=unchanged and all(hash_file(Path(v['object']))==v['sha256'] for v in external.values())
unchanged=unchanged and all(hash_file(Path(p))==h for p,h in ancillary.items())
passed=len(records)==7 and all(r['exit_code']==0 for r in records)
summary={'producer_receipt_sha256':rh,'all_passed':passed,'records':records,'dependencies_unchanged':unchanged,
 'root_dependency_snapshot_sha256':hash_file(V/'dependencies-before-root.json'),
 'clean_axiom_checks':sum(r['named_clean_axiom_checks'] for r in records),
 'noncircular_checks':sum(r['noncircular_checks'] for r in records),
 'frozen_artifacts_verified':len(receipt['artifacts'])}
write_json(V/'independent-build-receipt.json',summary)
assert passed and unchanged
assert summary['clean_axiom_checks']==38 and summary['noncircular_checks']==6
print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2))
