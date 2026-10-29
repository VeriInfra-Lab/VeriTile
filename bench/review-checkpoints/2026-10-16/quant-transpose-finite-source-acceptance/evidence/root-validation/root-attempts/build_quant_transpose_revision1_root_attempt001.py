from pathlib import Path
from datetime import datetime,timezone
import os,json,sys,subprocess,shutil,time
sys.dont_write_bytecode=True
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');T=Path('/tmp/veritile-standardization-20261015');P=T/'quant-transpose-finite-acceptance-revision1-wip';V=T/'quant-transpose-revision1-root-validation';W=V/'build-attempt001'
sys.path.insert(0,str(R/'bench'));from adapter_common import hash_file,load,write_json,imports
assert not W.exists();W.mkdir(parents=True);r=load(P/'evidence/receipt.json');rh=hash_file(P/'evidence/receipt.json');assert rh=='63392bac5342812d467c2bc499e7870e2609d49fa74faaf01731c1391b1dcf83'
project={};external={}
for p in r['dependencies']['captures']:
 dep=load(P/p);project.update(dep.get('project',{}));external.update(dep.get('external_objects',{}))
leanpath=r['final_clean_attempts'][0]['LEAN_PATH'].split(':');assert leanpath[0]==str(P);leanpath[0]=str(W)
known=set(project)|set(external)|{p.removesuffix('.lean').replace('/','.') for p in r['modules']};missing={mod for p in r['modules'] for mod in imports((P/p).read_text()) if mod not in known};assert missing=={'Mathlib.Data.List.Range'}
additional={}
for mod in missing:
 rel=Path(mod.replace('.','/')+'.olean');obj=next(Path(p)/rel for p in leanpath[1:] if (Path(p)/rel).is_file());additional[mod]={'olean':str(obj),'sha256':hash_file(obj)};external[mod]=additional[mod]
for v in project.values():assert hash_file(Path(v['source']))==v['sha256'];assert hash_file(Path(v['olean']))==v['olean_sha256']
for v in external.values():assert hash_file(Path(v['olean']))==v['sha256']
snapshot={'recorded_before_root_first_compile':datetime.now(timezone.utc).isoformat(),'producer_receipt_sha256':rh,'project':project,'external_objects':external,'additional_direct_external_not_in_producer_snapshots':additional,'note':'One original direct external import was absent from producer snapshot: captured here before root independent rebuild; no claim about its historical pre-producer hash is made.'};write_json(V/'dependencies-before-root.json',snapshot)
for p,h in r['modules'].items():
 assert hash_file(P/p)==h;dest=W/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(P/p,dest)
records=[];env=os.environ.copy();env['LEAN_PATH']=':'.join(leanpath)
for i,p in enumerate(r['modules'],1):
 src=W/p;obj=src.with_suffix('.olean');log=V/f'build-attempt001-{i:02d}-{src.stem}.log';cmd=['/tmp/veritile-lean-toolchain/bin/lean','--root='+str(W),str(src),'-o',str(obj)];start=time.monotonic()
 with log.open('w') as f:proc=subprocess.run(cmd,env=env,cwd=W,stdout=f,stderr=subprocess.STDOUT)
 record={'module':p,'source_sha256':hash_file(src),'source_unchanged':hash_file(src)==r['modules'][p],'command':cmd,'LEAN_PATH':env['LEAN_PATH'],'exit_code':proc.returncode,'seconds':round(time.monotonic()-start,3),'log':log.relative_to(V).as_posix(),'log_sha256':hash_file(log),'object_sha256':hash_file(obj) if obj.exists() else None};records.append(record);write_json(V/f'build-attempt001-{i:02d}-{src.stem}.json',record)
 print(p,proc.returncode,flush=True)
 if proc.returncode:break
unchanged=all(hash_file(Path(v['source']))==v['sha256'] and hash_file(Path(v['olean']))==v['olean_sha256'] for v in project.values()) and all(hash_file(Path(v['olean']))==v['sha256'] for v in external.values())
write_json(V/'independent-build-receipt.json',{'producer_receipt_sha256':rh,'records':records,'all_passed':len(records)==18 and all(x['exit_code']==0 and x['source_unchanged'] for x in records),'dependencies_unchanged':unchanged,'dependency_snapshot_sha256':hash_file(V/'dependencies-before-root.json'),'additional_dependency_capture':additional})
assert len(records)==18 and all(x['exit_code']==0 for x in records) and unchanged
