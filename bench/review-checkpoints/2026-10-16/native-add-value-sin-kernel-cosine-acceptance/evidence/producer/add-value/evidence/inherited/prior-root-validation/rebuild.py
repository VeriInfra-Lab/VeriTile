from pathlib import Path
import hashlib,json,os,shutil,subprocess,time
w=Path(__file__).resolve().parent
p=Path('/tmp/veritile-standardization-20261015/add-value-finite-acceptance-wip')
repo=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
r=json.loads((p/'evidence/receipt.json').read_text())
assert sha(p/'evidence/receipt.json')=='0ba656d2df77a707a3faf1b82e46bb5304e0801aeca81082a9295528061d1373'
build=w/'build-attempt-002';build.mkdir(exist_ok=False)
deps=json.loads((p/'evidence/dependencies-before.json').read_text())
captured=[]
for mod,entry in deps['project'].items():
 rel=mod.replace('.','/')+'.lean'
 src=repo/rel
 assert sha(src)==entry['sha256']
 obj=repo/'.lake/build/lib/lean'/Path(rel).with_suffix('.olean')
 assert obj.exists() and sha(obj)==entry['olean_sha256'],obj
 dest=build/Path(rel).with_suffix('.olean');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(obj,dest)
 captured.append({'source':str(src),'source_sha256':sha(src),'object':str(obj),'object_sha256':sha(obj)})
for rel,digest in r['modules'].items():
 src=p/r['producer_module_source_paths'][rel];assert sha(src)==digest
 dest=build/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
records=[]
for info in json.loads((p/'evidence/final-compiles.json').read_text())['modules']:
 old=json.loads((p/info['receipt']).read_text())
 rel=info['source'].removeprefix('after/')
 src=build/rel;obj=src.with_suffix('.olean')
 env=os.environ.copy();env['LEAN_PATH']=old['LEAN_PATH'].replace(str(p),str(build))
 cmd=['/tmp/veritile-lean-toolchain/bin/lean','--root='+str(build),str(src),'-o',str(obj)]
 begin=time.monotonic();result=subprocess.run(cmd,env=env,capture_output=True,text=True)
 log=w/(src.stem+'.log');log.write_text(result.stdout+result.stderr)
 rec={'module':rel,'source_sha256':sha(src),'command':cmd,'LEAN_PATH':env['LEAN_PATH'],'exit_code':result.returncode,'seconds':time.monotonic()-begin,'log':str(log),'log_sha256':sha(log),'object_sha256':sha(obj) if obj.exists() else None}
 records.append(rec);print(src.stem,result.returncode,round(rec['seconds'],2),flush=True)
 if result.returncode:break
for d in captured:d['unchanged']=sha(d['source'])==d['source_sha256'] and sha(d['object'])==d['object_sha256']
out={'reviewer':'root','producer_receipt_sha256':sha(p/'evidence/receipt.json'),'records':records,'dependencies':captured,'all_passed':len(records)==5 and all(r['exit_code']==0 for r in records),'dependencies_unchanged':all(d['unchanged'] for d in captured)}
f=w/'independent-build-receipt.json';f.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'receipt':str(f),'sha256':sha(f),'all_passed':out['all_passed']}))
