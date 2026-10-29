import hashlib,json,os,re,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent
repo=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
evidence=root/'evidence';evidence.mkdir(exist_ok=True)
modules=['bench/tritonbench_g/vector_addition/'+p+'.lean' for p in ('Proofs/PointwiseSchedule','Proofs/ValueSchedules','ValueScheduleCorrect')]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def deps():
 files={};queue=modules.copy();external=set()
 while queue:
  rel=queue.pop()
  if rel in files:continue
  p=root/rel;kind='package'
  if not p.is_file():p=repo/rel;kind='repository'
  if not p.is_file():external.add(rel);continue
  assert not p.is_symlink()
  info={'source_kind':kind,'source_path':str(p),'sha256':sha(p)}
  if kind=='repository':
   compiled=repo/'.lake/build/lib/lean'/Path(rel).with_suffix('.olean')
   assert compiled.is_file(),str(compiled)
   info.update(olean_path=str(compiled),olean_sha256=sha(compiled))
  files[rel]=info
  for line in p.read_text().splitlines():
   if line.startswith('import '):
    for name in line[7:].split():
     if name.startswith('--'):break
     queue.append(name.replace('.','/')+'.lean')
 for rel in ('bench/tritonbench_g/vector_addition/vector_addition.py','lakefile.toml','lake-manifest.json','lean-toolchain'):
  files[rel]={'source_kind':'repository','source_path':str(repo/rel),'sha256':sha(repo/rel)}
 return {'files':dict(sorted(files.items())),'external_imports':sorted(external)}
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
assert not (evidence/'final-dependencies-before.json').exists()
before=deps();write(evidence/'final-dependencies-before.json',before)
env=dict(os.environ);env['LEAN_PATH']=str(root)+':'+env.get('LEAN_PATH','')
results=[]
for rel in modules:
 source=root/rel;log=evidence/(source.stem+'.compile.log');assert not log.exists()
 before_hash=sha(source)
 command=['/tmp/veritile-lean-toolchain/bin/lean','--root='+str(root),str(source),'-o',str(source.with_suffix('.olean'))]
 start=time.monotonic()
 with log.open('w') as out:result=subprocess.run(command,env=env,stdout=out,stderr=subprocess.STDOUT)
 row={'module':rel,'command':command,'exit_code':result.returncode,'seconds':time.monotonic()-start,
  'source_sha256_before':before_hash,'source_sha256_after':sha(source),'log':str(log.relative_to(root)),'log_sha256':sha(log)}
 results.append(row);write(evidence/'verification-results.json',results);print(json.dumps(row),flush=True)
 assert result.returncode==0,log.read_text()
after=deps();assert before==after;write(evidence/'final-dependencies-after.json',after)
logs='\n'.join((root/r['log']).read_text() for r in results)
assert logs.count('axiom footprint ⊆ standard base ✓')==16 and logs.count('non-circular ✓')==2
assert not re.search(r'error:|warning:|sorryAx|DISALLOWED',logs)
print(json.dumps({'modules':len(modules),'dependencies':len(before['files']),'all_exit_zero':True,'axiom_checks':16,'noncircularity_checks':2,'warnings':0}),flush=True)
