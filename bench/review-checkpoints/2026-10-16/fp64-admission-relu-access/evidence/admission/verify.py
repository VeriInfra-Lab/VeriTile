import hashlib,json,os,re,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent
repo=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
evidence=root/'evidence';evidence.mkdir(exist_ok=True)
modules=['bench/tritonbench_g/vector_addition/'+p+'.lean' for p in ('Proofs/DTypeAdmission','DTypeAdmissionCorrect')]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):
 assert not p.exists(),str(p)
 p.write_text(json.dumps(d,indent=2)+'\n')
def deps(search):
 files={};queue=modules.copy();external={}
 while queue:
  rel=queue.pop()
  if rel in files or rel in external:continue
  p=root/rel;kind='package'
  if not p.is_file():p=repo/rel;kind='repository'
  if not p.is_file():
   compiled=next((base/Path(rel).with_suffix('.olean') for base in search if (base/Path(rel).with_suffix('.olean')).is_file()),None)
   assert compiled is not None,rel
   external[rel]={'olean_path':str(compiled),'olean_sha256':sha(compiled)}
   continue
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
 return {'files':dict(sorted(files.items())),'external_direct_import_oleans':dict(sorted(external.items())),
  'toolchain_binary_sha256':{str(p):sha(p) for p in (Path('/tmp/veritile-lean-toolchain/bin/lean'),Path('/tmp/veritile-lean-toolchain/bin/lake'))},
  'external_library_build_rechecked':False}
mode=sys.argv[1]
run=evidence/mode;run.mkdir(exist_ok=False)
leanpath=subprocess.check_output(['/tmp/veritile-lean-toolchain/bin/lake','env','printenv','LEAN_PATH'],cwd=repo,text=True).strip()
env=dict(os.environ);env['LEAN_PATH']=str(root)+':'+leanpath
search=[Path(p) for p in leanpath.split(':')]
before=deps(search);write(run/'dependencies-before.json',before)
results=[]
for rel in modules:
 source=root/rel;log=run/(source.stem+'.compile.log');before_hash=sha(source)
 command=['/tmp/veritile-lean-toolchain/bin/lean','--root='+str(root),str(source),'-o',str(source.with_suffix('.olean'))]
 start=time.monotonic()
 with log.open('w') as out:result=subprocess.run(command,env=env,stdout=out,stderr=subprocess.STDOUT)
 row={'module':rel,'command':command,'exit_code':result.returncode,'seconds':time.monotonic()-start,
  'source_sha256_before':before_hash,'source_sha256_after':sha(source),'log':str(log.relative_to(root)),'log_sha256':sha(log)}
 results.append(row)
 (run/'verification-results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(row),flush=True)
 assert before_hash==sha(source)
 if result.returncode:
  print(log.read_text()[:8000],flush=True)
  write(run/'dependencies-after.json',deps(search))
  raise SystemExit(result.returncode)
after=deps(search);write(run/'dependencies-after.json',after);assert before==after
logs='\n'.join((root/r['log']).read_text() for r in results)
assert logs.count('axiom footprint ⊆ standard base ✓')==17
assert not re.search(r'error:|warning:|sorryAx|DISALLOWED',logs)
print(json.dumps({'modules':len(modules),'dependencies':len(before['files']),'all_exit_zero':True,'named_axiom_checks':17,'noncircularity_checks':logs.count('non-circular ✓'),'warnings':0}),flush=True)
