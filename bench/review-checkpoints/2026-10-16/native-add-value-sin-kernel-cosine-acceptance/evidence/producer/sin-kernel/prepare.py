from pathlib import Path
import ast,hashlib,json,shutil,re,subprocess,os
W=Path(__file__).resolve().parent;R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');E=W/'evidence'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
src=R/'bench/tritonbench_g/sin_kernel/sin_kernel.py'
text=src.read_text();prefix=text.split('#'*100)[0]
(E/'source-original.py.txt').write_text(text);(E/'source-prefix.py').write_text(prefix)
old=E/'retained-prior-proofs';old.mkdir(exist_ok=True)
for p in src.parent.rglob('*'):
 if p.is_file():
  q=old/p.relative_to(src.parent);q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes())
template=W.parent/'add-value-finite-acceptance-revision1-wip'
shutil.copyfile(template/'compile.py',W/'compile.py')
env=dict(os.environ,PATH='/tmp/veritile-lean-toolchain/bin:'+os.environ['PATH'],LAKE_HOME='/tmp/veritile-lean-toolchain')
paths=subprocess.check_output(['/tmp/veritile-lean-toolchain/bin/lake','env','printenv','LEAN_PATH'],cwd=R,env=env,text=True).strip().split(':')
paths=[Path(x) if Path(x).is_absolute() else R/x for x in paths];paths.append(Path('/tmp/veritile-lean-toolchain/lib/lean'))
project={};external={}
def visit(mod):
 if mod in project or mod in external:return
 source=R/(mod.replace('.','/')+'.lean');rel=Path(mod.replace('.','/')+'.olean')
 obj=next((p/rel for p in paths if (p/rel).is_file()),None)
 assert obj,mod
 if not source.is_file():external[mod]={'olean':str(obj),'sha256':sha(obj)};return
 project[mod]={'source':str(source),'sha256':sha(source),'olean':str(obj),'olean_sha256':sha(obj)}
 for line in source.read_text().splitlines():
  m=re.match(r'^import\s+([A-Za-z0-9_.]+)',line)
  if m:visit(m[1])
roots=['bench.tritonbench_g.vector_addition.Proofs.NativeArithmetic','bench.tritonbench_g.sin_kernel.Proofs.Precision']
for root in roots:visit(root)
record={'purpose':'before first sin_kernel Lean import','roots':roots,'project':project,'external':external,'LEAN_PATH':[str(p) for p in paths]}
out=E/'dependencies-before.json';assert not out.exists();out.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'source_sha256':sha(src),'dependencies_sha256':sha(out),'project':len(project),'external':len(external)}))
