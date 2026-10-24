from pathlib import Path
import re,hashlib,json,subprocess
W=Path(__file__).resolve().parent;R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
paths=subprocess.check_output(['/tmp/veritile-lean-toolchain/bin/lake','env','printenv','LEAN_PATH'],cwd=R,text=True).strip().split(':')
paths=[Path(x) if Path(x).is_absolute() else R/x for x in paths]
paths.append(Path('/tmp/veritile-lean-toolchain/lib/lean'))
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
roots=['bench.tritonbench_g.vector_addition.Proofs.FiniteRoutes','bench.tritonbench_g.vector_addition.Proofs.MixedFP32Defined','bench.tritonbench_g.vector_addition.Proofs.MixedSmallDefined','bench.tritonbench_g.add_example.Kernels']
for root in roots:visit(root)
record={'purpose':'before first add_example Lean import','roots':roots,'project':project,'external':external,'LEAN_PATH':[str(p) for p in paths]}
out=W/'evidence/dependencies-before.json';assert not out.exists();out.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'path':str(out),'sha256':sha(out),'project':len(project),'external':len(external)}))
