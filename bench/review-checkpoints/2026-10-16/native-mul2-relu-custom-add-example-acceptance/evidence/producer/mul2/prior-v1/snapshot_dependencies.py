import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json
W=Path(__file__).resolve().parent
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sys.path.insert(0,str(R/'bench'))
from adapter_common import imports
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
roots=[Path('/tmp/veritile-tritonbench-lake/packages')/p/'.lake/build/lib/lean' for p in ['Cli','batteries','Qq','aesop','proofwidgets','importGraph','LeanSearchClient','plausible','mathlib']]+[Path('/tmp/veritile-lean-toolchain/lib/lean')]
project={};external={}
def visit(name):
 if name in project or name in external:return
 rel=Path(*name.split('.'));src=R/rel.with_suffix('.lean')
 if src.exists():
  obj=R/'.lake/build/lib/lean'/rel.with_suffix('.olean')
  assert obj.is_file(),str(obj)
  project[name]={'source':str(src),'relative_source':str(src.relative_to(R)),'sha256':sha(src),'olean':str(obj),'olean_sha256':sha(obj)}
  for child in imports(src.read_text()):visit(child)
 else:
  obj=next((r/rel.with_suffix('.olean') for r in roots if (r/rel.with_suffix('.olean')).is_file()),None)
  assert obj,name
  external[name]={'olean':str(obj),'sha256':sha(obj)}
for name in ['bench.tritonbench_g.relu_triton_kernel.Proofs.ByteCodec','bench.tritonbench_g.triton_mul2.Kernels']:visit(name)
record={'project':dict(sorted(project.items())),'external_objects':dict(sorted(external.items())),'input_sha256':{str(R/p):sha(R/p) for p in ['lean-toolchain','lakefile.toml','lake-manifest.json','bench/tritonbench_g/triton_mul2/triton_mul2.py','bench/adapter_common.py']}}
target=W/'evidence'/sys.argv[1]
assert not target.exists(),'Preserve every snapshot'
target.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'snapshot':str(target),'sha256':sha(target),'project':len(project),'external':len(external)}))
