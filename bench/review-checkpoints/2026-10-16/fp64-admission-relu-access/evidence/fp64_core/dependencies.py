"""Snapshot project source closure and imported project olean bytes."""
import hashlib,json,re,sys
from pathlib import Path
root=Path(__file__).resolve().parent
repo=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
modules=sorted(root.glob('bench/**/*.lean'))
seen={};local={str(p.relative_to(root)):sha(p) for p in modules};todo=list(modules)
while todo:
 p=todo.pop()
 for line in p.read_text().splitlines():
  if not line.startswith('import '):continue
  for name in line.removeprefix('import ').split():
   if not name.startswith(('VeriTile.','bench.')):continue
   rel=Path(*name.split('.')).with_suffix('.lean')
   if (root/rel).exists():continue
   if str(rel) in seen:continue
   source=repo/rel;compiled=repo/'.lake/build/lib/lean'/rel.with_suffix('.olean')
   assert source.is_file() and compiled.is_file(),rel
   seen[str(rel)]={'source_sha256':sha(source),'olean_path':str(compiled),'olean_sha256':sha(compiled)}
   todo.append(source)
result={'package_sources':local,'project_dependencies':dict(sorted(seen.items())),'toolchain':{'lean_toolchain_source_sha256':sha(repo/'lean-toolchain'),'lean_executable_sha256':sha(Path('/tmp/veritile-lean-toolchain/bin/lean'))},'external_libraries':'Trusted existing Lean core libraries; no fresh rebuild claimed.'}
out=root/'evidence'/sys.argv[1];out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'snapshot':str(out),'project_dependencies':len(seen),'sha256':sha(out)}))
