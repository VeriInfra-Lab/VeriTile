"""Read-only closure/symlink/source/olean fingerprint for the isolated grid."""
import hashlib,json,sys
from pathlib import Path
root=Path(__file__).resolve().parent;base=root.parent
core=base/'vector-addition-fp64-core-wip'
repo=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
assert sha(core/'evidence/receipt.json')=='de82b457d5717243302331868d49aed72430991a03c24362ba570a3d2d132448'
assert sha(core/'evidence/scope-review.json')=='6b8e4b5ce3fd501d86db34c3c259ab88dce4e169d2ae0ede219578acdf358e78'
review=base/'vector-addition-fp64-core-independent-review.json'
assert sha(review)=='e0025b76d42db20f9ca76fccdb6552f04cc40be43292adb4d18d4624366f0aa6'
receipt=read(core/'evidence/receipt.json')
for rel,h in receipt['modules'].items():assert sha(core/rel)==h
for rel,h in receipt['compiled_olean_sha256'].items():
 assert sha(core/rel)==h
 assert (root/rel).is_symlink() and (root/rel).resolve()==core/rel and sha(root/rel)==h
modules=sorted(root.glob('bench/**/*.lean'));seen={};todo=[str(p.relative_to(root)) for p in modules];external=set()
while todo:
 rel=todo.pop()
 if rel in seen:continue
 for directory,kind in [(root,'package'),(core,'frozen_core'),(repo,'repository')]:
  p=directory/rel
  if p.is_file():break
 else:external.add(rel);continue
 row={'source_kind':kind,'source_path':str(p),'sha256':sha(p)}
 if kind!='package':
  compiled=(directory if kind=='frozen_core' else repo/'.lake/build/lib/lean')/Path(rel).with_suffix('.olean')
  assert compiled.is_file();row.update({'olean_path':str(compiled),'olean_sha256':sha(compiled)})
 seen[rel]=row
 for line in p.read_text().splitlines():
  if not line.startswith('import '):continue
  for name in line.removeprefix('import ').split():
   if name.startswith('--'):break
   todo.append(name.replace('.','/')+'.lean')
for rel in ['bench/tritonbench_g/vector_addition/vector_addition.py','lakefile.toml','lake-manifest.json','lean-toolchain']:
 seen[rel]={'source_kind':'repository','source_path':str(repo/rel),'sha256':sha(repo/rel)}
route=core/'evidence/source-routes/receipt.json';rr=read(route)
assert sha(repo/'bench/tritonbench_g/vector_addition/vector_addition.py')==rr['source_sha256']
for row in rr['profiles']:assert sha(core/'evidence/source-routes'/row['ttir_path'])==row['ttir_sha256']
result={'files':dict(sorted(seen.items())),'external_imports':sorted(external),'frozen_core_receipt_sha256':sha(core/'evidence/receipt.json'),'frozen_core_scope_sha256':sha(core/'evidence/scope-review.json'),'frozen_core_independent_review_sha256':sha(review),'source_route_receipt_sha256':sha(route),'source_route_count':len(rr['profiles']),'core_import_olean_symlinks':{rel:str((root/rel).resolve()) for rel in receipt['compiled_olean_sha256']}}
out=root/'evidence'/sys.argv[1];out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'files':len(seen),'imported_oleans':sum('olean_sha256' in row for row in seen.values()),'sha256':sha(out)}))
