"""Pin source closure and the exact project objects resolved in this overlay."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sys.path.insert(0,str(REPO/'bench'))
from adapter_common import imports
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
modules=json.loads((ROOT/'modules.json').read_text())
queue=modules[:];seen={};external=set()
while queue:
    rel=queue.pop()
    if rel in seen:continue
    source=ROOT/rel if (ROOT/rel).exists() else REPO/rel
    if not source.is_file():external.add(rel);continue
    row={'source_path':str(source),'source_sha256':sha(source)}
    if source.is_relative_to(REPO):
        obj=ROOT/Path(rel).with_suffix('.olean')
        assert obj.is_symlink() and obj.is_file(),obj
        row.update(olean_path=str(obj.resolve()),olean_sha256=sha(obj))
    seen[rel]=row
    queue.extend(m.replace('.','/')+'.lean' for m in imports(source.read_text()))
old=json.loads((ROOT/'evidence/preexisting-source-hashes.json').read_text())
assert all(sha(REPO/p)==h for p,h in old.items())
result={'schema':1,'files':dict(sorted(seen.items())),'external_library_imports':sorted(external),
        'preexisting_task_files_unchanged':old,
        'toolchain':{str(p):sha(p) for p in [REPO/'lean-toolchain',REPO/'lakefile.toml',REPO/'lake-manifest.json',Path('/tmp/veritile-lean-toolchain/bin/lean')]},
        'external_library_boundary':'Existing Lean/Mathlib/dependency libraries are trusted; no fresh rebuild is claimed.'}
mode=sys.argv[1];assert mode in ['before','after']
out=ROOT/('evidence/final-dependencies-'+mode+'.json')
out.write_text(json.dumps(result,indent=2)+'\n')
if mode=='after':assert result==json.loads((ROOT/'evidence/final-dependencies-before.json').read_text())
print(json.dumps({'mode':mode,'source_files':len(seen),'imported_project_oleans':sum('olean_sha256' in r for r in seen.values()),'sha256':sha(out)}))
