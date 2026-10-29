import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,ctypes,hashlib,json,subprocess,sysconfig
W=Path(__file__).resolve().parent;R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');E=W/'evidence'
sys.path.insert(0,str(R/'bench'))
from adapter_common import imports
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();load=lambda p:json.loads(Path(p).read_text())
def save(n,d):(E/n).write_text(json.dumps(d,indent=2)+'\n')
modules={str(p.relative_to(W/'after')):sha(p) for p in sorted((W/'after').rglob('*.lean'))};seen=set();project={};external=set()
def visit(rel):
 if rel in seen:return
 seen.add(rel);p=W/'after'/rel if rel in modules else R/rel
 if not p.is_file():external.add(rel);return
 if rel not in modules:
  obj=(W/'build-overlay'/rel).with_suffix('.olean').resolve();assert obj.is_file(),(rel,obj)
  project[rel]={'sha256':sha(p),'object':str(obj),'object_sha256':sha(obj)}
 for m in imports(p.read_text()):visit(m.replace('.','/')+'.lean')
for rel in modules:visit(rel)
assert not any('/sin_kernel/' in p or '/add_value/' in p or '/vector_addition_custom/' in p for p in project)
save('dependencies-before.json',{'capture_phase':'before final rebuild; after iterative development checks','project':dict(sorted(project.items())),'external_imports':sorted(external),'new_modules':modules})
print(json.dumps({'project_dependencies':len(project),'external_imports':len(external),'modules':len(modules)}))
