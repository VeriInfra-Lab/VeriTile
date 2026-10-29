"""Fail-closed frozen-candidate integrity/compile checks; no Lean invocation."""
import json,re,sys
from pathlib import Path
sys.dont_write_bytecode=True
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P))
from audit_util import sha,imports,lean_code_mask

def verify():
 r=json.loads((P/'receipt.json').read_text());checks=0
 def must(ok,msg):
  nonlocal checks
  checks+=1
  if not ok:raise ValueError(msg)
 def bound(path,digest):must(Path(path).is_file() and sha(path)==digest,'hash mismatch '+str(path))
 for rel,h in r['artifact_sha256'].items():
  q=P/rel;must(q.resolve().is_relative_to(P),'artifact escaped '+rel);bound(q,h)
 for path,h in r['external_input_sha256'].items():bound(path,h)
 rows=json.loads((P/'last-build.json').read_text());seen=set();axioms=0
 roots=[Path(x) for x in r['ordered_roots']]
 for row in rows:
  module=row['module'];must(module not in seen,'duplicate module');seen.add(module)
  source=Path(row['source']);bound(source,row['source_before_sha256'])
  must(row['source_before_sha256']==row['source_after_sha256'],'source drift '+module)
  must(row['exit_code']==0 and row['changed_inputs']==[],'failed compile '+module)
  must(row['ordered_roots']==r['ordered_roots'],'roots mismatch')
  must(row['direct_objects_before']==row['direct_objects_after'],'dependency drift '+module)
  must(set(row['direct_objects_before'])==set(imports(source.read_text())),'imports mismatch '+module)
  bound(row['log'],row['log_sha256']);bound(row['object'],row['object_sha256']);bound(row['lean'],row['lean_sha256'])
  code=lean_code_mask(source.read_text());must(not re.search(r'\b(sorry|admit|axiom)\b',code),'unproved declaration '+module)
  log=Path(row['log']).read_text();must('error:' not in log and 'DISALLOWED' not in log,'bad log '+module)
  named=re.findall(r'^#axiomsClean\s+(\S+)',code,re.M)
  for name in named:
   must(re.search(r'(?:^|\.)'+re.escape(name)+r': axiom footprint ⊆ standard base ✓',log,re.M) is not None,'missing axiom check '+name)
  axioms+=len(named)
  for dep,objects in row['direct_objects_before'].items():
   first=next((base/(dep.replace('.','/')+'.olean') for base in roots if (base/(dep.replace('.','/')+'.olean')).is_file()),None)
   must(first is not None and str(first) in objects,'not first-hit '+dep)
   must(any(first.resolve().is_relative_to(base.resolve()) for base in roots),'object target escaped approved roots '+dep)
   for q,h in objects.items():bound(q,h)
 closure=json.loads((P/'project-closure.json').read_text())
 for dep,info in closure['project_modules'].items():
  bound(info['source'],info['source_sha256'])
  must(imports(Path(info['source']).read_text())==info['imports'],'closure imports mismatch '+dep)
 for dep,objects in {**{k:v['objects'] for k,v in closure['project_modules'].items()},**closure['external_direct_modules']}.items():
  first=next((base/(dep.replace('.','/')+'.olean') for base in roots if (base/(dep.replace('.','/')+'.olean')).is_file()),None)
  must(first is not None and str(first) in objects,'closure first-hit changed '+dep)
  must(any(first.resolve().is_relative_to(base.resolve()) for base in roots),'closure object target escaped '+dep)
  for q,h in objects.items():bound(q,h)
 must(len(rows)==r['new_module_count'],'module count')
 must(axioms==r['named_axiom_checks'],'axiom count')
 must(r['public_specifications']==re.findall(r'^specification\s+(\S+)',(P/r['public_entry']).read_text(),re.M),'public inventory')
 print(json.dumps({'status':'passed','checks':checks,'new_modules':len(rows),'named_axioms':axioms,
  'public_specifications':len(r['public_specifications']),'gpu_run':False},indent=2))
if __name__=='__main__':verify()
