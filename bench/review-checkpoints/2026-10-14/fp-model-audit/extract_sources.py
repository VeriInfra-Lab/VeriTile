import ast,json,re,hashlib
from pathlib import Path
ROOT=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');out=Path('/tmp/veritile-fp-adequacy-20261014')
rows=json.loads((out/'public-contracts.json').read_text());by={}
for r in rows:by.setdefault(r['port'],[]).append(r)
allrows=[]
for task,rs in by.items():
 p=ROOT/rs[0]['python_file'];src=p.read_text();tree=ast.parse(src);needed={(f['name'],f['line']) for r in rs for f in r['python_functions']};nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and (n.name,n.lineno) in needed];assert len(nodes)==len(needed),(task,needed)
 records=[]
 for n in nodes:
  calls=[]
  for c in ast.walk(n):
   if isinstance(c,ast.Call):
    name=ast.unparse(c.func)
    if name.startswith(('tl.','libdevice.')) or name.endswith('.to'):calls.append(dict(operation=name,line=c.lineno,expression=ast.get_source_segment(src,c)))
  records.append(dict(name=n.name,line=n.lineno,end_line=n.end_lineno,source=ast.get_source_segment(src,n),calls=calls))
 lean=ROOT/rs[0]['file'];text=lean.read_text();imports=re.findall(r'(?m)^import (.+)$',text)
 kernellines=[dict(line=i+1,text=l) for i,l in enumerate(text.splitlines()) if re.match(r'(?:private |noncomputable )?(?:def|abbrev) \S+',l) and ('kernel' in l.lower() or 'surface' in l.lower())]
 record=dict(task=task,python_file=rs[0]['python_file'],python_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),functions=records,lean_file=rs[0]['file'],lean_sha256=hashlib.sha256(lean.read_bytes()).hexdigest(),lean_imports=imports,kernel_definitions=kernellines)
 allrows.append(record)
 print(task,':',','.join(sorted({c['operation'] for rec in records for c in rec['calls']})))
(out/'source-operations.json').write_text(json.dumps(allrows,ensure_ascii=False,indent=2)+'\n')
