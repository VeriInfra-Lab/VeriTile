"""Read-only source renderer; writes review inputs only under this /tmp root."""
import ast,hashlib,json,re,sys
from pathlib import Path
root=Path(__file__).resolve().parent
repo=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
inv=json.loads((root/'inventory.json').read_text())
cat={r['task']:r for r in json.loads(Path(inv['catalog_path']).read_text())['tasks']}
for task in sys.argv[1:]:
 row=next(r for r in inv['tasks'] if r['task']==task)
 files=list(dict.fromkeys(row['python_files']+[d['file'] for d in cat[task]['declarations']]+row['source_files']))
 print('\nTASK '+task)
 observed={}
 for rel in files:
  p=repo/rel;text=p.read_text();lines=text.splitlines();observed[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
  print('\nFILE '+rel)
  if p.suffix=='.py':
   tree=ast.parse(text)
   tests=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name.startswith('test_')]
   excluded={i for n in tests for i in range(min([n.lineno]+[d.lineno for d in n.decorator_list]),n.end_lineno+1)}
   selected=[(i,s) for i,s in enumerate(lines,1) if i not in excluded]
  else:
   without=re.sub(r'/\-[\s\S]*?\-/',lambda m:'\n'*m.group().count('\n'),text)
   selected=[(i,s) for i,s in enumerate(without.splitlines(),1) if s.strip() and not s.lstrip().startswith('--')]
  for i,s in selected: print(f'{i:4d} {s}')
 out=root/'read-inputs';out.mkdir(exist_ok=True)
 (out/(task+'.json')).write_text(json.dumps({'task':task,'files':observed},indent=2)+'\n')
