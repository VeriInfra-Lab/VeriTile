import ast,hashlib,json,sys
from pathlib import Path
p=Path(__file__).parent;r=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');sys.dont_write_bytecode=True;sys.path.insert(0,str(r/'bench'));import adapter_common as a
inv=json.loads((p/'inventory.json').read_text());cat={x['task']:x for x in json.loads(Path(inv['catalog_path']).read_text())['tasks']}
for t in sys.argv[1:]:
 row=next(x for x in inv['tasks'] if x['task']==t);files=list(dict.fromkeys(row['python_files']+[x['file'] for x in cat[t]['declarations']]+row['source_files']));obs={};print('\nTASK',t)
 for f in files:
  b=(r/f).read_bytes();obs[f]=hashlib.sha256(b).hexdigest();s=b.decode();print('\nFILE',f)
  if f.endswith('.py') and f not in {p['file'] for p in row['production']}:continue
  if f.endswith('.py'):
   starts=[x for x in ast.walk(ast.parse(s)) if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and not x.name.startswith('test_')];indices={i for x in starts for i in range(min([x.lineno]+[d.lineno for d in x.decorator_list]),x.end_lineno+1)}
   for i,l in enumerate(s.splitlines(),1):
    if i in indices:print(f'{i:4d} {l}')
  else:
   ds=a.declarations(r/f)
   for x in cat[t]['declarations']:
    if x['file']==f:print(a.split_statement(ds[x.get('parser_name') or x['declaration']]['text'])[0])
 (p/'read-inputs'/f'{t}.json').write_text(json.dumps({'task':t,'files':obs},indent=2)+'\n')
