import ast,hashlib,json,re,sys
from pathlib import Path
p=Path(__file__).parent;r=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');inv=json.loads((p/'inventory.json').read_text());cat={x['task']:x for x in json.loads(Path(inv['catalog_path']).read_text())['tasks']}
for t in sys.argv[1:]:
 row=next(x for x in inv['tasks'] if x['task']==t);files=list(dict.fromkeys(row['python_files']+[x['file'] for x in cat[t]['declarations']]+row['source_files']));obs={};print('\nTASK',t)
 for f in files:
  b=(r/f).read_bytes();obs[f]=hashlib.sha256(b).hexdigest();s=b.decode();print('\nFILE',f)
  if f.endswith('.py'):
   starts=[x for x in ast.walk(ast.parse(s)) if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and not x.name.startswith('test_')];indices={i for x in starts for i in range(min([x.lineno]+[d.lineno for d in x.decorator_list]),x.end_lineno+1)}
   for i,l in enumerate(s.splitlines(),1):
    if i in indices:print(f'{i:4d} {l}')
  else:
   s=re.sub(r'/\-[\s\S]*?\-/',lambda m:'\n'*m.group().count('\n'),s);skip=False;statement=False
   for i,l in enumerate(s.splitlines(),1):
    if re.match(r'^(?:private )?(?:theorem|specification|lemma)\b',l):skip=False;statement=True
    elif re.match(r'^(?:noncomputable |private |protected )*(?:def|abbrev|structure|inductive|namespace|end|import|syntax|macro_rules|set_option|instance|variable)\b',l):skip=False;statement=False
    if not skip and l.strip() and not l.lstrip().startswith('--'):print(f'{i:4d} {l}')
    if statement and ':= by' in l:skip=True;statement=False
 (p/'read-inputs'/f'{t}.json').write_text(json.dumps({'task':t,'files':obs},indent=2)+'\n')
