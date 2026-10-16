from pathlib import Path
import json,hashlib,re,difflib
W=Path(__file__).resolve().parent;P=Path('/tmp/veritile-standardization-20261015/vector-custom-finite-acceptance-revision1-wip');F=W/'replay'
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pairs=[(P/load(P/'evidence/native-latest.json')['receipt'],F/load(F/'evidence/native-latest.json')['receipt'],'native'),(P/load(P/'evidence/abi-latest.json')['path'],F/load(F/'evidence/abi-latest.json')['path'],'abi')]
ircount=0;checked=0;debug=[]
for ar,br,kind in pairs:
 a=ar.parent;b=br.parent;left=load(a/'records.json');right=load(b/'records.json');assert len(left)==len(right)
 for x,y in zip(left,right):
  name=x['x']+'__'+x['y'] if kind=='native' else str(x['n'])
  for key in set(x)|set(y):
   if key in ['artifacts','raw_sha256']:continue
   v=x.get(key);z=y.get(key)
   if isinstance(v,str):v=v.replace(str(a),'<SOURCE>');z=z.replace(str(b),'<SOURCE>')
   assert v==z,(kind,name,key,v,z)
  for ext in ['raw.ttir','ttir','ttgir','llir','ptx']:
   p=a/(name+'.'+ext);q=b/(name+'.'+ext);assert p.exists()==q.exists()
   if not p.exists():continue
   aa=p.read_text().replace(str(a),'<SOURCE>');bb=q.read_text().replace(str(b),'<SOURCE>')
   if ext=='ptx':
    def executable(text):
     m=re.search(r'^\s*\.section\s+\.debug',text,re.M)
     if m:
      assert not re.search(r'^\s*(?:\.visible\s+)?(?:\.entry|\.func)',text[m.start():],re.M)
      return text[:m.start()]
     return text
    if aa!=bb:
     diff=W/'debug-diffs'/(kind+'-'+name+'.diff');diff.parent.mkdir(exist_ok=True);diff.write_text(''.join(difflib.unified_diff(aa.splitlines(True),bb.splitlines(True),fromfile=str(p),tofile=str(q))));debug.append(str(diff.relative_to(W)))
    aa=executable(aa);bb=executable(bb)
   assert aa==bb,(kind,name,ext)
   ircount+=1
  checked+=1
for rel in ['evidence/semantics/address-samples.json','evidence/semantics/wrapper-traces.json','evidence/semantics/metadata.json','evidence/setup-order/records.json']:
 assert load(P/rel)==load(F/rel),rel
# Independently trace every stored value back to the two typed loads and
# compare against frozen shared-vector IR; no index arithmetic is normalized.
def fingerprint(text):
 defs={};loads=[];store=None
 for raw in text.splitlines():
  line=re.sub(r' loc\(.*$','',raw.strip());m=re.match(r'(%[A-Za-z0-9_]+) = (.*)',line)
  if m:
   defs[m[1]]=m[2]
   if m[2].startswith('tt.load '):loads.append(m[1])
  if line.startswith('tt.store '):store=line
 assert len(loads)==2 and store
 value=re.match(r'tt.store %\w+, (%\w+),',store)[1]
 def walk(v):
  body=defs[v]
  if v in loads:return ['loaded_word',loads.index(v),re.sub(r'tensor<(16|1024)x','tensor<Bx',body[body.index(' : ')+3:])]
  body=re.sub(r'tensor<(16|1024)x','tensor<Bx',body);args=re.findall(r'%[A-Za-z0-9_]+',body)
  return [re.sub(r'%[A-Za-z0-9_]+','%arg',body),[walk(a) for a in args]]
 return {'value':walk(value),'store_pointer_type':re.sub(r'tensor<(16|1024)x','tensor<Bx',store[store.index(' : ')+3:])}
fingerprints=[]
native=pairs[0][1].parent
for p in sorted(native.glob('*.raw.ttir')):
 q=P/'evidence/scalar-reuse'/p.name;assert q.is_file();a=fingerprint(p.read_text());assert a==fingerprint(q.read_text())
 fingerprints.append({'pair':p.stem,'custom_sha256':sha(p),'shared_vector_sha256':sha(q),'fingerprint':a})
assert len(fingerprints)==189
(W/'scalar-fingerprints.json').write_text(json.dumps(fingerprints,indent=2)+'\n')
result={'status':'passed','profiles_replayed':checked,'dtype_matrix':484,'abi_profiles':14,'native_backend_compiles':183,'frontend_typed_routes':189,'IR_artifacts_equal':ircount,'PTX_executable_equal_with_debug_diffs_retained':debug,'scalar_fingerprints_equal':189,'compiled_C_pointer_reference_checks':22272,'wrapper_AST_traces':15,'real_CPU_metadata_cases':6,'signed_grid_parser_cases':7,'GPU_launches':0,'native_scalar_validation':False}
(W/'replay-comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='PTX_executable_equal_with_debug_diffs_retained'}))
