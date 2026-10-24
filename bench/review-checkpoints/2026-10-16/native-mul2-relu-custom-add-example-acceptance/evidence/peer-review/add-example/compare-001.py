from pathlib import Path
import json,hashlib,re,difflib,collections
W=Path(__file__).resolve().parent;P=Path('/tmp/veritile-standardization-20261015/add-example-finite-acceptance-wip');F=W/'replay'
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
oldfront=P/'evidence/native-attempts/20261015T215117570205Z';newfront=next((F/'evidence/native-attempts').glob('*/receipt.json')).parent
oldback=P/'evidence/backend-attempts/20261015T215542684768Z';newback=next((F/'evidence/backend-attempts').glob('*/receipt.json')).parent
front=load(newfront/'records.json');back=load(newback/'records.json')
ircount=0;debug=[]
for a,b,kind in [(oldfront,newfront,'frontend'),(oldback,newback,'backend')]:
 left=load(a/'records.json');right=load(b/'records.json');assert len(left)==len(right)
 for x,y in zip(left,right):
  name=x['x']+'__'+x['y']
  for key in set(x)|set(y):
   if key in ['artifacts','ttir_sha256']:continue
   v=x.get(key);z=y.get(key)
   if isinstance(v,str):v=v.replace(str(P),'<PACKAGE>');z=z.replace(str(F),'<PACKAGE>')
   assert v==z,(kind,name,key,v,z)
  for ext in (['ttir'] if kind=='frontend' else ['ttir','ttgir','llir','ptx']):
   p=a/(name+'.'+ext);q=b/(name+'.'+ext);assert p.exists()==q.exists()
   if not p.exists():continue
   aa=p.read_text().replace(str(P),'<PACKAGE>');bb=q.read_text().replace(str(F),'<PACKAGE>')
   if ext=='ptx':
    def executable(text):
     m=re.search(r'^\s*\.section\s+\.debug',text,re.M)
     if m:
      assert not re.search(r'^\s*(?:\.visible\s+)?(?:\.entry|\.func)',text[m.start():],re.M)
      return text[:m.start()]
     return text
    if aa!=bb:
     diff=W/'debug-diffs'/(name+'.diff');diff.parent.mkdir(exist_ok=True);diff.write_text(''.join(difflib.unified_diff(aa.splitlines(True),bb.splitlines(True),fromfile=str(p),tofile=str(q))));debug.append(str(diff.relative_to(W)))
    aa=executable(aa);bb=executable(bb)
   assert aa==bb,(kind,name,ext)
   ircount+=1
assert load(oldfront/'binder.json')==load(newfront/'binder.json')
oldhost=P/'evidence/host-attempts/20261015T215521664431Z';newhost=next((F/'evidence/host-attempts').glob('*/receipt.json')).parent
def scrub(x):
 if isinstance(x,dict):return {k:scrub(v) for k,v in x.items() if k not in ['data_ptr','storage_base']}
 if isinstance(x,list):return [scrub(v) for v in x]
 return x
left=load(oldhost/'records.json');right=load(newhost/'records.json');assert len(right)==73 and scrub(left)==scrub(right)
assert load(oldhost/'abi.json')==load(newhost/'abi.json')
returns=0
for row in right:
 if row['status']!='returned':continue
 returns+=1;x=row['launch']['x'];out=row['output'];launch=row['launch']
 assert out['shape']==x['shape'] and out['dtype']==x['dtype'] and out['device']==x['device']
 assert out['storage_offset']==0 and out['data_ptr']==(out['storage_base'] if out['numel'] else 0)
 assert all(b==0 for b in launch['storage_bytes']) and launch['n']==x['numel'] and launch['grid']==[(x['numel']+3)//4] and launch['block']==4
 if x['storage_bytes'] and out['storage_bytes']:assert x['storage_base']+x['storage_bytes']<=out['storage_base'] or out['storage_base']+out['storage_bytes']<=x['storage_base']
assert returns==72
# Independently inspect the actual frontend dataflow and compare stored scalar
# graph with the already frozen shared-vector raw IR, without reusing custom
# source or any custom registration.
V=Path('/tmp/veritile-standardization-20261015/vector-finite-acceptance-revision2-wip')
vp=V/load(V/'evidence/native-latest.json')['receipt'];vreceipt=load(vp)
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
  if v in loads:return ['loaded_word',loads.index(v),re.sub(r'tensor<(4|1024)x','tensor<Bx',body[body.index(' : ')+3:])]
  body=re.sub(r'tensor<(4|1024)x','tensor<Bx',body);args=re.findall(r'%[A-Za-z0-9_]+',body)
  return [re.sub(r'%[A-Za-z0-9_]+','%arg',body),[walk(a) for a in args]]
 return {'value':walk(value),'store_pointer_type':re.sub(r'tensor<(4|1024)x','tensor<Bx',store[store.index(' : ')+3:])}
graphs=[]
for row in front:
 if row['status']!='frontend_generated':continue
 p=newfront/row['ttir'];q=vp.parent/(row['x']+'__'+row['y']+'.raw.ttir')
 assert sha(q)==vreceipt['artifacts'][q.name]
 raw=p.read_text();assert 'arith.cmpi slt' in raw and 'tensor<4xi32>' in raw and raw.index('tt.load')<raw.rindex('tt.load')<raw.index('tt.store')
 a=fingerprint(raw);assert a==fingerprint(q.read_text()),row
 graphs.append({'pair':[row['x'],row['y']],'add_raw_sha256':sha(p),'shared_vector_raw_sha256':sha(q),'fingerprint':a})
(W/'scalar-fingerprints.json').write_text(json.dumps({'shared_vector_receipt':str(vp),'shared_vector_receipt_sha256':sha(vp),'records':graphs},indent=2)+'\n')
result={'status':'passed','frontend_dtype_pairs_replayed':len(front),'backend_profiles_replayed':len(back),'backend_compiled':sum(r['status']=='compiled' for r in back),'backend_rejected':sum(r['status']=='rejected' for r in back),'IR_artifacts_equal':ircount,'PTX_debug_diffs':debug,'scalar_fingerprints_equal':len(graphs),'exact_wrapper_CPU_records':73,'zero_allocations_returned':72,'actual_allocation_errors':1,'binder_i32_profiles':8,'ABI_reference_profiles':10,'GPU_launches':0,'native_scalar_validation':False}
(W/'replay-comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='PTX_debug_diffs'}))
