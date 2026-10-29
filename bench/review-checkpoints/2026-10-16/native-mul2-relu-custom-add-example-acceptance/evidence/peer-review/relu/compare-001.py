from pathlib import Path
import json,hashlib,re,difflib,collections
W=Path(__file__).resolve().parent;P=Path('/tmp/veritile-standardization-20261015/relu-finite-acceptance-revision1-wip')
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=next((P/'evidence/native-attempts').glob('*/receipt.json')).parent;b=next((W/'replay/evidence/native-attempts').glob('*/receipt.json')).parent
before=load(a/'receipt.json');after=load(b/'receipt.json');left=load(a/'records.json');right=load(b/'records.json');assert len(left)==len(right)==33
for key in ['source','source_sha256','source_ast_identity','torch','triton','target','names','gpu_executed','native_backend_attempted','cases','frontend_generated','native_compiled']:assert before[key]==after[key],key
ircount=0;debugdiffs=[]
for lhs,rhs in zip(left,right):
 for key in set(lhs)|set(rhs):
  if key in ['artifacts','raw_ttir_sha256']:continue
  x=lhs.get(key);y=rhs.get(key)
  if isinstance(x,str):x=x.replace(str(a),'<SOURCE>');y=y.replace(str(b),'<SOURCE>')
  assert x==y,(lhs['case'],key,x,y)
 for ext in ['raw.ttir','ttir','ttgir','llir','ptx']:
  pa=a/(lhs['case']+'.'+ext);pb=b/(lhs['case']+'.'+ext);assert pa.exists()==pb.exists()
  if not pa.exists():continue
  aa=pa.read_text().replace(str(a),'<SOURCE>');bb=pb.read_text().replace(str(b),'<SOURCE>')
  if ext=='ptx':
   def executable(text):
    match=re.search(r'^[ \t]*\.section[ \t]+\.debug',text,re.M)
    if match:
     assert not re.search(r'^[ \t]*(?:\.visible[ \t]+)?(?:\.entry|\.func)',text[match.start():],re.M)
     return text[:match.start()]
    return text
   if aa!=bb:
    path=W/'debug-diffs'/(lhs['case']+'.ptx.diff');path.parent.mkdir(exist_ok=True);path.write_text(''.join(difflib.unified_diff(aa.splitlines(True),bb.splitlines(True),fromfile=str(pa),tofile=str(pb))));debugdiffs.append(str(path.relative_to(W)))
   aa=executable(aa);bb=executable(bb)
  assert aa==bb,(lhs['case'],ext)
  ircount+=1
old=load(next((P/'evidence/host-attempts').glob('*/records.json')));new=load(next((W/'replay/evidence/host-attempts').glob('*/records.json')))
def scrub(obj):
 if isinstance(obj,dict):return {k:scrub(v) for k,v in obj.items() if k not in ['data_ptr','storage_base','storage_bytes'] or k=='storage_bytes' and not isinstance(v,list)}
 if isinstance(obj,list):return [scrub(v) for v in obj]
 return obj
assert len(old)==len(new)==73 and scrub(old)==scrub(new)
sizes={'torch.bool':1,'torch.int8':1,'torch.uint8':1,'torch.int16':2,'torch.uint16':2,'torch.int32':4,'torch.uint32':4,'torch.int64':8,'torch.uint64':8,'torch.float16':2,'torch.bfloat16':2,'torch.float32':4,'torch.float64':8,'torch.complex32':4,'torch.complex64':8,'torch.complex128':16,'torch.float8_e4m3fn':1,'torch.float8_e5m2':1,'torch.float8_e4m3fnuz':1,'torch.float8_e5m2fnuz':1,'torch.float8_e8m0fnu':1,'torch.float4_e2m1fn_x2':1}
metadata=0
for row in new:
 for field in ['x','output']:
  if field not in row:continue
  m=row[field];assert m['data_ptr']==(m['storage_base']+m['storage_offset']*sizes[m['dtype']] if m['numel'] else 0)
  metadata+=1
 if row['status']=='returned':
  assert row['output']['shape']==row['x']['shape'] and row['output']['dtype']=='torch.float32' and row['output']['device']==row['x']['device']
  if row['x']['storage_bytes'] and row['output']['storage_bytes']:
   x=row['x'];y=row['output'];assert x['storage_base']+x['storage_bytes']<=y['storage_base'] or y['storage_base']+y['storage_bytes']<=x['storage_base']
result={'status':'passed','offline_profiles_replayed':33,'frontend_generated':23,'native_compiled':23,'IR_artifacts_equal_after_path_normalization':ircount,'PTX_debug_diffs':debugdiffs,'PTX_scope':'Executable PTX equal after path normalization; debug sections excluded only after proving no later entry/function. Full diffs retained.','host_records_replayed':73,'metadata_records_checked':metadata,'metadata_comparison':'All semantic fields agree. Raw allocator addresses checked independently; uninitialized output bytes retained as arbitrary and excluded from cross-run equality.','gpu_executions':0,'native_numeric_validation':False}
(W/'replay-comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
