from pathlib import Path
import difflib,hashlib,json,re
W=Path(__file__).resolve().parent;P=Path('/tmp/veritile-standardization-20261015/sin-kernel-finite-acceptance-wip');F=W/'replay'
load=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=P/load(P/'evidence/native-latest.json')['receipt'];new=F/load(F/'evidence/native-latest.json')['receipt']
[oa]=list((P/'evidence/abi-profiles').glob('*/receipt.json'));[na]=list((F/'evidence/abi-profiles').glob('*/receipt.json'))
def norm(t):
 for path in [old.parent,new.parent,oa.parent,na.parent]:t=t.replace(str(path),'<RUN>')
 return t.replace(str(P),'<PACKAGE>').replace(str(F),'<PACKAGE>')
def scrub(x,skip=()):
 if isinstance(x,str):return norm(x)
 if isinstance(x,list):return [scrub(v,skip) for v in x]
 if isinstance(x,dict):return {k:scrub(v,skip) for k,v in x.items() if k not in skip}
 return x
ir=0;debug=[]
def compare_ir(a,b,name,raw_ext):
 global ir
 for ext in [raw_ext,'ttir','ttgir','llir','ptx']:
  p=a/(name+'.'+ext);q=b/(name+'.'+ext);assert p.exists()==q.exists()
  if not p.exists():continue
  left,right=norm(p.read_text()),norm(q.read_text())
  if ext=='ptx':
   def executable(s):
    m=re.search(r'^\s*\.section\s+\.debug',s,re.M)
    if m:
     assert not re.search(r'^\s*(?:\.visible\s+)?(?:\.entry|\.func)',s[m.start():],re.M)
     return s[:m.start()]
    return s
   if left!=right:
    out=W/'debug-diffs'/(('abi-' if a==oa.parent else 'dtype-')+name+'.diff');out.parent.mkdir(exist_ok=True)
    out.write_text(''.join(difflib.unified_diff(left.splitlines(True),right.splitlines(True),fromfile=str(p),tofile=str(q))));debug.append(str(out.relative_to(W)))
   left,right=executable(left),executable(right)
  assert left==right,(name,ext)
  ir+=1
left,right=load(old.with_name('records.json')),load(new.with_name('records.json'));assert len(left)==len(right)==22
for a,b in zip(left,right):
 assert scrub(a,['artifacts','sha256'])==scrub(b,['artifacts','sha256']),a['dtype']
 compare_ir(old.parent,new.parent,a['dtype'],'frontend.ttir')
assert load(old.with_name('binder.json'))==load(new.with_name('binder.json'))
arows,brows=load(oa.with_name('records.json')),load(na.with_name('records.json'));assert len(arows)==len(brows)==30
for a,b in zip(arows,brows):
 assert scrub(a,['raw_ttir_sha256','artifacts'])==scrub(b,['raw_ttir_sha256','artifacts'])
 compare_ir(oa.parent,na.parent,a['dtype']+'_'+str(a['n']),'raw.ttir')
oh=P/load(P/'evidence/host-latest.json')['receipt'];nh=F/load(F/'evidence/host-latest.json')['receipt']
a,b=load(oh.with_name('records.json')),load(nh.with_name('records.json'));assert len(a)==len(b)==75
assert scrub(a,['storage_base','data_ptr','output_id'])==scrub(b,['storage_base','data_ptr','output_id'])
metadata=0
for row in b:
 assert row['status']=='returned';x=row['x'];o=row['output'];launch=row['launch']
 assert o['dtype']==x['dtype'] and o['shape']==x['shape'] and o['device']==x['device'] and o['storage_offset']==0
 assert launch['n']==x['numel'] and launch['grid']==[(x['numel']+1023)//1024] and launch['block']==1024
 assert o['data_ptr']==(o['storage_base'] if o['numel'] else 0)
 if x['storage_bytes'] and o['storage_bytes']:assert x['storage_base']+x['storage_bytes']<=o['storage_base'] or o['storage_base']+o['storage_bytes']<=x['storage_base']
 if x['numel']==0:assert x['data_ptr']==0
 metadata+=2
assert load(oh.with_name('ast-traces.json'))==load(nh.with_name('ast-traces.json'))
for name in ['records.json','extract-records.json']:assert load(P/'evidence/setup-order'/name)==load(F/'evidence/setup-order'/name)
graphs=[]
for name,bits in [('float32',32),('float64',64)]:
 text=(new.parent/(name+'.frontend.ttir')).read_text();defs={};store=None;loaded=[]
 for line in text.splitlines():
  line=re.sub(r' loc\(.*$','',line.strip());m=re.match(r'(%\w+) = (.*)',line)
  if m:
   defs[m[1]]=m[2]
   if m[2].startswith('tt.load '):loaded.append(m[1])
  if line.startswith('tt.store '):store=line
 assert len(loaded)==1 and store
 value=re.match(r'tt.store %\w+, (%\w+),',store)[1]
 assert defs[value]==f'math.sin {loaded[0]} : tensor<1024xf{bits}>'
 assert store.endswith(f'tensor<1024x!tt.ptr<f{bits}>>')
 graphs.append({'dtype':name,'raw_sha256':sha(new.parent/(name+'.frontend.ttir')),'loaded_word':defs[loaded[0]],'stored_expression':defs[value],'store':store,'narrowing':False})
(W/'scalar-dataflow.json').write_text(json.dumps({'status':'passed','graphs':graphs,'native_scalar_validation':False},indent=2)+'\n')
assert ir==160 and metadata==150
result={'status':'passed','frontend_dtype_profiles':22,'backend_compiled':2,'binder_rejected':5,'architecture_rejected':2,'sine_type_rejected':13,'ABI_profiles':30,'equal_IR_artifacts':ir,'PTX_debug_diffs':debug,'scalar_dataflow_profiles':2,'real_CPU_metadata_cases':75,'tensor_metadata_records':metadata,'original_wrapper_AST_order_traces':18,'exact_signed_grid_parser_cases':7,'exact_i32_extractor_cases':15,'binder_extent_profiles':10,'GPU_launches':0,'native_scalar_validation':False,'cubin_comparison':'retained but not compared; no GPU execution claim'}
(W/'replay-comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='PTX_debug_diffs'}))
