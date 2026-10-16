from pathlib import Path
import ast,difflib,hashlib,json,re
W=Path(__file__).resolve().parent;P=Path('/tmp/veritile-standardization-20261015/add-value-finite-acceptance-revision1-wip');F=W/'replay'
load=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=P/load(P/'evidence/native-latest.json')['receipt'];new=F/load(F/'evidence/native-latest.json')['receipt']
oa=P/load(P/'evidence/abi-latest.json')['path'];na=F/load(F/'evidence/abi-latest.json')['path']
def norm(t):
 for path in [old.parent,new.parent,oa.parent,na.parent]:t=t.replace(str(path),'<RUN>')
 return t.replace(str(P),'<PACKAGE>').replace(str(F),'<PACKAGE>')
def scrub(x):
 if isinstance(x,str):return norm(x)
 if isinstance(x,list):return [scrub(v) for v in x]
 if isinstance(x,dict):return {k:scrub(v) for k,v in x.items()}
 return x
original=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile/bench/tritonbench_g/add_value/add_value.py')
nodes=lambda t:[ast.dump(n) for n in ast.parse(t).body if isinstance(n,ast.FunctionDef) and n.name in ['puzzle1','puzzle1_kernel']]
assert nodes(original.read_text())==nodes((new.parent/'source.py').read_text())
oldrows=load(old)['records'];newrows=load(new)['records'];assert len(oldrows)==len(newrows)==22
ir=0;debug=[]
def compare_ir(a,b,name):
 global ir
 for ext in ['raw.ttir','ttir','ttgir','llir','ptx']:
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
for a,b in zip(oldrows,newrows):
 assert a['dtype']==b['dtype'] and a['exit_code']==b['exit_code']==0
 assert scrub(a['result'])==scrub(b['result']),a['dtype']
 compare_ir(old.parent,new.parent,a['dtype'])
arows,brows=load(oa.with_name('records.json')),load(na.with_name('records.json'));assert len(arows)==len(brows)==14
for a,b in zip(arows,brows):
 assert {k:v for k,v in a.items() if k not in ['raw_sha256','artifacts']}=={k:v for k,v in b.items() if k not in ['raw_sha256','artifacts']}
 compare_ir(oa.parent,na.parent,str(a['n']))
for rel in ['evidence/semantics/address-samples.json','evidence/semantics/wrapper-traces.json','evidence/semantics/metadata.json','evidence/setup-order/records.json']:
 assert load(P/rel)==load(F/rel),rel
sem=load(F/'evidence/semantics/receipt.json');assert sem['compiled_C_vs_integer_reference_pointer_checks']==1425408 and sem['original_wrapper_AST_cases']==18 and sem['CPU_metadata_examples']==6
inventory=load(P/'evidence/scalar-operation-inventory.json');graphs=[]
for profile in inventory['profiles']:
 name=profile['dtype'];p=new.parent/(name+'.raw.ttir');defs={};loadvars=[];store=None
 for line in p.read_text().splitlines():
  line=re.sub(r' loc\(.*$','',line.strip());m=re.match(r'(%\w+) = (.*)',line)
  if m:
   defs[m[1]]=m[2]
   if m[2].startswith('tt.load '):loadvars.append(m[1])
  if line.startswith('tt.store '):store=line
 assert len(loadvars)==1 and store
 stored=re.match(r'tt.store %\w+, (%\w+),',store)[1];ops=[];leaves=[]
 def walk(v):
  if v=='%value':leaves.append('runtime_fp32');return ['runtime_fp32']
  if v==loadvars[0]:leaves.append('input');return ['loaded_word',defs[v].split(' : ')[-1]]
  body=defs[v];args=re.findall(r'%\w+',body);children=[walk(a) for a in args]
  if not body.startswith(('tt.splat ','arith.constant ')):ops.append(body.split(' ')[0])
  return [re.sub(r'%\w+','%arg',body),children]
 graph=walk(stored)
 if name=='bool':expected=['arith.cmpi','arith.uitofp','arith.addf','arith.fptosi']
 elif name.startswith('uint'):expected=['arith.uitofp','arith.addf','arith.fptoui']
 elif name.startswith('int'):expected=['arith.sitofp','arith.addf','arith.fptosi']
 elif name in ['float16','bfloat16']:expected=['arith.extf','arith.addf','arith.truncf']
 elif name=='float32':expected=['arith.addf']
 elif name=='float64':expected=['arith.extf','arith.addf']
 else:expected=['tt.fp_to_fp','arith.addf','tt.fp_to_fp']
 assert ops==expected and sorted(leaves)==['input','runtime_fp32'],(name,ops,leaves)
 if name=='bool':assert 'cmpi ne,' in p.read_text() and store.endswith('tensor<1024x!tt.ptr<i8>>')
 if name.startswith('float8'):assert ', rounding = rtne' in defs[stored]
 graphs.append({'dtype':name,'observed_scalar_operations':ops,'numeric_ports':[op['port'] for op in profile['operations']],'dataflow':graph,'stored_pointer_type':store.split(' : ')[-1],'raw_sha256':sha(p)})
(W/'scalar-dataflow.json').write_text(json.dumps({'status':'passed','profiles':graphs,'runtime_argument_bits':0x41200000,'native_numerical_validation':False},indent=2)+'\n')
result={'status':'passed','original_AST_equal':True,'dtype_profiles':22,'dtype_backend_compiled':15,'binder_rejected':5,'architecture_rejected':2,'ABI_profiles':14,'equal_IR_artifacts':ir,'PTX_debug_diffs':debug,'scalar_dataflow_profiles':len(graphs),'compiled_C_pointer_checks':1425408,'exact_wrapper_AST_stub_traces':18,'real_CPU_metadata_cases':6,'exact_signed_grid_parser_cases':7,'GPU_launches':0,'native_scalar_validation':False,'cubin_comparison':'retained but not compared; debug/path-dependent executable-container bytes do not establish device execution'}
assert ir==145
(W/'replay-comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='PTX_debug_diffs'}))
