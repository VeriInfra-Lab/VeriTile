import sys
sys.dont_write_bytecode=True
import ast,hashlib,json,re,shutil
from pathlib import Path
W=Path(__file__).resolve().parent;R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');V=W.parent/'vector-finite-acceptance-revision2-wip';OLD=W.parent/'vector-custom-finite-acceptance-v2'
sys.path.insert(0,str(R/'bench'))
from adapter_common import imports
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
def save(name,obj):(W/'evidence'/name).write_text(json.dumps(obj,indent=2)+'\n')
modules={str(p.relative_to(W/'after')):sha(p) for p in sorted((W/'after').rglob('*.lean'))}
prior=load(V/'evidence/dependencies-before.json');seen=set();project={};external=set()
def visit(rel):
 if rel in seen:return
 seen.add(rel);p=W/'after'/rel if rel in modules else R/rel
 if not p.is_file():external.add(rel);return
 if rel not in modules:
  assert rel in prior['project'],rel
  v=prior['project'][rel];assert sha(p)==v['sha256'] and sha(v['object'])==v['object_sha256'],rel
  project[rel]=v
 for m in imports(p.read_text()):visit(m.replace('.','/')+'.lean')
for m in modules:visit(m)
assert len(project)==200 and len(external)==37
save('dependencies-before.json',{'capture_phase':'before final rebuild; after iterative development checks','project':dict(sorted(project.items())),'external_imports':sorted(external),'new_modules':modules})
inherit=W/'evidence/inherited';hist=inherit/'historical-custom-bounded-v2';hist.mkdir(exist_ok=True)
for name in ['receipt.json','catalog-supplement-task.proposed.json','declaration-proposal.json','domain-review.json','final-verification.json']:
 p=OLD/'evidence'/name
 if p.exists():shutil.copyfile(p,hist/name)
oldreceipt=load(OLD/'evidence/receipt.json')
for rel,h in oldreceipt['modules'].items():
 src=OLD/'after'/rel;assert sha(src)==h
 dst=hist/(rel+'.txt');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
save('historical-evidence-policy.json',{'old_package':str(OLD),'old_receipt_sha256':sha(OLD/'evidence/receipt.json'),'old_ports':616,'old_module_count':5,'active_registration':False,'policy':'Frozen bounded proposal preserved unchanged as historical evidence only. No new declaration, source view, module source map, or numerical authority assumes its five uninstalled modules. Current native proposal has its own seven modules and296 ports.'})
source=R/'bench/tritonbench_g/vector_addition_custom/vector_addition_custom.py';tree=ast.parse(source.read_text());f={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
review={'original_source_sha256':sha(source),'statements':{n:[ast.unparse(s) for s in f[n].body] for n in ['_add_kernel','custom_add']},'source_order':['empty_like','output.size(0); rankzero fails','BLOCK16','original first-dimension host grid','ordinary context/signature/compilation/grid parse/pointer packing/launch','return_same_allocated_tensor'],'has_cuda_guard':False,'host_size_is_numel':False,'full_dimension_domain':'Every shape dimension is nonnegative signed64, including arbitrarily large dimensions paired with zero numel.','pointer_extractor':load(V/'evidence/source-review.json')['pointer_extractor'],'all_extents_bound_i32':True,'host_extent_upper_exclusive':2**63,'source_return_at_two31':'vector_addition_custom_two31_return constructs return with identical allocated tensor and unchanged memory. CPU shape(2^31,0) has actual zero storage/data_ptr; ordinary null extraction can succeed. No CUDA premise.','masked_lanes':'no invented other value and no memory access','numerical_undefinedness':'scalarDefined guards actual conversion; no policy value can be stored outside domain','observation_semantics':'pure mathematical access/encoding/conversion/ownership observations; no native trap, rollback, race winner, or postfailure memory','trusted_memory':'ordinary typed load and pure byte packing/store; no arbitrary carrier readback law; final bytes independently characterized','test_harness':'Non-JIT test_add has no production caller. Four initial examples, then four dictionary calls on final empty tensors; import-time result_gold=test_add retained as harness.','gpu_launches':0,'large_allocations':0}
save('source-review.json',review)
# Backward scalar dataflow from the stored value, retaining each conversion,
# constant and exact opcode; only SSA/location/tile dimension are normalized.
O=W/'evidence/scalar-reuse';O.mkdir(exist_ok=True)
vlatest=load(V/'evidence/native-latest.json');clatest=load(W/'evidence/native-latest.json');vp=V/vlatest['receipt'];cp=W/clatest['receipt']
shutil.copyfile(vp,O/'vector-native-receipt.json')
def fingerprint(text):
 defs={};loads=[];store=None
 for raw in text.splitlines():
  line=re.sub(r' loc\(.*$','',raw.strip())
  m=re.match(r'(%[A-Za-z0-9_]+) = (.*)',line)
  if m:
   defs[m[1]]=m[2]
   if m[2].startswith('tt.load '):loads.append(m[1])
  if line.startswith('tt.store '):store=line
 assert len(loads)==2 and store is not None
 value=re.match(r'tt.store %\w+, (%\w+),',store)[1]
 def walk(v):
  body=defs[v]
  if v in loads:return ['loaded_word',loads.index(v),re.sub(r'tensor<(16|1024)x','tensor<Bx',body[body.index(' : ')+3:])]
  body=re.sub(r'tensor<(16|1024)x','tensor<Bx',body)
  args=re.findall(r'%[A-Za-z0-9_]+',body)
  return [re.sub(r'%[A-Za-z0-9_]+','%arg',body),[walk(a) for a in args]]
 return {'value':walk(value),'store_pointer_type':re.sub(r'tensor<(16|1024)x','tensor<Bx',store[store.index(' : ')+3:])}
records=[]
for p in sorted(cp.parent.glob('*.raw.ttir')):
 q=vp.parent/p.name;assert q.is_file()
 a=fingerprint(p.read_text());b=fingerprint(q.read_text());assert a==b,p.name
 shutil.copyfile(q,O/p.name)
 records.append({'pair':p.name.removesuffix('.raw.ttir'),'custom_path':str(p.relative_to(W)),'custom_sha256':sha(p),'vector_path':str((O/p.name).relative_to(W)),'vector_sha256':sha(q),'fingerprint':a})
assert len(records)==189
(O/'records.json').write_text(json.dumps(records,indent=2)+'\n')
save('scalar-reuse/receipt.json',{'status':'passed','paths':len(records),'method':'Backward stored-value dataflow preserving typed load leaves, exact scalar opcodes/conversions/constants, and store type. Only SSA names, locations and16/1024 vector lengths normalized. Index/host execution is custom and independently audited.','vector_receipt_sha256':sha(vp),'custom_receipt_sha256':sha(cp),'records_sha256':sha(O/'records.json'),'active_native_routes':183,'six_backend_reject_routes_preserved':True})
print(json.dumps({'dependencies':len(project),'external':len(external),'scalar_fingerprints':len(records),'historical_registration':False}))
