"""Read-only independent checks of the frozen candidate; prints review evidence only."""
from pathlib import Path
import hashlib,json,re
root=Path(__file__).resolve().parents[5]
task=root/'bench/repairs/tritonbench_g/int8_dequant_matmul_scalar_cast'
e=task/'kernel-evidence';path=e/'delivery.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
expected='11f4ef71a014f6e916ccf9fa1e1c264536655767bbebfea0ef896789f9c4913d'
assert sha(path)==expected
p=json.loads(path.read_text());seen=[]
def resolve(s):
 q=Path(s);return q if q.is_absolute() else root/q
def visit(x):
 if isinstance(x,dict):
  if isinstance(x.get('path'),str) and isinstance(x.get('sha256'),str):
   assert sha(resolve(x['path']))==x['sha256'],x['path'];seen.append(x['path'])
  for y in x.values():visit(y)
 elif isinstance(x,list):
  for y in x:visit(y)
visit(p)
for name,digest in p['source_pins'].items():assert sha(resolve(name))==digest,name
for name,digest in p['predecessor_all_pins_preserved'].items():assert sha(resolve(name))==digest,name
receipts=p['module_receipts'];assert len(receipts)==21
paths=None
for name,entry in receipts.items():
 r=json.loads(resolve(entry['receipt']['path']).read_text());assert r==entry['build'];assert r['exit_code']==0 and r['source']==name;assert r['source_sha256']==sha(resolve(name))
 for obj,digest in r['objects'].items():assert sha(obj)==digest,obj
 if paths is None:paths=r['lean_path']
 else:assert paths==r['lean_path']
def imports(source):
 names=[]
 for line in source.read_text().splitlines():
  m=re.match(r'^(?:public\s+)?import\s+(.+)$',line)
  if m:names.extend(n for n in m[1].split() if re.fullmatch(r'[A-Za-z0-9_.]+',n))
 return names
def closure(roots):
 done=set();queue=list(roots)
 while queue:
  x=queue.pop()
  if x in done:continue
  done.add(x)
  for name in imports(x):
   child=root/(name.replace('.','/')+'.lean')
   if child.is_file():queue.append(child)
 return done
c=json.loads(resolve(p['lean_checks']['closure']['path']).read_text());visit(c)
repo=c['repository_imports'];assert len(repo)==248
assert {str(x.relative_to(root)) for x in closure([resolve(m) for m in receipts])}==set(repo)
ilean_checked=[];missing=[];effective_checked=[]
for name,item in repo.items():
 source=resolve(name);assert sha(source)==item['source_sha256'],name
 rel=Path(name).with_suffix('.olean');effective=next(Path(d)/rel for d in paths if (Path(d)/rel).is_file())
 recorded=resolve(item['artifact']['path']);assert effective.resolve()==recorded.resolve(),(name,effective,recorded)
 assert sha(effective)==item['artifact']['sha256']
 ilean=effective.with_suffix('.ilean')
 if ilean.is_file():
  idat=json.loads(ilean.read_text());declared=imports(source);actual=[x[0] for x in idat['directImports']]
  assert declared==actual,(name,declared,actual)
  assert idat['module']=='.'.join(Path(name).with_suffix('').parts),(name,idat['module'])
  ilean_checked.append(str(ilean))
 else:
  assert name not in receipts and item['status']=='existing dependency cache pinned; not rebuilt here'
  missing.append({'source':name,'effective_object':str(effective),'sha256':sha(effective),'boundary':item['status']})
 effective_checked.append(str(effective))
 if 'receipt' in item:
  r=json.loads(resolve(item['receipt']['path']).read_text());assert r['exit_code']==0 and r['source_sha256']==sha(source),name
  for obj,digest in r['objects'].items():assert sha(obj)==digest,obj
 if name in receipts:assert str(effective) in receipts[name]['build']['objects'],name
assert len(missing)==37 and len(ilean_checked)==211
public=closure([task/'Correct.lean',task/'FPEquiv.lean',task/'Contract.lean']);audit=closure([task/'Audit.lean'])
a=(e/'combined-audit.log').read_text();assert a.count('axiom footprint ⊆ standard base ✓')==51 and a.count('non-circular ✓')==9
assert ': error:' not in a and 'DISALLOWED' not in a and 'sorryAx' not in a
for token in ['Quote.fullResult','Quote.Result.replay_eq','Quote.sourceCommit_eq','Quote.sourceGrid_eq','zero_hook_prepares','ScalarCast.correct','ScalarCast.fp_equiv']:
 assert token+': axiom footprint ⊆ standard base ✓' in a,token
original=task.with_name('int8_dequant_matmul')/'int8_dequant_matmul_repaired.py';source=task/'int8_dequant_matmul_scalar_cast.py'
text=original.read_text()
for name in ['M','N','K','stride_am','stride_ak','stride_bk','stride_bn','stride_cm','stride_cn']:
 old=f'{name} = {name}.to(tl.int64)';new=f'{name} = tl.cast({name}, tl.int64)';assert text.count(old)==1;text=text.replace(old,new)
assert source.read_text()==text
binder=json.loads((e/'binder-validation.json').read_text());visit(binder)
assert len(binder['cases'])==10 and all(x['compilation']=='passed' for x in binder['cases'])
assert binder['no_specialization_disabled'] and not binder['GPU_execution'] and not binder['actual_JIT_launch']
assert binder['source_audit']['configurations']==418
assert all(c['signature']['stride_ak']==c['signature']['stride_bn']==c['signature']['stride_cn']=='constexpr' for c in binder['cases'])
assert all(c['signature']['M']=='constexpr' for c in binder['cases'] if c['shape'][0]==1)
assert p['source_execution']['whole_quote_lowering_assumption']==False and p['source_execution']['numerical_result_assumptions']==[]
result={'all_available_checks_passed':True,'reviewed_delivery_sha256':expected,'current_candidate_module_receipts':21,
 'repository_sources':len(repo),'effective_objects_checked':len(effective_checked),'source_ilean_direct_imports_checked':len(ilean_checked),
 'public_source_closure':len(public),'audit_source_closure':len(audit),'predecessor_pins_preserved':len(p['predecessor_all_pins_preserved']),
 'record_pins_verified':len(seen),'unique_record_files':len(set(seen)),'exact_nine_cast_only_byte_delta':True,'axiom_checks':51,
 'non_circular_checks':9,'binder_profiles':10,'compiler_artifacts':sum(len(c['artifacts']) for c in binder['cases']),
 'recompiled':False,'compiler_or_numeric_experiments_rerun':False,'GPU_execution':False,
 'reused_dependency_cache_without_ilean':missing,
 'first_review_attempt':'Session4729 exit1 because it required an ilean sidecar for every reused dependency. Inventory showed37 absent sidecars, all already marked reused cache; no candidate receipt/object/source mismatch was found. No build was run.'}
print(json.dumps(result,indent=2))
