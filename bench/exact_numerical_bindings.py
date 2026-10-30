"""Add exact scalar numerical bindings without accepting original programs.

Every source and evidence path is resolved inside the requested repository root. Structural source/graph/layout/host obligations are never rules.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from copy import deepcopy
import argparse,hashlib,json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from adapter_common import declarations,canonical,sha
CHECKPOINT = 'bench/review-checkpoints/2026-10-16/'
PACKAGE_PATHS = {
 'repo': '',
 'fp64_core': CHECKPOINT+'fp64-admission-relu-access/evidence/fp64_core',
 'fp64_grid': CHECKPOINT+'fp64-admission-relu-access/evidence/fp64_grid',
 'relu_access': CHECKPOINT+'fp64-admission-relu-access/evidence/relu_access',
 'relu_original': CHECKPOINT+'vector-integers-relu-program-order/evidence/relu',
 'relu_words': CHECKPOINT+'vector-mixed-fp32-relu-words/evidence/words',
}
SUPPLEMENTAL_INPUTS = {
 ('fp64_core','evidence/source-routes/receipt.json'):
  CHECKPOINT+'exact-numerical-bindings/evidence/fp64-source-routes/receipt.json',
}
PINS={
 ('fp64_core','evidence/receipt.json'):'de82b457d5717243302331868d49aed72430991a03c24362ba570a3d2d132448',
 ('fp64_grid','evidence/receipt.json'):'ae6a3ac04b77e40856757d7c0bde3c55e74fc8ea72e60e2986cceaae8159cab2',
 ('relu_access','evidence/receipt.json'):'0f845c08f838bb9f2f36c9c0fbcdbd799ba2ec7d5e394d41afa13ef9b4aa8228',
 ('relu_original','evidence/receipt.json'):'ed4a852afafce5efd027f62565e32afe42146eb9c0b140d872990a5da211fd90',
 ('relu_words','evidence/receipt.json'):'bf37c9bb89af701ca3737ce96fbc21d9a707d8a3d7c41e1a47d11a2e73c31276',
 ('fp64_grid','independent-review.json'):'0a9a5565f656af532a5f40a4de41d462b07e1a8fd81830781d82dbc8a63f7b13',
 ('relu_access','independent-review.json'):'3ca92cda61c0326ff431b6768d340af8f842deaead3da900ae5703373e515459',
}
FLOATS={'float16':'fp16','bfloat16':'bf16','float32':'fp32','float64':'fp64',
 'float8_e4m3fn':'fp8e4m3fn','float8_e5m2':'fp8e5m2'}
INTS=('bool','int8','uint8','int16','uint16','int32','uint32','int64','uint64')
CANONICAL={**{x:x for x in INTS},**FLOATS}
FP64_PREFIX='VeriTile.Bench.TritonBenchG.VectorAddition.FP64TypedCore.Numerics.'
FOLLOWUP=CHECKPOINT+'fp64-admission-relu-access/numerical-followup.json'
KIND='exact_scalar_numerical_bindings'
NEW_STATUS='pending_numeric_validation'


def fail(condition,message):
 if not condition:raise ValueError(message)

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

class BoundInputs:
 def __init__(self,root):
  self.root=Path(root).resolve();self.hashes={}
  self.roots={name:(self.root/path).resolve() for name,path in PACKAGE_PATHS.items()}
  fail(all(p.is_relative_to(self.root) for p in self.roots.values()),'package root escapes repository')
 def bind(self,package,path,expected=None):
  if (package,path) in SUPPLEMENTAL_INPUTS:
   path=SUPPLEMENTAL_INPUTS[package,path];package='repo'
  relative=Path(path);fail(not relative.is_absolute() and '..' not in relative.parts,'external binding path')
  root=self.roots[package].resolve();p=(root/relative).resolve()
  fail(p.is_relative_to(root) and p.is_file(),'missing or escaping input: '+package+':'+path)
  data=p.read_bytes();actual=sha(data)
  fail(expected is None or actual==expected,'bound source/hash changed: '+package+':'+path)
  ref={'package':package,'path':path,'sha256':actual};self.hashes[package+':'+path]=actual
  return data,ref
 def read(self,package,path,expected=None):
  data,ref=self.bind(package,path,expected);return json.loads(data),ref
 def public(self,package,row,receipt):
  data,source_ref=self.bind('repo',row['file'],receipt['modules'][row['file']])
  parsed=declarations(self.root/row['file'])
  p=parsed.get(row['parser_name']);fail(p is not None and p['kind']=='specification' and p['statement_sha256']==row['statement_sha256'],'exact public statement changed')
  return {'file':row['file'],'declaration':row['parser_name'],'statement_sha256':row['statement_sha256']},source_ref


def case(task,x,y,config):
 # The production scope checker imports this collector through its exact gate.
 # Resolve the shared case validator only when invoked, after module loading.
 from check_original_program_scope import validate_case
 profile={'inputs':{'x':CANONICAL[x]},'outputs':{'output':CANONICAL[x] if task=='vector_addition' else 'fp32'},'compute':{'value':'fp64' if task=='vector_addition' else 'fp32'}}
 if y is not None:profile['inputs']['y']=CANONICAL[y]
 result={'dtype_profile':profile,'configuration':config};validate_case(result);return result


def expected_fp64_ports(x,y):
 """Declared typed route, independently selected from actual source dtype cases."""
 f={'float16':'f16','bfloat16':'bf16','float32':'f32','float64':'f64','float8_e4m3fn':'f8E4M3FN','float8_e5m2':'f8E5M2'}
 out=[]
 def add(name,side,types,result,operation):out.append((name,side,types,result,operation))
 for side,d in (('x',x),('y',y)):
  if d.startswith('int'):add('signedTo64',side,['i'+d[3:]],'f64','arith.sitofp')
  elif d.startswith('uint'):add('unsignedTo64',side,['i'+d[4:]],'f64','arith.uitofp')
  elif d in ('float16','bfloat16'):
   add('halfTo32' if d=='float16' else 'bfloatTo32',side,[f[d]],'f32','arith.extf')
   add('singleTo64',side,['f32'],'f64','arith.extf')
  elif d=='float32':add('singleTo64',side,['f32'],'f64','arith.extf')
  elif d in ('float8_e4m3fn','float8_e5m2'):add('e4To64' if d=='float8_e4m3fn' else 'e5To64',side,[f[d]],'f64','tt.fp_to_fp')
  else:fail(d in ('bool','float64'),'unsupported fp64 input')
 add('add64','compute',['f64','f64'],'f64','arith.addf')
 if x=='bool' or x.startswith('int'):add('toSigned','output',['f64'],'i'+('8' if x=='bool' else x[3:]),'arith.fptosi')
 elif x.startswith('uint'):add('toUnsigned','output',['f64'],'i'+x[4:],'arith.fptoui')
 elif x!='float64':
  add({'float16':'toHalf','bfloat16':'toBFloat','float32':'toSingle','float8_e4m3fn':'toE4RTNE','float8_e5m2':'toE5RTNE'}[x],'output',['f64'],f[x], 'tt.fp_to_fp rounding=rtne' if x.startswith('float8_') else 'arith.truncf')
 return out


def derive_authority(root=ROOT):
 b=BoundInputs(root);receipts={};reviews=[]
 for (package,path),h in PINS.items():
  obj,ref=b.read(package,path,h)
  if path=='independent-review.json':reviews.append(ref)
  else:receipts[package]=obj
 grid=receipts['fp64_grid'];access=receipts['relu_access']
 access_deps,_=b.read('relu_access','evidence/final-dependencies-before.json',access['artifacts']['evidence/final-dependencies-before.json'])
 candidates,candidates_ref=b.read('fp64_grid','evidence/numerical-bindings.json',grid['numerical_bindings_sha256'])
 cfg64,cfg64_ref=b.read('fp64_grid','evidence/frontend-configuration.json',grid['frontend_configuration_sha256'])
 core=receipts['fp64_core'];routes,routes_ref=b.read('fp64_core','evidence/source-routes/receipt.json',grid['inherited_source_routes']['receipt_sha256'])
 fail(routes_ref['sha256']==core['source_route_audit']['receipt_sha256'],'core/grid source-route receipt references differ')
 numerics,registry_ref=b.read('fp64_core','evidence/numerical-correspondence.json',candidates['inherited_registry_sha256'])
 fail(candidates['families']==numerics['families'] and len(numerics['families'])==15,'changed numeric family inventory')
 corepath='bench/tritonbench_g/vector_addition/Proofs/FP64TypedCore.lean'
 core_source,core_ref=b.bind('repo',corepath,core['modules'][corepath])
 declared=set(__import__('re').findall(r'^  (\w+) :',core_source.decode().split('structure Numerics where\n')[1].split('\ninductive Cast')[0],__import__('re').M))
 fail(declared=={f['callback'] for f in numerics['families']},'callback field inventory mismatch')
 profilepath='bench/tritonbench_g/vector_addition/Proofs/FP64Profiles.lean'
 _,profiles_ref=b.bind('repo',profilepath,core['modules'][profilepath])
 original64='bench/tritonbench_g/vector_addition/vector_addition.py';_,original64_ref=b.bind('repo',original64,cfg64['source_sha256'])
 fp64_public=[b.public('fp64_grid',p,grid) for p in grid['public_contracts']]
 fail(len(fp64_public)==2,'fp64 declaration inventory mismatch')
 config64={'backend':'cuda','architecture':'sm90','warp_size':32,'BLOCK_SIZE':1024,
  'runtime_signature':{'n_elements':'i32'},'frontend_configuration_sha256':cfg64_ref['sha256'],
  'triton_version':cfg64['environment']['triton'],'python_version':cfg64['environment']['python'],
  'native_execution':'NOT_RUN'}
 expected_pairs={(x,y) for x in CANONICAL for y in CANONICAL if x=='float64' or y=='float64'}
 fail(len(expected_pairs)==29 and {(r['x'],r['y']) for r in routes['profiles']}==expected_pairs,'incomplete/unsupported fp64 profile inventory')
 bypair=defaultdict(list)
 for instance in candidates['instances']:bypair[instance['dtype_profile']['x'],instance['dtype_profile']['y']].append(instance)
 fail(set(bypair)==expected_pairs,'missing/extra fp64 candidate case')
 rules={};provenance={};requirements=[];case_rows=defaultdict(list)
 def rule(ident,task,relation):
  fail(ident not in rules,'duplicate numerical family')
  rules[ident]={'id':ident,'kind':'numerical_rule_conformance','operation_class':'scalar_primitive_conformance',
   'status':NEW_STATUS,'queue_status':'exact_profile_bound_validation_pending','affected_tasks':[task],
   'required_validation':relation,'native_whole_kernel_validated':False,'validation_evidence':None,'scope_bindings':[]}
 def register(ident,task,refs,callback,c,port):
  body={'task':task,'declarations':refs,'callbacks':[callback],**c}
  binding={'id':task+':'+sha(canonical({'rule':ident,'binding':body,'port':port['port']}))[:32],**body}
  rules[ident]['scope_bindings'].append(binding)
  key=ident+'::'+binding['id'];fail(key not in provenance,'duplicate scalar instance')
  provenance[key]={'assumption_id':ident,'scope_binding_id':binding['id'],'kind':'scalar_primitive_conformance',
   'status':'explicitly_deferred_unvalidated','validation_evidence':None,**port}
  for ref in refs:
   case_rows[(task,canonical(ref),canonical(c))].append({'callback':callback,'assumption_id':ident,'scope_binding_id':binding['id']})
 for family in numerics['families']:rule(family['id'],'vector_addition',family['required_correspondence'])
 route_map={(row['x'],row['y']):row for row in routes['profiles']}
 for x,y in sorted(expected_pairs):
  rows=bypair[x,y];expected=expected_fp64_ports(x,y)
  actual=[(i['callback'],i['port']['side'],i['port']['input_types'],i['port']['output_type'],i['port']['operation']) for i in rows]
  fail(actual==expected,'missing/extra/wrong fp64 scalar port')
  rr=route_map[x,y];_,raw_ref=b.bind('fp64_core','evidence/source-routes/'+rr['ttir_path'],rr['ttir_sha256'])
  refs=[ref for ref,_ in fp64_public] if x in FLOATS else [fp64_public[0][0]]
  c=case('vector_addition',x,y,config64)
  for i in rows:
   fail(i['callback_declaration']==FP64_PREFIX+i['callback'],'callback declaration mismatch')
   fail(i['callback_source_sha256']==core_ref['sha256'] and i['configuration_sha256']==cfg64_ref['sha256'] and
    i['raw_ttir_sha256']==raw_ref['sha256'] and i['original_source_sha256']==original64_ref['sha256'],'fp64 candidate provenance mismatch')
   fail([{k:p['parser_name'] if k=='declaration' else p[k] for k in ('file','declaration','statement_sha256')} for p in i['declaration_bindings']]==refs,'candidate statement binding mismatch')
   fail(i['status']=='explicitly_deferred_unvalidated' and i['validation_evidence'] is None,'candidate is not unvalidated')
   original=dict(i);key=original.pop('instance_key');fail(sha(json.dumps(original,sort_keys=True,separators=(',',':')).encode())==key,'candidate instance hash mismatch')
   register(i['rule_id'],'vector_addition',refs,i['callback_declaration'],c,
    {'callback_declaration':i['callback_declaration'],'callback_source':core_ref,'original_source':original64_ref,
     'raw_ttir':raw_ref,'configuration_ref':cfg64_ref,'port':i['port'],'scalar_domain':i['scalar_domain'],
     'scalar_path_predicate':'active_address; integer output additionally requires successful exact-domain conversion',
     'candidate_instance_key':key,'candidate_registry':candidates_ref,
     'structural_prerequisites_not_waived':i['structural_prerequisites_not_waived']})
 # ReLU source-selected scalar branches. Unsigned/bool zero remains fromNat;
 # signed literal is required only on the negative comparison branch.
 ra,ra_ref=b.read('relu_original','evidence/dtype-audit/receipt.json',receipts['relu_original']['artifacts']['evidence/dtype-audit/receipt.json'])
 _,relu_original_ref=b.bind('repo','bench/tritonbench_g/relu_triton_kernel/relu_triton_kernel.py',ra['source_sha256'])
 wp='bench/tritonbench_g/relu_triton_kernel/Proofs/IntegerWords.lean'
 word_source,words_ref=b.bind('repo',wp,receipts['relu_words']['modules'][wp])
 text=word_source.decode()
 for snippet in ('if signedComparison w v then M.fromInt none v.toInt else M.literal none .fp32 0',
  'if unsignedComparison w v then M.fromNat none v.toNat else M.literal none .fp32 0',
  'if booleanComparison v then M.fromNat none (boolBits v).toNat else M.literal none .fp32 0',
  'unsignedComparison w v = Bool.true','booleanComparison v = Bool.true'):
  fail(snippet in text,'ReLU source branch/callback changed')
 _,algebra_ref=b.bind('repo','VeriTile/Triton/Float/Structural.lean',access_deps['files']['VeriTile/Triton/Float/Structural.lean']['sha256'])
 relu_public,relu_source_ref=b.public('relu_access',access['public_contract'],access)
 relu_callbacks=access['public_contract']['unclosed_numeric_callbacks'];fail(len(relu_callbacks)==3,'ReLU callback inventory changed')
 fail(ra['constexprs']=={'N':1025,'block_size':1024},'ReLU frozen configuration changed')
 config_relu={'backend':'cuda','architecture':'sm90','warp_size':32,'N':1025,'block_size':1024,
  'frontend_configuration_sha256':ra_ref['sha256'],'triton_version':ra['environment']['triton'],
  'python_version':ra['environment']['python'],'native_execution':'NOT_RUN'}
 relu_defs=[('RELU-W-N1-FROMINT',relu_callbacks[0],'fromInt','Direct nonnegative signed-source integer to fp32, with exact native rounding/subnormal/overflow policy.'),
  ('RELU-W-N1-FROMNAT',relu_callbacks[1],'fromNat','Direct unsigned-source word or normalized unsigned bool i1 to fp32, including zero; exact native conversion policy.'),
  ('RELU-W-N2-LITERAL',relu_callbacks[2],'literal','Native positive fp32 zero for the signed-negative selected result branch only.')]
 for ident,_,_,relation in relu_defs:rule(ident,'relu_triton_kernel',relation)
 for d in INTS:
  c=case('relu_triton_kernel',d,None,config_relu)
  selected=[relu_defs[0],relu_defs[2]] if d.startswith('int') else [relu_defs[1]]
  raw_hashes=[h for path,h in access_deps['audited_ttir_sha256'].items() if path.endswith('/dtype-audit/'+d+'.raw.ttir')]
  fail(len(raw_hashes)==1,'missing/duplicate inherited ReLU TTIR')
  _,raw_ref=b.bind('relu_original','evidence/dtype-audit/'+d+'.raw.ttir',raw_hashes[0])
  for ident,callback,field,_ in selected:
   predicate='signed_negative' if field=='literal' else 'signed_nonnegative' if field=='fromInt' else 'all_unsigned_words' if d.startswith('uint') else 'all_256_bool_bytes_normalized_to_u1'
   source_type='signed'+d[3:] if d.startswith('int') else 'unsigned'+d[4:] if d.startswith('uint') else 'unsigned1_after_byte_nonzero'
   register(ident,'relu_triton_kernel',[relu_public],callback,c,
    {'callback_declaration':'VeriTile.Triton.FP.Structural.Algebra.'+field,'callback_source':algebra_ref,
     'source_use':words_ref,'original_source':relu_original_ref,'raw_ttir':raw_ref,'configuration_ref':ra_ref,
     'port':{'side':'result','original_storage_dtype':d,'input_types':['literal0' if field=='literal' else source_type],
      'output_type':'f32','operation':'fp32_literal_zero' if field=='literal' else 'arith.sitofp' if field=='fromInt' else 'arith.uitofp'},
     'scalar_domain':predicate,'scalar_path_predicate':predicate,'original_rule_family':'RELU-W-N2' if field=='literal' else 'RELU-W-N1',
     'structural_prerequisites_not_waived':['native storage/load-to-word representation','source/TTIR/compiler and host/allocator correspondence','all other N specializations remain unregistered']})
 # Entire supported case universe is derived independently of binding presence.
 contracts=[('vector_addition',ref,source_ref,
   [case('vector_addition',x,y,config64) for x,y in sorted(expected_pairs) if idx==0 or x in FLOATS])
  for idx,(ref,source_ref) in enumerate(fp64_public)]
 contracts.append(('relu_triton_kernel',relu_public,relu_source_ref,[case('relu_triton_kernel',d,None,config_relu) for d in INTS]))
 for task,ref,source_ref,supported in contracts:
  rows=[]
  for c in supported:
   links=case_rows.pop((task,canonical(ref),canonical(c)),None);fail(bool(links),'missing supported declaration case')
   rows.append({**c,'required_callbacks':sorted({l['callback'] for l in links}),
    'required_binding_refs':sorted(links,key=lambda x:(x['callback'],x['assumption_id'],x['scope_binding_id']))})
  requirements.append({'task':task,'declaration':ref,'source_ref':source_ref,
   'supported_cases_sha256':sha(canonical(sorted(supported,key=canonical))),
   'callback_union':sorted({c for row in rows for c in row['required_callbacks']}),'cases':rows,
   'support_boundary':'All listed dtype cases at the exact frozen frontend configuration only; no unlisted N/target/options or native execution.'})
 fail(not case_rows,'binding outside the supported case universe')
 for rr in rules.values():rr['scope_bindings'].sort(key=lambda x:x['id'])
 followup,follow_ref=b.read('repo',FOLLOWUP)
 fail(len(followup['rules'])==35,'descriptive followup baseline changed')
 unmigrated=[{'task':row['task'],'id':row['id'],'source_entry_sha256':sha(canonical(row)),
  'disposition':'not_registered_as_a_whole; requires reviewed scalar/structural split and exact declaration/case binding'} for row in followup['rules']]
 data={'schema':1,'kind':KIND,'rules':list(rules.values()),'provenance':provenance,'requirements':requirements,
  'unmigrated_descriptive_followups':{'source':follow_ref,'whole_entries_auto_registered':0,'entries':unmigrated},
  'input_sha256':dict(sorted(b.hashes.items())),'independent_source_reviews':reviews,
  'structural_policy':'Native source/graph/layout/address/allocator/host/errors remain blocking. A numeric queue never marks an original program complete.',
  'summary':{'new_scalar_families':18,'fp64_families':15,'relu_families':3,'scope_bindings':len(provenance),
   'fp64_ports':73,'fp64_declaration_port_bindings':120,'relu_ports':13,'relu_declaration_port_bindings':13,
   'declaration_cases':sum(len(c['cases']) for c in requirements),'registered_declarations':3,
   'unvalidated_numeric_instances':len(provenance),'original_program_completion_increment':0,'native_executions':0}}
 fail(len(provenance)==86 and sum(len(c['cases']) for c in requirements)==58,'unexpected exact scope counts')
 from reviewed_numerical_extensions import extend_reviewed_authority
 return extend_reviewed_authority(root,data)


def extend_queue(legacy,authority):
 result=deepcopy(legacy);fail(legacy['schema']==1,'legacy queue schema changed')
 oldids={r['id'] for r in legacy['rules']};fail(not oldids&{r['id'] for r in authority['rules']},'rule collision')
 result['rules']+=deepcopy(authority['rules'])
 result['exact_binding_provenance']=deepcopy(authority['provenance'])
 result['exact_callback_requirements']=deepcopy(authority['requirements'])
 result['exact_numerical_inputs']=deepcopy(authority['input_sha256'])
 result['unmigrated_descriptive_followups']=deepcopy(authority['unmigrated_descriptive_followups'])
 result['exact_numerical_summary']=deepcopy(authority['summary'])
 result['summary']['rule_profiles']=len(result['rules'])
 result['summary']['rule_statuses']=dict(sorted(Counter(r['queue_status'] for r in result['rules']).items()))
 for task in result['tasks']:
  additions=[r['id'] for r in authority['rules'] if task['task'] in r['affected_tasks']]
  task['rule_ids']+=additions
  if additions:task['exact_numerical_declarations']=[r['declaration'] for r in authority['requirements'] if r['task']==task['task']]
 return result


def validate_exact_queue(queue,authority,legacy):
 """Fail closed against source-derived authority, not self-rehashed claims."""
 fail(queue==extend_queue(legacy,authority),'exact numerical queue differs from frozen source/statement/case/port authority')
 return {'status':'passed','exact_bindings':len(authority['provenance']),
  'declaration_cases':sum(len(r['cases']) for r in authority['requirements']),
  'native_executions':0,'structural_acceptance':False}


def collect(root=ROOT):
 from deferred_numerical_experiments import collect_legacy
 authority=derive_authority(root);legacy=collect_legacy(root)
 queue=extend_queue(legacy,authority);validate_exact_queue(queue,authority,legacy)
 return legacy,authority,queue

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--root',type=Path,default=ROOT)
 parser.add_argument('--output',type=Path)
 args=parser.parse_args()
 legacy,authority,queue=collect(args.root)
 if args.output:
  args.output.mkdir(parents=True,exist_ok=True)
  for name,data in [('authority.json',authority),('deferred_numerical_experiments.json',queue)]:
   (args.output/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
 print(json.dumps({'status':'passed','rules':len(queue['rules']),**authority['summary']}))
