"""Build concrete deferred numerical queue candidates for both public specs."""
import hashlib,json,sys
from pathlib import Path
root=Path(__file__).resolve().parent;base=root.parent;core=base/'vector-addition-fp64-core-wip'
repo=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
write=lambda p,d:p.write_text(json.dumps(d,indent=2)+'\n')
sys.path.insert(0,str(repo/'bench'))
from adapter_common import declarations
publicfile='bench/tritonbench_g/vector_addition/FP64GridCorrect.lean'
parsed=declarations(root/publicfile)
names=['vector_addition_fp64_defined_grid_correctness','vector_addition_fp64_floating_grid_correctness']
contracts=[{'file':publicfile,'module':'bench.tritonbench_g.vector_addition.FP64GridCorrect','parser_name':n,'declaration':'VeriTile.Bench.TritonBenchG.VectorAddition.FP64GridCorrect.'+n,'kind':parsed[n]['kind'],'statement_sha256':parsed[n]['statement_sha256']} for n in names]
assert all(c['kind']=='specification' for c in contracts)
audit=base/'vector-addition-dtype-extension-wip/evidence/extension-audit/receipt.json';old=read(audit)
routes=read(core/'evidence/source-routes/receipt.json')
assert routes['prior_audit_sha256']==sha(audit)
config={'source_audit_receipt_sha256':sha(audit),'source_sha256':old['source_sha256'],'extracted_kernel_sha256':old['extracted_kernel_sha256'],'environment':old['environment'],'offline_target':old['scope']['offline_target'],'constexprs':old['scope']['constexprs'],'frontend_options':old['scope']['frontend_options'],'binding_limit':'Observed frontend configuration only; native lowering, numerical behavior and GPU execution remain unvalidated.'}
configfile=root/'evidence/frontend-configuration.json';write(configfile,config)
families=read(core/'evidence/numerical-correspondence.json')['families'];family_by_name={r['callback']:r for r in families}
floatmap={'float16':'f16','bfloat16':'bf16','float32':'f32','float64':'f64','float8_e4m3fn':'f8E4M3FN','float8_e5m2':'f8E5M2'}
instances=[]
def add(row,name,side,source_types,target_type,operation,original_dtype,domain):
 family=family_by_name[name]
 applicable=contracts if row['x'] in floatmap else contracts[:1]
 d={'rule_id':family['id'],'callback':name,'callback_declaration':'VeriTile.Bench.TritonBenchG.VectorAddition.FP64TypedCore.Numerics.'+name,'callback_source':'bench/tritonbench_g/vector_addition/Proofs/FP64TypedCore.lean','callback_source_sha256':sha(core/'bench/tritonbench_g/vector_addition/Proofs/FP64TypedCore.lean'),'dtype_profile':{'x':row['x'],'y':row['y'],'output':row['x'],'compute':'float64'},'port':{'side':side,'original_storage_dtype':original_dtype,'input_types':source_types,'output_type':target_type,'operation':operation},'configuration_path':'evidence/frontend-configuration.json','configuration_sha256':sha(configfile),'original_source_sha256':routes['source_sha256'],'raw_ttir_path':str(core/'evidence/source-routes'/row['ttir_path']),'raw_ttir_sha256':row['ttir_sha256'],'declaration_bindings':applicable,'scalar_domain':domain,'numerical_correspondence':family['required_correspondence'],'status':'explicitly_deferred_unvalidated','validation_evidence':None,'queue_registration_status':'concrete_candidate_ready_for_parent_registration; not an accepted numerical result','structural_prerequisites_not_waived':['task-local AST to original source/compiler correspondence','native storage/address/allocator/host and inactive-lane/poison behavior']}
 d['instance_key']=hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest();instances.append(d)
for row in routes['profiles']:
 for side in ['x','y']:
  d=row[side]
  if d.startswith('int'):
   add(row,'signedTo64',side,['i'+d[3:]],'f64','arith.sitofp',d,'Every source-width two\'s-complement word at an active address; direct conversion.')
  elif d.startswith('uint'):
   add(row,'unsignedTo64',side,['i'+d[4:]],'f64','arith.uitofp',d,'Every source-width unsigned word at an active address; direct conversion.')
  elif d in ['float16','bfloat16']:
   add(row,'halfTo32' if d=='float16' else 'bfloatTo32',side,[floatmap[d]],'f32','arith.extf',d,'Every raw source word, including signed zero, subnormal, infinity and NaN payloads.')
   add(row,'singleTo64',side,['f32'],'f64','arith.extf',d,'The raw f32 intermediate produced by the preceding explicit extension; no omitted stage.')
  elif d=='float32':
   add(row,'singleTo64',side,['f32'],'f64','arith.extf',d,'Every raw f32 word, including specials and payloads.')
  elif d.startswith('float8_'):
   add(row,'e4To64' if d=='float8_e4m3fn' else 'e5To64',side,[floatmap[d]],'f64','tt.fp_to_fp',d,'Every raw fp8 word; direct widening without a synthesized fp32 stage.')
  else:assert d in ['bool','float64']
 add(row,'add64','compute',['f64','f64'],'f64','arith.addf',row['x']+' + '+row['y'],'Every ordered pair of promoted f64 words at active addresses, including all exceptional encodings; no finiteness premise.')
 d=row['x']
 if d=='bool' or d.startswith('int'):
  width='8' if d=='bool' else d[3:]
  add(row,'toSigned','output',['f64'],'i'+width,'arith.fptosi',d,'Active outputs only: callback must implement direct toward-zero conversion on the exact finite/range domain. OutputDefined requires some; out-of-domain none is retained. Bool stores preserve raw signed8 result byte.')
 elif d.startswith('uint'):
  add(row,'toUnsigned','output',['f64'],'i'+d[4:],'arith.fptoui',d,'Active outputs only: direct toward-zero conversion on the exact finite/range domain. OutputDefined requires some; out-of-domain none is retained.')
 elif d!='float64':
  name={'float16':'toHalf','bfloat16':'toBFloat','float32':'toSingle','float8_e4m3fn':'toE4RTNE','float8_e5m2':'toE5RTNE'}[d]
  add(row,name,'output',['f64'],floatmap[d],('tt.fp_to_fp rounding=rtne' if d.startswith('float8_') else 'arith.truncf'),d,'Every raw f64 result at active addresses; direct narrowing, including signed zero/subnormal/overflow/infinity/NaN policy. No f32 intermediate and no finiteness premise.')
assert len(instances)==73 and len({i['instance_key'] for i in instances})==73
assert {i['callback'] for i in instances}==set(family_by_name)
assert sum(len(i['declaration_bindings']) for i in instances)==120
report={'schema':1,'kind':'concrete_numerical_correspondence_queue_candidates','public_contracts':contracts,'families':families,'family_count':15,'port_instance_count':73,'declaration_port_bindings':120,'inherited_registry_sha256':sha(core/'evidence/numerical-correspondence.json'),'configuration_sha256':sha(configfile),'model':'task-local typed scalar/memory/grid interpreter; not shared FP.Structural','new_numerical_families':0,'numerical_callbacks_remain_unvalidated':True,'instances':instances}
write(root/'evidence/numerical-bindings.json',report)
print(json.dumps({'families':15,'port_instances':73,'declaration_port_bindings':120,'contracts':contracts,'sha256':sha(root/'evidence/numerical-bindings.json')},indent=2))
