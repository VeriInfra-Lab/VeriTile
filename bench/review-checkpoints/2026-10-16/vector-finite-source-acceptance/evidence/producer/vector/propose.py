"""Exact finite scope and scalar correspondence proposal, never self-approval."""
from collections import Counter
from copy import deepcopy
import hashlib,json,re,sys
from pathlib import Path
W=Path(__file__).resolve().parent
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sys.path.insert(0,str(R/'bench'))
from adapter_common import canonical,declarations,sha
from check_original_program_scope import declaration_scope,validate_case,validate_deferrals
load=lambda p:json.loads(Path(p).read_text())
hash_file=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):(W/'evidence'/name).write_text(json.dumps(obj,indent=2)+'\n')
TASK='vector_addition';BASE='bench/tritonbench_g/vector_addition/';PUBLIC=BASE+'FiniteCorrect.lean'
NAME='vector_addition_finite_source_outcomes';NS='VeriTile.Bench.TritonBenchG.VectorAddition.'
PYTHON=BASE+'vector_addition.py'
modules=list(load(W/'evidence/staging-map.json').values())
module_hashes={p:hash_file(W/'after'/p) for p in modules}
ref={'file':PUBLIC,'declaration':NAME,'statement_sha256':declarations(W/'after'/PUBLIC)[NAME]['statement_sha256']}
proposal=load(W/'evidence/inherited/typed-host-numerical-proposal.json')
profiles={(p['x_dtype'],p['y_dtype']):p for p in proposal['profiles']}
frontend=load(W/'evidence/frontend/frontend-probes.json');fr={(x['x'],x['y']):x for x in frontend}
promotions={(x['x'],x['y']):x for x in load(W/'evidence/frontend/promotion-matrix.json')}
receipt=load(W/'evidence/frontend/receipt.json')
assert len(profiles)==189 and sum(len(p['operations']) for p in profiles.values())==308
for p in profiles.values():
 for op in p['operations']:
  assert hash_file(op['source'])==op['source_sha256']
for pair,row in fr.items():assert (row['status']=='frontend_generated')==(pair in profiles)
ledger=load(R/'bench/original_program_scope.json');row=deepcopy(next(x for x in ledger['tasks'] if x['task']==TASK));inventory=row['source_inventory']
stages=[x['id'] for x in inventory['stages'] if not x['may_be_test_harness']]
dtypeids=[x['id'] for x in inventory['dtypes']];branchids=[x['id'] for x in inventory['branches']]
def dtype(d):return {'float16':'fp16','float32':'fp32','float64':'fp64','bfloat16':'bf16','int1':'bool',
 'float8_e4m3fn':'fp8e4m3fn','float8_e5m2':'fp8e5m2','float8_e4m3fnuz':'fp8e4m3fnuz','float8_e5m2fnuz':'fp8e5m2fnuz',
 'fp8e4nv':'fp8e4m3fn','fp8e5':'fp8e5m2','f8e4':'fp8e4m3fn','f8e5':'fp8e5m2','float8e4nv':'fp8e4m3fn','float8e5':'fp8e5m2'}.get(d,d)
configs=[]
for pair,f in fr.items():
 x,y=pair;p=profiles.get(pair)
 terminal='returned' if p else 'signature_rejected' if 'fnuz' in x or 'fnuz' in y else 'addition_rejected'
 for size,lo,hi in [('empty',0,1),('one_block',1,1025),('multiple_blocks',1025,2**31)]:
  for outcome in ['allocation_error','cuda_assertion',terminal]:
   compute={'index':'int32'}
   if p:compute.update(value=dtype(p['compute_dtype']),store=dtype(p['native_store_dtype']))
   c={'dtype_profile':{'inputs':{'x':dtype(x),'y':dtype(y)},'outputs':{'output':dtype(x)},'compute':compute},
    'configuration':{'entrypoint':'add','source_variant':'original','BLOCK_SIZE':1024,'n_elements':{'lower_inclusive':lo,'upper_exclusive':hi},
     'runtime_n_signature':'i32','size_class':size,'outcome':outcome,'grid':'ceil_div_output_numel_1024',
     'target':receipt['scope']['offline_target'],'triton_version':receipt['environment']['triton'],'torch_version':receipt['environment']['torch'],
     'frontend_options_sha256':sha(canonical(receipt['scope']['frontend_options'])),'debug':False,'instrumentation_mode':'none',
     'layouts':'all_actual_strided_views_with_safe_flat_source_spans','offsets':'nonnegative_storage_offsets_and_uint64_no_wrap',
     'permissions':'stable_readable_x_y_and_readable_writable_output_bytes','alignment':'natural_element_width','view_metadata':'trusted_Torch_storage_base_extent_data_ptr_and_empty_like_strides',
     'allocation':'fresh_empty_like_x_preserve_format_metadata','input_aliasing':'arbitrary_x_y_storage_overlap','y_logical_numel':'unrestricted_when_actual_backing_span_covers_n',
     'device':'same_cuda_device_for_success','values':'ValueDomain_typed_inputs_and_defined_active_float_to_integer_conversions',
     'first_rejection_argument':'x' if 'fnuz' in x else 'y' if 'fnuz' in y else 'none'}}
   validate_case(c);configs.append((pair,c))
cases=[c for _,c in configs];assert len(cases)==2601 and len({canonical(c) for c in cases})==2601

def callback(port):
 if port.startswith('b.rest.binary'):return 'VeriTile.Triton.FP.Structural.Algebra.binary'
 if port.startswith('b.rest.cast'):return 'VeriTile.Triton.FP.Structural.Algebra.cast'
 if port.startswith('b.wide.'):return NS+'FP64TypedCore.Numerics.'+port.split('.')[-1]
 if port.startswith('FP.'):return 'VeriTile.Triton.'+port
 return NS+port.split(' ')[0]

def operation_types(pair,p,op):
 port=op['port'];x,y=pair;compute=dtype(p['compute_dtype']);cfg=op['configuration']
 if port.startswith('b.rest.binary'):t=dtype(port.split()[2]);return [t,t],t
 if port.startswith('b.rest.cast'):a,b=port.split()[2:4];return [dtype(a)],dtype(b)
 if port in ['MixedSmall.integerRounded','MixedFP32.integerRounded']:
  integer=x if x=='bool' or x.startswith(('int','uint')) else y
  target='fp16' if '-> fp16' in op['operation'] else 'fp32'
  return ['uint1_after_bool_normalization' if integer=='bool' else integer],target
 if port=='MixedSmall.addTarget .fp16':return ['fp16','fp16'],'fp16'
 if port=='MixedSmall.widenHalf':return ['fp16'],'fp32'
 if port=='MixedSmall.widenBFloat':return ['bf16'],'fp32'
 if port=='MixedSmall.narrowBFloat':return ['fp32'],'bf16'
 if port=='FP.add':return ['fp32','fp32'],'fp32'
 if port.startswith('FP.IntegerConversion.'):
  return ['fp32'],('int' if port.endswith('toSigned') else 'uint')+str(cfg['width'])
 if port.startswith('b.wide.'):
  name=port.split('.')[-1]
  if name=='add64':return ['fp64','fp64'],'fp64'
  if name in ['signedTo64','unsignedTo64']:return [('int' if name=='signedTo64' else 'uint')+str(cfg['width'])],'fp64'
  if name in ['halfTo32','bfloatTo32']:return ['fp16' if name=='halfTo32' else 'bf16'],'fp32'
  if name=='singleTo64':return ['fp32'],'fp64'
  if name in ['e4To64','e5To64']:return ['fp8e4m3fn' if name=='e4To64' else 'fp8e5m2'],'fp64'
  if name in ['toSigned','toUnsigned']:return ['fp64'],('int' if name=='toSigned' else 'uint')+str(cfg['width'])
  return ['fp64'],{'toHalf':'fp16','toBFloat':'bf16','toSingle':'fp32','toE4RTNE':'fp8e4m3fn','toE5RTNE':'fp8e5m2'}[name]
 raise ValueError(port)

rules={};provenance={};requirements=[];links={}
for pair,c in configs:
 p=profiles.get(pair);selected=p['operations'] if p and c['configuration']['outcome']=='returned' and c['configuration']['size_class']!='empty' else []
 refs=[]
 for index,op in enumerate(selected):
  port=op['port'];cb=callback(port);ident='VECTOR-FINITE-'+re.sub('[^A-Z0-9]+','-',port.upper()).strip('-')
  if ident not in rules:rules[ident]={'id':ident,'kind':'numerical_rule_conformance','operation_class':'scalar_primitive_conformance',
   'status':'pending_numeric_validation','queue_status':'exact_profile_bound_validation_pending','affected_tasks':[TASK],
   'required_validation':'Exact native scalar correspondence for '+port+' at every registered operand/output dtype and configuration, including rounding, subnormal, NaN, infinity and signed-zero behavior on the explicitly defined source domain.',
   'native_whole_kernel_validated':False,'validation_evidence':None,'scope_bindings':[]}
  body={'task':TASK,'declarations':[ref],'callbacks':[cb],**deepcopy(c)}
  bid=TASK+':finite:'+sha(canonical({'rule':ident,'binding':body,'source_port':op,'ordinal':index}))[:32]
  rules[ident]['scope_bindings'].append({'id':bid,**body});link={'assumption_id':ident,'scope_binding_id':bid}
  links.setdefault(cb,[]).append(link);refs.append({'callback':cb,**link})
  src=str(Path(op['source']).relative_to(R));raw=fr[pair];it,ot=operation_types(pair,p,op)
  provenance[ident+'::'+bid]={'kind':'scalar_primitive_conformance','status':'explicitly_deferred_unvalidated','validation_evidence':None,**link,
   'callback_declaration':cb,'correspondence_kind':'parametric_callback' if port.startswith('b.') else 'concrete_scalar_port',
   'port':{'name':port,'ordinal':index,'operation':op['operation'],'input_types':it,'output_type':ot,'scalar_configuration':op['configuration']},
   'scalar_domain':'All active typed inputs satisfying ValueDomain; active float-to-integer outputs must be defined; no assumption supplies a result for undefined conversion.',
   'scalar_path_predicate':'active_source_element; selected_route='+p['family'],'native_execution':False,'source_statement':op['statement'],
   'original_source':{'path':PYTHON,'sha256':hash_file(R/PYTHON)},'public_statement':ref,'lean_source_sha256':module_hashes,
   'scalar_source':{'path':src,'sha256':op['source_sha256']},
   'execution_source':p['execution_binding'],'host_source':p['host_binding'],
   'raw_ttir':{'path':'evidence/frontend/'+raw['ttir_path'],'sha256':raw['ttir_sha256']},
   'frontend_receipt':{'path':'evidence/frontend/receipt.json','sha256':hash_file(W/'evidence/frontend/receipt.json')},
   'producer_scalar_inventory':{'path':'evidence/inherited/typed-host-numerical-proposal.json','sha256':hash_file(W/'evidence/inherited/typed-host-numerical-proposal.json')},
   'configuration':c['configuration'],'structural_prerequisites_not_waived':['SafeAPI finite view extents and no-wrap','fresh allocation and immutable inputs','actual typed source execution and host order','ValueDomain excludes undefined native conversions','trusted ordinary metadata/allocation/frontend/load/store semantics']}
 requirements.append({**deepcopy(c),'required_callbacks':sorted({r['callback'] for r in refs}),'required_binding_refs':refs})
for r in rules.values():r['scope_bindings'].sort(key=lambda b:b['id'])
deferrals=[{'kind':'numerical_rule_conformance','assumption_id':ident,'rule_sha256':sha(canonical(r)),
 'scope_binding_id':b['id'],'dtype_profile':b['dtype_profile'],'configuration':b['configuration'],'required_validation':r['required_validation']}
 for ident,r in rules.items() for b in r['scope_bindings']]
union=sorted(links)
scope={'source_inventory_sha256':sha(canonical(inventory)),'stages':stages,'dtypes':dtypeids,'branches':branchids,
 'cases':{ident:deepcopy(cases) for ident in stages+dtypeids+branchids}}
declaration={**ref,'kind':'specification','model_kind':'backend_value_contract','role':'parametric_value_contract',
 'scope_relation':'full_declared_stage','source_variant':'original_source_translation','covers_stages':stages,
 'dtype_profile':'All289 ordered canonical dtype pairs:189 source execution routes,64 first signature and36 addition rejections; empty, one-block and multiple-block signed-i32 length classes. Actual strided flat views and fresh output allocation.',
 'dtype_scope_relation':'complete_enumerated_wrapper_dtype_domain','unclosed_numeric_callbacks':union,'extra_numeric_law_ids':[],
 'native_binding_status':'not_established','review_basis':'PRODUCER PROPOSAL requiring independent substantive root review. Actual source-order host, selected existing typed execution and physical source memory replay are connected; allocation/CUDA/frontend failures, all result values, frames, aliasing, indices and pointer widths are explicit.',
 'original_scope':scope}
declaration_scope(declaration,inventory)
coverage={k:[] for k in ['stages','dtypes','branches']}
for dimension,items in inventory.items():
 for item in items:
  if dimension=='stages' and item['may_be_test_harness']:
   coverage[dimension].append({'id':item['id'],'status':'test_harness','explanation':'Non-JIT test_add has no production caller, calls add four times and is invoked by import-time result_gold initialization. Existing HostWrapper retains this test call order; no RNG implementation proof is required.'})
  else:coverage[dimension].append({'id':item['id'],'status':'covered','cases':deepcopy(cases),'proofs':[ref],
   'explanation':'Complete finite source-target API: all289 dtype pairs, first allocation/CUDA/signature/addition outcomes, all189 actual execution routes, linear storage-offset addresses despite arbitrary strides, complete grid, finite signed indices, byte widths, input/output frames and arbitrary active-lane schedules. See domain-review.json for explicit unsafe and undefined cases.'})
row.update(status='pending',scope_schema=2,coverage=coverage,structural_gaps=[],review=None,deferred_numerical_assumptions=deferrals,
 callback_assumptions=links,proposed_status_after_independent_root_review='verified_with_deferred_numerics')
row['input_sha256']['lean'].update(module_hashes)
row['acceptance_boundary_review']={'status':'producer_proposal_awaiting_independent_root_review','policy':'bench/original_program_acceptance.md',
 'policy_sha256':hash_file(R/'bench/original_program_acceptance.md'),'finite_domain':'SafeAPI and ValueDomain on actual Torch tensors with canonical17 dtypes, CUDA sm90, debug=false, Triton3.7.1 and ordinary trusted primitives. All189 previously admitted routes retained.',
 'layout_behavior':'Source reads/writes flat storage spans, not arbitrary Torch elementwise logical indexing.','native_correctness':False,
 'queue_integration':'Root-reviewed frozen authority delta and unchanged whole-queue equality are required; author metadata alone does not grant acceptance.'}
authority={'schema':1,'kind':'vector_finite_exact_numeric_authority_delta','task':TASK,'rules':list(rules.values()),'provenance':provenance,
 'requirements':[{'task':TASK,'declaration':ref,'source_ref':{'path':PUBLIC,'sha256':module_hashes[PUBLIC]},
 'supported_cases_sha256':sha(canonical(sorted(cases,key=canonical))),'callback_union':union,'cases':requirements,
 'support_boundary':'All2601 exact source outcome cases;616 scalar ports on positive successful cases; errors and zero-grid launches have no required numerical operations.'}],
 'summary':{'scalar_families':len(rules),'scope_bindings':len(provenance),'declaration_cases':len(cases),'registered_declarations':1,
 'admitted_dtype_pairs':189,'canonical_dtype_pairs':289,'signature_rejections':64,'addition_rejections':36,
 'inherited_scalar_operations':308,'native_executions':0,'original_program_completion_increment':0}}
assert len(provenance)==616
validate_deferrals(TASK,deferrals,{'rules':list(rules.values())})
assert {(r['assumption_id'],r['scope_binding_id']) for c in requirements for r in c['required_binding_refs']}=={(d['assumption_id'],d['scope_binding_id']) for d in deferrals}
supp=deepcopy(next(x for x in load(R/'bench/review-checkpoints/2026-10-16/vector-typed-host-and-rotary-grid/catalog-supplement.json')['tasks'] if x['task']==TASK))
supp['declarations'].append(declaration);supp['source_sha256'].update(module_hashes);supp['source_sha256'][PYTHON]=hash_file(R/PYTHON);supp['review']=None
supp['required_value_stages']=list(dict.fromkeys(supp['required_value_stages']+stages))
supp['source_views'].append({'id':'finite_typed_host_physical_views','variant':'original_source_translation','frontend':'task_local_exact_ast',
 'ordinary_shared_dsl_admission':False,'memory_semantics':'sequentially_consistent_schedule','original_python':PYTHON,'candidate_python':None,
 'selected_definitions':{BASE+'Proofs/TypedHostExecution.lean':['execute'],BASE+'Proofs/FiniteRoutes.lean':['scalarSource'],BASE+'Proofs/FiniteSchedule.lean':['runBatches']},
 'entry_aliases':{BASE+'Proofs/HostWrapper.lean':['add'],BASE+'Proofs/TypedHostWrapper.lean':['launch']},
 'translation_trust_boundary':'Reviewed unchanged Python AST and exact17x17 target frontend partition; trusted Torch metadata/fresh allocation and typed memory primitives. The actual existing typed interpreters construct execution; independent physical source replay proves output/input/byte frames and finite pointer/index applicability. No generic compiler/allocator/MMU/scheduler implementation proof or native correctness claim.'})
for name,data in [('declaration-proposal.json',declaration),('scope-row-proposal.json',row),('scope-cases.json',cases),('catalog-supplement-task.proposed.json',supp),('numerical-authority-delta.json',authority)]:save(name,data)
save('proposal-validation.json',{'status':'passed','counts':authority['summary'],'checks':['exact current specification parser','unchanged schema2 declaration_scope','all2601 validate_case calls','unchanged validate_deferrals','616 per-case scalar port bijection','all189 inherited value routes retained'],'independent_root_review':'pending','global_queue_check':'pending reviewed authority integration','scope_acceptance_increment':0})
print(json.dumps(authority['summary']))
