import sys;sys.dont_write_bytecode=True
from copy import deepcopy
from pathlib import Path
import hashlib,json,re
W=Path(__file__).resolve().parent;R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sys.path.insert(0,str(R/'bench'))
from adapter_common import canonical,declarations,sha
from check_original_program_scope import declaration_scope,validate_case,validate_deferrals
load=lambda p:json.loads(Path(p).read_text())
hash_file=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):(W/'evidence'/n).write_text(json.dumps(x,indent=2)+'\n')
TASK='add_value';BASE='bench/tritonbench_g/add_value/';PUBLIC=BASE+'NativeCorrect.lean';NAME='add_value_native_source_outcomes';PYTHON=BASE+'add_value.py'
modules=[str(p.relative_to(W/'after')) for p in sorted((W/'after').rglob('*.lean'))];module_hashes={p:hash_file(W/'after'/p) for p in modules}
ref={'file':PUBLIC,'declaration':NAME,'statement_sha256':declarations(W/'after'/PUBLIC)[NAME]['statement_sha256']}
native=str(Path(load(W/'evidence/native-latest.json')['receipt']).parent)+'/';receipt=load(W/native/'receipt.json');records={r['dtype']:r['result'] for r in receipt['records']}
def dtype(d):return {'float16':'fp16','bfloat16':'bf16','float32':'fp32','float64':'fp64','float8_e4m3fn':'fp8e4m3fn','float8_e5m2':'fp8e5m2','float8_e4m3fnuz':'fp8e4m3fnuz','float8_e5m2fnuz':'fp8e5m2fnuz','float8_e8m0fnu':'fp8e8m0fnu','float4_e2m1fn_x2':'packed_fp4e2m1'}.get(d,d)
profiles={}
def op(port,inputs,output,config=None):
 return {'port':port,'input_types':inputs,'output_type':output,'configuration':config or {},'operation':port.rsplit('.',1)[-1]}
for d,r in records.items():
 if r['status']!='compiled':continue
 t=dtype(d);operations=[]
 if d=='float64':operations=[op('wide.singleTo64',['fp32'],'fp64'),op('wide.add64',['fp64','fp64'],'fp64')]
 else:
  if d.startswith('int') or d.startswith('uint'):
   width=int(re.search(r'\d+',d)[0]);signed=d.startswith('int')
   operations.append(op('signedTo32' if signed else 'unsignedTo32',[d],'fp32',{'width':width,'rounding':'rtne'}))
  elif d in ['float16','bfloat16','float8_e4m3fn','float8_e5m2']:
   operations.append(op({'float16':'halfTo32','bfloat16':'bfloatTo32','float8_e4m3fn':'e4To32','float8_e5m2':'e5To32'}[d],[t],'fp32'))
  operations.append(op('add32',['fp32','fp32'],'fp32',{'runtime_right_argument_bits':0x41200000,'rounding':'rtne'}))
  if d=='bool' or d.startswith(('int','uint')):
   width=8 if d=='bool' else int(re.search(r'\d+',d)[0]);signed=d=='bool' or d.startswith('int')
   operations.append(op('FP.IntegerConversion.toSigned' if signed else 'FP.IntegerConversion.toUnsigned',['fp32'],('int' if signed else 'uint')+str(width),{'width':width,'rounding':'rtz','domain':'finite_truncation_in_range','raw_bool_store':d=='bool'}))
  elif d in ['float16','bfloat16','float8_e4m3fn','float8_e5m2']:
   operations.append(op({'float16':'toHalf','bfloat16':'toBFloat','float8_e4m3fn':'toE4RTNE','float8_e5m2':'toE5RTNE'}[d],['fp32'],t,{'rounding':'rtne'}))
 profiles[d]={'dtype':d,'compute':'fp64' if d=='float64' else 'fp32','operations':operations,'raw_ttir':native+d+'.raw.ttir','raw_ttir_sha256':hash_file(W/native/(d+'.raw.ttir'))}
assert len(profiles)==15 and sum(len(p['operations']) for p in profiles.values())==41
inventory_record={'task':TASK,'source':PYTHON,'source_sha256':hash_file(R/PYTHON),'scalar_module':BASE+'Proofs/NativeScalar.lean','scalar_sha256':module_hashes[BASE+'Proofs/NativeScalar.lean'],'profiles':list(profiles.values()),'runtime_value':{'type':'fp32','bits':0x41200000,'constexpr':False},'bool_structural_casts':'raw8!=0, exact predicate to fp32 0/1, signedStore8 raw byte without bool normalization','native_scalar_validation':False}
save('scalar-operation-inventory.json',inventory_record)
ledger=load(R/'bench/original_program_scope.json');row=deepcopy(next(x for x in ledger['tasks'] if x['task']==TASK));inventory=row['source_inventory']
stages=[x['id'] for x in inventory['stages'] if not x['may_be_test_harness']];dtypeids=[x['id'] for x in inventory['dtypes']];branchids=[x['id'] for x in inventory['branches']]
configs=[]
observations=['input_access','missing_bytes','undefined_conversion','output_access','ownership_conflict']
for d,r in records.items():
 p=profiles.get(d);terminal='returned_or_undefined_conversion' if p else 'architecture_rejected' if 'fnuz' in d else 'binder_rejected'
 outcomes=[('allocation_error','none'),('cuda_assertion','none'),('context_error','none'),(terminal,'none')]
 if p:outcomes += [(v,'none') for v in ['compilation_error','grid_limit','pointer_packing_error','launch_setup_error','two31_no_access_return']]+[('native_observation',v) for v in observations]
 for outcome,observation in outcomes:
  compute={'index':'int32','pointer':'uint64','value_argument':'fp32'}
  if p:compute.update(value=p['compute'],store='int8' if d=='bool' else dtype(d))
  c={'dtype_profile':{'inputs':{'x':dtype(d)},'outputs':{'output':dtype(d)},'compute':compute},'configuration':{
   'entrypoint':'puzzle1','source_variant':'original','BLOCK_SIZE':1024,'value':{'python':10,'runtime_type':'fp32','bits':0x41200000,'constexpr':False},
   'numel':({'query':'not_reached_allocation_or_cuda_failure'} if outcome in ['allocation_error','cuda_assertion'] else {'lower_inclusive':2**31 if outcome=='two31_no_access_return' else 0,'upper_exclusive':2**31+1 if outcome=='two31_no_access_return' else 2**63}),
   'outcome':outcome,'native_observation':observation,'grid':'ceil_div_original_output_numel_1024','rank':'arbitrary_including_rank0_numel1',
   'target':{'backend':'cuda','arch':90,'warp_size':32},'triton_version':receipt['triton_version'],'torch_version':receipt['torch_version'],'frontend_options_sha256':sha(canonical(records['float32']['options'])),
   'debug':False,'instrumentation_mode':'none','runtime_N':'annotated_tl.int32_for_all_host_extents','bound_conversion':'PyLong_AsLong_success_then_int32_assignment_mod2pow32',
   'lane_arithmetic':'i32_pid_times1024_plus_lane_wrap_before_signed_slt','pointer_arithmetic':'sign_extend_i32_offset_then_scale_storage_width_then_wrap_uint64',
   'setup_order':['context','signature_and_backend_compilation','signed_grid_tuple_parse','pointer_and_runtime_scalar_packing','launch'],'grid_limit':2**31-1,
   'layouts':'all_actual_dense_strided_Torch_metadata; original_source_flat_data_ptr_access','view_metadata':'unconditional_Torch_storage_data_and_extracted_pointer_binding; actual_empty_like_preserve_format_strides_even_with_no_active_masks',
   'allocation':'fresh_output_empty_like_input_metadata_and_exterior_frame','permissions':'observed_at_each_actual_request_in_current_launch_address_space;owner_device_metadata_separate',
   'device':'allocation_then_two_CUDA_assertion;no_same_device_assertion;peer_or_mapped_memory_allowed_by_permissions',
   'scalar_definedness':'partial_float_to_integer_conversion_observed_before_value_materialization;no_undefined_fallback','source_pointers':'exactly_one_input_pointer_and_one_output_pointer;value_is_runtime_fp32_scalar',
   'observations':'mathematical_requests_only;no_native_trap_race_winner_rollback_or_postfailure_memory',
   'numeric_claim':'actual_loaded_word_and_source_result_with_defined_conversion_partition' if outcome=='returned_or_undefined_conversion' else 'value_insensitive_constructor_or_first_setup_error_only',
   'output_bytes':'independent_initial_memory_write_plan_and_pure_packing_final_byte_formula;untouched_bytes_equal_allocated_contents'}}
  validate_case(c);configs.append((d,c))
cases=[c for _,c in configs];assert len(cases)==238 and len({canonical(c) for c in cases})==238
def callback(port):
 if port.startswith('FP.'):return 'VeriTile.Triton.'+port
 if port.startswith('wide.'):return 'VeriTile.Bench.TritonBenchG.VectorAddition.FP64TypedCore.Numerics.'+port.split('.')[-1]
 return 'AddValueNative.Numerics.'+port
rules={};provenance={};requirements=[];links={}
for d,c in configs:
 p=profiles.get(d);selected=p['operations'] if p and c['configuration']['outcome']=='returned_or_undefined_conversion' else [];refs=[]
 for ordinal,operation in enumerate(selected):
  port=operation['port'];cb=callback(port);ident='ADD-VALUE-NATIVE-'+re.sub('[^A-Z0-9]+','-',port.upper()).strip('-')
  if ident not in rules:rules[ident]={'id':ident,'kind':'numerical_rule_conformance','operation_class':'scalar_primitive_conformance','status':'pending_numeric_validation','queue_status':'exact_profile_bound_validation_pending','affected_tasks':[TASK],'required_validation':'Exact native scalar correspondence for '+port+' at each registered operand/output dtype and configuration, including rounding, subnormal, NaN, infinity and signed-zero behavior on the explicitly defined active source domain.','native_whole_kernel_validated':False,'validation_evidence':None,'scope_bindings':[]}
  body={'task':TASK,'declarations':[ref],'callbacks':[cb],**deepcopy(c)};bid=TASK+':native:'+sha(canonical({'rule':ident,'binding':body,'operation':operation,'ordinal':ordinal}))[:32]
  rules[ident]['scope_bindings'].append({'id':bid,**body});link={'assumption_id':ident,'scope_binding_id':bid};refs.append({'callback':cb,**link});links.setdefault(cb,[]).append(link)
  scalar='VeriTile/Triton/Float/IntegerConversion.lean' if port.startswith('FP.') else 'bench/tritonbench_g/vector_addition/Proofs/FP64TypedCore.lean' if port.startswith('wide.') else BASE+'Proofs/NativeScalar.lean'
  scalarhash=module_hashes[scalar] if scalar in module_hashes else hash_file(R/scalar)
  provenance[ident+'::'+bid]={'kind':'scalar_primitive_conformance','status':'explicitly_deferred_unvalidated','validation_evidence':None,**link,'callback_declaration':cb,'correspondence_kind':'concrete_scalar_port' if port.startswith('FP.') else 'parametric_callback',
   'port':{'name':port,'ordinal':ordinal,'operation':operation['operation'],'input_types':operation['input_types'],'output_type':operation['output_type'],'scalar_configuration':operation['configuration']},
   'scalar_domain':'Active loaded input with the actual runtime fp32 word10; actual partial float-to-integer conversion observed before materializing a store; no arbitrary fallback, native exception or postfailure-memory claim.','scalar_path_predicate':'active_source_element; input_dtype='+d,'native_execution':False,
   'source_statement':'output = x + value; tl.store(output_ptr + offsets, output, mask=mask); original lines13-14; typed storage conversion '+port,
   'original_source':{'path':PYTHON,'sha256':hash_file(R/PYTHON)},'public_statement':ref,'lean_source_sha256':module_hashes,'scalar_source':{'path':scalar,'sha256':scalarhash},
   'execution_source':{'module':BASE+'Proofs/NativeProgram.lean','sha256':module_hashes[BASE+'Proofs/NativeProgram.lean'],'declaration':'AddValueNative.executeProgram'},'host_source':{'module':BASE+'NativeCorrect.lean','sha256':module_hashes[BASE+'NativeCorrect.lean'],'declaration':'AddValueNative.executeHost'},
   'raw_ttir':{'path':p['raw_ttir'],'sha256':p['raw_ttir_sha256']},'frontend_receipt':{'path':native+'receipt.json','sha256':hash_file(W/native/'receipt.json')},'producer_scalar_inventory':{'path':'evidence/scalar-operation-inventory.json','sha256':hash_file(W/'evidence/scalar-operation-inventory.json')},
   'configuration':c['configuration'],'structural_prerequisites_not_waived':['actual annotated i32_ABI signed masks and wrapped64bit source addresses','fresh output input preservation and independently derived byte frames','unconditional actual allocation metadata and extracted-pointer binding','exact original unary source execution with runtime fp32 argument10','partial conversion observation before store values','ordinary trusted allocation/context/typed-load/pure-packing/store primitives']}
 requirements.append({**deepcopy(c),'required_callbacks':sorted({x['callback'] for x in refs}),'required_binding_refs':refs})
for r in rules.values():r['scope_bindings'].sort(key=lambda b:b['id'])
deferrals=[{'kind':'numerical_rule_conformance','assumption_id':i,'rule_sha256':sha(canonical(r)),'scope_binding_id':b['id'],'dtype_profile':b['dtype_profile'],'configuration':b['configuration'],'required_validation':r['required_validation']} for i,r in rules.items() for b in r['scope_bindings']]
union=sorted(links);scope={'source_inventory_sha256':sha(canonical(inventory)),'stages':stages,'dtypes':dtypeids,'branches':branchids,'cases':{i:deepcopy(cases) for i in stages+dtypeids+branchids}}
declaration={**ref,'kind':'specification','model_kind':'backend_value_contract','role':'parametric_value_contract','scope_relation':'full_declared_stage','source_variant':'original_source_translation','covers_stages':stages,
 'dtype_profile':'All22 dense Torch dtype profiles:15 full backend routes,2 fnuz architecture rejects,5 binder rejects. Full nonnegative signed64 host numel with actual annotated i32 arithmetic and wrapped64bit pointers; arbitrary rank including scalar tensors.',
 'dtype_scope_relation':'complete_enumerated_wrapper_dtype_domain','unclosed_numeric_callbacks':union,'extra_numeric_law_ids':[],'native_binding_status':'not_established','review_basis':'Producer proposal requiring independent root review. Complete original unary tile load, runtime fp32 value10, typed add/store and full numel grid. Actual signed32 masks and sign-extended wrapped64bit addresses; allocation, two-CUDA assertion, context/compilation/grid-parse/pointer-packing/launch outcomes. No active value/access/no-wrap premise on master; structural observations and independent final bytes. Historical bounded five-module proposal remains unregistered.','original_scope':scope}
declaration_scope(declaration,inventory)
coverage={k:[] for k in ['stages','dtypes','branches']}
for dimension,items in inventory.items():
 for item in items:
  if dimension=='stages' and item['may_be_test_harness']:coverage[dimension].append({'id':item['id'],'status':'test_harness','explanation':'Non-JIT test_puzzle has four ordinary puzzle1 calls and import-time result_gold; no production caller or RNG implementation obligation.'})
  else:coverage[dimension].append({'id':item['id'],'status':'covered','cases':deepcopy(cases),'proofs':[ref],'explanation':'Complete238 full-host-domain cases for all22 dense dtype profiles. Unary source loads and runtime fp32 value10; actual i32/u64 arithmetic; independent final byte result and frame. No request safety, defined-value or no-wrap premise is imposed on the master. Observations assert no native trap, race winner, rollback or postfailure memory.'})
row.update(status='pending',scope_schema=2,coverage=coverage,structural_gaps=[],review=None,deferred_numerical_assumptions=deferrals,callback_assumptions=links,proposed_status_after_independent_root_review='verified_with_deferred_numerics');row['input_sha256']['lean'].update(module_hashes)
row['acceptance_boundary_review']={'status':'producer_proposal_awaiting_independent_root_review','policy':'bench/original_program_acceptance.md','policy_sha256':hash_file(R/'bench/original_program_acceptance.md'),'finite_domain':'Actual Torch metadata, all22 dense dtype profiles, full nonnegative signed64 host extent, CUDA sm90 debug=false Triton3.7.1. Access and conversion safety are observations; original CUDA assertion remains. All allocation/setup failures and returned/observed paths explicit.','layout_behavior':'Flat physical pointers ignore logical strides; no generic Torch elementwise indexing claim.','native_correctness':False,'queue_integration':'Independent root review pinned by frozen receipt and review hash; unchanged whole-queue equality required.'}
authority={'schema':1,'kind':'add_value_native_exact_numeric_authority_delta','task':TASK,'rules':list(rules.values()),'provenance':provenance,'requirements':[{'task':TASK,'declaration':ref,'source_ref':{'path':PUBLIC,'sha256':module_hashes[PUBLIC]},'supported_cases_sha256':sha(canonical(sorted(cases,key=canonical))),'callback_union':union,'cases':requirements,'support_boundary':'238 exact source outcome cases;41 scalar ports for loaded-word returned/undefined partitions; value-insensitive constructor-only and setup-failure cases claim no numerical intermediates.'}],
 'summary':{'scalar_families':len(rules),'scope_bindings':len(provenance),'declaration_cases':len(cases),'registered_declarations':1,'admitted_dtype_profiles':15,'canonical_dtype_profiles':22,'signature_rejections':5,'architecture_rejections':2,'scalar_operations':41,'native_backend_compiles':15,'native_executions':0,'original_program_completion_increment':0}}
assert len(provenance)==41 and len(rules)==15
validate_deferrals(TASK,deferrals,{'rules':list(rules.values())})
assert {(r['assumption_id'],r['scope_binding_id']) for c in requirements for r in c['required_binding_refs']}=={(d['assumption_id'],d['scope_binding_id']) for d in deferrals}
supp={'task':TASK,'required_value_stages':stages,'source_sha256':{**row['input_sha256']['lean'],**row['input_sha256']['python']},'source_views':[{'id':'add_value_native_i32_full_host_extent','variant':'original_source_translation','frontend':'task_local_exact_ast','ordinary_shared_dsl_admission':False,'memory_semantics':'sequentially_consistent_schedule','original_python':PYTHON,'candidate_python':None,'selected_definitions':{BASE+'Proofs/NativeProgram.lean':['executeProgram','programPlan'],BASE+'Proofs/NativeScalar.lean':['source','result']},'entry_aliases':{BASE+'NativeCorrect.lean':['executeHost']},'translation_trust_boundary':'Exact original Python AST and all22 native profiles;14 ABI/binder/backend extent profiles; full signed32 masks and sign-extended wrapped addresses; one actual input pointer plus runtime fp32 scalar10. Ordinary Torch metadata/fresh allocation, successful mapped pointer extraction, typed loads and pure packing/store primitives trusted. No result/launch-success oracle or arbitrary-carrier readback axiom; no native scalar validation claim.'}],'declarations':[declaration],'review':None}
for n,x in [('declaration-proposal.json',declaration),('scope-row-proposal.json',row),('scope-cases.json',cases),('catalog-supplement-task.proposed.json',supp),('numerical-authority-delta.json',authority)]:save(n,x)
save('proposal-validation.json',{'status':'passed','counts':authority['summary'],'checks':['current specification parser','unchanged schema2 declaration_scope','238 validate_case calls','unchanged validate_deferrals','41 exact case-operation binding bijection','22 native profiles preserve first rejection and runtime argument specialization'],'independent_root_review':'pending','global_queue_check':'pending reviewed authority integration','scope_acceptance_increment':0})
print(json.dumps(authority['summary']))
