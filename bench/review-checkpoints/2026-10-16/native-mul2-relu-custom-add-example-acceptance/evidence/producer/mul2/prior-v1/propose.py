import sys
sys.dont_write_bytecode=True
from copy import deepcopy
from pathlib import Path
import hashlib,json,re
W=Path(__file__).resolve().parent;R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sys.path.insert(0,str(R/'bench'))
from adapter_common import canonical,declarations,sha
from check_original_program_scope import declaration_scope,validate_case,validate_deferrals
load=lambda p:json.loads(Path(p).read_text());hash_file=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):(W/'evidence'/n).write_text(json.dumps(x,indent=2)+'\n')
TASK='triton_mul2';BASE='bench/tritonbench_g/triton_mul2/';PUBLIC=BASE+'FiniteCorrect.lean';NAME='triton_mul2_finite_source_outcomes';PYTHON=BASE+'triton_mul2.py'
modules=list(load(W/'evidence/staging-map.json').values());module_hashes={p:hash_file(W/'after'/p) for p in modules}
ref={'file':PUBLIC,'declaration':NAME,'statement_sha256':declarations(W/'after'/PUBLIC)[NAME]['statement_sha256']}
native='native-attempts/20261015T214958399467Z/';receipt=load(W/native/'receipt.json')
records={r['result']['dtype']:r['result'] for r in receipt['records'] if r['result']['kernel']=='mul2_kernel' and r['result']['block']==16}
def dtype(d):return {'float16':'fp16','bfloat16':'bf16','float32':'fp32','float64':'fp64','float8_e4m3fn':'fp8e4m3fn','float8_e5m2':'fp8e5m2','float8_e4m3fnuz':'fp8e4m3fnuz','float8_e5m2fnuz':'fp8e5m2fnuz','float8_e8m0fnu':'fp8e8m0fnu','float4_e2m1fn_x2':'packed_fp4e2m1'}.get(d,d)
floats=['float16','bfloat16','float32','float64'];bits={'float16':0x4000,'bfloat16':0x4000,'float32':0x40000000,'float64':0x4000000000000000}
profiles=[]
for d,r in records.items():
 if r['status']!='compiled':continue
 profiles.append({'dtype':d,'compute':'int32' if d=='bool' else dtype(d),'numerical_operations':([{'callback':'TritonMul2Finite.Numerics.multiply','format':dtype(d),'operation':'multiply','input_types':[dtype(d),dtype(d)],'output_type':dtype(d),'literal_left_bits':bits[d]}] if d in floats else []),'structural':'rawbyte!=0 then i32 multiply2 then rawbyte0/2' if d=='bool' else 'modulo storage-width integer multiply2' if d.startswith(('int','uint')) else 'storage-width float multiply with exact literal2 bits'})
assert len(profiles)==13
save('scalar-operation-inventory.json',{'task':TASK,'source':PYTHON,'source_sha256':hash_file(R/PYTHON),'scalar_module':BASE+'Proofs/FiniteScalar.lean','scalar_sha256':module_hashes[BASE+'Proofs/FiniteScalar.lean'],'profiles':profiles,'float_to_integer_conversions':0,'all_admitted_storage_bit_patterns_defined':True,'integer_and_bool_numerical_deferrals':0,'native_scalar_validation':False})
ledger=load(R/'bench/original_program_scope.json');row=deepcopy(next(x for x in ledger['tasks'] if x['task']==TASK));inventory=row['source_inventory']
stages=[x['id'] for x in inventory['stages'] if not x['may_be_test_harness']];dtypeids=[x['id'] for x in inventory['dtypes']];branchids=[x['id'] for x in inventory['branches']]
block_classes=[('zero',{'value':0},False),('default',{'value':16},True),('harness512',{'value':512},True),('harness1024',{'value':1024},True),('other_valid',{'form':'2^k','exponent_lower_inclusive':0,'exponent_upper_inclusive':20,'excluded_exponents':[4,9,10]},True),('invalid_integer',{'predicate':'integer_nonzero_and_not_power_of_two_le_2^20'},False)]
sizes=[('empty',0,1),('narrow_positive',1,2**31),('i32_minimum',2**31,2**31+1),('wide',2**31+1,2**63)]
configs=[]
for d,r in records.items():
 admitted=r['status']=='compiled';sig='architecture_rejected' if 'fnuz' in d else 'signature_rejected'
 for wrapper,kernel,mode in [('triton_mul2','mul2_kernel','outplace'),('triton_mul2_inplace','mul2_inplace_kernel','inplace')]:
  for bc,bs,valid in block_classes:
   for size,lo,hi in sizes:
    outcomes=(['allocation_error'] if mode=='outplace' else [])
    if bc=='zero':outcomes+=['division_by_zero']
    else:
     outcomes+=['prepare_error']
     if not admitted and d not in ['float8_e4m3fn','float8_e5m2']:outcomes+=[sig]
     elif not valid:outcomes+=['block_frontend_rejected']
     elif not admitted:outcomes+=['fp8_backend_rejected']
     else:
      outcomes+=['backend_or_context_error']
      if size=='wide':outcomes+=['grid_parse_rejected']
      outcomes+=['pointer_setup_error']
      if size!='empty':outcomes+=['native_launch_error']
      if size in ['empty','i32_minimum']:outcomes+=['returned_masked']
      else:outcomes+=['input_access_observation','missing_input_bytes_observation','output_access_observation','byte_ownership_observation','returned_active']
      if size=='wide':outcomes+=['returned_masked']
    for outcome in outcomes:
     c={'dtype_profile':{'inputs':{'x':dtype(d)},'outputs':{'output':dtype(d)},'compute':{'index':'int32','pointer':'uint64','value':'int32' if d=='bool' else dtype(d),'store':'int8' if d=='bool' else dtype(d)}},'configuration':{
      'entrypoint':wrapper,'kernel':kernel,'mode':mode,'source_variant':'original','BLOCK_SIZE':bs,'block_class':bc,'numel':{'lower_inclusive':lo,'upper_exclusive':hi},'size_class':size,'outcome':outcome,
      'target':{'backend':'cuda','arch':90,'warp_size':32},'triton_version':receipt['triton_version'],'torch_version':receipt['torch_version'],'frontend_options_sha256':sha(canonical(records['float32']['options'])),'debug':False,'instrumentation_mode':'none',
      'size_abi':'annotated_i32_with_native_PyLongAsLong_then_int32_cast','host_grid':'Python_floor((numel+BLOCK_SIZE-1)/BLOCK_SIZE);signed_i32_grid_parse','indices':'i32(pid)*i32(BLOCK_SIZE)+i32(lane);signed_i32_mask','pointer':'sign_extend_offset_to64_then_width_scale_and_u64_wrap',
      'layouts':'arbitrary_actual_flat_storage_spans_with_logical_strides_ignored','metadata':'actual_shape_numel_dtype_device_base_extent_storage_offset_and_data_ptr;empty_data_ptr0_independent_of_affine_offset','allocation':'fresh_zeros_like_same_metadata_preserve_format_strides_zero_bytes_and_frame' if mode=='outplace' else 'same_input_object_and_memory;no_allocation',
      'schedule':'every_launched_pid_exactly_once_in_any_complete_order','tile_order':'load_complete_active_tile_before_any_stores','memory_observations':'actual_byte_bounds_device_read_write_missing_bytes_and_byte_ownership;no_native_trap_rollback_or_failure_memory_claim',
      'scalar_domain':'all_loaded_storage_bit_patterns;no_undefined_conversion','value_activity':'active_source_lane_exists' if outcome=='returned_active' else 'no_numeric_assumption_required','source_statements_per_kernel':7}}
     validate_case(c);configs.append((d,c))
cases=[c for _,c in configs];assert len({canonical(c) for c in cases})==len(cases)
rules={};provenance={};requirements=[];links={};cb='TritonMul2Finite.Numerics.multiply'
for d,c in configs:
 refs=[]
 if d in floats and c['configuration']['outcome']=='returned_active':
  typ=dtype(d);ident='TRITON-MUL2-FINITE-MULTIPLY-'+typ.upper();operation={'name':'multiply','format':typ,'input_types':[typ,typ],'output_type':typ,'literal_left_bits':bits[d],'rounding':'native_storage_width_rtne','arbitrary_right_storage_bits':True}
  if ident not in rules:rules[ident]={'id':ident,'kind':'numerical_rule_conformance','operation_class':'scalar_primitive_conformance','status':'pending_numeric_validation','queue_status':'exact_profile_bound_validation_pending','affected_tasks':[TASK],'required_validation':'Native '+typ+' storage-width multiplication by exact literal2 for every source-loaded bit pattern, including NaNs, infinities, subnormals, signed zero and overflow; no integer, Boolean, indexing, address, allocation or error-order deferral.','native_whole_kernel_validated':False,'validation_evidence':None,'scope_bindings':[]}
  body={'task':TASK,'declarations':[ref],'callbacks':[cb],**deepcopy(c)};bid=TASK+':finite:'+sha(canonical({'rule':ident,'binding':body,'operation':operation}))[:32]
  rules[ident]['scope_bindings'].append({'id':bid,**body});link={'assumption_id':ident,'scope_binding_id':bid};refs.append({'callback':cb,**link});links.setdefault(cb,[]).append(link)
  kernel=c['configuration']['kernel'];raw=native+kernel+'-'+d+'-b16.raw.ttir'
  provenance[ident+'::'+bid]={'kind':'scalar_primitive_conformance','status':'explicitly_deferred_unvalidated','validation_evidence':None,**link,'callback_declaration':cb,'correspondence_kind':'parametric_callback','port':{'name':'multiply','ordinal':0,'operation':'multiply','input_types':[typ,typ],'output_type':typ,'scalar_configuration':operation},
   'scalar_domain':'All source-loaded '+typ+' storage bits multiplied on the left by exact storage-width literal2. No float-to-int conversion or result-definedness assumption.','scalar_path_predicate':'active_source_lane_exists; input_dtype='+d,'native_execution':False,
   'source_statement':'output = 2 * x; tl.store(..., output, mask=mask), exact typed loads and stores with no fp32 promotion for half/bfloat',
   'original_source':{'path':PYTHON,'sha256':hash_file(R/PYTHON)},'public_statement':ref,'lean_source_sha256':module_hashes,'scalar_source':{'path':BASE+'Proofs/FiniteScalar.lean','sha256':module_hashes[BASE+'Proofs/FiniteScalar.lean']},
   'execution_source':{'module':BASE+'Proofs/FiniteSource.lean','sha256':module_hashes[BASE+'Proofs/FiniteSource.lean'],'declaration':'TritonMul2Finite.executeBody'},'host_source':{'module':BASE+'Proofs/FiniteHost.lean','sha256':module_hashes[BASE+'Proofs/FiniteHost.lean'],'declaration':'TritonMul2Finite.executeHost'},
   'raw_ttir':{'path':raw,'sha256':hash_file(W/raw)},'frontend_receipt':{'path':native+'receipt.json','sha256':hash_file(W/native/'receipt.json')},'producer_scalar_inventory':{'path':'evidence/scalar-operation-inventory.json','sha256':hash_file(W/'evidence/scalar-operation-inventory.json')},
   'native_ir_role':'Original B16 representative scalar lowering for this exact kernel/dtype; block-polymorphic source bindings are not native validation or a claim that every resource configuration compiles.',
   'configuration':c['configuration'],'structural_prerequisites_not_waived':['signed-i32 size cast and per-operation index wrap','actual wrapped-u64 addresses and byte load/store behavior','source-first binder/frontend/backend/grid/pointer failures','full complete grid and exactly-once native schedule','unconditional metadata and in-place identity','fresh zero allocation and input/exterior byte frames','empty and signed-minimum all-masked behavior']}
 requirements.append({**deepcopy(c),'required_callbacks':sorted({x['callback'] for x in refs}),'required_binding_refs':refs})
for r in rules.values():r['scope_bindings'].sort(key=lambda b:b['id'])
deferrals=[{'kind':'numerical_rule_conformance','assumption_id':i,'rule_sha256':sha(canonical(r)),'scope_binding_id':b['id'],'dtype_profile':b['dtype_profile'],'configuration':b['configuration'],'required_validation':r['required_validation']} for i,r in rules.items() for b in r['scope_bindings']]
union=sorted(links);scope={'source_inventory_sha256':sha(canonical(inventory)),'stages':stages,'dtypes':dtypeids,'branches':branchids,'cases':{i:deepcopy(cases) for i in stages+dtypeids+branchids}}
declaration={**ref,'kind':'specification','model_kind':'backend_value_contract','role':'parametric_value_contract','scope_relation':'full_declared_stage','source_variant':'original_source_translation','covers_stages':stages,
 'dtype_profile':'All22 canonical Torch dtype routes:13 admitted,5 binder rejects,2 architecture rejects,2 fp8 backend rejects. Both wrappers, all integer block outcomes and full Torch numel domain with actual signed32 casting.',
 'dtype_scope_relation':'complete_enumerated_wrapper_dtype_domain','unclosed_numeric_callbacks':union,'extra_numeric_law_ids':[],'native_binding_status':'not_established','review_basis':'PRODUCER PROPOSAL requiring independent root review. Original AST lowering for both seven-statement kernels; concrete byte interpreter with finite signed32/u64 addressing and in-place-aware independent last-write results; full wrappers/error order, metadata and zero allocation. Only four storage-width float multiplication families deferred.','original_scope':scope}
declaration_scope(declaration,inventory)
coverage={k:[] for k in ['stages','dtypes','branches']}
for dimension,items in inventory.items():
 for item in items:
  if dimension=='stages' and item['may_be_test_harness']:coverage[dimension].append({'id':item['id'],'status':'test_harness','explanation':'Non-JIT test_mul contains four ordinary wrapper calls with B1024/B512 and import-time result_gold; no production callers.'})
  else:coverage[dimension].append({'id':item['id'],'status':'covered','cases':deepcopy(cases),'proofs':[ref],'explanation':'Complete finite source outcome cases across22 dtype routes, both wrappers, all integer block classes, all Torch numel<2^63 including signed32 wrap, actual byte accesses/ownership and independent results. See domain-review.json.'})
row.update(status='pending',scope_schema=2,coverage=coverage,structural_gaps=[],review=None,deferred_numerical_assumptions=deferrals,callback_assumptions=links,proposed_status_after_independent_root_review='verified_with_deferred_numerics');row['input_sha256']['lean'].update(module_hashes)
row['acceptance_boundary_review']={'status':'producer_proposal_awaiting_independent_root_review','policy':'bench/original_program_acceptance.md','policy_sha256':hash_file(R/'bench/original_program_acceptance.md'),'finite_domain':'Actual Torch metadata;22 canonical storage dtypes;integer BLOCK_SIZE;full numel<2^63;CUDA sm90,debug=false,Triton3.7.1. Actual signed32/u64 wrapped operations remain in main theorem; no SafeAPI/narrow-size premise.','layout_behavior':'Flat physical pointers ignore logical strides. Dynamic observations assert no native trap, race result, rollback or failure memory.','native_correctness':False,'queue_integration':'Independent root receipt+review pins and unchanged whole-queue equality required.'}
summary={'scalar_families':len(rules),'scope_bindings':len(provenance),'declaration_cases':len(cases),'registered_declarations':1,'admitted_dtype_profiles':13,'canonical_dtype_profiles':22,'signature_rejections':5,'architecture_rejections':2,'backend_rejections':2,'scalar_operations':4,'integer_bool_numerical_deferrals':0,'native_backend_compiles':30,'native_executions':0,'original_program_completion_increment':0}
authority={'schema':1,'kind':'triton_mul2_finite_exact_numeric_authority_delta','task':TASK,'rules':list(rules.values()),'provenance':provenance,'requirements':[{'task':TASK,'declaration':ref,'source_ref':{'path':PUBLIC,'sha256':module_hashes[PUBLIC]},'supported_cases_sha256':sha(canonical(sorted(cases,key=canonical))),'callback_union':union,'cases':requirements,'support_boundary':'All finite source cases; four storage-width floating multiplication families only on returned-active branches. Empty/all-masked and rejection/observation branches have no numerical obligations.'}],'summary':summary}
assert len(rules)==4 and len(provenance)==64
validate_deferrals(TASK,deferrals,{'rules':list(rules.values())})
assert {(r['assumption_id'],r['scope_binding_id']) for c in requirements for r in c['required_binding_refs']}=={(d['assumption_id'],d['scope_binding_id']) for d in deferrals}
supp={'task':TASK,'required_value_stages':stages,'source_sha256':{**row['input_sha256']['lean'],**row['input_sha256']['python']},'source_views':[{'id':'triton_mul2_finite_original_host_grid','variant':'original_source_translation','frontend':'task_local_exact_ast','ordinary_shared_dsl_admission':False,'memory_semantics':'sequentially_consistent_schedule','original_python':PYTHON,'candidate_python':None,'selected_definitions':{BASE+'Kernels.lean':['mul2_kernel','mul2_inplace_kernel'],BASE+'Proofs/FiniteSource.lean':['lowerBody','runOriginalProgram','runInplaceProgram'],BASE+'Proofs/FiniteScalar.lean':['source','result']},'entry_aliases':{BASE+'Proofs/FiniteHost.lean':['executeHost','hostWith','launchWith']},'translation_trust_boundary':'Both unchanged seven-statement ASTs; original annotated Python offline dtype/block compilation. Concrete finite words/byte effects and full wrapper orchestration; ordinary Torch/runtime/compiler/MMU boundary explicit. No native numerical validation claim.'}],'declarations':[declaration],'review':None}
for name,value in [('declaration-proposal.json',declaration),('scope-row-proposal.json',row),('scope-cases.json',cases),('catalog-supplement-task.proposed.json',supp),('numerical-authority-delta.json',authority)]:save(name,value)
save('proposal-validation.json',{'status':'passed','counts':summary,'checks':['current specification parser','unchanged schema2 declaration_scope','all validate_case calls','unchanged validate_deferrals','64 exact case-operation bindings','22 original native dtype routes and correct first rejection','only4 floating multiply families;zero integer/bool/structural deferrals'],'independent_root_review':'pending','scope_acceptance_increment':0})
print(json.dumps(summary))
