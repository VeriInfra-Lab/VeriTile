"""Additive actual-i32 source authority proposal; independent review required."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from copy import deepcopy
import ast,hashlib,json,re
W=Path(__file__).resolve().parent;R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');OLD=Path('/tmp/veritile-standardization-20261015/vector-finite-acceptance-wip')
sys.path.insert(0,str(R/'bench'))
from adapter_common import canonical,declarations,sha
from check_original_program_scope import declaration_scope,validate_case,validate_deferrals
from scope_helpers import dtype,callback,operation_types
load=lambda p:json.loads(Path(p).read_text());hash_file=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):(W/'evidence'/name).write_text(json.dumps(obj,indent=2)+'\n')
TASK='vector_addition';BASE='bench/tritonbench_g/vector_addition/';PUBLIC=BASE+'NativeCorrect.lean';NAME='vector_addition_native_source_outcomes';PYTHON=BASE+'vector_addition.py'
module_hashes={str(p.relative_to(W/'after')):hash_file(p) for p in (W/'after').rglob('*.lean')}
ref={'file':PUBLIC,'declaration':NAME,'statement_sha256':declarations(W/'after'/PUBLIC)[NAME]['statement_sha256']}
profiles={(p['x_dtype'],p['y_dtype']):p for p in load(W/'evidence/inherited/typed-host-numerical-proposal.json')['profiles']}
latest=load(W/'evidence/native-latest.json');nr=W/latest['receipt'];receipt=load(nr);fr={(r['x'],r['y']):r for r in load(nr.with_name('records.json'))}
row=deepcopy(next(x for x in load(R/'bench/original_program_scope.json')['tasks'] if x['task']==TASK));inventory=row['source_inventory'];stages=[s['id'] for s in inventory['stages'] if not s['may_be_test_harness']];dtypeids=[s['id'] for s in inventory['dtypes']];branchids=[s['id'] for s in inventory['branches']]
configs=[];extras={'complex32','complex64','complex128','float8_e8m0fnu','float4_e2m1fn_x2'}
observations=['x_access','y_access','x_missing_bytes','y_missing_bytes','x_encoding','y_encoding','undefined_conversion','output_access','ownership_conflict']
for pair,f in fr.items():
 x,y=pair;p=profiles.get(pair)
 if f.get('backend')=='compiled':terminal='returned_or_undefined_conversion'
 elif f.get('backend')=='rejected':terminal='backend_rejected'
 elif f['error_type']=='KeyError':terminal='binder_rejected'
 elif 'fnuz' in x or 'fnuz' in y:terminal='signature_rejected'
 else:terminal='addition_rejected'
 outcomes=[('allocation_error','none'),('cuda_assertion','none'),('context_error','none'),(terminal,'none')]
 if terminal=='returned_or_undefined_conversion':outcomes += [(o,'none') for o in ['compilation_error','pointer_packing_error','grid_limit','launch_setup_error','two31_no_access_return']] + [('native_observation',o) for o in observations]
 for outcome,observation in outcomes:
  compute={'index':'int32','pointer':'uint64'}
  if p:compute.update(value=dtype(p['compute_dtype']),store=dtype(p['native_store_dtype']))
  c={'dtype_profile':{'inputs':{'x':dtype(x),'y':dtype(y)},'outputs':{'output':dtype(x)},'compute':compute},'configuration':{
    'entrypoint':'add','source_variant':'original','BLOCK_SIZE':1024,'target':receipt['target'],'torch_version':receipt['torch'],'triton_version':receipt['triton'],
    'debug':False,'instrumentation_mode':'none','outcome':outcome,'native_observation':observation,
    'n_elements':{'lower_inclusive':2**31 if outcome=='two31_no_access_return' else 0,'upper_exclusive':2**31+1 if outcome=='two31_no_access_return' else 2**63},
    'host_grid':'ceil_div_original_output_numel_1024','setup_order':['context','signature_and_backend_admission','compilation','signed_grid_tuple_parse','pointer_packing','launch'],'grid_limit':2**31-1,'grid_limit_outcome':'genuine_launch_setup_failure',
    'runtime_n_signature':'annotated_i32_for_all_host_extents','bound_conversion':'PyLong_AsLong_success_then_int32_assignment_mod2pow32',
    'lane_arithmetic':'i32_pid_times1024_plus_lane_wrap_before_signed_slt','pointer_arithmetic':'sign_extend_i32_offset_then_scale_storage_width_then_wrap_uint64',
    'host_extent_regimes':['zero','positive_below2pow31','exact2pow31','negative_i32_bound','repeated_i32_offsets','native_grid_limit'],
    'layouts':'all_actual_dense_strided_Torch_metadata; source_flat_data_ptr_access','storage_offsets':'actual_nonnegative_element_offset; empty_data_ptr_zero',
    'input_aliasing':'arbitrary_x_y_overlap','y_logical_numel':'unrestricted; wrapper_has_no_shape_equality_or_y_extent_check',
    'device':'original_three_CUDA_assertion; actual_access_permissions_observed; no_same_device_guard_added',
    'allocation':'fresh_empty_like_x_with_actual_preserve_format_strides_and_initial_exterior_frame',
    'scalar_definedness':'partial_conversion_observation; no_undefined_fallback_is_stored','observations':'mathematical_requests_only; no_native_trap_race_winner_rollback_or_postfailure_memory_claim',
    'numeric_claim':'actual_loaded_words_and_source_result_with_defined_conversion_partition' if outcome=='returned_or_undefined_conversion' else 'value_insensitive_constructor_or_first_setup_error_only',
    'first_rejection_argument':('x' if x in extras else 'y') if terminal=='binder_rejected' else ('x' if 'fnuz' in x else 'y') if terminal=='signature_rejected' else 'none'}}
  validate_case(c);configs.append((pair,c))
cases=[c for _,c in configs];assert len({canonical(c) for c in cases})==len(cases)
rules={};provenance={};requirements=[];links={}
for pair,c in configs:
 p=profiles.get(pair);selected=p['operations'] if c['configuration']['outcome']=='returned_or_undefined_conversion' else [];refs=[]
 for index,op in enumerate(selected):
  port=op['port'];cb=callback(port);ident='VECTOR-NATIVE-'+re.sub('[^A-Z0-9]+','-',port.upper()).strip('-')
  if ident not in rules:rules[ident]={'id':ident,'kind':'numerical_rule_conformance','operation_class':'scalar_primitive_conformance','status':'pending_numeric_validation','queue_status':'exact_profile_bound_validation_pending','affected_tasks':[TASK],'required_validation':'Native scalar correspondence for '+port+' on the actual active loaded words, operand/output dtypes and selected sm90 configuration, including rounding, subnormal, NaN, infinity and signed-zero behavior. Undefined conversion observations are structural and cannot be waived.','native_whole_kernel_validated':False,'validation_evidence':None,'scope_bindings':[]}
  body={'task':TASK,'declarations':[ref],'callbacks':[cb],**deepcopy(c)};bid=TASK+':native:'+sha(canonical({'rule':ident,'binding':body,'source_port':op,'ordinal':index}))[:32]
  rules[ident]['scope_bindings'].append({'id':bid,**body});link={'assumption_id':ident,'scope_binding_id':bid};links.setdefault(cb,[]).append(link);refs.append({'callback':cb,**link});it,ot=operation_types(pair,p,op);raw=fr[pair]
  provenance[ident+'::'+bid]={'kind':'scalar_primitive_conformance','status':'explicitly_deferred_unvalidated','validation_evidence':None,**link,'callback_declaration':cb,'correspondence_kind':'parametric_callback' if port.startswith('b.') else 'concrete_scalar_port','port':{'name':port,'ordinal':index,'operation':op['operation'],'input_types':it,'output_type':ot,'scalar_configuration':op['configuration']},'scalar_domain':'Actual active typed input words; scalarDefined observes partial conversions before values materialize. Input masks and addresses are full source i32/u64 functions without a no-wrap bound.','scalar_path_predicate':'actual_signed_mask_active; selected_route='+p['family'],'native_execution':False,'source_statement':op['statement'],'original_source':{'path':PYTHON,'sha256':hash_file(R/PYTHON)},'public_statement':ref,'lean_source_sha256':module_hashes,'scalar_source':{'path':str(Path(op['source']).relative_to(R)),'sha256':op['source_sha256']},'execution_source':p['execution_binding'],'host_source':p['host_binding'],'raw_ttir':{'path':str((nr.parent/(pair[0]+'__'+pair[1]+'.raw.ttir')).relative_to(W)),'sha256':raw['raw_sha256']},'frontend_receipt':{'path':latest['receipt'],'sha256':latest['sha256']},'producer_scalar_inventory':{'path':'evidence/inherited/typed-host-numerical-proposal.json','sha256':hash_file(W/'evidence/inherited/typed-host-numerical-proposal.json')},'configuration':c['configuration'],'structural_prerequisites_not_waived':['actual annotated_i32_ABI and signed masks','actual wrapped64bit addresses','ordinary fresh allocation and input byte frame','concrete typed scalar source correspondence','undefined conversion observation without arbitrary stored value','missing byte and ownership observations without native exception claims']}
 requirements.append({**deepcopy(c),'required_callbacks':sorted({r['callback'] for r in refs}),'required_binding_refs':refs})
for rule in rules.values():rule['scope_bindings'].sort(key=lambda b:b['id'])
deferrals=[{'kind':'numerical_rule_conformance','assumption_id':ident,'rule_sha256':sha(canonical(rule)),'scope_binding_id':binding['id'],'dtype_profile':binding['dtype_profile'],'configuration':binding['configuration'],'required_validation':rule['required_validation']} for ident,rule in rules.items() for binding in rule['scope_bindings']]
union=sorted(links);scope={'source_inventory_sha256':sha(canonical(inventory)),'stages':stages,'dtypes':dtypeids,'branches':branchids,'cases':{ident:deepcopy(cases) for ident in stages+dtypeids+branchids}}
declaration={**ref,'kind':'specification','model_kind':'backend_value_contract','role':'parametric_value_contract','scope_relation':'full_declared_stage','source_variant':'original_source_translation','covers_stages':stages,'dtype_profile':'All22x22 original dense dtype signatures:195 binder,64 target-signature,36 addition and6 backend rejections;183 native setup routes. Full nonnegative signed64 metadata extent, actual i32 offsets/masks and wrapped64bit addresses.','dtype_scope_relation':'complete_enumerated_wrapper_dtype_domain','unclosed_numeric_callbacks':union,'extra_numeric_law_ids':[],'native_binding_status':'not_established','review_basis':'Producer proposal requiring independent substantive root review; original AST/binder/backend matrix and exact installed ABI extraction are receipt-bound. Complete actual-address source load/compute/store and original host outcomes; conditional prior Nat formula retained.','original_scope':scope}
declaration_scope(declaration,inventory);validate_deferrals(TASK,deferrals,{'rules':list(rules.values())})
coverage={k:[] for k in ['stages','dtypes','branches']}
for dimension,items in inventory.items():
 for item in items:
  if dimension=='stages' and item['may_be_test_harness']:coverage[dimension].append({'id':item['id'],'status':'test_harness','explanation':'Original non-JIT test_add has no production caller; four wrapper cases and import-time result_gold remain inventoried in inherited HostWrapper proof.'})
  else:coverage[dimension].append({'id':item['id'],'status':'covered','cases':deepcopy(cases),'proofs':[ref],'explanation':'All original22 dense signatures and full nonnegative signed64 extents; actual annotated32bit masks and sign-extended wrapped pointers. Independent write plan/byte frames and metadata; explicit constructor observations never claim native trap, race winner, rollback or universal reachability. Prior safe Nat prefix theorem remains conditional.'})
row.update(status='pending',scope_schema=2,coverage=coverage,structural_gaps=[],review=None,deferred_numerical_assumptions=deferrals,callback_assumptions=links,proposed_status_after_independent_root_review='verified_with_deferred_numerics');row['input_sha256']['lean'].update(module_hashes)
row['acceptance_boundary_review']={'status':'producer_proposal_awaiting_independent_root_review','policy':'bench/original_program_acceptance.md','policy_sha256':hash_file(R/'bench/original_program_acceptance.md'),'source_domain':'Original22 dense Torch dtype names; nonnegative signed64 tensor extents; actual metadata/ordinary allocation; CUDA sm90 debug=false Triton3.7.1. Access and value safety are observations, not bounds on master applicability.','native_correctness':False,'prior_evidence':'All prior typed and bounded finite declarations/registrations retained; the6 exact backend failures are additional setup outcomes, not deleted source routes.','queue_integration':'Reviewed frozen additive authority and unchanged whole-queue equality required.'}
summary={'scalar_families':len(rules),'scope_bindings':len(provenance),'declaration_cases':len(cases),'registered_declarations':1,'original_dtype_pairs':484,'binder_rejections':195,'signature_rejections':64,'addition_rejections':36,'backend_rejections':6,'backend_compiles':183,'frontend_typed_routes_retained':189,'native_executions':0,'original_program_completion_increment':0}
authority={'schema':1,'kind':'vector_native_exact_numeric_authority_delta','task':TASK,'rules':list(rules.values()),'provenance':provenance,'requirements':[{'task':TASK,'declaration':ref,'source_ref':{'path':PUBLIC,'sha256':module_hashes[PUBLIC]},'supported_cases_sha256':sha(canonical(sorted(cases,key=canonical))),'callback_union':union,'cases':requirements,'support_boundary':'Actual signed-i32 requests over full host metadata, all setup failures and structural observations. Numeric ports apply to loaded-word returned/undefined partition; value-insensitive constructor-only cases assert no numeric intermediate or native failure memory.'}],'summary':summary}
supp=load(OLD/'evidence/catalog-supplement-task.proposed.json');supp['declarations'].append(declaration);supp['source_sha256'].update(module_hashes);supp['review']=None;supp['source_views'].append({'id':'native_i32_full_host_extent','variant':'original_source_translation','frontend':'task_local_exact_ast','ordinary_shared_dsl_admission':False,'memory_semantics':'sequentially_consistent_schedule','original_python':PYTHON,'candidate_python':None,'selected_definitions':{BASE+'Proofs/NativeProgram.lean':['executeProgram'],BASE+'Proofs/NativeGrid.lean':['executeLaunch']},'entry_aliases':{PUBLIC:['executeHost']},'translation_trust_boundary':'Frozen original22x22 binder/frontend/backend audit; exact installed i32 extraction; actual source typed scalar routes; ordinary Torch metadata/fresh allocation and typed local memory primitives. Grid/load/store constructed; no result or launch-success oracle.'})
for name,data in [('declaration-proposal.json',declaration),('scope-row-proposal.json',row),('scope-cases.json',cases),('catalog-supplement-task.proposed.json',supp),('numerical-authority-delta.json',authority)]:save(name,data)
save('proposal-validation.json',{'status':'passed','counts':summary,'checks':['unchanged validate_case','unchanged declaration_scope','unchanged validate_deferrals','native22x22 partition','additive scalar authority retains prior evidence'],'independent_review':'pending','acceptance_increment':0});print(json.dumps(summary))
