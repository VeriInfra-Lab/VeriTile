"""Producer proposal for full original-source add_example; review/acceptance remain zero."""
from pathlib import Path
from copy import deepcopy
import ast,hashlib,json,re,sys
sys.dont_write_bytecode=True
W=Path(__file__).resolve().parent;R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sys.path.insert(0,str(R/'bench'))
from adapter_common import canonical,declarations,sha
from check_original_program_scope import declaration_scope,validate_case,validate_deferrals
from scope_helpers import dtype,callback,operation_types
TASK='add_example';BASE='bench/tritonbench_g/add_example/';PUBLIC=BASE+'FiniteCorrect.lean';NAME='add_example_actual_source_outcomes';PYTHON=BASE+'add_example.py'
MODULES=[BASE+'Proofs/Finite'+n+'.lean' for n in ['Indices','Scalar','Source','Physical','Program','Grid','Admission','Host']]+[PUBLIC]
FRONT='evidence/native-attempts/20261015T215117570205Z/receipt.json'
BACK='evidence/backend-attempts/20261015T215542684768Z/receipt.json'
HOST='evidence/host-attempts/20261015T215521664431Z/receipt.json'
load=lambda p:json.loads(Path(p).read_text());hash_file=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,data):(W/'evidence'/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def generate():
 source=(R/PYTHON).read_text();fs={n.name:n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
 host=[ast.unparse(n) for n in fs['add_wrapper'].body];kernel=[ast.unparse(n) for n in fs['add_kernel'].body]
 assert host==['out = torch.zeros_like(x)','BLOCK_SIZE = 4','n_elements = x.numel()','num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE','add_kernel[num_blocks,](x, y, out, n_elements, BLOCK_SIZE)','return out']
 assert kernel==['pid = tl.program_id(axis=0)','block_start = pid * BLOCK_SIZE','offsets = block_start + tl.arange(0, BLOCK_SIZE)','mask = offsets < n_elements','x = tl.load(in_ptr0 + offsets, mask=mask)','y = tl.load(in_ptr1 + offsets, mask=mask)','output = x + y','tl.store(out_ptr + offsets, output, mask=mask)']
 frontend=load(W/FRONT);backend=load(W/BACK);cpu=load(W/HOST)
 prefix=(W/'evidence/source-prefix.py').read_text()
 assert ast.dump(ast.parse(prefix),include_attributes=False)==ast.dump(ast.Module(body=ast.parse(source).body[:5],type_ignores=[]),include_attributes=False)
 for path in [FRONT,BACK,HOST]:
  rec=load(W/path)
  assert rec['gpu_executed'] is False
  for name,h in rec['artifacts'].items():assert hash_file((W/path).parent/name)==h
 assert frontend['cases']==484 and frontend['frontend_generated']==189
 records={(a['x'],a['y']):a for a in load((W/FRONT).with_name('records.json'))}
 back={(a['x'],a['y']):a for a in load((W/BACK).with_name('records.json'))}
 assert len(back)==189 and sum(a['status']=='compiled' for a in back.values())==183
 names=frontend['canonical_names'];assert len(names)==22
 profiles={(p['x_dtype'],p['y_dtype']):p for p in load(W/'evidence/inherited/typed-host-numerical-proposal.json')['profiles']}
 refs=[{'path':p,'sha256':hash_file(W/p)} for p in [FRONT,BACK]]
 modules={p:hash_file(W/p) for p in MODULES};ref={'file':PUBLIC,'declaration':NAME,'statement_sha256':declarations(W/PUBLIC)[NAME]['statement_sha256']}
 original={'path':PYTHON,'sha256':hash_file(R/PYTHON)}
 row=deepcopy(next(x for x in load(R/'bench/original_program_scope.json')['tasks'] if x['task']==TASK));inventory=row['source_inventory']
 stages=[s['id'] for s in inventory['stages'] if not s['may_be_test_harness']];dtypeids=[s['id'] for s in inventory['dtypes']];branchids=[s['id'] for s in inventory['branches']]
 configs=[];extras=set(names[17:]);counts={}
 for x in names:
  for y in names:
   pair=x,y;p=profiles.get(pair);f=records[pair];native=back.get(pair)
   terminal=('binder_rejected' if x in extras or y in extras else 'signature_rejected' if 'fnuz' in x or 'fnuz' in y else 'addition_rejected' if f['status']!='frontend_generated' else 'backend_rejected' if native['status']!='compiled' else 'admitted')
   counts[terminal]=counts.get(terminal,0)+1
   outcomes=['host_or_native_setup_failure',terminal] if terminal!='admitted' else ['host_or_native_setup_failure','native_grid_parse_rejected','returned_empty','returned_all_masked','source_applicability_observation','returned_defined']
   for outcome in outcomes:
    compute={'index':'int32','pointer':'uint64'}
    if p:compute.update(value=dtype(p['compute_dtype']),store=dtype(p['native_store_dtype']))
    extent=({'equals':0} if outcome=='returned_empty' else {'native_launchable':True,'residue_mod_2pow32':2**31} if outcome=='returned_all_masked' else {'lower_exclusive':(2**31-1)*4} if outcome=='native_grid_parse_rejected' else {'lower_inclusive':1,'upper_inclusive':(2**31-1)*4,'residue_mod_2pow32_not_equal':2**31} if outcome in ['returned_defined','source_applicability_observation'] else {'lower_inclusive':0,'upper_exclusive':'unbounded_actual_Torch_numel'})
    case={'dtype_profile':{'inputs':{'x':dtype(x),'y':dtype(y)},'outputs':{'out':dtype(x)},'compute':compute},'configuration':{'entrypoint':'add_wrapper','source_variant':'original','BLOCK_SIZE':4,'outcome':outcome,'n_elements':extent,'target':frontend['target'],'torch_version':frontend['torch'],'triton_version':frontend['triton'],'native_grid':'ceil_div_original_x_numel_4','native_grid_max':2**31-1,'native_setup_order':['allocation_zeros_like_x','x_numel_and_grid','JIT_current_device_and_stream','binder_frontend_backend','native_launcher_context','signed_int_grid_tuple_parse','pointer_extraction_and_i32_argument','enqueue'],'runtime_n_signature':'annotated_i32_including_specialization_n1','finite_index':'i32_pid_mul4_add_lane_then_signed_slt_i32_n','pointer_arithmetic':'sign_extend_i32_offset_scale_storage_width_wrap_uint64','devices':'all_actual_CPU_or_CUDA_dense_tensor_metadata; permissions_on_current_launch_device; no_owner_equality','layouts':'rank0_empty_noncontiguous_offset_broadcast_and_all_actual_dense_views; source_uses_flat_pointer','input_relation':'arbitrary_x_y_aliasing_dtypes_shapes_lengths_strides_devices; no_wrapper_guards','allocation':'fresh_zeros_like_x; actual_preserve_format_metadata; all_allocated_output_bytes_zero','translation':'actual_extracted_ABI_pointers; shared_native_address_to_canonical_backing_translation; null_data_ptr_no_translated_span_requirement','schedule':'every_complete_program_schedule; prestore_output_overlap_is_an_observation_without_a_selected_winner','memory':'ordinary_local_byte_decode_and_encode; active_reads_from_actual_backing_bytes; independent_initial_snapshot_formula_and_full_byte_frame','observations':'actual_input_access_encoding_conversion_output_access_ownership; no_native_exception_return_rollback_or_final_memory_claim','numeric_domain':'active_defined_returned_scalar_ports_only' if outcome=='returned_defined' else 'no_numeric_result_claim'}}
    validate_case(case);configs.append((pair,case))
 assert counts=={'admitted':183,'addition_rejected':36,'signature_rejected':64,'binder_rejected':195,'backend_rejected':6},counts
 cases=[c for _,c in configs];assert len({canonical(c) for c in cases})==len(cases)
 rules={};provenance={};requirements=[];links={}
 for pair,c in configs:
  p=profiles.get(pair);selected=p['operations'] if c['configuration']['outcome']=='returned_defined' else [];required=[]
  for index,op in enumerate(selected):
   port=op['port'];cb=callback(port);ident='ADD-EXAMPLE-ACTUAL-'+re.sub('[^A-Z0-9]+','-',port.upper()).strip('-')
   if ident not in rules:rules[ident]={'id':ident,'kind':'numerical_rule_conformance','operation_class':'scalar_primitive_conformance','status':'pending_numeric_validation','queue_status':'exact_profile_bound_validation_pending','affected_tasks':[TASK],'required_validation':'Exact native scalar correspondence for '+port+' on active loaded words and this dtype/target path, including rounding, subnormals, NaN, infinity and signed zero. Partial conversion must be defined; structural observations cannot be waived.','native_whole_kernel_validated':False,'validation_evidence':None,'scope_bindings':[]}
   body={'task':TASK,'declarations':[ref],'callbacks':[cb],**deepcopy(c)};bid=TASK+':actual:'+sha(canonical({'rule':ident,'binding':body,'source_port':op,'ordinal':index}))[:32]
   rules[ident]['scope_bindings'].append({'id':bid,**body});link={'assumption_id':ident,'scope_binding_id':bid};links.setdefault(cb,[]).append(link);required.append({'callback':cb,**link});it,ot=operation_types(pair,p,op)
   rawpath=(Path(FRONT).parent/(pair[0]+'__'+pair[1]+'.ttir')).as_posix()
   provenance[ident+'::'+bid]={'kind':'scalar_primitive_conformance','status':'explicitly_deferred_unvalidated','validation_evidence':None,**link,'callback_declaration':cb,'correspondence_kind':'parametric_callback' if port.startswith('b.') else 'concrete_scalar_port','port':{'name':port,'ordinal':index,'operation':op['operation'],'input_types':it,'output_type':ot,'scalar_configuration':op['configuration']},'scalar_domain':'Canonical actual active input words with defined float-to-integer conversions; finite i32/u64 addressing is proved separately.','scalar_path_predicate':'active source signed-i32 mask; scalar conversion defined; native sm90 route compiled; '+p['family'],'native_execution':False,'source_statement':op['statement'],'original_source':original,'public_statement':ref,'lean_source_sha256':modules,'scalar_source':{'path':str(Path(op['source']).relative_to(R)),'sha256':op['source_sha256']},'execution_source':p['execution_binding'],'raw_ttir':{'path':rawpath,'sha256':hash_file(W/rawpath)},'native_receipts':refs,'producer_scalar_inventory':{'path':'evidence/inherited/typed-host-numerical-proposal.json','sha256':hash_file(W/'evidence/inherited/typed-host-numerical-proposal.json')},'configuration':c['configuration'],'structural_prerequisites_not_waived':['exact original eight-statement AST','actual runtime i32 ABI and signed masks','actual pointer64 and shared byte translation','allocation zero bytes and exterior frame','all tile x loads then all tile y loads','conversion definedness before stores','output ownership checks without choosing a race winner']}
  requirements.append({**deepcopy(c),'required_callbacks':sorted({q['callback'] for q in required}),'required_binding_refs':required})
 for rule in rules.values():rule['scope_bindings'].sort(key=lambda b:b['id'])
 deferrals=[{'kind':'numerical_rule_conformance','assumption_id':ident,'rule_sha256':sha(canonical(rule)),'scope_binding_id':binding['id'],'dtype_profile':binding['dtype_profile'],'configuration':binding['configuration'],'required_validation':rule['required_validation']} for ident,rule in rules.items() for binding in rule['scope_bindings']]
 union=sorted(links);scope={'source_inventory_sha256':sha(canonical(inventory)),'stages':stages,'dtypes':dtypeids,'branches':branchids,'cases':{ident:deepcopy(cases) for ident in stages+dtypeids+branchids}}
 declaration={**ref,'kind':'specification','model_kind':'backend_value_contract','role':'parametric_value_contract','scope_relation':'full_declared_stage','source_variant':'original_source_translation','covers_stages':stages,'dtype_profile':'All22x22 actual dense dtype pairs, full original numel/grid and runtime-i32 extent, native failure priority, all actual dense CPU/CUDA layouts and canonical backing bytes.','dtype_scope_relation':'complete_enumerated_wrapper_dtype_domain','unclosed_numeric_callbacks':union,'extra_numeric_law_ids':[],'native_binding_status':'not_established','review_basis':'PRODUCER PROPOSAL; independent root substantive review required. Exact source AST, full finite grid, typed routes, ordinary zero allocation, complete byte formula and all complete schedules.','original_scope':scope}
 declaration_scope(declaration,inventory);validate_deferrals(TASK,deferrals,{'rules':list(rules.values())})
 coverage={k:[] for k in ['stages','dtypes','branches']}
 for dimension,items in inventory.items():
  for item in items:
   if dimension=='stages' and item['may_be_test_harness']:coverage[dimension].append({'id':item['id'],'status':'test_harness','explanation':'Original non-JIT test_add_kernel contains four fixed wrapper examples and import-time result_gold assignment; production wrapper and kernel are fully covered.'})
   else:coverage[dimension].append({'id':item['id'],'status':'covered','cases':deepcopy(cases),'proofs':[ref],'explanation':'Complete original dtype/setup/finite-access partition; source AST and ordinary zero allocation feed an initial-snapshot independent full-byte formula. No added source guard or chosen undefined/race result.'})
 row.update(status='pending',scope_schema=2,coverage=coverage,structural_gaps=[],review=None,deferred_numerical_assumptions=deferrals,callback_assumptions=links,proposed_status_after_independent_root_review='verified_with_deferred_numerics');row['input_sha256']['lean'].update(modules)
 row['acceptance_boundary_review']={'status':'producer_proposal_awaiting_independent_root_review','policy':'bench/original_program_acceptance.md','policy_sha256':hash_file(R/'bench/original_program_acceptance.md'),'finite_domain':'Full original host tensor metadata and finite runtime-i32 requests; all22x22 dtype routes; no rank, dtype, shape, stride, y-length or CUDA assertion.','native_correctness':False,'native_order':'JIT context; compile; native context; signed-grid parse; pointer/i32 arguments; enqueue','queue_integration':'Root must pin frozen producer receipt and independent root review receipt; retain exact whole-queue equality.'}
 summary={'scalar_families':len(rules),'scope_bindings':len(provenance),'declaration_cases':len(cases),'registered_declarations':1,'original_dtype_pairs':484,**counts,'frontend_typed_routes_retained':189,'native_executions':0,'original_program_completion_increment':0}
 delta={'schema':1,'kind':'add_example_actual_exact_numeric_authority_delta','task':TASK,'original_source':original,'dependency_snapshots':[{'path':'evidence/dependencies-before.json','sha256':hash_file(W/'evidence/dependencies-before.json')}],'native_evidence':[{'role':'full22x22_frontend','receipt':refs[0]},{'role':'full189_backend','receipt':refs[1]}],'rules':list(rules.values()),'provenance':provenance,'requirements':[{'task':TASK,'declaration':ref,'source_ref':{'path':PUBLIC,'sha256':modules[PUBLIC]},'supported_cases_sha256':sha(canonical(sorted(cases,key=canonical))),'callback_union':union,'cases':requirements,'support_boundary':'Only defined active returned values require numeric conformance; setup failures, empty/all-masked returns and applicability observations make no numeric-result claim.'}],'summary':summary}
 assert {(v['assumption_id'],v['scope_binding_id']) for c in requirements for v in c['required_binding_refs']}=={(v['assumption_id'],v['scope_binding_id']) for v in deferrals}
 save('source-check.json',{'original_source':original,'source_prefix_sha256':hash_file(W/'evidence/source-prefix.py'),'AST_identity':True,'host_statements':host,'kernel_statements':kernel,'frontend_receipt':refs[0],'backend_receipt':refs[1],'cpu_metadata_receipt':{'path':HOST,'sha256':hash_file(W/HOST)},'runtime_i32':'n_elements annotation overrides binder specialization, LP64 extractI32 narrows PyLong_AsLong to int32; native grid parse bounds reachable host n before extraction','gpu_executed':False,'numerical_validation':False})
 for name,data in [('declaration-proposal.json',declaration),('scope-row-proposal.json',row),('scope-cases.json',cases),('numerical-authority-delta.json',delta)]:save(name,data)
 save('proposal-validation.json',{'status':'passed','counts':summary,'checks':['unchanged validate_case','unchanged declaration_scope','unchanged validate_deferrals','all484 native route partition','source-derived numerical port/deferral bijection'],'independent_review':'pending','acceptance_increment':0})
 print(json.dumps(summary))
if __name__=='__main__':generate()
