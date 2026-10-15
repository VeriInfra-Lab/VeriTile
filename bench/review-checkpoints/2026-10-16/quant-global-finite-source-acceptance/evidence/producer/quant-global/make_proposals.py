from pathlib import Path
import ast,copy,hashlib,json,sys
sys.dont_write_bytecode=True
REPO=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sys.path.insert(0,str(REPO/'bench'))
from adapter_common import canonical,sha,hash_file,declarations
import check_original_program_scope as scope_gate
ROOT=Path(sys.argv[1]).resolve();TASK='quantize_global';BASE='bench/tritonbench_g/quantize_global/'
PYTHON=BASE+'quantize_global.py';PUBLIC=BASE+'FullSourceCorrect.lean';NAME='quantize_global_full_source_outcomes'
NS='QuantizeGlobalFinite.'
def load(p):return json.loads(Path(p).read_text())
def save(name,data):
 p=ROOT/'evidence'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,indent=2)+'\n')
def refpath(p):return {'path':str(p.relative_to(ROOT)),'sha256':hash_file(p)}
MAPPING=load(ROOT/'evidence/import-mapping.json');MODULES=MAPPING['modules']
NATIVE=ROOT/'evidence/native-attempts/20261015T204252377299Z'
native=load(NATIVE/'receipt.json')
assert hash_file(REPO/PYTHON)==native['source_sha256']=='163f6e73234587c1d05763ed60ccc1ac4ca7d7d384f18b7af17399509205032f'
assert native['production_ast_unchanged'] and native['gpu_launches']==0
records=native['records'];assert len(records)==52
assert sum(r['status']=='compiled' for r in records)==32
for r in records:
 for art in r.get('artifacts',{}).values():assert hash_file(NATIVE/Path(art['path']).name)==art['sha256']
source=ast.parse((REPO/PYTHON).read_text())
functions={n.name:n for n in source.body if isinstance(n,ast.FunctionDef)}
production=['_quantize_global','quantize_global']
copied=ast.parse((NATIVE/'quantize_global.py').read_text())
cp={n.name:n for n in copied.body if isinstance(n,ast.FunctionDef)}
assert all(ast.dump(functions[n],include_attributes=False)==ast.dump(cp[n],include_attributes=False) for n in production)
assert not functions['test_quantize_global'].decorator_list
assert all(not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='test_quantize_global' for n in ast.walk(functions[f])) for f in production)
ref={'file':PUBLIC,'declaration':NAME,'statement_sha256':declarations(ROOT/'overlay'/PUBLIC)[NAME]['statement_sha256']}
ledger=load(REPO/'bench/original_program_scope.json');row=copy.deepcopy(next(r for r in ledger['tasks'] if r['task']==TASK));inventory=row['source_inventory']
names=['bool','uint8','int8','int16','int32','int64','uint16','uint32','uint64','float16','bfloat16','float32','float64','complex32','complex64','complex128','float8_e4m3fn','float8_e5m2','float8_e4m3fnuz','float8_e5m2fnuz','float8_e8m0fnu','float4_e2m1fn_x2']
admitted=set(names[:6]+['float32','float64']);fp8=set(names[16:20]);abs_always=set(names[1:6]+names[9:16])
def dt(d):return {'float16':'fp16','bfloat16':'bf16','float32':'fp32','float64':'fp64','float8_e4m3fn':'fp8e4m3fn','float8_e5m2':'fp8e5m2','float8_e4m3fnuz':'fp8e4m3fnuz','float8_e5m2fnuz':'fp8e5m2fnuz','float8_e8m0fnu':'fp8e8m0fnu','float4_e2m1fn_x2':'packed_fp4e2m1'}.get(d,d)
def maxdt(d):return {'complex32':'fp16','complex64':'fp32','complex128':'fp64'}.get(d,dt(d))
def invdt(d):return 'fp32' if d in set(names[:6]) else maxdt(d)
added={'complex32','complex64','complex128','fp8e8m0fnu'}-scope_gate.DTYPES
save('dtype-enum-extension.json',{'schema':1,'kind':'additive_exact_rejected_dtype_labels','checker':'bench/check_original_program_scope.py','existing_checker_sha256':hash_file(REPO/'bench/check_original_program_scope.py'),'add_to_DTYPES':sorted(added),'existing_fp4_label':'packed_fp4e2m1','reason':'Canonical installed Torch dense dtype families must retain exact rejection labels; no validation rule is removed or relaxed.','canonical_integration':'pending independent root review','producer_validation':'unchanged validator functions with only these exact labels added in memory; canonical validation is not thereby claimed'})
scope_gate.DTYPES.update(added)
def classify(d,dev,shape):
 reached=['abs']
 if not (d in abs_always or (d=='bool' and dev=='cuda') or (d in fp8 and dev=='cpu')):return reached,'unsupported_abs'
 reached+=['maximum']
 if shape=='empty':return reached,'empty_maximum'
 if d in fp8:return reached,'unsupported_maximum'
 reached+=['unsqueeze','reciprocal']
 if shape=='rank_zero':return reached,'missing_size'
 reached+=['outputAllocation']
 if dev=='cpu':return reached,'cuda_assertion'
 reached+=['signature']
 if d.startswith('complex'):return reached,'unsupported_signature'
 reached+=['compilation']
 if d in ['float16','bfloat16']:return reached,'unsupported_llrint'
 reached+=['launch']
 assert d in admitted
 if shape=='large_unsafe':return reached,'i32_offset_wrap'
 return reached,'returned_or_undefined_conversion'
configs=[]
for d in names:
 for dev in ['cpu','cuda']:
  for block in [1024,2048]:
   shapes=[('empty',0,1),('rank_zero',1,2),('one_block',1,block+1),('multiple_blocks',block+1,2**31),('i64_n_boundary',2**31,2**31+1),('large_unsafe',2**31+1,2**63)]
   for shape,lo,hi in shapes:
    reached,terminal=classify(d,dev,shape)
    outcomes=['primitive_failure:'+stage for stage in reached]+[terminal]
    if terminal=='returned_or_undefined_conversion':outcomes+=['linear_view_overrun']
    for outcome in outcomes:
     effective=reached[:reached.index(outcome.split(':',1)[1])] if outcome.startswith('primitive_failure:') else reached
     compute={}
     if 'reciprocal' in effective:compute['inverse']=invdt(d)
     if 'compilation' in effective:compute['kernel_arithmetic']=invdt(d)
     if 'launch' in effective:compute.update(offset='int32',n_elements='int64' if lo>=2**31 else 'int32')
     c={'dtype_profile':{'inputs':{'x':dt(d)},'outputs':{'output':'int8','absmax':maxdt(d)},'compute':compute},
      'configuration':{'entrypoint':'quantize_global','source_variant':'original','device':dev,'torch_default_dtype':'fp32',
       'block_size':block,'num_warps':4,'num_stages':3 if block==1024 else 1,'num_ctas':1,
       'N':{'lower_inclusive':lo,'upper_exclusive':hi},'shape_class':shape,'rank':'zero' if shape=='rank_zero' else 'positive',
       'outcome':outcome,'first_failure_semantics':'first_reached_source_stage; all earlier primitive outcomes succeed',
       'grid':'ceil_div_N_BLOCK_SIZE','trials':'arbitrary_finite_successful_autotune_trials_from_both_source_configs_then_final_selected_config',
       'target':{'backend':'cuda','arch':90,'warp_size':32},'torch_version':native['torch_version'],'triton_version':native['triton_version'],
       'layout':'all_actual_dense_strided_views; abs_max_logical_view_then_kernel_linear_data_ptr',
       'physical_binding':'actual_base_storage_offset_byte_capacity_alignment_no_wrap_lifetime_permissions_and_pairwise_fresh_intervals',
       'devices':'input_max_inverse_on_input_cuda_ordinal; output_on_current_default_cuda_ordinal; actual_launch_read_write_permissions',
       'output_layout':'fresh_contiguous_int8_shape_x_shape_offset_zero; arbitrary_torch_empty_payload',
       'absmax_return':'unsqueeze_0_alias_of_rank_zero_max; shape1_stride1_input_dtype_for_admitted_routes',
       'numerical_values':'all_finite_storage_words_including_NaN_Inf_signed_min_and_noncanonical_bool; undefined_llrint_has_no_value'}}
     scope_gate.validate_case(c);configs.append((d,c))
cases=[c for _,c in configs];assert len({canonical(c) for c in cases})==len(cases)
stage_ids=[s['id'] for s in inventory['stages'] if not s['may_be_test_harness']];dtype_ids=[s['id'] for s in inventory['dtypes']];branch_ids=[s['id'] for s in inventory['branches']]
scope={'source_inventory_sha256':sha(canonical(inventory)),'stages':stage_ids,'branches':branch_ids,'dtypes':dtype_ids,'cases':{i:copy.deepcopy(cases) for i in stage_ids+dtype_ids+branch_ids}}
families={
 'ABS':(NS+'Numerics.abs','torch_abs','Source dtype-specific abs on every logical view word, including noncanonical bool normalization, signed-min behavior, signed zeros, NaNs/infinities and native floating policy.'),
 'MAX':(NS+'Numerics.maximum','torch_whole_tensor_max','Invocation-bound trusted whole tensor reduction over source-produced abs values. Bind actual shape, strides, reduction implementation/order and NaN/tie payload behavior; not a fabricated left fold or shape-independent NaN oracle.'),
 'RECIP':(NS+'Numerics.reciprocal','torch_reciprocal_then_wrapped_scalar_1_multiply','Exact unchanged 1.0/absmax producer; bool/integer result fp32 under pinned default dtype float32, fp32/fp64 retained. Include runtime reciprocal/multiply rounding, zeros, infinities, NaNs and subnormal behavior.'),
 'U8TO32':(NS+'Numerics.unsignedTo32','arith_uitofp_u8_f32','Unsigned8 to fp32 conversion on the exact loaded byte.'),
 'I8TO32':(NS+'Numerics.signed8To32','arith_sitofp_i8_f32','Signed8 to fp32 conversion on the exact loaded word.'),
 'I16TO32':(NS+'Numerics.signed16To32','arith_sitofp_i16_f32','Signed16 to fp32 conversion on the exact loaded word.'),
 'I32TO32':(NS+'Numerics.signed32To32','arith_sitofp_i32_f32','Signed32 to fp32 native rounding from exact source word.'),
 'I64TO32':(NS+'Numerics.signed64To32','arith_sitofp_i64_f32','Signed64 to fp32 native rounding from exact source word.'),
 'BOOL32':(NS+'promoted','bool_byte_nonzero_then_uitofp_i1_f32','All256 raw bool storage bytes normalize by byte!=0, then yield exact fp32 zero/one; signature is *u1, not a raw i1 pointer.'),
 'MUL32':(NS+'Numerics.mul32','arith_mulf_f32','Native fp32 multiplication on exact input words, both parenthesized source sites separately bound; include NaNs/infinities/signed zero/rounding/denormals and permitted compiler policy.'),
 'MUL64':(NS+'Numerics.mul64','arith_mulf_f64','Native fp64 multiplication on exact input words, both parenthesized source sites separately bound; include floating specials and rounding policy.'),
 'LIT32':(NS+'scale127','literal_127_f32','Exact source127.0 materialization as fp32 word0x42fe0000.'),
 'LIT64':(NS+'scale127','literal_127_f64','Exact source127.0 materialization as fp64 word0x405fc00000000000.'),
 'LLRINT32':(NS+'llrint','__nv_llrintf','Concrete nearest-even integer result only when the fp32 operand decodes finite and rounded result lies in[-2^63,2^63). NaN, infinity and out-of-range conversions remain explicit undefined observations; no arbitrary return integer may discharge them.'),
 'LLRINT64':(NS+'llrint','__nv_llrint','Concrete nearest-even integer result only when the fp64 operand decodes finite and rounded result lies in[-2^63,2^63). Undefined operands are not value assumptions.'),
 'STORE8':(NS+'result','arith_trunci_i64_i8','For defined llrint64 only, source store truncates low8 bits of the signed64 result into int8 storage; no clamp or saturation.')}
def ports(d,c):
 if c['configuration']['outcome']!='returned_or_undefined_conversion':return []
 ar='64' if d=='float64' else '32'
 z=[('ABS','logical_abs'),('MAX','whole_tensor_maximum'),('RECIP','produced_reciprocal')]
 cast={'bool':'BOOL32','uint8':'U8TO32','int8':'I8TO32','int16':'I16TO32','int32':'I32TO32','int64':'I64TO32'}.get(d)
 if cast:z.append((cast,'loaded_x_promotion'))
 return z+[('MUL'+ar,'x_times_inverse'),('LIT'+ar,'literal_127'),('MUL'+ar,'literal_times_inner_product'),('LLRINT'+ar,'round64_defined_domain'),('STORE8','store_int8_defined_domain')]
def op_types(family,d,port):
 ar='fp64' if d=='float64' else 'fp32'
 if family=='ABS':return [dt(d)],dt(d)
 if family=='MAX':return ['tensor_'+dt(d)],dt(d)
 if family=='RECIP':return [dt(d),'wrapped_python_float1'],ar
 if 'TO32' in family or family=='BOOL32':return [dt(d)],'fp32'
 if family.startswith('MUL'):return [ar,ar],ar
 if family.startswith('LIT'):return ['python_literal127'],ar
 if family.startswith('LLRINT'):return [ar],'int64'
 return ['int64'],'int8'
rules={s:{'id':'QG-FINITE-'+s,'status':'pending_numeric_validation','queue_status':'awaiting_explicit_numeric_validation','affected_tasks':[TASK],'required_validation':v[2],'native_whole_kernel_validated':False,'validation_evidence':None,'scope_bindings':[]} for s,v in families.items()}
provenance={};requirements=[];links={}
for d,c in configs:
 refs=[]
 for suffix,port in ports(d,c):
  callback,operation,validation=families[suffix];ident=rules[suffix]['id']
  body={'task':TASK,'declarations':[ref],'callbacks':[callback],**copy.deepcopy(c)}
  bid=TASK+':finite:'+sha(canonical({'rule':ident,'binding':body,'port':port}))[:32]
  rules[suffix]['scope_bindings'].append({'id':bid,**body});link={'assumption_id':ident,'scope_binding_id':bid};links.setdefault(callback,[]).append(link)
  refs.append({'callback':callback,**link});nwidth='i64' if c['configuration']['shape_class']=='i64_n_boundary' else 'i32'
  native_label={'uint8':'u8','int8':'i8','int16':'i16','int32':'i32','int64':'i64','float32':'fp32','float64':'fp64'}.get(d,d)
  raw=NATIVE/f'{native_label}-{c["configuration"]["block_size"]}-{nwidth}.ttir';assert raw.is_file()
  input_types,out_type=op_types(suffix,d,port)
  provenance[ident+'::'+bid]={'kind':'scalar_or_tensor_primitive_conformance','status':'explicitly_deferred_unvalidated','validation_evidence':None,**link,
   'callback_declaration':callback,'correspondence_kind':'concrete_source_port' if suffix in ['BOOL32','LIT32','LIT64','LLRINT32','LLRINT64','STORE8'] else 'parametric_callback',
   'port':{'name':port,'operation':operation,'storage_dtype':dt(d),'input_types':input_types,'output_type':out_type},
   'scalar_domain':'only finite decoded operand and representable rounded signed64 result' if suffix.startswith('LLRINT') else 'only already-defined signed64 conversion result' if suffix=='STORE8' else 'all source storage words, including exceptional floating values and signed integer minimum',
   'source_statement':{'ABS':'absmax = x.abs().max().unsqueeze(0)','MAX':'absmax = x.abs().max().unsqueeze(0)','RECIP':'absmax_inv = 1.0 / absmax','STORE8':'tl.store(output_ptr + offsets, output, mask=mask)'}.get(suffix,'output = tl.extra.cuda.libdevice.llrint(127.0 * (x * absmax_inv))'),
   'original_source':{'path':PYTHON,'sha256':hash_file(REPO/PYTHON)},'public_statement':ref,'lean_source_sha256':MODULES,
   'raw_ttir':refpath(raw),'native_compile_receipt':refpath(NATIVE/'receipt.json'),
   'host_wiring_receipt':refpath(ROOT/'evidence/host-finite-probes/receipt.json'),'torch_cuda_sources':refpath(ROOT/'evidence/torch-source/receipt-escalated.json'),
   'configuration':c['configuration'],'native_execution':False,
   'operand_order':{'model':['literal127','inner_product'],'native_ttir':['inner_product','literal127'],'obligation':'Exact correspondence at this finite127 port, including exceptional inner values; no general floating commutativity law assumed.'} if port=='literal_times_inner_product' else {'binding':'exact_source_argument_order'},
   'invocation_binding':'Numerics closes over this original invocation metadata/reduction context; maximum is the actual whole tensor primitive, not a universal shape-independent scalar fold.',
   'structural_prerequisites_not_waived':['source-order admission and primitive errors','valid metadata and exact physical capacities/no-wrap/permissions','actual host-produced maximum/inverse wiring','all actual sequential trial grids','original masked linear loads and signed32 offsets','no output value or successful-launch oracle','no invented llrint value for undefined operands']}
 requirements.append({**copy.deepcopy(c),'required_callbacks':sorted({r['callback'] for r in refs}),'required_binding_refs':refs})
for r in rules.values():r['scope_bindings'].sort(key=lambda b:b['id'])
deferrals=[]
for r in rules.values():
 assert r['scope_bindings']
 for b in r['scope_bindings']:
  deferrals.append({'kind':'numerical_rule_conformance','assumption_id':r['id'],'rule_sha256':sha(canonical(r)),'scope_binding_id':b['id'],'dtype_profile':b['dtype_profile'],'configuration':b['configuration'],'required_validation':r['required_validation']})
union=sorted(links)
declaration={**ref,'kind':'specification','model_kind':'backend_value_contract','role':'parametric_value_contract','scope_relation':'full_declared_stage','source_variant':'original_source_translation','covers_stages':stage_ids,
 'dtype_profile':'All22 canonical dense Torch dtype families, CPU/CUDA first rejections and eight CUDA typed execution routes; both original block configs; all finite source metadata0<=numel<2^63 with explicit i32-overflow/view hazards and undefined conversion outcomes.',
 'dtype_scope_relation':'complete_enumerated_wrapper_dtype_domain','unclosed_numeric_callbacks':union,'extra_numeric_law_ids':[],'native_binding_status':'not_established',
 'review_basis':'PRODUCER PROPOSAL: independent root substantive review required. The master statement derives complete original source outcomes, physical byte store traces through every trial, producer/consumer values, exact return metadata and frames.','original_scope':scope}
scope_gate.declaration_scope(declaration,inventory)
coverage={dim:[] for dim in ['stages','dtypes','branches']}
for dim,items in inventory.items():
 for item in items:
  if dim=='stages' and item['may_be_test_harness']:
   coverage[dim].append({'id':item['id'],'status':'test_harness','explanation':'Non-JIT test_quantize_global has no production caller and invokes the production wrapper. Import-time test invocation is harness initialization; no RNG implementation correctness is claimed.'})
  else:coverage[dim].append({'id':item['id'],'status':'covered','cases':copy.deepcopy(cases),'proofs':[ref],'explanation':'Complete finite source-target API outcome contract retains all22 dtype families, source-ordered failures, empty/rankzero cases, physical dense strided views, both configs and all trial grids. Concrete partial llrint has explicit undefined outcomes. Native numerical conformance is exactly registered for execution cases; no discarded numerical equality is asserted for earlier rejection cases.'})
row.update(status='pending',scope_schema=2,coverage=coverage,structural_gaps=[],review=None,deferred_numerical_assumptions=deferrals,callback_assumptions=links,proposed_status_after_independent_root_review='verified_with_deferred_numerics')
row['input_sha256']['lean'].update(MODULES)
row['acceptance_boundary_review']={'status':'producer_proposal_awaiting_independent_root_review','policy':'bench/original_program_acceptance.md','policy_sha256':hash_file(REPO/'bench/original_program_acceptance.md'),
 'finite_domain':'Ordinary dense strided CPU/CUDA Torch2.13.0+cu130 tensors, default floating dtype float32; current-launch CUDA sm90, Triton3.7.1. Finite native valid metadata numel<2^63; all22 canonical dtype families retained.',
 'trusted_boundary':'Exact metadata, fresh allocator payload/ownership/device/permission/lifetime and typed native primitive mechanics; source wiring, execution, byte-store refinement, finite indexes and result/frame are proved.',
 'source_hazards':'numel>2^31 reaches signed i32 offset wrap; source-linear view span overrun is explicit; llrint NaN/Inf/out-of-range produces no logical value. These are observations, not added native checks or native trap/rollback claims.',
 'native_correctness':False,'queue_integration':'Exact numerical authority delta and reviewed additive dtype labels must be canonically integrated; candidate JSON alone does not pass canonical acceptance.'}
authority={'schema':1,'kind':'quantize_global_finite_exact_numeric_authority_delta','task':TASK,'rules':list(rules.values()),'provenance':provenance,
 'requirements':[{'task':TASK,'declaration':ref,'source_ref':{'path':PUBLIC,'sha256':MODULES[PUBLIC]},'supported_cases_sha256':sha(canonical(sorted(cases,key=canonical))),'callback_union':union,'cases':requirements,
 'support_boundary':'All explicit finite source outcome cases; numeric ports only for the48 returned-or-undefined execution cases. Earlier rejection/failure/hazard claims have no returned numerical equality. Undefined llrint is not admitted by any numerical rule.'}],
 'summary':{'scalar_families':len(rules),'scope_bindings':len(provenance),'declaration_cases':len(cases),'numeric_execution_cases':sum(bool(c['required_callbacks']) for c in requirements),'registered_declarations':1,'native_executions':0,'original_program_completion_increment':0}}
assert authority['summary']['numeric_execution_cases']==48
scope_gate.validate_deferrals(TASK,deferrals,{'rules':list(rules.values())})
expected={(r['assumption_id'],r['scope_binding_id']) for c in requirements for r in c['required_binding_refs']}
assert expected=={(d['assumption_id'],d['scope_binding_id']) for d in deferrals}
assert union==sorted({c for r in requirements for c in r['required_callbacks']})
save('source-check.json',{'source':PYTHON,'source_sha256':hash_file(REPO/PYTHON),'production_ast_unchanged':True,'native_compile_attempts':52,'native_compile_successes':32,'expected_kernel_rejections':20,'gpu_executions':0,
 'dtype_families':names,'successful_cuda_routes':sorted(admitted),'bool_signature':'*u1','n_integer_specializations':['i32','i64'],'host_probe_receipt':refpath(ROOT/'evidence/host-finite-probes/receipt.json'),'autotuner_probe_receipt':refpath(ROOT/'evidence/autotune-probes/receipt.json'),
 'native_probe_limitation':'Exact unchanged AST compiled offline to TTIR/TTGIR/LLIR/PTX; no GPU execution. CPU primitive probes are not CUDA evidence; CUDA dispatcher review uses pinned installed-commit sources.'})
save('declaration-proposal.json',declaration);save('scope-row-proposal.json',row);save('scope-cases.json',cases);save('numerical-authority-delta.json',authority)
save('proposal-validation.json',{'status':'passed_candidate_validation_with_explicit_additive_dtype_enum','checks':['current public statement parser','all validate_case checks with only proposed dtype additions','unchanged schema2 declaration_scope','unchanged exact validate_deferrals','case/port/callback/deferral bijection','all native artifact hashes and unchanged production AST'],
 'counts':authority['summary'],'added_dtype_labels':sorted(added),'independent_root_review':'pending','canonical_schema_queue_check':'pending enum/authority/catalog integration; not claimed passed','original_program_completion_increment':0})
print(json.dumps(authority['summary']),flush=True)
