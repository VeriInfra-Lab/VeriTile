"""Exact finite transpose proposals; independent root review is mandatory."""
import ast,hashlib,json,sys
sys.dont_write_bytecode = True
from pathlib import Path
from copy import deepcopy
ROOT=Path(__file__).resolve().parent
REPO=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sys.path.insert(0,str(REPO/'bench'))
from adapter_common import canonical,declarations,sha
from check_original_program_scope import declaration_scope,validate_case,validate_deferrals
from authority_adapter import native_receipt_inputs,project_dependency_inputs
import check_original_program_scope as scope_validator
ADDITIVE_DTYPES={'complex32','complex64','complex128','fp8e8m0fnu'}
scope_validator.DTYPES |= ADDITIVE_DTYPES
TASK='quant_transpose_kernel';BASE='bench/tritonbench_g/'+TASK+'/'
PYTHON=BASE+TASK+'.py';PUBLIC=BASE+'FiniteAcceptance.lean';NAME='quant_transpose_finite_acceptance';NS='QuantTransposeFinite.'
load=lambda p:json.loads(p.read_text())
hash_file=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
MODULES=load(ROOT/'evidence/module-layout.json')['modules']
def save(n,d):(ROOT/'evidence'/n).write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
source=(REPO/PYTHON).read_text();tree=ast.parse(source)
functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
source_statements={n:[ast.unparse(v) for v in functions[n].body] for n in ['_quantize_global_transpose','quantize_global_transpose']}
native_path=next((ROOT/'inherited-v1/native-attempts').glob('*/receipt.json'));native=load(native_path)
assert native['source_sha256']==hash_file(REPO/PYTHON) and native['production_ast_unchanged']
assert len(native['records'])==28 and sum(r['status']=='compiled' for r in native['records'])==18
original_source={'path':PYTHON,'sha256':hash_file(REPO/PYTHON)}
native_evidence=[]
for role,path in [('original_dtype_config_matrix',native_path),('independent_mixed_width_metadata',ROOT/'peer-review/native-review-receipt.json')]:
 reference={'path':str(path.relative_to(ROOT)),'sha256':hash_file(path)}
 native_evidence.append({'role':role,'receipt':reference,'retained_inputs':native_receipt_inputs(ROOT,reference,original_source['sha256'])})
peer_native=load(ROOT/'peer-review/native-review-receipt.json')
assert len(peer_native['records'])==18 and all(r['status']=='compiled' for r in peer_native['records'])
native_references=[e['receipt'] for e in native_evidence]
snapshot_paths=[ROOT/'evidence/dependencies-before.json']+sorted((ROOT/'evidence').glob('dependencies-added-before-*.json'))
dependency_snapshots=[{'path':str(p.relative_to(ROOT)),'sha256':hash_file(p)} for p in snapshot_paths]
for path,expected in project_dependency_inputs(ROOT,dependency_snapshots).items():assert hash_file(REPO/path)==expected,path
assert 'pid_m = group_id * GROUP_M + pid % group_size' in source_statements['_quantize_global_transpose']
assert source_statements['quantize_global_transpose'][:4]==['absmax = input.abs().max().unsqueeze(0)','absmax_inv = 1.0 / absmax','M, N = input.shape',"out = torch.empty(N, M, device='cuda', dtype=torch.int8)"]
ledger=load(REPO/'bench/original_program_scope.json');row=deepcopy(next(r for r in ledger['tasks'] if r['task']==TASK));inventory=row['source_inventory']
ref={'file':PUBLIC,'declaration':NAME,'statement_sha256':declarations(ROOT/PUBLIC)[NAME]['statement_sha256']}
profiles=['bool','uint8','int8','int16','int32','int64','uint16','uint32','uint64','fp16','bf16','fp32','fp64','complex32','complex64','complex128','fp8e4m3fn','fp8e5m2','fp8e4m3fnuz','fp8e5m2fnuz','fp8e8m0fnu','packed_fp4e2m1']
admitted={'bool','uint8','int8','int16','int32','int64','fp32','fp64'}
fp8=set(profiles[16:20]);complexes=set(profiles[13:16]);absbad={'uint16','uint32','uint64','fp8e8m0fnu','packed_fp4e2m1'}
def absdtype(d):return {'complex32':'fp16','complex64':'fp32','complex128':'fp64'}.get(d,d)
def compute(d):return 'fp32' if d in {'bool','uint8','int8','int16','int32','int64'} else absdtype(d)
def sequence(device,d,shape):
 stages=['abs']
 if d in absbad or (device=='cuda' and d in fp8) or (device=='cpu' and d=='bool'):return stages,'unsupported_abs'
 stages+=['maximum']
 if shape=='empty':return stages,'empty_maximum'
 if d in fp8:return stages,'unsupported_maximum'
 stages+=['unsqueeze','reciprocal']
 if shape=='wrong_rank':return stages,'shape_unpack'
 stages+=['outputAllocation']
 if shape=='bad_stride':return stages,'input_stride_assertion'
 stages+=['signature']
 if d in complexes:return stages,'unsupported_signature'
 stages+=['compilation']
 if d in {'fp16','bf16'}:return stages,'unsupported_llrint_overload'
 stages+=['driverContext','gridParse']
 if shape=='oversized_grid':return stages,'grid_parse_limit'
 stages+=['launchEnterHook','pointerPacking']
 if device=='cpu':return stages,'cpu_pointer'
 stages+=['launch','launchExitHook']
 assert d in admitted
 return stages,'returned_or_undefined_conversion'
configs=[]
for device in ['cpu','cuda']:
 for d in profiles:
  for shape in ['empty','wrong_rank','bad_stride','valid_matrix','oversized_grid']:
   stages,terminal=sequence(device,d,shape)
   for outcome,site in [(terminal,None)]+[('primitive_error',s) for s in stages if s!='gridParse']:
    for selected in ([0,1] if outcome=='returned_or_undefined_conversion' else ['not_reached_or_either_duplicate']):
     c={'dtype_profile':{'inputs':{'input':d},'outputs':{'out':'int8','absmax':absdtype(d)},'compute':{'inverse':compute(d),'value_arithmetic':compute(d),'llrint':'int64','pid':'int32','arithmetic32':'int32','arithmetic64':'int64','pointer':'uint64'}},
      'configuration':{'entrypoint':'quantize_global_transpose','source_variant':'original','device':device,'shape_class':shape,'outcome':outcome,
       'first_primitive_error_stage':site or 'none','native_admission':'dimension_bearing_nativeAdmission','grid_class':'gridM_times_gridN_at_least_2pow31' if shape=='oversized_grid' else 'gridM_times_gridN_below_2pow31_when_rank2','driver_order':['driverContext','signed_grid_ParseTuple','launchEnterHook','pointerPacking','launch','launchExitHook'],'selected_config_index':selected,
       'source_stages_reached':stages[:stages.index(site)+1] if site else stages,
       'numeric_claim':'produced_and_source_scalar_words_with_defined_or_undefined_conversion_partition' if outcome=='returned_or_undefined_conversion' else 'value_insensitive_first_error_only; no_numeric_intermediate_or_postfailure_memory_claim',
       'native_observation':'none_or_returned_conversion_partition',
       'autotune_entries':[{'index':i,'BLOCK_M':128,'BLOCK_N':128,'GROUP_M':8,'num_warps':4,'num_stages':3} for i in [0,1]],
       'autotune_schedule':'arbitrary_finite_successful_warmups_and_trials_then_final; swallowed_resource_failures_do_not_launch',
       'target':{'backend':'cuda','arch':90,'warp_size':32},'torch_version':native['torch_version'],'torch_commit':native['torch_commit'],'triton_version':native['triton_version'],'torch_default_float':'fp32',
       'layout':'dense_strided_logical_view_with_actual_nonnegative_strides_offsets_extent; one_input_stride_equals1_after_assertion',
       'output':'fresh_cuda_contiguous_[N,M]_strides_[M,1]_int8; return_original_absmax_[1]',
       'metadata_specialization':'Each actual nonnegative M/N/stride argument below 2^63 specializes independently: value1 is constexpr; other values below2^31 are signed_i32; remaining values are signed_i64. Output strides are coupled to [N,M]: stride_bm=1, stride_bn=M.',
       'integer_expression_semantics':'pid and lane ranges are i32; every expression node promotes constants to the other operand width, i32+i32 stays i32, otherwise i64. Each i32/i64 arithmetic node wraps before later promotion. Constant arithmetic is unbounded; all-constant division/modulo use Python floor semantics, while runtime signed division truncates toward zero and remainder follows it. Division by zero or signed_min/-1 is an integerDivision observation.',
       'address_domain':'Actual signed selected-width offsets are extended to64, scaled by element bytes and added modulo2^64 to actual pointers. Masked-off input/output pointers impose no access bound; only resulting active addresses are checked against regions/permissions. The unmasked inverse load is checked once per program. No padded-i32/no-wrap or ArithmeticRefines prerequisite; safe wrapped-but-different accesses remain covered.',
       'grid_domain':'Host cdiv grid uses exact nonnegative integers; signed-int grid-x parsing requires gridM*gridN<2^31, otherwise genuine gridParse/gridLimit first rejection before entry hook, CPU-pointer extraction, launch and exit hook. This is a target launch prerequisite, not an address-safety inference.',
       'returned_semantics':'NativeResult specifies independent actual-address write plan, returned tensors, every final byte, and untouched-byte frame. Mathematical transpose additionally follows under optional ArithmeticRefines; it is not a premise of the main outcome theorem.',
       'numerical_domain':'All storage words; LlrintDefined partitions exact returned result versus semantic conversionUndefined observation. Defined iff decoded finite and nearest-even rounded integer is in [-2^63,2^63); no native trap or rollback claim and no domain waiver',
       'empty_rank_stride_partition':'empty numel=0 dominates; else rank!=2; else both strides!=1; else rank2 valid_matrix with supported grid, or oversized_grid; no metadata extent omitted'}}
     validate_case(c);configs.append((d,c))
base_case_count=len(configs)
numeric_configs=[(d,c) for d,c in configs if c['configuration']['outcome']=='returned_or_undefined_conversion']
observations=['integerDivision','inputAccess','inverseAccess','outputAccess','missingInputBytes','missingInverseBytes','outputOwnershipConflict']
for d,c in numeric_configs:
 for observation in observations:
  observed=deepcopy(c);cfg=observed['configuration']
  cfg.update(outcome='native_observation',native_observation=observation,
    numeric_claim='value_insensitive_interpreter_observation_only; no_numeric_intermediate_equality_or_native_trap_race_winner_rollback_or_postfailure_memory_claim',
    observation_scope='Classification of the model observation if reached; not an empirical rejection or a reachability assertion for every PhysicalBinding. Inverse access/missing-byte observations are excluded by successful host preparation and PhysicalBinding.',
    observation_order='input tile integer arithmetic and active loads; unmasked inverse load; source conversion then output-access/ownership checks; stores only on successful tile plan')
  validate_case(observed);configs.append((d,observed))
cases=[c for _,c in configs];assert len({canonical(c) for c in cases})==len(cases)
stages=[s['id'] for s in inventory['stages'] if not s['may_be_test_harness']];branches=[s['id'] for s in inventory['branches']];dtypes=[s['id'] for s in inventory['dtypes']]
scope={'source_inventory_sha256':sha(canonical(inventory)),'stages':stages,'branches':branches,'dtypes':dtypes,'cases':{i:deepcopy(cases) for i in stages+branches+dtypes}}
families={
 'ABS':('Numerics.abs','torch.abs','Exact CUDA Torch abs on the source storage dtype, including signed minimum overflow, signed zero, NaN, infinities and bool byte normalization.'),
 'MAX':('Numerics.maximum','torch.max_all','Exact Torch full nonempty logical-tensor maximum after abs; no arbitrary association or left-fold NaN rule is assumed.'),
 'RECIP':('Numerics.reciprocal','torch_rtruediv_1','Exact source 1.0 / absmax scalar chain, including Torch reciprocal, dtype promotion to default fp32 for integer/bool, and scalar multiplication semantics.'),
 'UCAST':('Numerics.unsignedTo32','arith.uitofp','Exact unsigned uint8 to fp32 conversion.'),
 'SCAST8':('Numerics.signed8To32','arith.sitofp','Exact signed int8 to fp32 conversion.'),
 'SCAST16':('Numerics.signed16To32','arith.sitofp','Exact signed int16 to fp32 conversion.'),
 'SCAST32':('Numerics.signed32To32','arith.sitofp','Exact signed int32 to fp32 conversion with native rounding.'),
 'SCAST64':('Numerics.signed64To32','arith.sitofp','Exact signed int64 to fp32 conversion with native rounding.'),
 'BOOL':('promoted','bool_nonzero_to_fp32','Concrete byte!=0 normalization followed by exact fp32 zero/one, as confirmed by generated LLVM icmp ne i8.'),
 'LIT32':('scale127','fp32_literal127','The native source floating literal 127.0 has exact fp32 word 0x42fe0000.'),
 'LIT64':('scale127','fp64_literal127','The native source floating literal 127.0 has exact fp64 word 0x405fc00000000000.'),
 'MUL32':('Numerics.mul32','arith.mulf_fp32','Native fp32 multiplication agrees at this exact source port, with source operand order, rounding, denormal, signed-zero and special-value policy.'),
 'MUL64':('Numerics.mul64','arith.mulf_fp64','Native fp64 multiplication agrees at this exact source port, with source operand order, rounding, denormal, signed-zero and special-value policy.'),
 'LLRINT32':('llrint','__nv_llrintf','Native fp32 llrint agrees with exact rational nearest-even and signed64 encoding only on LlrintDefined; undefined branch has no fabricated output.'),
 'LLRINT64':('llrint','__nv_llrint','Native fp64 llrint agrees with exact rational nearest-even and signed64 encoding only on LlrintDefined; undefined branch has no fabricated output.'),
 'NARROW8':('result','int64_to_int8_store','Concrete BitVec.ofInt8 low-byte signed narrowing agrees with native int64-to-int8 store conversion.')}
def ports(d):
 c=compute(d);suffix='64' if c=='fp64' else '32'
 out=[('ABS','abs',[d],d,'input.abs()'),('MAX','maximum',[d,'nonempty_list_'+d],d,'input.abs().max()'),('RECIP','inverse',[d],c,'1.0 / absmax')]
 cast={'bool':'BOOL','uint8':'UCAST','int8':'SCAST8','int16':'SCAST16','int32':'SCAST32','int64':'SCAST64'}.get(d)
 if cast:out.append((cast,'promote_input',[d],c,'a * absmax_inv'))
 out += [('LIT'+suffix,'literal127',['literal_127'],c,'127.0'),('MUL'+suffix,'inner_multiply',[c,c],c,'a * absmax_inv'),('MUL'+suffix,'outer_multiply',[c,c],c,'127.0 * (a * absmax_inv)'),('LLRINT'+suffix,'llrint',[c],'int64','tl.extra.cuda.libdevice.llrint(127.0 * (a * absmax_inv))'),('NARROW8','int8_store',['int64'],'int8','tl.store(B, output, mask=mask)')]
 return out
rules={k:{'id':'QT-FINITE-'+k,'status':'pending_numeric_validation','queue_status':'awaiting_explicit_numeric_validation','affected_tasks':[TASK],'required_validation':v[2],'native_whole_kernel_validated':False,'validation_evidence':None,'scope_bindings':[]} for k,v in families.items()}
provenance={};requirements=[];links={};deferrals=[]
for d,c in configs:
 refs=[]
 for suffix,port,inputs,output,expression in (ports(d) if c['configuration']['outcome']=='returned_or_undefined_conversion' else []):
  callback,operation,_=families[suffix];callback=NS+callback;ident=rules[suffix]['id']
  body={'task':TASK,'declarations':[ref],'callbacks':[callback],**deepcopy(c)}
  bid=TASK+':finite:'+sha(canonical({'rule':ident,'binding':body,'port':port}))[:32]
  binding={'id':bid,**body};rules[suffix]['scope_bindings'].append(binding)
  link={'assumption_id':ident,'scope_binding_id':bid};links.setdefault(callback,[]).append(link);refs.append({'callback':callback,**link})
  provenance[ident+'::'+bid]={'kind':'scalar_primitive_conformance','status':'explicitly_deferred_unvalidated','validation_evidence':None,**link,'callback_declaration':callback,
   'correspondence_kind':'parametric_callback' if callback.startswith(NS+'Numerics.') else 'concrete_scalar_port',
   'port':{'name':port,'operation':operation,'storage_dtype':d,'input_types':inputs,'output_type':output},
   'scalar_domain':'All source words at executed producer/multiply ports; host max nonempty logical list. llrint output/narrowing correspondence only on defined partition; undefined partition has no fabricated output.',
   'scalar_path_predicate':'CUDA admitted positive rank2 unit-stride-axis matrix; actual mask-selected finite-address lane; structural conversion-domain partition; native actual-address return plan and byte replay, including safe wrapped addresses distinct from mathematical coordinates',
   'native_execution':False,'source_statement':expression,'original_source':original_source,'public_statement':ref,
   'lean_source_sha256':{f:hash_file(ROOT/f) for f in MODULES},'native_receipts':native_references,
   'legacy_route_source':{'path':BASE+'IntegerCorrect.lean','sha256':hash_file(REPO/(BASE+'IntegerCorrect.lean'))},'configuration':c['configuration'],
   'structural_prerequisites_not_waived':['exact source host producer and return tuple','exact defined-or-undefined llrint64 partition; no undefined output','actual rotated grouped expression, exact constexpr1/i32/i64 promotion and per-node wrap','complete ordered supported grid and actual-address active accesses; grid/division/access/missing-byte/ownership observations remain structural','fresh disjoint allocations and independent actual-byte result; optional ArithmeticRefines bridge only','all existing successful typed routes retained']}
 requirements.append({**deepcopy(c),'required_callbacks':sorted({x['callback'] for x in refs}),'required_binding_refs':refs})
for r in rules.values():
 r['scope_bindings'].sort(key=lambda b:b['id'])
 for b in r['scope_bindings']:deferrals.append({'kind':'numerical_rule_conformance','assumption_id':r['id'],'rule_sha256':sha(canonical(r)),'scope_binding_id':b['id'],'dtype_profile':b['dtype_profile'],'configuration':b['configuration'],'required_validation':r['required_validation']})
union=sorted(links)
declaration={**ref,'kind':'specification','model_kind':'backend_value_contract','role':'parametric_value_contract','scope_relation':'full_declared_stage','source_variant':'original_source_translation','covers_stages':stages,
 'dtype_profile':'All22 canonical dense CPU/CUDA source profiles, Torch default fp32;8 CUDA admitted storage dtypes. All first abs/max/rank/stride/signature/llrint-overload/grid-parse/pointer errors retained with exact driver order. Actual constexpr1/i32/i64 metadata expression widths, uint64 wrapped pointers, two duplicate128x128/group8 configs.',
 'dtype_scope_relation':'complete_enumerated_wrapper_dtype_domain','unclosed_numeric_callbacks':union,'extra_numeric_law_ids':[],'native_binding_status':'not_established',
 'review_basis':'PRODUCER REVISION PROPOSAL, independent root review pending. Full host producers and assertions; actual per-expression constexpr1/i32/i64 grouped arithmetic and uint64 wrapped pointers; masks then actual input/output access checks plus unmasked inverse load; complete ordered programs and finite autotune trials. Main result has no Safe or ArithmeticRefines premise and partitions genuine signed-grid parse rejection before pointers, interpreter observations, and independent actual-address returned bytes. Optional ArithmeticRefines recovers logical transpose and legacy fp32 relation. Observations claim no native trap, race winner, rollback or postfailure state; inverse access/missing inverse bytes are general interpreter constructors but unreachable under successful PhysicalBinding and host preparation.',
 'original_scope':scope}
declaration_scope(declaration,inventory)
coverage={d:[] for d in ['stages','dtypes','branches']}
for dim,items in inventory.items():
 for item in items:
  if dim=='stages' and item['may_be_test_harness']:coverage[dim].append({'id':item['id'],'status':'test_harness','explanation':'Non-JIT test_quantize_global_transpose has no production caller and invokes the entrypoint four times with fixed example shapes. Import-time result_gold is test initialization; no RNG implementation proof required.'})
  else:coverage[dim].append({'id':item['id'],'status':'covered','cases':deepcopy(cases),'proofs':[ref],'explanation':'Complete finite constexpr1/i32/i64 metadata and wrapped uint64 actual-address CUDA-sm90 outcome contract, plus canonical dtype/first-error classes. Returned bytes follow the actual address plan; logical transpose is conditional on ArithmeticRefines. Structural observation cases classify a possible model constructor, do not assert each is reachable under PhysicalBinding, and claim no native trap, rollback, race winner or postfailure state. No padded-lane no-wrap exclusion.'})
row.update(status='pending',scope_schema=2,coverage=coverage,structural_gaps=[],review=None,deferred_numerical_assumptions=deferrals,callback_assumptions=links,proposed_status_after_independent_root_review='verified_with_deferred_numerics')
row['input_sha256']['lean'].update({f:hash_file(ROOT/f) for f in MODULES})
row['acceptance_boundary_review']={'status':'producer_proposal_awaiting_independent_root_review','policy':'bench/original_program_acceptance.md','policy_sha256':hash_file(REPO/'bench/original_program_acceptance.md'),'finite_domain':'Domain and PhysicalBinding bind source logical views, actual allocation bytes and ordinary metadata below2^63. Each actual metadata argument independently specializes constexpr1/i32/i64; every expression wraps before promotion and pointers wrap modulo2^64. Main theorem includes actual-address return, conversionUndefined and structural observations, with no Safe/ArithmeticRefines premise. Padded overflow and safe wrapped-but-different accesses are retained. Default float32, CUDA sm90, both source entries.','observation_boundary':'No native trap, race winner, rollback or postfailure state is asserted. Observation cases classify model results if reached, not empirical native rejection or universal reachability; inverseAccess/missingInverseBytes cannot arise under successful PhysicalBinding and host preparation.','native_correctness':False,'queue_integration':'Original full equality retained; independent reviewed receipt and root-review pins required; only repository sources and retained checkpoint evidence become queue authority.'}
authority={'schema':1,'kind':'quant_transpose_finite_exact_numeric_authority_delta','task':TASK,'rules':list(rules.values()),'provenance':provenance,
 'original_source':original_source,'native_evidence':native_evidence,'dependency_snapshots':dependency_snapshots,
 'requirements':[{'task':TASK,'declaration':ref,'source_ref':{'path':PUBLIC,'sha256':hash_file(ROOT/PUBLIC)},'supported_cases_sha256':sha(canonical(sorted(cases,key=canonical))),'callback_union':union,'cases':requirements,'support_boundary':'All finite source dtype/geometry/first-reachable-primitive-error cases including CPU/CUDA oversized-grid rejection and explicit division/access/missing-byte/ownership observations. Numeric ports remain on admitted CUDA returned-or-undefined outcomes at actual finite addresses. Value-insensitive rejections/observations assert no numeric intermediate equality, native trap, race winner, rollback or postfailure memory; no universal reachability claim. Undefined llrint is structural and cannot be waived.'}],
 'summary':{'scalar_families':len(rules),'scope_bindings':len(provenance),'declaration_cases':len(cases),'registered_declarations':1,'native_executions':0,'original_program_completion_increment':0}}
validate_deferrals(TASK,deferrals,{'rules':list(rules.values())})
assert {(r['assumption_id'],r['scope_binding_id']) for c in requirements for r in c['required_binding_refs']}=={(r['assumption_id'],r['scope_binding_id']) for r in deferrals}
for n,d in [('declaration-proposal.json',declaration),('scope-row-proposal.json',row),('numerical-authority-delta.json',authority),('scope-cases.json',cases)]:save(n,d)
save('source-check.json',{'source':PYTHON,'sha256':hash_file(REPO/PYTHON),'source_statements':source_statements,'native_receipts':native_references,'native_successes':18,'native_rejections':10,'independent_metadata_native_successes':len(peer_native['records']),'gpu_executed':False,'default_dtype':'torch.float32','partial_group_rotation_preserved':True,'per_expression_mixed_width_wrap':True,'original_config_entries':2,'test_harness_review':'Non-JIT; no production caller; four source wrapper examples.'})
save('dtype-enum-extension.json',{'add':sorted(ADDITIVE_DTYPES),'rationale':'Exact canonical source dtypes with first-error proof; no validator logic changed; existing packed_fp4e2m1 reused.'})
save('proposal-validation.json',{'status':'passed','checks':['current public statement hash','unchanged schema2 declaration_scope','all validate_case checks with explicit additive four-name dtype enum proposal','unchanged numerical validate_deferrals','exact per-case required-port and deferral bijection','both native receipts and all copied native artifacts bind unchanged production source','all before-first-use dependency snapshots and repository sources'],'case_partition':{'base_cases':base_case_count,'numeric_cases':len(numeric_configs),'structural_observation_kinds':observations,'added_observation_cases':len(numeric_configs)*len(observations)},'counts':authority['summary'],'root_review':'pending','global_queue':'pending reviewed authority integration','acceptance_increment':0})
print(json.dumps(authority['summary']))
