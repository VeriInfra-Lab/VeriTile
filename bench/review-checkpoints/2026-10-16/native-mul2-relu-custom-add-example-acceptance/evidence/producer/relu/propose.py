"""Producer-only full original-source ReLU authority proposal; acceptance stays zero."""
from pathlib import Path
import ast,hashlib,json,sys
from copy import deepcopy
sys.dont_write_bytecode=True
W=Path(__file__).resolve().parent
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sys.path.insert(0,str(R/'bench'))
from adapter_common import canonical,declarations,sha
from check_original_program_scope import validate_case,declaration_scope,validate_deferrals
TASK='relu_triton_kernel';BASE='bench/tritonbench_g/'+TASK+'/'
MODULES=[BASE+'Proofs/Actual'+n+'.lean' for n in ['Indices','Scalar','Source','Physical','Grid','Admission','Host']]+[BASE+'ActualCorrect.lean']
PUBLIC=MODULES[-1];NAME='relu_actual_source_outcomes';NS='VeriTile.Bench.TritonBenchG.ReluTritonKernel.'
PYTHON=BASE+TASK+'.py'
hash_file=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
def save(name,data):
 p=W/'evidence'/name;p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

def generate():
 source=(R/PYTHON).read_text();tree=ast.parse(source);fs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
 host=[ast.unparse(n) for n in fs['relu'].body]
 kernel=[ast.unparse(n) for n in fs['relu_kernel'].body]
 assert host==['out = torch.empty_like(x, dtype=torch.float32, device=x.device)','N = out.numel()','BLOCK_SIZE = 1024','num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE','grid = lambda meta: (triton.cdiv(N, BLOCK_SIZE),)','relu_kernel[grid](x, out, N, BLOCK_SIZE)','return out']
 assert kernel==['pid = tl.program_id(0)','block_start = pid * block_size','offsets = block_start + tl.arange(0, block_size)','mask = offsets < N','x = tl.load(x_ptr + offsets, mask=mask)','result = tl.where(x >= 0, x, 0.0)','if pid == 0:\n    tl.store(out_ptr + offsets, result, mask=mask)']
 native='evidence/native-attempts/20261015T221045164642Z/receipt.json';receipt=load(W/native)
 assert receipt['source_sha256']==hash_file(R/PYTHON) and receipt['source_ast_identity'] and not receipt['gpu_executed']
 for rel,h in receipt['artifacts'].items():assert hash_file((W/native).parent/rel)==h
 names=receipt['names'];assert len(names)==22
 admitted=names[:13]
 refs={'path':native,'sha256':hash_file(W/native)}
 modulehash={p:hash_file(W/p) for p in MODULES}
 public={'file':PUBLIC,'declaration':NAME,'statement_sha256':declarations(W/PUBLIC)[NAME]['statement_sha256']}
 row=deepcopy(next(r for r in load(R/'bench/original_program_scope.json')['tasks'] if r['task']==TASK))
 inventory=row['source_inventory']
 ranges=[('empty',0,1),('first_tile',1,1025),('multiple_tiles',1025,2**31),('unsigned_i32',2**31,2**32),('signed_i64_launchable',2**32,(2**31-1)*1024+1),('signed_i64_grid_overflow',(2**31-1)*1024+1,2**63),('unsigned_i64',2**63,2**64),('nonrepresentable',2**64,None)]
 dtype_name=lambda d:{'float16':'fp16','bfloat16':'bf16','float32':'fp32','float64':'fp64','float8_e4m3fn':'fp8e4m3fn','float8_e5m2':'fp8e5m2','float8_e4m3fnuz':'fp8e4m3fnuz','float8_e5m2fnuz':'fp8e5m2fnuz','float8_e8m0fnu':'fp8e8m0fnu','float4_e2m1fn_x2':'packed_fp4e2m1'}.get(d,d)
 cases=[];indexed=[]
 for d in names:
  for size,lo,hi in ranges:
   terminal=('missing_signature_map' if d in names[17:] else 'unsupported_signature' if d in names[15:17] else 'nonrepresentable_constexpr' if lo>=2**64 else 'ordered_comparison_rejected' if d in names[13:15] else 'grid_parse_rejected' if lo>(2**31-1)*1024 else 'returned_empty' if size=='empty' else 'returned_defined')
   outcomes=['host_or_native_preparation_failure',terminal]
   if terminal=='returned_defined':outcomes.append('source_access_outside_model')
   for outcome in outcomes:
    case={'dtype_profile':{'inputs':{'x':dtype_name(d)},'outputs':{'out':'fp32'},'compute':{'index':'int32','mask':'int32' if lo<2**31 else 'uint32' if lo<2**32 else 'int64' if lo<2**63 else 'uint64','output':'fp32'}},'configuration':{'entrypoint':'relu','source_variant':'original','block_size':1024,'N':{'lower_inclusive':lo,'upper_exclusive':hi if hi is not None else 'unbounded'},'size_class':size,'outcome':outcome,'grid':'ceil_div_N_1024','target':receipt['target'],'triton_version':receipt['triton'],'torch_version':receipt['torch'],'input_devices':'all_actual_dense_CPU_or_CUDA_metadata; permissions_relative_to_current_launch_device','layout':'all_actual_dense_views_including_rank0_empty_noncontiguous_offset_broadcast','pointer':'actual_extracted_ABI_pointer_with_bytewise_backing_translation','allocation':'actual_empty_like_float32_metadata_and_arbitrary_fresh_contents','native_failure_sites':['device','empty_like','numel','JIT_current_device','compile_and_module_load','launcher_context_and_grid_parse','pointer_extraction','enqueue'],'native_order':'compile; launcher_context/grid_parse; pointer_extraction; enqueue','source_access_observation':'all_programs_load_before_pid0_gate; unsafe_access_is_not_a_native_exception_or_final_memory','result':'pid0_active_stores_only; every_other_byte_retains_allocated_snapshot'}}
    validate_case(case);cases.append(case);indexed.append((d,case))
 assert len({canonical(c) for c in cases})==len(cases)
 stage=[s['id'] for s in inventory['stages'] if not s['may_be_test_harness']];branch=[s['id'] for s in inventory['branches']];dtype=[s['id'] for s in inventory['dtypes']]
 scope={'source_inventory_sha256':sha(canonical(inventory)),'stages':stage,'branches':branch,'dtypes':dtype,'cases':{k:deepcopy(cases) for k in stage+branch+dtype}}
 # Each port below is selected from the source/value route, never from a supplied binding.
 family={
 'SINT32':('VeriTile.Triton.FP.Structural.Algebra.fromInt','arith.sitofp','Exact selected signed finite integer to fp32 conversion, including rounding and overflow.'),
 'UINT32':('VeriTile.Triton.FP.Structural.Algebra.fromNat','arith.uitofp','Exact unsigned finite integer or normalized bool i1 to fp32 conversion.'),
 'CMP16':(NS+'Actual.Numerics.cmp16','arith.cmpf_oge_fp16','Exact ordered x>=+0 comparison for every fp16 word, including NaNs and signed zero.'),
 'CMP32':(NS+'Actual.Numerics.cmp32','arith.cmpf_oge_fp32','Exact ordered x>=+0 comparison for every fp32 word, including NaNs and signed zero.'),
 'EXT16':('VeriTile.Triton.FP.Structural.Algebra.cast','arith.extf_fp16_fp32','Exact fp16 to fp32 extension, including specials and denormal behavior.'),
 'EXTBF16':('VeriTile.Triton.FP.Structural.Algebra.cast','arith.extf_bf16_fp32','Exact bf16 to fp32 extension at comparison and store ports.'),
 'CMP64':(NS+'FP64Source.orderedGeZero64','arith.cmpf_oge_fp64','Concrete finite IEEE word comparison agrees with native ordered fp64 comparison.'),
 'NARROW64':(NS+'FP64Source.Numerics.narrow64to32','arith.truncf_fp64_fp32','Exact native fp64 to fp32 narrowing, including ties, overflow, underflow, specials and signed zero.')}
 def ports(d):
  if d.startswith('int'):return [('SINT32','selected_nonnegative','selection')]
  if d.startswith('uint') or d=='bool':return [('UINT32','all_unsigned_words' if d!='bool' else 'all_256_bool_storage_bytes_normalized_to_i1','selection')]
  if d=='float16':return [('CMP16','all_fp16_words','comparison'),('EXT16','all_selected_fp16_words','store')]
  if d=='bfloat16':return [('EXTBF16','all_bf16_words','comparison_cast'),('CMP32','all_extended_bf16_words','comparison'),('EXTBF16','all_selected_bf16_words','store')]
  if d=='float32':return [('CMP32','all_fp32_words','comparison')]
  if d=='float64':return [('CMP64','all_fp64_words','comparison'),('NARROW64','all_selected_fp64_words','store')]
  return []
 rules={suffix:{'id':'RELU-ACTUAL-'+suffix,'status':'pending_numeric_validation','queue_status':'awaiting_explicit_numeric_validation','affected_tasks':[TASK],'required_validation':data[2],'native_whole_kernel_validated':False,'validation_evidence':None,'scope_bindings':[]} for suffix,data in family.items()}
 provenance={};requirements=[];links={}
 for d,c in indexed:
  selected=ports(d) if c['configuration']['outcome']=='returned_defined' else []
  rs=[]
  for suffix,domain,port in selected:
   callback,operation,_=family[suffix];rule=rules[suffix]
   body={'task':TASK,'declarations':[public],'callbacks':[callback],**deepcopy(c)}
   bid=TASK+':actual:'+sha(canonical({'rule':rule['id'],'binding':body,'port':port}))[:32]
   rule['scope_bindings'].append({'id':bid,**body});link={'assumption_id':rule['id'],'scope_binding_id':bid}
   links.setdefault(callback,[]).append(link);rs.append({'callback':callback,**link})
   provenance[rule['id']+'::'+bid]={'kind':'scalar_primitive_conformance','status':'explicitly_deferred_unvalidated','validation_evidence':None,**link,'callback_declaration':callback,'correspondence_kind':'concrete_scalar_port' if suffix=='CMP64' else 'parametric_callback','port':{'name':port,'operation':operation,'storage_dtype':d},'scalar_domain':domain,'scalar_path_predicate':'active source lane; '+domain+'; store conversion only in pid0 gate','native_execution':False,'original_source':{'path':PYTHON,'sha256':hash_file(R/PYTHON)},'public_statement':public,'lean_source_sha256':modulehash,'native_receipts':[refs],'raw_ttir':{'path':str(Path(native).parent/('dtype-'+d+'.raw.ttir')),'sha256':hash_file((W/native).parent/('dtype-'+d+'.raw.ttir'))},'configuration':c['configuration'],'structural_prerequisites_not_waived':['actual finite mask/pointer calculations','all pre-gate input accesses','bytewise ABI-to-backing mapping','native current-device permissions','allocator freshness','source pid0-only stores; untouched output bytes retained'],'fixed_zero_encoding':'all-zero IEEE positive zero is concrete in source and reference, not an arbitrary Algebra.literal callback'}
  requirements.append({**deepcopy(c),'required_callbacks':sorted({r['callback'] for r in rs}),'required_binding_refs':rs})
 deferrals=[]
 for rule in rules.values():
  rule['scope_bindings'].sort(key=lambda b:b['id'])
  for binding in rule['scope_bindings']:deferrals.append({'kind':'numerical_rule_conformance','assumption_id':rule['id'],'rule_sha256':sha(canonical(rule)),'scope_binding_id':binding['id'],'dtype_profile':binding['dtype_profile'],'configuration':binding['configuration'],'required_validation':rule['required_validation']})
 union=sorted(links)
 declaration={**public,'kind':'specification','model_kind':'backend_value_contract','role':'parametric_value_contract','scope_relation':'full_declared_stage','source_variant':'original_source_translation','covers_stages':stage,'dtype_profile':'All22 canonical dtype/setup routes; all actual constexpr branches and dense CPU/CUDA metadata;13 compilable routes; source access observations preserve pre-gate reads.','dtype_scope_relation':'complete_enumerated_wrapper_dtype_domain','unclosed_numeric_callbacks':union,'extra_numeric_law_ids':[],'native_binding_status':'not_established','review_basis':'PRODUCER PROPOSAL pending independent root substantive review. Public theorem composes exact AST lowering, finite words, canonical backing reads/writes, full grid initial-snapshot theorem, host source order and fallible native boundaries.','original_scope':scope}
 declaration_scope(declaration,inventory)
 coverage={dim:[] for dim in ['stages','dtypes','branches']}
 for dim,items in inventory.items():
  for item in items:
   if dim=='stages' and item['may_be_test_harness']:coverage[dim].append({'id':item['id'],'status':'test_harness','explanation':'Non-JIT fixed examples call relu; existing HostWrapper.initializeModule retains all calls, dict insertion, first-error propagation and import-time result_gold assignment.'})
   else:coverage[dim].append({'id':item['id'],'status':'covered','cases':deepcopy(cases),'proofs':[public],'explanation':'All source dtype, constexpr and actual access outcomes. No source N or device guard added. Invalid accesses are observations without native trap/return/rollback claims. Safe executions derive complete backing bytes from independent scalar and reverse-last-write formulas.'})
 row.update(status='pending',scope_schema=2,coverage=coverage,structural_gaps=[],review=None,deferred_numerical_assumptions=deferrals,callback_assumptions=links,proposed_status_after_independent_root_review='verified_with_deferred_numerics')
 row['input_sha256']['lean'].update(modulehash)
 row['acceptance_boundary_review']={'status':'producer_proposal_awaiting_independent_root_review','policy':'bench/original_program_acceptance.md','policy_sha256':hash_file(R/'bench/original_program_acceptance.md'),'finite_domain':'All actual source cases; no arbitrary SafeDomain, CUDA owner equality, rank restriction or N≤2^31 assumption. Native grid bounds arise from ordinary launcher parsing.','source_bug':'Only pid0 stores; all other output bytes retain arbitrary empty_like contents.','native_correctness':False,'queue_integration':'Root must pin frozen producer receipt and independent review receipt, then integrate exact authority.'}
 delta={'schema':1,'kind':'relu_actual_exact_numeric_authority_delta','task':TASK,'original_source':{'path':PYTHON,'sha256':hash_file(R/PYTHON)},'dependency_snapshots':[{'path':'evidence/dependencies-before.json','sha256':hash_file(W/'evidence/dependencies-before.json')}],'native_evidence':[{'role':'full22_and_constexpr_boundaries','receipt':refs}],'rules':list(rules.values()),'provenance':provenance,'requirements':[{'task':TASK,'declaration':public,'source_ref':{'path':PUBLIC,'sha256':hash_file(W/PUBLIC)},'supported_cases_sha256':sha(canonical(sorted(cases,key=canonical))),'callback_union':union,'cases':requirements,'support_boundary':'Exact all-source observation cases; only nonempty defined returned bytes require numeric conformance. Access observations and setup/empty outcomes are value-independent.'}],'summary':{'scalar_families':len(rules),'scope_bindings':len(provenance),'declaration_cases':len(cases),'registered_declarations':1,'native_executions':0,'original_program_completion_increment':0}}
 validate_deferrals(TASK,deferrals,{'rules':list(rules.values())})
 assert {(r['assumption_id'],r['scope_binding_id']) for c in requirements for r in c['required_binding_refs']}=={(r['assumption_id'],r['scope_binding_id']) for r in deferrals}
 save('source-check.json',{'original_source':delta['original_source'],'host_statements':host,'kernel_statements':kernel,'AST_identity':True,'native_receipt':refs,'native_backend_compiled':True,'gpu_executed':False,'generalization_basis':'Pinned constexpr promotion rule plus exact finite source AST, not inference from sampled N.','numerical_ports':family})
 for name,data in [('declaration-proposal',declaration),('scope-row-proposal',row),('numerical-authority-delta',delta),('scope-cases',cases)]:save(name+'.json',data)
 save('proposal-validation.json',{'status':'passed','checks':['unchanged validate_case on all cases','unchanged schema2 declaration_scope','unchanged exact validate_deferrals','per-case callback and port/deferral bijection'],'counts':delta['summary'],'scope_acceptance_increment':0,'independent_review':'pending'})
 print(json.dumps(delta['summary']))
if __name__=='__main__':generate()
