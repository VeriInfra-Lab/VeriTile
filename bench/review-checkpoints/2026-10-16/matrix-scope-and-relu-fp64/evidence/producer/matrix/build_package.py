"""Construct reviewed-producer proposals; never impersonate the root review."""
from pathlib import Path
import copy,hashlib,json,sys
ROOT=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
OUT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'bench'))
from adapter_common import canonical,sha,hash_file,load,write_json,declarations
from check_original_program_scope import source_inventory,declaration_scope,case_set,full_execution_evidence,validate_case
from current_source_catalog import verify_source_view
CAT='bench/review-checkpoints/2026-10-16/quantization-and-mixed-softmax/current-source-catalog.json'
SUP='bench/review-checkpoints/2026-10-16/quantization-and-mixed-softmax/catalog-supplement.json'
BASE='bench/tritonbench_g/matrix_transpose/'
PYTHON=BASE+'matrix_transpose.py';NEW=BASE+'FullScope.lean'
catalog=load(ROOT/CAT);task=next(t for t in catalog['tasks'] if t['task']=='matrix_transpose')
ledger=load(ROOT/'bench/original_program_scope.json');old=next(t for t in ledger['tasks'] if t['task']=='matrix_transpose')
supplement=copy.deepcopy(next(t for t in load(ROOT/SUP)['tasks'] if t['task']=='matrix_transpose'))
source_hash=hash_file(OUT/'after'/NEW)
statement=declarations(OUT/'after'/NEW)['matrix_transpose_full_source_outcomes']
inventory=source_inventory(ROOT,{PYTHON:hash_file(ROOT/PYTHON)})
kernel_id,wrapper_id,harness_id=[s['id'] for s in inventory['stages']]
dtype_id=inventory['dtypes'][0]['id']
profile={'inputs':{'matrix':'fp16'},'outputs':{'out':'fp16'},'compute':{'arange':'int32','stride_products':'int32','transpose':'fp16'}}
def case(configuration):return {'dtype_profile':copy.deepcopy(profile),'configuration':configuration}
common={'entrypoint':'wrapper','grid':[1],'shape_api':'natural_dimensions_after_trusted_Torch_conversion','source_variant':'original','storage_allocation':'fresh_contiguous_cuda_fp16'}
success=[case({**common,'outcome':'returned','SIZE_M':2**a,'D_HEAD':2**b,'randn_error':False,'zeros_error':False,'launch_setup_error':False,'backend_error':False}) for a in range(21) for b in range(21-a)]
valid='1 <= dimension <= 1048576 and (dimension & (dimension-1)) == 0'
rejections=[]
for stage,dim,prior in [('sizeMRange','SIZE_M',[]),('dHeadRange','D_HEAD',['arangeCheck(SIZE_M) = none'])]:
 for reason,predicate in [
  ('indexWidth',f'{dim} >= 4294967296'),
  ('nonPositive',f'{dim} = 0'),
  ('nonPowerOfTwo',f'0 < {dim} < 4294967296 and ({dim} & ({dim}-1)) != 0'),
  ('tensorSize',f'1048576 < {dim} < 4294967296 and ({dim} & ({dim}-1)) = 0')]:
  rejections.append(case({**common,'outcome':'frontend_rejection','partition':stage+'.'+reason,'SIZE_M_domain':'Nat','D_HEAD_domain':'Nat','predicate_conjunction':prior+[predicate],'randn_error':False,'zeros_error':False,'launch_setup_error':False,'backend_error':'arbitrary_bool_ignored_after_earlier_failure'}))
rejections.append(case({**common,'outcome':'frontend_rejection','partition':'matrixPointerTensorSize','SIZE_M_domain':'Nat','D_HEAD_domain':'Nat','predicate_conjunction':['arangeCheck(SIZE_M) = none','arangeCheck(D_HEAD) = none','SIZE_M * D_HEAD > 1048576'],'randn_error':False,'zeros_error':False,'launch_setup_error':False,'backend_error':'arbitrary_bool_ignored_after_earlier_failure'}))
errors=[]
for index,name in enumerate(['randn','zeros','launch_setup','backend']):
 config={**common,'outcome':'primitive_error','first_failure':name,'SIZE_M_domain':'Nat','D_HEAD_domain':'Nat'}
 flags=['randn','zeros','launch_setup','backend']
 config.update({f+'_error':False if i<index else True if i==index else 'arbitrary_bool_ignored_after_earlier_failure' for i,f in enumerate(flags)})
 if name=='backend':config['shape_predicate']='SourceShape(SIZE_M,D_HEAD)'
 errors.append(case(config))
allcases=success+rejections+errors
kernelcases=success+rejections+errors[2:]
assert len(success)==231 and len(rejections)==9 and len(errors)==4
proof={'file':NEW,'declaration':'matrix_transpose_full_source_outcomes','statement_sha256':statement['statement_sha256']}
original_scope={'source_inventory_sha256':sha(canonical(inventory)),'stages':[kernel_id,wrapper_id],'dtypes':[dtype_id],'branches':[],
 'cases':{kernel_id:kernelcases,wrapper_id:allcases,dtype_id:allcases}}
declaration={**proof,'kind':'specification','model_kind':'bit_interpreter_contract','role':'parametric_value_contract','scope_relation':'full_declared_stage','source_variant':'original_source_translation','covers_stages':[kernel_id,wrapper_id],'execution_frontend':'shared_triton_dsl','dtype_scope_relation':'complete_enumerated_wrapper_dtype_domain','unclosed_numeric_callbacks':[],'extra_numeric_law_ids':[],'native_binding_status':'not_established','original_scope':original_scope,
 'review_basis':'Producer proposal awaiting independent root review. Existing actual fp16 six-statement interpreter is composed with exact source-ordered arange/broadcast admission and the wrapper allocation/launch/return wiring under explicit trusted primitive contracts. Every admitted shape is retained: 231 pairs m=2^a,d=2^b,a+b<=20. The admission theorem covers the entire natural-shape complement using nine first-failure partitions. Ordinary errors are propagated at randn, zeros, launch setup, or post-frontend backend/runtime; no arbitrary successful-run or final-value oracle is used. Returned metadata, all fp16 payloads, all output cells, input preservation and full memory frame are included. There is no floating arithmetic callback or numerical deferral.'}
assert full_execution_evidence(task,declaration)
assert declaration_scope(declaration,inventory)==original_scope
for cases in original_scope['cases'].values():case_set(cases)
row=copy.deepcopy(old)
row.update(scope_schema=2,status='pending',proposed_status_after_independent_root_review='verified',structural_gaps=[],review=None,deferred_numerical_assumptions=[],callback_assumptions={})
row['input_sha256']['lean'][NEW]=source_hash
row['source_inventory']=inventory
row['coverage']={'stages':[
 {'id':kernel_id,'status':'covered','explanation':'Exact original kernel source; all 231 frontend-admitted fp16 configurations, the universal first-rejection partition and launch errors. The returned route executes all six statements through Host.run, discharges output injectivity, proves all payloads/frame and bounds integer/byte offsets. Frontend/native failures are propagated, never hidden as successful transpose.','cases':kernelcases,'proofs':[proof]},
 {'id':wrapper_id,'status':'covered','explanation':'Original randn/zeros shape, fp16 dtype, CUDA request, stride arguments, grid=(1,), original size parameters and exact Out return identity are bound to trusted primitive contracts. The unconditional public theorem covers all primitive outcomes; there is no successful-execution premise. Supported shape API is natural dimensions; unsupported Python caller conversion and negative dimensions belong to Torch primitive errors, with observed boundaries retained separately.','cases':allcases,'proofs':[proof]},
 {'id':harness_id,'status':'test_harness','explanation':'Non-JIT test_triton_vs_torch is not called by production kernel/wrapper. It calls wrapper for 16x16 and 32x64 and clones returned tensors. Import-time random test execution is a harness under the finite acceptance boundary; both production configurations are among the 231 proved cases.','proofs':[]}],
 'dtypes':[{'id':dtype_id,'status':'covered','explanation':'The original production wrapper hardcodes torch.float16 on both allocations; kernel pointers therefore load/store fp16 and transpose has no cast or arithmetic. arange and annotated stride products are int32; admitted element offsets are <2^20 and both fp16 bytes <2^21. All 65536 sampled words are covered without RNG/arithmetic assumptions.','cases':allcases,'proofs':[proof]}], 'branches':[]}
row['acceptance_boundary_review']={'status':'producer_proposal_awaiting_independent_root_review','policy':'bench/original_program_acceptance.md','policy_sha256':hash_file(ROOT/'bench/original_program_acceptance.md'),'domain':'Original wrapper, canonical natural matrix dimensions, fixed fp16/CUDA allocations, one program; natural-shape rejection complement is preserved, arbitrary Python objects are not asserted supported.','known_rejections':'Nine complete natural-shape frontend partitions plus ordinary primitive error sites; negative/nonshape observed errors are preserved in wrapper-binding.json.','native_functional_correctness':'not_established','numerical_deferrals':0}
view={'id':'original_fp16_wrapper_admission_and_full_execution','variant':'original_source_translation','selected_definitions':{BASE+'Kernels.lean':['typedKernel','transposeSource%']},'entry_aliases':{NEW:['wrapper'],BASE+'HostCorrect.lean':['run'],BASE+'Proofs/FPBits.lean':['source']},'entry_alias_validation':'FullScope.wrapper preserves primitive call order, source admission and invokes Host.run; Host.run executes FPBits.source Bool.false with parameters h and launchState h, selecting the original typedKernel. Exact wrapper AST probe binds trusted allocation/metadata/return calls.','authority':'explicit_supplement_review','frontend':'shared_triton_dsl','ordinary_shared_dsl_admission':True,'memory_semantics':'single_program_total_typed_memory','translation_trust_boundary':'Reviewed original Python/Lean source wiring, explicit trusted Torch/Triton frontend/allocation/launch/load/store semantics. No universal compiler or GPU implementation theorem.','original_python':PYTHON,'candidate_python':None}
verify_source_view(ROOT,view,{p:hash_file(ROOT/p) for p in view['selected_definitions']})
supplement['source_sha256'].update({NEW:source_hash})
supplement['source_views'].append(view)
supplement['declarations'].append(declaration)
supplement['required_value_stages']=list(dict.fromkeys(supplement['required_value_stages']+[kernel_id,wrapper_id]))
supplement['review']=None
write_json(OUT/'catalog-supplement-task.proposed.json',supplement)
write_json(OUT/'scope-row.proposed.json',row)
write_json(OUT/'scope-cases.json',{'successes':success,'rejections':rejections,'primitive_errors':errors})
write_json(OUT/'metadata-check.json',{'status':'passed_proposal_structure_only','full_execution_evidence':True,'declaration_scope_exact':True,'original_source_view_valid':True,'exact_success_cases':231,'complete_rejection_partitions':9,'primitive_error_partitions':4,'kernel_cases':len(kernelcases),'wrapper_dtype_cases':len(allcases),'case_uniqueness_checked':True,'accepted_scope_gate':'not_run_pending_independent_root_review_and_catalog_integration','numerical_callbacks':[],'numerical_deferrals':[]})
write_json(OUT/'integration-inputs.json',{'current_catalog':{'path':CAT,'sha256':hash_file(ROOT/CAT),'canonical_sha256':sha(canonical(catalog))},'current_supplement':{'path':SUP,'sha256':hash_file(ROOT/SUP)},'current_ledger_sha256':hash_file(ROOT/'bench/original_program_scope.json'),'new_source':{'path':NEW,'sha256':source_hash},'public_statement':proof,'new_library_registration':'TritonBenchGVerified through existing matrix_transpose.+ glob; no lakefile edit','root_review_required':True,'repository_modified_by_producer':False})
print(json.dumps({'status':'passed_proposal_structure_only','success_cases':231,'rejection_partitions':9,'primitive_error_partitions':4,'statement_sha256':statement['statement_sha256']}))
