from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import json,sys,shutil
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');T=Path('/tmp/veritile-standardization-20261015')
P=T/'vector-finite-acceptance-revision2-wip';P1=T/'vector-finite-acceptance-revision1-wip';V=T/'vector-native-revision2-root-validation'
sys.path.insert(0,str(R/'bench'))
from adapter_common import hash_file,load,write_json,canonical,sha,declarations
from check_original_program_scope import declaration_scope,validate_case,validate_deferrals
assert not (V/'review.json').exists()
receipt=load(P/'evidence/receipt.json');receipt_sha=hash_file(P/'evidence/receipt.json')
assert receipt_sha=='8679242cdbdbe9d8972fe59c011fc89f51ba723de0b233bbe8af07ec32312cc4'
old=load(P1/'evidence/receipt.json');assert hash_file(P1/'evidence/receipt.json')=='f90d5fcdb2eaa946cf05fb884f0e0964e35ce4d1e207a96f46142b164780131f'
for package,r in [(P,receipt),(P1,old)]:
 for path,h in r['artifacts'].items():assert hash_file(package/path)==h,path
 for path,h in r['modules'].items():assert hash_file(package/'after'/path)==h,path
assert set(receipt['modules'])==set(old['modules'])
assert [p for p,h in receipt['modules'].items() if h!=old['modules'][p]]==['bench/tritonbench_g/vector_addition/NativeCorrect.lean']
build=load(V/'independent-build-receipt.json')
assert build['all_passed'] and build['dependencies_unchanged'] and build['producer_receipt_sha256']==receipt_sha
assert (len(build['records']),build['clean_axiom_checks'],build['noncircular_checks'])==(7,38,6)
dep=load(V/'dependencies-before-root.json')
assert (len(dep['project']),len(dep['external_objects']))==(200,37)
for path,entry in dep['project'].items():
 assert hash_file(R/path)==entry['sha256'],path
 assert hash_file(Path(entry['object']))==entry['object_sha256'],path
for entry in dep['external_objects'].values():assert hash_file(Path(entry['object']))==entry['sha256']
for path,h in dep['linked_objects'].items():assert hash_file(Path(path))==h
for record in build['records']:
 assert record['exit_code']==0 and record['source_sha256']==receipt['modules'][record['module']]
 assert hash_file(V/record['log'])==record['log_sha256']

semantic=load(V/'semantics-replay/receipt.json')
assert semantic['compiled_C_vs_integer_reference_pointer_checks']==1425408
assert semantic['raw_source_ir_checks']==189 and semantic['original_wrapper_AST_cases']==14
assert semantic['all22_stage_counts']=={'binder':195,'signature':64,'addition':36,'backend':6,'compiled':183}
special=load(V/'specialization-probe/receipt.json')
assert special['cases']==18 and special['all_expected_backend_rejections']
for parent,record in [(V/'semantics-replay',semantic),(V/'specialization-probe',special)]:
 assert record['source_sha256']==hash_file(R/'bench/tritonbench_g/vector_addition/vector_addition.py')
 assert record['gpu_launches']==0 and record['large_allocations']==0
 for path,h in record['artifacts'].items():assert hash_file(parent/path)==h,path
authority=load(V/'root-authority-check.json')
assert authority['status']=='passed' and authority['synthetic_review_fixture_only']
assert authority['case_links_checked']==4498 and authority['omitted_port_mutations_rejected']==296
assert len(authority['adapter_mutations_rejected'])==6 and authority['old_authority_unchanged']
assert authority['unchanged_whole_queue_equality_test']=={'status':'passed','exact_bindings':1642,'declaration_cases':10760,'native_executions':0,'structural_acceptance':False}

row=load(P/'evidence/scope-row-proposal.json');declaration=load(P/'evidence/declaration-proposal.json')
current=next(t for t in load(R/'bench/original_program_scope.json')['tasks'] if t['task']=='vector_addition')
assert row['source_inventory']==current['source_inventory']
declaration_scope(declaration,current['source_inventory'])
actual=declarations(P/'after'/declaration['file'])[declaration['declaration']]
assert actual['statement_sha256']==declaration['statement_sha256']=='aa7377b0ec370120c45a4fb542cdf1e8acc789e9e85a6044780fa6017fb7e35a'
cases=load(P/'evidence/scope-cases.json');assert len(cases)==4498
expected_order=['context','signature_and_backend_admission','compilation','signed_grid_tuple_parse','pointer_packing','launch']
for case in cases:
 validate_case(case);assert case['configuration']['setup_order']==expected_order
assert len({canonical(c) for c in cases})==4498
delta=load(P/'evidence/numerical-authority-delta.json')
assert len(delta['rules'])==37 and len(delta['provenance'])==296 and len(delta['requirements'])==1
validate_deferrals('vector_addition',row['deferred_numerical_assumptions'],{'rules':delta['rules']})
requirements=delta['requirements'][0]
assert len(requirements['cases'])==4498
assert Counter(c['configuration']['outcome'] for c in cases)['two31_no_access_return']==183
for case in requirements['cases']:
 if case['configuration']['outcome']!='returned_or_undefined_conversion':
  assert case['required_callbacks']==[] and case['required_binding_refs']==[]
for dimension,items in row['coverage'].items():
 assert [x['id'] for x in items]==[x['id'] for x in current['source_inventory'][dimension]]
 for item in items:
  if item['status']=='covered':assert item['cases']==cases and item['proofs']==[{k:declaration[k] for k in ['file','declaration','statement_sha256']}]
  else:assert item['status']=='test_harness'
setup=load(P/'evidence/setup-order/receipt.json')
assert setup['actual_parser_tests']==7 and setup['counterexample']['pointer_stage_reached'] is False
assert setup['counterexample']['host_n']==2**41 and setup['counterexample']['grid_x']==2**31
for name in ['build_vector_native_revision2_root.py','probe_vector_native_specializations_root.py','finalize_vector_native_revision2_root_review.py']:
 shutil.copyfile(T/name,V/name)
for name in ['vector-native-revision2-root-build.log','vector-native-specializations-root.log','vector-native-revision2-root-authority.log','vector-native-revision2-root-semantics.log']:
 shutil.copyfile(T/name,V/name)
review={
 'schema':1,'reviewer':'root','status':'substantively_reviewed_for_finite_scope_integration',
 'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'task':'vector_addition',
 'producer_receipt_sha256':receipt_sha,'producer_receipt':'evidence/receipt.json',
 'superseded_producer_receipt_sha256':hash_file(P1/'evidence/receipt.json'),
 'superseded_finding':'Revision1 allowed pointerPacking to precede oversized signed grid parsing. Revision2 repairs source ordering and adds a direct regression theorem plus exact installed-parser evidence.',
 'public_statement':{k:declaration[k] for k in ['file','declaration','statement_sha256']},
 'reviewed_candidate_source_sha256':receipt['modules'],
 'reviewed_existing_repo_source_sha256':{**{p:e['sha256'] for p,e in dep['project'].items()},'bench/tritonbench_g/vector_addition/vector_addition.py':hash_file(R/'bench/tritonbench_g/vector_addition/vector_addition.py')},
 'semantic_review':{
  'source_and_dispatch':'Original add allocates empty_like(x), checks all three CUDA flags, queries output numel and launches block1024. The 22x22 canonical binder/target/addition/backend partition retains 189 typed frontend routes and identifies the six genuine backend failures separately. Root checked the original Python, scalar execution relations and all seven new modules.',
  'integer_domain':'Host extents remain nonnegative signed64. Every bound is the actual annotated i32 word; pid multiply and lane add wrap32, comparisons are signed, and signed offsets scale/wrap into uint64 addresses. No extent-below-2^31 or no-wrap hypothesis restricts the master. The explicit 2^31 theorem returns the exact allocated tensor and unchanged allocated memory without input-access or numerical premises.',
  'source_execution':'Each program completes the active x load statement before y loads, uses previously checked exact typed scalar source functions, and then performs ordinary typed stores. Initial-memory plans are justified by fresh output disjointness, rather than a final-value callback. Undefined active conversions and invalid accesses are explicitly observed, with no fabricated numerical fallback.',
  'memory_and_metadata':'Torch raw storage/data pointers and extracted ABI pointers are distinct. Affine mapped allocation metadata binds source addresses to the launch space; permissions are indexed by the launch device, independent of storage owner. Empty null pointers, view offsets, preserve-format output strides, arbitrary x/y overlap and unconstrained y extent are represented. Pure packing and load/store semantics are explicit trusted primitives; no arbitrary narrow-storage carrier round-trip law is assumed.',
  'result_and_frames':'The returned specification binds actual store replay, an independent reverse-last-write formula for every byte, untouched bytes, and the output-allocation exterior frame. The allocator supplies metadata/freshness and its initial exterior frame. The prior bounded logical result is retained as a separate conditional refinement.',
  'failure_order':'Revision2 tests grid-x signed representability after compilation and before pointer packing. The direct theorem does not assume pointer-packing success. Ordinary runtime setup remains a fallible primitive boundary; no result or universal launch-success oracle is used.',
  'numerical_scope':'The 4498 source cases bind 296 exact scalar ports in 37 additive families. Value-insensitive rejected/observation/2^31-no-access claims use no numerical ports. Every required-port omission fails. Earlier bounded declarations and numerical rules remain preserved; future native scalar conformance is explicitly deferred.',
  'acceptance_boundary':'This review approves integration of the finite source model under the documented trusted primitive boundary. It does not establish native GPU correctness, official comparator success, or native behavior after an invalid/racing request.'},
 'verification':{'root_modules_rebuilt':7,'root_clean_axiom_checks':38,'root_noncircular_checks':6,
  'unchanged_project_sources_and_objects':200,'root_direct_external_objects_captured':37,
  'producer_artifacts_verified':1677,'previous_revision_artifacts_verified':1639,
  'root_pointer_checks':1425408,'root_wrapper_AST_cases':14,'root_CPU_metadata_examples':4,
  'source_dtype_pairs':484,'native_frontend_IR_checks':189,'root_additional_backend_specialization_cases':18,
  'scope_cases':4498,'exact_scalar_ports':296,'numerical_families':37,
  'root_missing_port_mutations_rejected':296,'root_adapter_mutations_rejected':6},
 'dependency_history_limit':dep['historical_limit'],
 'current_repository_integration':'pending; exact final combined catalog/queue/ledger/build checks still required',
 'gpu_launches':0,'native_numerical_conformance_validated':False,'official_comparator_passed':False,
 'original_program_completion_increment':0,
}
write_json(V/'review.json',review)
artifacts={}
for p in sorted(V.rglob('*')):
 if p.is_file() and not p.is_symlink() and not any(part in {'build-attempt001','cache','tmp','__pycache__'} for part in p.relative_to(V).parts):
  artifacts[p.relative_to(V).as_posix()]=hash_file(p)
write_json(V/'review-receipt.json',{'schema':1,'kind':'immutable_root_source_review','frozen':True,
 'producer_receipt_sha256':receipt_sha,'review_sha256':hash_file(V/'review.json'),
 'artifacts':artifacts,'excluded_generated_paths':['build-attempt001/**','**/cache/**','**/tmp/**','**/__pycache__/**'],
 'source_acceptance_increment':0})
print(json.dumps({'review_sha256':hash_file(V/'review.json'),'review_receipt_sha256':hash_file(V/'review-receipt.json'),'artifacts':len(artifacts),'status':review['status']},indent=2))
