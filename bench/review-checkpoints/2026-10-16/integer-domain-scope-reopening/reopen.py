from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import copy,json,shutil,subprocess,sys
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');T=Path('/tmp/veritile-standardization-20261015');P=T/'annotated-i32-scope-review'
B=R/'bench/review-checkpoints/2026-10-16/integer-domain-scope-reopening';rel=B.relative_to(R).as_posix()
sys.path.insert(0,str(R/'bench'))
from adapter_common import load,hash_file,canonical,sha,write_json
assert not B.exists()
assert not subprocess.check_output(['git','status','--porcelain'],cwd=R)
base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()
assert base.startswith('992346ec')
receipt_hash='0cc8683df7cf0ee9a3497c00082e93f4d125cde63f9c4c5e3ef55af98d8bbb00'
assert hash_file(P/'receipt.json')==receipt_hash
r=load(P/'receipt.json'); assert r['frozen'] and r['status']=='confirmed_scope_gap'
for p,h in r['artifacts'].items(): assert hash_file(P/p)==h,p
probe=load(P/'probe-results.json');assert probe['status']=='passed'
assert probe['driver_extraction']['success'] and probe['driver_extraction']['stored_signed_i32']==-2**31
for x in probe['records']:
 assert hash_file(Path(x['original_source']))==x['source_sha256']
 assert x['N_python']==2**31 and x['compiled']=='TTIR_LLVM_PTX' and x['gpu_launches']==0
 assert x['all_lane_argument']['all_masks_false'] and x['all_lane_argument']['input_loads']==x['all_lane_argument']['output_stores']==0
log=(P/'lean-attempts/0002-Narrowing.log').read_text();assert log.count('axiom footprint ⊆ standard base')==7 and 'sorryAx' not in log
before=load(R/'bench/original_program_scope.json'); assert Counter(x['status'] for x in before['tasks'])=={'pending':179,'verified':1,'verified_with_deferred_numerics':3}
B.mkdir(parents=True);retained={};omitted={}
for p,h in {**r['artifacts'],'receipt.json':receipt_hash}.items():
 if Path(p).suffix in ['.olean','.so']: omitted[p]=h;continue
 target='evidence/peer/'+p+('.txt' if p.endswith('.lean') else '')
 q=B/target;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(P/p,q);retained[target]=h
shutil.copyfile(__file__,B/'reopen.py')
write_json(B/'evidence-index.json',{'retained_sha256':retained,'generated_objects_not_copied':omitted,'note':'Exact compiler sources, logs and object hashes retained; no generated objects are registered as project modules.'})
review={'schema':1,'kind':'original_program_scope_reopening','reviewer':'root','status':'confirmed_structural_omission_acceptance_withdrawn','recorded_at_utc':datetime.now(timezone.utc).isoformat(),'base_commit':base,'previous_ledger_sha256':hash_file(R/'bench/original_program_scope.json'),'peer_receipt':{'path':rel+'/evidence/peer/receipt.json','sha256':receipt_hash},'affected_tasks':['vector_addition','vector_addition_custom','add_value'],'source_hashes':{str(Path(x['original_source']).relative_to(R)):x['source_sha256'] for x in probe['records']},'semantic_review':'Root read all three original annotations/wrappers, compiled signed-mask evidence, exact installed extractI32 source/probe, universal Lean no_i32_below_bound/all_source_masks_false/empty_active_tile proofs and all three explicit SafeAPI exclusions. Python extent2^31 is accepted by the annotated binder and 64bit-host extractor, narrows to signed-minimum32, and produces legal source grids. Every source mask is false, so ordinary successful allocation/setup returns its original allocated payload without active loads/stores. This deterministic no-access route is absent from the bounded finite theorems. Existing bounded theorems and exact numerical registrations remain valid in their stated domain; their whole-source acceptance recommendation is withdrawn. The omission is structural and cannot be a numerical deferral. No GPU execution or large allocation is claimed.','retained_lean_checks':7,'previous_root_recommendations_superseded':[{'path':str(T/'vector-custom-finite-root-validation'/p),'sha256':hash_file(T/'vector-custom-finite-root-validation'/p)} for p in ['review.json','review-final.json']],'remaining_work':'Add actual runtime i32 narrowing, signed index/mask wrap and u64 actual-address execution/outcome coverage; retain mathematical prefix formula as conditional refinement; independently review full original dtype/setup scope.','whole_source_acceptance_change':-1,'numerical_conformance_validated':False,'gpu_executed':False}
write_json(B/'root-review.json',review);ref={'path':rel+'/root-review.json','sha256':hash_file(B/'root-review.json')}
ledger=copy.deepcopy(before)
for row in ledger['tasks']:
 if row['task'] not in review['affected_tasks']:continue
 task=row['task'];gap={'id':task.upper()+'-ANNOTATED-I32','kind':'runtime_integer_scope_omission','finding':'The original annotated runtime length admits Python2^31, casts it to signed-minimum32 and takes a valid all-masks-false no-access return path; the finite SafeAPI extent<2^31 hypothesis excludes it.','why_blocks_full_original_scope':'An actual source route is omitted even though no numerical scalar operation or active access is required. The existing finite theorem remains valid under its explicit hypotheses.','can_defer_as_numeric_assumption':False,'review':ref}
 row['structural_gaps'].append(gap)
 if task=='vector_addition':
  row.setdefault('accepted_scope_history',[]).append({'withdrawn_at_review':ref,'status':row['status'],'review':row['review'],'accepted_scope_progress_status':row['accepted_scope_progress']['status'],'coverage_sha256':sha(canonical(row['coverage'])),'original_status_explanation':'Historical finite acceptance retained; full-source recommendation withdrawn by concrete omitted route.'})
  row['accepted_scope_progress']['previous_status_before_reopening']=row['accepted_scope_progress']['status'];row['accepted_scope_progress']['status']='historical_acceptance_withdrawn_structural_gap'
  row['status']='pending';row['review']=None
  for dim,items in row['coverage'].items():
   for item in items:
    if item['status']=='covered':
     item['historical_status_before_reopening']=item['status'];item['status']='partial_pending_scope_repair';item['explanation']+=' Full-source acceptance withdrawn: annotated i32 runtime extent2^31 is an omitted source route; see scope_reopening.'
 row['scope_reopening']={'review':ref,'status':'pending_actual_i32_execution_repair','finite_proofs_and_numerical_registrations_retained':True}
ledger['scope_reopening_checkpoint']=rel+'/README.md'
write_json(R/'bench/original_program_scope.json',ledger)
f=R/'bench/original_program_scope.md';s=f.read_text();s=s.replace('**4 source scopes accepted and 179 pending**','**3 source scopes accepted and 180 pending**').replace('`relu_triton_kernel`, `vector_addition` and `quantize_global`','`relu_triton_kernel` and `quantize_global`')
s=s.replace('ReLU and\nvector addition and quantize_global have passed with exactly registered numerical\ndeferrals; 179 scopes remain pending.','ReLU and quantize_global have passed with exactly registered numerical\ndeferrals; 180 scopes remain pending. Vector addition\'s earlier acceptance was withdrawn after an actual runtime-i32 source route was found outside its finite theorem.')
s=s.replace('counted as accepted. Vector addition now connects all 189 typed source routes','counted as accepted. Within its explicitly bounded theorem domain, vector addition connects 189 typed source routes')
s += '\nThe [annotated i32 scope reopening](review-checkpoints/2026-10-16/integer-domain-scope-reopening/README.md) preserves the earlier vector review and numerical registrations. Python length `2^31` is accepted by the real binder and narrows to `-2^31`; every signed mask is false, so ordinary successful setup returns unchanged allocation bytes. The prior `<2^31` premise omits this source path. Vector addition is pending again, and the custom-vector/add-value candidates remain pending until actual runtime arithmetic and outcomes are covered. This is a structural omission, not a numerical deferral.\n'
f.write_text(s)
(B/'README.md').write_text('# Annotated i32 scope reopening\n\nCurrent source acceptance is 3/183, with 180 pending. Vector addition\'s earlier whole-source acceptance is withdrawn; matrix transpose, ReLU and quantize_global retain their current accepted statuses.\n\nThe unchanged vector_addition, vector_addition_custom and add_value kernels accept a Python runtime extent of 2^31 through their annotated i32 binder. The installed driver stores -2^31 without rejection. Their original legal grids run only false signed masks, hence no input loads or output stores. Under ordinary successful allocation/setup, the wrappers return the unchanged allocated output. Existing finite proofs require extent<2^31 and omit that route.\n\nThe checkpoint retains original-source compilation, exact C extractor evidence, all-lane Lean proofs, the unsuccessful proof attempt, candidate scope snapshots and independent root review. No GPU execution or large tensor allocation was performed. Custom addition also has an actual zero-storage Torch shape [2^31,0] example.\n\nOriginal proofs, frozen reviews, catalogs and exact numerical bindings remain unchanged. Historical acceptance is preserved in the ledger\'s accepted_scope_history; current production coverage is partial pending repair. Custom-vector and add-value candidates are not integrated or accepted. Their prior candidate recommendations are superseded for whole-source acceptance.\n\nThe repair must model actual narrowing, signed masks and uint64 addresses and cover returned/undefined/access outcomes. No checker was weakened and no numerical assumption can discharge this gap.\n')
assert Counter(x['status'] for x in ledger['tasks'])=={'pending':180,'verified':1,'verified_with_deferred_numerics':2}
print(json.dumps({'checkpoint':rel,'root_review_sha256':ref['sha256'],'accepted':3,'pending':180,'retained_files':len(retained)}))
