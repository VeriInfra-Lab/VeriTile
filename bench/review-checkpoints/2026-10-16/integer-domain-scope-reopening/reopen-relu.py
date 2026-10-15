from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import shutil,sys,json
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');T=Path('/tmp/veritile-standardization-20261015')
P=T/'add-example-finite-acceptance-wip/evidence/relu-scope-review/20261015T215740139914Z'
B=R/'bench/review-checkpoints/2026-10-16/integer-domain-scope-reopening';rel=B.relative_to(R).as_posix()
sys.path.insert(0,str(R/'bench'))
from adapter_common import load,hash_file,canonical,sha,write_json
rh='ff0c42a49e95755cf0e90133adaac0d60c627a67483c5810d1a676c460fd01ce';assert hash_file(P/'receipt.json')==rh
r=load(P/'receipt.json');assert r['source_ast_identity'] and r['native_backend_compiled'] and not r['gpu_executed']
assert hash_file(Path(r['source']))==r['source_sha256']
for p,h in r['artifacts'].items():assert hash_file(P/p)==h,p
assert len(r['records'])==5 and all(x['status']=='compiled' and x['grid_x_less_than_2p31'] for x in r['records'])
assert any(x['N']==2**31+1 and x['grid_x']==2**21+1 for x in r['records'])
assert not (B/'relu-root-review.json').exists()
idx=load(B/'evidence-index.json')
for p,h in {**r['artifacts'],'receipt.json':rh}.items():
 if p.endswith('.cubin'):idx['generated_objects_not_copied']['relu/'+p]=h;continue
 q='evidence/relu/'+p;dest=B/q;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(P/p,dest);idx['retained_sha256'][q]=h
for name in ['reopen_vector_i32_scope_attempt001.py','reopen_vector_i32_scope_attempt001.log','reopen_vector_i32_scope_attempt002.log']:
 dest=B/'evidence/preparation-attempts'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(T/name,dest);idx['retained_sha256'][dest.relative_to(B).as_posix()]=hash_file(dest)
shutil.copyfile(__file__,B/'reopen-relu.py')
write_json(B/'evidence-index.json',idx)
proofs=['bench/tritonbench_g/relu_triton_kernel/Proofs/FiniteIndices.lean','bench/tritonbench_g/relu_triton_kernel/Proofs/FiniteBoundary.lean','bench/tritonbench_g/relu_triton_kernel/FiniteCorrect.lean']
review={'schema':1,'kind':'original_program_scope_reopening','reviewer':'root','status':'confirmed_structural_omission_acceptance_withdrawn','recorded_at_utc':datetime.now(timezone.utc).isoformat(),'task':'relu_triton_kernel','native_receipt':{'path':rel+'/evidence/relu/receipt.json','sha256':rh},'reviewed_inputs':{p:hash_file(R/p) for p in proofs}|{str(Path(r['source']).relative_to(R)):r['source_sha256']},'semantic_review':'Root checked unchanged Python constexpr N and pid0-only store condition, the current SourceDomain numel≤2^31 restriction, existing kernel-checked next_block_overflows_before_gate example, and all five preserved exact-original compilation profiles. At N=2^31+1 the legal grid contains pid2^21; the source i32 block offset wraps negative while its constexpr-promoted unsigned mask is true. Original code does not reject this length. The source may perform invalid actual-address input loads before the pid0 store gate. Current finite proof excludes the admitted path instead of classifying actual-address/undefined outcomes. This is a structural omission, not a floating-numerical obligation. Existing bounded prefix behavior, proofs and exact numeric registrations remain valid under their hypotheses and are retained.','native_compiled_profiles':5,'gpu_executed':False,'whole_source_acceptance_change':-1,'remaining_work':'Cover actual constexpr integer promotion, wrapped indices/pointers, all-source access observations and real return behavior; retain pid0-only stores and conditional bounded mathematical result.'}
write_json(B/'relu-root-review.json',review);ref={'path':rel+'/relu-root-review.json','sha256':hash_file(B/'relu-root-review.json')}
ledger=load(R/'bench/original_program_scope.json');row=next(x for x in ledger['tasks'] if x['task']=='relu_triton_kernel');assert row['status']=='verified_with_deferred_numerics'
row.setdefault('accepted_scope_history',[]).append({'withdrawn_at_review':ref,'status':row['status'],'review':row['review'],'accepted_scope_progress_status':row['accepted_scope_progress']['status'],'coverage_sha256':sha(canonical(row['coverage'])),'original_status_explanation':'Historical bounded acceptance retained; full-source recommendation withdrawn.'})
row['accepted_scope_progress']['previous_status_before_reopening']=row['accepted_scope_progress']['status'];row['accepted_scope_progress']['status']='historical_acceptance_withdrawn_structural_gap';row['status']='pending';row['review']=None
row['structural_gaps'].append({'id':'RELU-CONSTEXPR-INDEX-SCOPE','kind':'constexpr_integer_scope_omission','finding':'Original constexpr N=2^31+1 compiles with legal grid; i32 block offsets wrap while unsigned mask can remain true. The finite SourceDomain numel≤2^31 excludes this actual source path.','why_blocks_full_original_scope':'Source loads execute in all programs before the pid0-only store gate. Wrapped access observations and their defined returned alternatives require actual-address coverage; bounded safety is not a source rejection.','can_defer_as_numeric_assumption':False,'review':ref})
for items in row['coverage'].values():
 for item in items:
  if item['status']=='covered':item['historical_status_before_reopening']='covered';item['status']='partial_pending_scope_repair';item['explanation']+=' Full-source acceptance withdrawn: actual larger constexpr lengths are outside this bounded finite theorem; see scope_reopening.'
row['scope_reopening']={'review':ref,'status':'pending_actual_constexpr_and_i32_execution_repair','finite_proofs_and_numerical_registrations_retained':True}
write_json(R/'bench/original_program_scope.json',ledger);assert Counter(x['status'] for x in ledger['tasks'])=={'pending':181,'verified':1,'verified_with_deferred_numerics':1}
f=R/'bench/original_program_scope.md';s=f.read_text().replace('**3 source scopes accepted and 180 pending**','**2 source scopes accepted and 181 pending**').replace('`relu_triton_kernel` and `quantize_global` are verified with exactly registered','`quantize_global` is verified with exactly registered')
s=s.replace('ReLU and quantize_global have passed with exactly registered numerical\ndeferrals; 180 scopes remain pending.', 'Quantize_global has passed with exactly registered numerical\ndeferrals; 181 scopes remain pending. ReLU\'s earlier acceptance was also withdrawn because larger constexpr lengths were excluded.')
s=s.replace('ReLU now has complete finite source coverage for all 17 canonical CUDA profiles,','Within its bounded theorem domain, ReLU has finite source coverage for 17 canonical CUDA profiles,')
s=s.replace('Vector addition is pending again, and the custom-vector/add-value candidates remain pending','Vector addition and ReLU are pending again, and the custom-vector/add-value candidates remain pending')
s += '\nThe same checkpoint records ReLU\'s independently compiled `N=2^31+1` route: its grid is valid, i32 offsets wrap before the pid-zero store gate, and its constexpr-promoted mask can remain active. The previous `N≤2^31` theorem remains available, but its applicability bound cannot establish complete original-source acceptance.\n'
f.write_text(s)
f=B/'README.md';s=f.read_text().replace('# Annotated i32 scope reopening','# Integer-domain scope reopening').replace('Current source acceptance is 3/183, with 180 pending. Vector addition\'s earlier whole-source acceptance is withdrawn; matrix transpose, ReLU and quantize_global retain their current accepted statuses.','Current source acceptance is 2/183, with 181 pending. Vector addition and ReLU\'s earlier whole-source acceptance is withdrawn; matrix transpose and quantize_global retain their current accepted statuses. The vector review first reduced the count to3; the subsequently confirmed ReLU omission reduced it to2.')
s += '\nReLU uses constexpr N, not a runtime annotated i32 bound. Five exact-original offline compilation profiles include N=2^31+1 with legal grid2^21+1. The current SourceDomain excludes this path, while the existing proof already exhibits negative wrapped offsets and an active mask. Its full-source claim is withdrawn separately in relu-root-review.json. Bounded proofs and all numerical registrations remain unchanged.\n'
f.write_text(s)
print(json.dumps({'accepted':2,'pending':181,'relu_review_sha256':ref['sha256'],'retained_files':len(idx['retained_sha256'])}))
