from pathlib import Path
from datetime import datetime, timezone
import json, os, shutil, subprocess, sys
sys.dont_write_bytecode=True
T=Path('/tmp/veritile-standardization-20261015')
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
P=T/'quant-transpose-finite-acceptance-revision2-wip'
V=T/'quant-transpose-revision2-root-validation'
W=V/'build-attempt001'; replay=V/'producer-replay'
sys.path.insert(0,str(R/'bench'))
from adapter_common import hash_file,load,write_json,declarations
r=load(P/'evidence/receipt.json'); rh=hash_file(P/'evidence/receipt.json')
assert rh=='883f7e8e086acb7b20ac7a3f20ba16dfdf697b425ef123d0af7509c877bf188a'
build=load(V/'independent-build-receipt.json')
assert build['all_passed'] and build['dependencies_unchanged']
logs='\n'.join((V/x['log']).read_text() for x in build['records'])
assert logs.count('axiom footprint ⊆ standard base ✓')==77
assert logs.count('non-circular ✓')==14
old=T/'quant-transpose-finite-acceptance-revision1-wip'
oldr=load(old/'evidence/receipt.json')
for package,receipt in [(P,r),(old,oldr)]:
 for rel,h in receipt['artifacts'].items():assert hash_file(package/rel)==h,rel
changed=[p for p,h in r['modules'].items() if oldr['modules'][p]!=h]
assert {Path(p).name for p in changed}=={'FiniteHost.lean','FiniteAdmission.lean','FiniteNativeHost.lean','FiniteAcceptance.lean'}
delta=load(P/'evidence/numerical-authority-delta.json')
assert (len(delta['rules']),len(delta['provenance']),sum(len(t['cases']) for t in delta['requirements']))==(16,140,1144)
adapter=load(replay/'evidence/adapter-check.json')
assert adapter['status']=='passed' and adapter['case_links_checked']==1144
assert adapter['omitted_port_mutations_rejected']==140 and len(adapter['adapter_mutations_rejected'])==23
assert adapter['first100_rules_equal_committed_queue']
write_json(V/'adapter-replay.json',adapter)

# Independently rebuild the exact pinned C parser and rerun the original wrapper AST.
parser=replay/'evidence/driver-order/parser-audit'
cmd=['cc','-shared','-fPIC','-O2','-I/usr/local/include/python3.12',str(parser/'parser.c'),'-o',str(parser/'parser.so')]
with (V/'parser-build.log').open('w') as f:result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
assert result.returncode==0
with (V/'host-order.log').open('w') as f:result=subprocess.run([sys.executable,str(replay/'audit_host_order.py')],stdout=f,stderr=subprocess.STDOUT)
assert result.returncode==0
for name in ['records.json','receipt.json']:shutil.copyfile(replay/'evidence/host-order'/name,V/('host-order-'+name))
records=load(V/'host-order-records.json')
assert len(records)==7 and all(x['downstream_pointer_stage_reached']==(x['grid_x']<2**31) for x in records)
write_json(V/'parser-root-build.json',{'command':cmd,'exit_code':0,'parser_source_sha256':hash_file(parser/'parser.c'),'parser_object_sha256':hash_file(parser/'parser.so'),'log_sha256':hash_file(V/'parser-build.log')})

# Check all generated admissions with independently rebuilt imports.
src=W/'AdmissionCases.lean';shutil.copyfile(P/'evidence/case-admission/AdmissionCases.lean',src)
env=dict(os.environ,LEAN_PATH=build['records'][0]['LEAN_PATH'])
cmd=['/tmp/veritile-lean-toolchain/bin/lean','--root='+str(W),str(src),'-o',str(src.with_suffix('.olean'))]
with (V/'case-admission.log').open('w') as f:result=subprocess.run(cmd,cwd=W,env=env,stdout=f,stderr=subprocess.STDOUT)
assert result.returncode==0
mapping=load(P/'evidence/case-admission/mapping.json')
assert mapping['scope_cases']==1144 and mapping['unique_admission_profiles']==1024
write_json(V/'case-admission.json',{'command':cmd,'exit_code':0,'LEAN_PATH':env['LEAN_PATH'],'source_sha256':hash_file(src),'mapping_sha256':hash_file(P/'evidence/case-admission/mapping.json'),'scope_cases':1144,'profiles':1024,'log_sha256':hash_file(V/'case-admission.log')})

for rel,h in r['artifacts'].items():assert hash_file(P/rel)==h,rel
semantic={
 'unchanged_execution_review':'The previous substantive read of all18 modules, original source, actual constexpr/i32/i64 expression widths, signed masks and uint64 addresses, logical abs/max/reciprocal producer, concrete full-tile loads/stores, every program and finite autotune trial, independent last-write byte results and exterior/returned-scalar frames remains applicable to the14 byte-identical modules. Earlier proofs and failed attempts are preserved. The mathematical transpose refinement is conditional and does not exclude safe wrapped actual-address outcomes.',
 'corrected_first_error':'Root found and withdrew revision1 because CPU fp32 M=2^38,N=1 reaches signed-grid parsing before CPU-pointer rejection. Root read all four changed modules and the original installed launch source. Revision2 explicitly orders context, signed grid parse, entry hook, pointer extraction, launch and exit hook; nativeAdmission receives actual M/N. Compilation still precedes the launcher, so fp16/bf16 reject before grid parsing. Primitive failure branches claim no postfailure memory, including exit-hook failure after possible side effects.',
 'public_scope':'The public specification uses dimension-bearing nativeAdmission for universal rejected outcomes and executeNativeAPI for complete admitted executions. Host extent metadata remains signed64; no inferred allocation capacity or new no-wrap premise excludes the repaired CPU route. The dimension-free admission remains historical small-profile evidence and supplies no full-domain authority.',
 'independent_verification':'Root rebuilt18 modules with77 clean-axiom and14 noncircular checks, then checked all1144 table cases/1024 distinct representative admission profiles using those independent objects. Root rebuilt the exact pinned CPython signed-grid parser and replayed seven original-wrapper AST cases, confirming actual OverflowError precedes downstream pointer instrumentation. No large allocation, CUDA launch, or native numerical result is claimed.',
 'numerical_scope':'The exact authority retains16 numerical families and140 required ports across1144 cases. Root replayed all140 omitted-port mutations and23 adapter mutations with the exact committed pre-quant authority; prior100 rules remain unchanged. Full integrated queue equality is still required. Numerical conformance remains explicitly deferred and cannot excuse structural gaps.',
 'dependency_boundary':'The177 project source/object dependencies and33 producer-recorded external objects remain unchanged. Root separately captured Mathlib.Data.List.Range before its independent build and verified it afterward; the missing historical producer capture is not retroactively asserted.',
 'acceptance_boundary':'Approved for integration under bench/original_program_acceptance.md after current-source catalog, queue, ledger and build checks. This is not native GPU or official comparator validation.'}
review={'schema':1,'kind':'quant_transpose_revision2_finite_source_root_review','reviewer':'root','status':'substantively_reviewed_for_finite_scope_integration','recorded_at_utc':datetime.now(timezone.utc).isoformat(),'producer_receipt_sha256':rh,'reviewed_candidate_source_sha256':r['modules'],'changed_source_files':changed,'semantic_review':semantic,'superseded_review_sha256':hash_file(T/'quant-transpose-revision1-root-validation/review.json'),'superseded_review_status':'withdrawn before commit due to grid/pointer precedence error','verification':{'modules':18,'clean_axioms':77,'noncircular':14,'cases':1144,'representative_profiles':1024,'parser_AST_cases':7,'omitted_ports_rejected':140,'adapter_mutations_rejected':23,'producer_artifacts':len(r['artifacts']),'old_producer_artifacts':len(oldr['artifacts'])},'numerical_conformance_validated':False,'native_execution':False,'acceptance_increment_before_integration':0}
write_json(V/'review.json',review)
shutil.copyfile(Path(__file__),V/'review-package.py')
shutil.copyfile(T/'build_quant_transpose_revision2_root.py',V/'build-package.py')
artifacts={}
for p in V.rglob('*'):
 if p.is_file() and not p.is_symlink() and 'producer-replay' not in p.relative_to(V).parts and p.name!='review-receipt.json' and not p.name.endswith(('.olean','.ilean','.ir','.olean.private','.olean.server')):
  artifacts[p.relative_to(V).as_posix()]=hash_file(p)
write_json(V/'review-receipt.json',{'schema':1,'frozen':True,'producer_receipt_sha256':rh,'review_sha256':hash_file(V/'review.json'),'artifacts':artifacts,'excluded_generated_paths':['producer-replay','generated Lean objects'],'source_acceptance_increment':0})
print(json.dumps({'status':'passed','review_sha256':hash_file(V/'review.json'),'review_receipt_sha256':hash_file(V/'review-receipt.json'),'artifacts':len(artifacts)},indent=2))
