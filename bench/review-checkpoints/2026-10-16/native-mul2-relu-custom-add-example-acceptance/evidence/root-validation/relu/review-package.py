from pathlib import Path
from datetime import datetime,timezone
import json,sys,shutil
sys.dont_write_bytecode=True
T=Path('/tmp/veritile-standardization-20261015');R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
P=T/'relu-finite-acceptance-revision1-wip';PEER=T/'relu-revision1-independent-review';V=T/'relu-revision1-root-review'
sys.path.insert(0,str(R/'bench'));from adapter_common import hash_file,load,write_json,declarations
assert not V.exists();V.mkdir()
rh='e219fb011932538d411b02b85a5e624e7d617198faca6730a953ddac2db5e46e'
assert hash_file(P/'evidence/receipt.json')==rh
assert hash_file(PEER/'peer-review.json')=='9213de1d4df58ed27951dd37f42dfb1a6236f9f13b2f5233890aca9c87f5c7a8'
assert hash_file(PEER/'review-receipt.json')=='dbe669133941360a710067786eaaa111ef07ee466b361c34e059b96851b0d487'
r=load(P/'evidence/receipt.json');pr=load(PEER/'review-receipt.json');peer=load(PEER/'peer-review.json')
for rel,h in r['artifacts'].items():assert hash_file(P/rel)==h,rel
for rel,h in pr['artifacts'].items():assert hash_file(PEER/rel)==h,rel
for rel,h in r['modules'].items():assert hash_file(P/rel)==h,rel
delta=load(P/'evidence/numerical-authority-delta.json')
assert (len(delta['rules']),len(delta['provenance']),sum(len(x['cases']) for x in delta['requirements']))==(8,68,404)
dependencies={}
for ref in delta['dependency_snapshots']:
 exact=load(P/ref['path']);assert hash_file(P/ref['path'])==ref['sha256']
 for module,v in exact['project'].items():
  rel=module.replace('.','/')+'.lean';assert hash_file(R/rel)==v['sha256'],rel;dependencies[rel]=v['sha256']
contract=delta['requirements'][0]['declaration']
assert declarations(P/contract['file'])[contract['declaration']]['statement_sha256']==contract['statement_sha256']
semantics={
 'substantive_read':'Root read the original Python and all eight Actual modules in full: Indices, Scalar, Source, Physical, Grid, Admission, Host and ActualCorrect, together with the frozen producer/independent review and exact authority adapter. The seven-statement original AST lowering preserves pre-gate load and comparison/select; only the store and its destination conversion are gated by pid==0.',
 'full_integer_domain':'Index multiply/add remain finite i32 operations. Comparison against constexpr N uses the actual signed-i32, unsigned-i32, sign-extended signed-i64 and unsigned-i64 promotion branches; N>=2^64 is classified by frontend admission. There is no old N<=2^31 applicability bound. Native signed-grid parsing separately bounds launched pids and precedes pointer extraction. All launched programs retain active input-read applicability even if they store nothing.',
 'scalar_scope':'All22 canonical dtypes remain explicit:13 scalar routes,5 absent signature mappings,2 unsupported fnuz signatures and2 ordered-comparison failures. Integer/bool comparisons and normalization, and literal positive-zero encoding, are concrete. Exact remaining integer-to-fp32, floating comparison, extension and narrowing correspondence is deferred in8 families/68 ports over404 cases. Select is computed before the gate; output narrowing belongs to the gated store.',
 'memory_boundary':'PhysicalView distinguishes canonical backing, native mapped base and extracted ABI pointer. AccessContext supplies a shared byte mapping and byte permissions indexed by current launch device; owner device is not an access guard. Every active native span has width/alignment/bounds/translation/permission conditions. Pointer bindings relate successful extraction to actual raw Tensor metadata and do not impose nonnull translation on empty null routes. No primitive supplies loaded values or final kernel memory.',
 'execution_result':'Concrete source execution composes every program and is related to an independent initial-memory store list and reverse-last-write byte formula. Fresh allocation and frames justify the initial input snapshot. Complete schedules permute all launched pids; the real native-grid bound proves only pid0 stores, while every other program retains its reads. Untouched output bytes remain arbitrary allocator contents, and input/exterior bytes are preserved. No whole-tensor mathematical ReLU claim is introduced.',
 'host_outcomes':'The host retains x.device, empty_like(float32,x.device), output.numel, unused num_blocks computation, independent launch grid and returned metadata. Current-device acquisition precedes dtype admission; successful compilation precedes launcher context/grid parse, pointers and enqueue. Fallible native operations preserve their actual state. Access outside the model is an explicit observation carrying the allocated snapshot, not a native exception, successful return, rollback or postfailure-memory claim.',
 'independent_evidence':'Peer rebuilt all8 exact modules with36 clean-axiom/6 noncircular checks and7 additional Lean regressions; replayed33 offline profiles/115 normalized IR artifacts,264 additional dtype/constexpr profiles,73 original-wrapper CPU/stub cases and12 actual signed-grid parser boundaries. All404 schema cases/68 port omissions and20 adapter mutations were checked. Root reverified every frozen producer/peer artifact and unchanged182 repository source dependencies. These peer runs are not relabelled as separate root executions.',
 'acceptance_boundary':'Suitable for integration under the documented finite-source primitive boundary after current catalog/ledger/queue/build checks. The prior bounded proofs and numerical rules remain unchanged. No GPU execution, live CUDA permission test, native scalar conformance or official comparator pass is established.'}
out={'schema':1,'reviewer':'root','status':'substantively_reviewed_for_finite_scope_integration','task':'relu_triton_kernel','revision':1,'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'producer_receipt_sha256':rh,'peer_review_sha256':hash_file(PEER/'peer-review.json'),'peer_receipt_sha256':hash_file(PEER/'review-receipt.json'),'reviewed_candidate_source_sha256':r['modules'],'reviewed_existing_repo_source_sha256':dependencies,'public_statement':contract,'semantic_review':semantics,'verified_artifact_counts':{'producer':len(r['artifacts']),'independent_peer':len(pr['artifacts'])},'current_repository_integration':'pending','native_execution':False,'numerical_conformance_validated':False,'acceptance_increment_before_integration':0}
write_json(V/'review.json',out);shutil.copyfile(Path(__file__),V/'review-package.py')
write_json(V/'review-receipt.json',{'schema':1,'frozen':True,'producer_receipt_sha256':rh,'review_sha256':hash_file(V/'review.json'),'artifacts':{p.name:hash_file(p) for p in V.iterdir() if p.is_file()},'source_acceptance_increment':0})
print(json.dumps({'review_sha256':hash_file(V/'review.json'),'receipt_sha256':hash_file(V/'review-receipt.json'),'status':'approved_for_integration_only'},indent=2))
