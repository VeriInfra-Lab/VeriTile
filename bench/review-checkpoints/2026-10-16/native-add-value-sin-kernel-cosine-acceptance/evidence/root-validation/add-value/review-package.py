from pathlib import Path
from datetime import datetime, timezone
import json, sys, shutil
sys.dont_write_bytecode=True
T=Path('/tmp/veritile-standardization-20261015'); R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
P=T/'add-value-finite-acceptance-revision1-wip'; PEER=T/'add-value-revision1-independent-review'; V=T/'add-value-revision1-root-review'
sys.path.insert(0,str(R/'bench'))
from adapter_common import hash_file,load,write_json,declarations
assert not V.exists()
rh='4ad8aff1121ae9a10d70952456aaf75f31c26c408ecc37bb8f641a9c712ef774'
assert hash_file(P/'evidence/receipt.json')==rh
assert hash_file(PEER/'peer-review.json')=='c50a99038d8ff77cc1bd57f6e44a3a04a6efec081f547b87451bc84f5c5197cf'
assert hash_file(PEER/'review-receipt.json')=='21cf9ef56ac47d47b52862aee57b3c40005cb4c05d7c594d665ff3d34baa07b3'
r=load(P/'evidence/receipt.json');pr=load(PEER/'review-receipt.json');review=load(PEER/'peer-review.json')
for folder,receipt in [(P,r),(PEER,pr)]:
 for rel,h in receipt['artifacts'].items(): assert hash_file(folder/rel)==h,rel
for rel,h in r['modules'].items():assert hash_file(P/'after'/rel)==h,rel
assert not review['blockers'] and review['candidate_and_repository_dependencies_unchanged']
assert review['verification']['modules_rebuilt']==7 and review['verification']['candidate_named_clean_axiom_checks']==30
assert review['verification']['candidate_noncircular_checks']==6
deps=load(PEER/'dependencies-before.json');assert deps==load(PEER/'dependencies-after.json')
for rel,v in deps['project'].items():
 assert hash_file(Path(v['source']))==v['sha256'],rel
 assert hash_file(Path(v['olean']))==v['olean_sha256'],rel
decl=load(P/'evidence/declaration-proposal.json')
assert declarations(P/'after'/decl['file'])[decl['declaration']]['statement_sha256']==decl['statement_sha256']
assert decl['statement_sha256']=='a63f6673991eae5bb39b7fb9095b5f7170713d9c129766286185f53540981d8e'
delta=load(P/'evidence/numerical-authority-delta.json')
assert (len(delta['rules']),len(delta['provenance']),sum(len(x['cases']) for x in delta['requirements']))==(15,41,238)
semantics={
 'substantive_review':'Root read the original Python, all seven candidate Lean modules in full (NativeCorrect, Admission, Arithmetic, Grid, Memory, Program, Scalar), the producer README and authority adapter, the independent source review and all ten independent regression proofs. The source interpreter is connected to an initial-memory plan through a separate scalar formula, byte packing and frame laws; build success alone is not source correspondence.',
 'source_and_scalar':'The source has one input pointer and a non-constexpr runtime fp32 operand10. Exact installed extractFP32 produces0x41200000, and argument_exact decodes it to10. A tile loads active inputs before scalar evaluation/stores. Bool normalizes nonzero loads but stores raw i8 arithmetic bytes10/11; fp64 keeps the loaded fp64 input and widens only the runtime fp32 scalar. Integer result conversion is partial, and undefined conversion becomes an observation before any result is stored. Fifteen admitted scalar graphs match the original compiled source.',
 'host_and_domains':'empty_like precedes the original conjunction of input/output CUDA assertions, then output.numel, BLOCK1024 grid, launch and identical allocation return. Scalar rank, arbitrary preserve-format strides and nonzero view offsets remain represented. All22 dtype routes remain:15 backend compiles,5 binder rejects,2 architecture rejects. Ordinary context/compilation/grid-parser/pointer-and-scalar-packing/enqueue failures retain the reviewed ordering; signed-grid parse precedes extraction.',
 'finite_indices':'Host extents cover all nonnegative signed64 values. N and lane arithmetic are actual i32, masks signed, pointers sign-extended and width-scaled modulo2^64. There is no no-wrap or valid-access premise restricting the master. At2^31 every mask is false and the exact allocated memory returns unchanged after successful setup; n=2^41 exceeds the signed grid limit. The independent plan records unsafe, missing-byte, undefined-conversion and overlapping-owner observations without asserting a native trap or postfailure result.',
 'pointer_and_memory_boundary':'Torch storage/data pointers are distinct from extracted mapped launch pointers. Buffer memory uses the mapped launch-address coordinate system, with permission indexed by current device independently of allocation owner. No separate canonical-backing translation theorem is claimed. Total metadata on early-failure branches are unconsumed affine completions, not unconditional extraction success. Local typed load presence/congruence, pure dtype packing and store-byte/frame laws form the explicit trusted primitive boundary allowed by the acceptance policy. Fresh output keeps initial input reads stable, returned bytes follow independent last-touch packing, and untouched bytes retain arbitrary allocation contents.',
 'schedule_scope':'The catalog explicitly records sequentially_consistent_schedule. The public launch chooses the complete List.range grid. The lower grid theorem is valid for arbitrary pid lists; fresh input frames prevent store/read interference, and the returned route checks nonoverlapping byte owners. Peer proves bytewise commutation of disjoint stores. Root accepts this declared finite source boundary under the policy excluding a per-task native scheduler proof; neither an all-native-schedules theorem nor native scheduling behavior is claimed. No observed/racing execution is promoted to a successful returned-value claim.',
 'numerical_registration':'238 schema2 cases bind41 exact operation ports in15 families, including runtime argument/ABI, widths, conversions, rounding, target and source provenance. Only native numerical conformance is deferred; malformed access, missing bytes and undefined conversions are structural observations, not numerical exceptions. Authority installation still needs independent literal pins and unchanged whole-queue equality.',
 'verification_attribution':'Peer rebuilt7 modules with identical objects,30 clean-axiom and6 noncircular checks, plus10 independent regressions and the22-profile admission theorem. Peer replayed22 dtype and14 extent profiles,145 equal IR artifacts,15 scalar graphs,1425408 C pointer checks,16 i32 boundaries, exact fp32 scalar extraction,18 AST traces,6 CPU metadata and49 real CPU allocation/assertion cases,7 grid parser tests,41 missing-port and20 adapter mutations. Root reverified434 producer artifacts,361 peer artifacts,201 source/object dependencies and the exact public statement/counts; root does not claim an additional independent rebuild or runtime replay.',
 'limits':'No GPU execution, CUDA mapping/access measurement, native numerical conformance or official comparator pass. The historical bounded five-module candidate and failed attempts remain evidence only. This review alone increments acceptance by zero; exact repository integration and gates remain required.'}
V.mkdir()
out={'schema':1,'reviewer':'root','status':'substantively_reviewed_for_finite_scope_integration','task':'add_value','revision':1,'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'producer_receipt_sha256':rh,'peer_review_sha256':hash_file(PEER/'peer-review.json'),'peer_receipt_sha256':hash_file(PEER/'review-receipt.json'),'reviewed_candidate_source_sha256':r['modules'],'public_statement':{k:decl[k] for k in ['file','declaration','statement_sha256']},'semantic_review':semantics,'verification':review['verification'],'verified_artifact_counts':{'producer':len(r['artifacts']),'independent_peer':len(pr['artifacts'])},'current_repository_integration':'pending','native_execution':False,'numerical_conformance_validated':False,'acceptance_increment_before_integration':0}
write_json(V/'review.json',out);shutil.copyfile(Path(__file__),V/'review-package.py')
write_json(V/'review-receipt.json',{'schema':1,'frozen':True,'producer_receipt_sha256':rh,'review_sha256':hash_file(V/'review.json'),'artifacts':{p.name:hash_file(p) for p in V.iterdir() if p.is_file()},'source_acceptance_increment':0})
print(json.dumps({'review_sha256':hash_file(V/'review.json'),'receipt_sha256':hash_file(V/'review-receipt.json'),'status':'approved_for_integration_only'},indent=2))
