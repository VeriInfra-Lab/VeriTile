from pathlib import Path
from datetime import datetime, timezone
import json, sys, shutil
sys.dont_write_bytecode=True
T=Path('/tmp/veritile-standardization-20261015');R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
P=T/'sin-kernel-finite-acceptance-wip';PEER=T/'sin-kernel-independent-review';V=T/'sin-kernel-root-review'
sys.path.insert(0,str(R/'bench'))
from adapter_common import hash_file,load,write_json,declarations
assert not V.exists()
rh='faf9aaf762de78aadfb536e11e5655af31075637fd536d3d6c04dc8bda226ddb'
assert hash_file(P/'evidence/receipt.json')==rh
assert hash_file(PEER/'peer-review.json')=='08150e5bfce4adefb119e0e6c45e95bdd1ab43c1c9ed68da5a60191fcfbaf9b0'
assert hash_file(PEER/'review-receipt.json')=='0d92828619b20bf6c6dd181f86f5da56fa9d0fcd3e202989ce74037e3f6e10b5'
r=load(P/'evidence/receipt.json');pr=load(PEER/'review-receipt.json');review=load(PEER/'peer-review.json')
for folder,receipt in [(P,r),(PEER,pr)]:
 for rel,h in receipt['artifacts'].items():assert hash_file(folder/rel)==h,rel
for rel,h in r['modules'].items():assert hash_file(P/'after'/rel)==h,rel
assert not review['blockers'] and review['candidate_and_repository_dependencies_unchanged']
assert review['verification']['modules_rebuilt']==8 and review['verification']['candidate_named_clean_axiom_checks']==40
assert review['verification']['candidate_noncircular_checks']==7
deps=load(PEER/'dependencies-before.json');assert deps==load(PEER/'dependencies-after.json')
for rel,v in deps['project'].items():
 assert hash_file(Path(v['source']))==v['sha256'],rel
 assert hash_file(Path(v['olean']))==v['olean_sha256'],rel
decl=load(P/'evidence/declaration-proposal.json')
assert declarations(P/'after'/decl['file'])[decl['declaration']]['statement_sha256']==decl['statement_sha256']
assert decl['statement_sha256']=='8dd5d8a4010a567f2da733f66023700c8e5c86350a7188c1bf3f09c9fddcda3e'
delta=load(P/'evidence/numerical-authority-delta.json')
assert (len(delta['rules']),len(delta['provenance']),sum(len(x['cases']) for x in delta['requirements']))==(2,2,52)
semantics={
 'substantive_review':'Root read the original Python and all eight candidate modules in full: NativeCorrect, Admission, Arithmetic, Grid, Memory, Program, Scalar and Source, plus producer README, authority adapter and independent peer source review. The proof explicitly connects executed finite source instructions to initial reads and a separate raw-byte result formula; it is not a mere assertion of source or build equality.',
 'actual_source':'lowerBody consumes all seven retained original AST statements. compiledSource uses a proved successful lowering without a fallback stream; sourceActiveTile/sourcePointer execute its i32 register and signed-mask prefix and are proved equal to independent closed requests. compiledScalar is extracted from instruction5, and complete_source_suffix fixes the masked-load/sine/store order. The existing typed precision proof preserves fp32/fp64 without implicit narrowing. Source tiles load inputs before scalar computation/stores.',
 'host_and_domains':'x.numel precedes empty_like even on allocation failure, followed by BLOCK1024 ceiling grid, launch and identical output return. There is no source CUDA, rank, contiguity or dtype assertion added. All22 actual dense dtype profiles remain represented:5 binder rejects,2 fnuz architecture rejects,13 sine frontend rejects,2 backend successes. JIT device/stream, compilation, native context, grid parsing, pointer packing and launch remain ordinary ordered setup primitives. Allocation bytes, preserve-format strides, scalar and empty/offset/broadcast metadata remain explicit.',
 'finite_extent':'Host numel covers nonnegative signed64 values, while actual N and index arithmetic are i32 at every specialization. Signed masks and sign-extended width-scaled u64 pointer wrap retain negative/repeated requests. No no-wrap/access premise excludes a source path. Empty and2^31 all-masked routes return the exact allocation unchanged after successful setup, independently of sine/input values. Exact grid-limit evidence distinguishes parsing failure from downstream pointer failure.',
 'memory_and_pointer':'TensorBinding separates Torch data/storage from actual extracted pointers in the mapped launch-address memory coordinate system. Current-device permissions do not follow from allocation-owner equality. Early-failure pointer fields are unconsumed total affine completions, not extraction-success claims. No separate native-to-canonical translation proof is asserted. WordMemory forces exact little-endian raw BitVec32/64 decoding/packing, byte presence, and local store/frame laws; it cannot supply a sine or final kernel result.',
 'execution_result':'Schedule quantifies over every complete grid permutation. Fresh output and frame lemmas preserve initial input reads across stores. The complete returned-byte formula uses raw bits of the final touching store and retains allocator bytes elsewhere, with the full exterior/input frame. Access, missing-byte and ownership conflicts remain mathematical observations without a native trap, rollback, race-winner or postfailure return claim. Although the generic evaluator has an undefined-expression case, the actual sine expression is total.',
 'numerics_and_authority':'52 exact cases bind only2 numerical ports in2 families, sin32 and sin64, across all raw words including NaNs/infinities/subnormals/signed zeros. Only native scalar conformance is deferred. Empty, masked and setup/error cases do not consume irrelevant sine ports. Receipt, root review, original AST, dependency snapshots, raw IR and exact public statement remain fail-closed bound inputs; integration must preserve whole-queue equality.',
 'verification_attribution':'Peer independently rebuilt8 modules with identical objects,40 clean-axiom and7 noncircular checks;10 independent regressions plus fresh22-profile admission equality passed. Peer replayed22 dtype and30 ABI profiles,160 equal IR artifacts,2 scalar graphs,75 real CPU wrapper cases/150 metadata records,18 AST traces,7 grid parser checks and15 i32 extractor cases, adding22 actual JIT binder checks. Both omissions and20 authority tamper cases were rejected. Root reverified535 producer artifacts,365 peer artifacts and204 unchanged project source/object pairs and exact statement/counts; root does not claim an additional rebuild or native replay.',
 'limits':'Existing source/proofs and producer failed attempts remain frozen. No GPU execution, live CUDA mapping/access measurement, numerical sine conformance or official comparator pass is established. Ordinary Torch and Triton primitives remain the declared trusted boundary. This root review alone increments accepted programs by zero; reviewed repository integration/gates are still required.'}
V.mkdir()
out={'schema':1,'reviewer':'root','status':'substantively_reviewed_for_finite_scope_integration','task':'sin_kernel','revision':1,'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'producer_receipt_sha256':rh,'peer_review_sha256':hash_file(PEER/'peer-review.json'),'peer_receipt_sha256':hash_file(PEER/'review-receipt.json'),'reviewed_candidate_source_sha256':r['modules'],'public_statement':{k:decl[k] for k in ['file','declaration','statement_sha256']},'semantic_review':semantics,'verification':review['verification'],'verified_artifact_counts':{'producer':len(r['artifacts']),'independent_peer':len(pr['artifacts'])},'current_repository_integration':'pending','native_execution':False,'numerical_conformance_validated':False,'acceptance_increment_before_integration':0}
write_json(V/'review.json',out);shutil.copyfile(Path(__file__),V/'review-package.py')
write_json(V/'review-receipt.json',{'schema':1,'frozen':True,'producer_receipt_sha256':rh,'review_sha256':hash_file(V/'review.json'),'artifacts':{p.name:hash_file(p) for p in V.iterdir() if p.is_file()},'source_acceptance_increment':0})
print(json.dumps({'review_sha256':hash_file(V/'review.json'),'receipt_sha256':hash_file(V/'review-receipt.json'),'status':'approved_for_integration_only'},indent=2))
