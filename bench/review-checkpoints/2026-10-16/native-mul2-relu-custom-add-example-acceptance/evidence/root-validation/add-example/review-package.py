from pathlib import Path
from datetime import datetime,timezone
import json,sys,shutil
sys.dont_write_bytecode=True
T=Path('/tmp/veritile-standardization-20261015');R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
P=T/'add-example-finite-acceptance-wip';PEER=T/'add-example-independent-review';V=T/'add-example-root-review'
sys.path.insert(0,str(R/'bench'));from adapter_common import hash_file,load,write_json,declarations
assert not V.exists()
rh='83920c47c558019d291230261b96f0827376831ae091430ce6971e37a640d8f5'
assert hash_file(P/'evidence/receipt.json')==rh
assert hash_file(PEER/'peer-review.json')=='57fbc53c339778d19220d8900d36bf55fd129b606a1d7d1c42cee5fcdc187963'
assert hash_file(PEER/'review-receipt.json')=='c865d6177d3973451e8f2b589bb49df8d1fbe5477c71dd02c5c897441f2f0cb4'
r=load(P/'evidence/receipt.json');pr=load(PEER/'review-receipt.json');review=load(PEER/'peer-review.json')
for folder,receipt in [(P,r),(PEER,pr)]:
 for rel,h in receipt['artifacts'].items():assert hash_file(folder/rel)==h,rel
for rel,h in r['modules'].items():assert hash_file(P/rel)==h,rel
assert not review['blockers'] and review['candidate_and_repository_dependencies_unchanged']
assert review['verification']['modules_rebuilt']==9 and review['verification']['candidate_named_clean_axiom_checks']==46
assert review['verification']['candidate_noncircular_checks']==7
deps=load(PEER/'dependencies-before.json');assert deps==load(PEER/'dependencies-after.json')
for rel,v in deps['project'].items():
 assert hash_file(Path(v['source']))==v['sha256'],rel
 assert hash_file(Path(v['olean']))==v['olean_sha256'],rel
decl=load(P/'evidence/declaration-proposal.json')
assert declarations(P/decl['file'])[decl['declaration']]['statement_sha256']==decl['statement_sha256']
assert decl['statement_sha256']=='508207088924a33110b60f931868c3dcae65e0fde34f21a48f5dfd9a99c83493'
delta=load(P/'evidence/numerical-authority-delta.json')
assert (len(delta['rules']),len(delta['provenance']),sum(len(x['cases']) for x in delta['requirements']))==(37,296,1700)
semantics={
 'substantive_review':'Root read the original Python and all nine candidate modules in full: FiniteIndices, Scalar, Source, Physical, Program, Grid, Admission, Host and public FiniteCorrect, together with the producer REVIEW.md, authority adapter and independent peer source review. The public theorem relates observed execution of the exact source interpreter to a separate host/initial-read/byte formula; it does not merely restate source execution or rely on build success as source correspondence.',
 'source_execution':'The lowerer verifies the exact original eight statements. Index registers hold BitVec32 values, masks use signed comparison, and pointers sign-extend offsets and wrap at64 bits after scaling by actual storage width. Source x loads and then y loads complete before output stores. The separately bounded Nat coverage theorem remains a refinement only; the master imposes no no-wrap bound or y-shape/length/device equality. All actual dense signed64 host extents fit its model; larger abstract Nat values do not remove any ordinary source path.',
 'host_and_first_failures':'Original zeros_like(x) precedes BLOCK4, x.numel, grid construction and launch. It adds no CUDA or rank guard; rank0 means one element, while an empty tensor still traverses setup. Native current-device/stream, dtype admission, compilation, launcher context, signed-grid parsing, argument extraction and enqueue remain ordered fallible ordinary primitives. The native grid bound is4*(2^31-1). Exact signed-minimum residues2^31 and2^31+2^32 are reachable all-masked routes; they return the original zero allocation without input/scalar/access premises under successful ordinary setup. The next grid-limit integer is rejected before pointer extraction.',
 'actual_pointer_boundary':'PhysicalView distinguishes extracted ABI pointer, device-space allocation base and canonical backing base/capacity. All views share the same address translation; byte permissions use actual current launch device rather than allocation-owner equality. Successful extraction ties every actual view pointer to metadata.extractedPointer, with canonical byte translation only for nonnull data_ptr. Null raw pointers bypass unused translated-span requirements. AllocationBinding requires extraction evidence only after dtype admission, all setup stages and grid parsing succeed, so early failures do not assume a missing successful result. Ordinary allocation/freshness/frame facts do not supply any kernel values.',
 'memory_values_and_schedule':'MemoryPrimitives has complete observed backing bytes and local width-specific dtype decode/encode laws. Active loads/stores separately inspect native and backing bounds, alignment, permissions and within-word translation. There is no missing-byte default or arbitrary tagged-carrier roundtrip law. Initial-read plans use the allocated snapshot; freshness and read/frame lemmas prove earlier output writes preserve later inputs. Metadata admits every permutation of the entire grid. Intra/inter-program overlapping output spans become applicability observations before any returned result is claimed. Successful final bytes use independent reverse-last-write lookup, with complete input/exterior frames and untouched zero bytes.',
 'observations_and_numerics':'Source applicability failures are mathematical observations only: no native trap, race winner, rollback, postfailure memory or successful return is claimed. The original 22x22 dtype domain preserves195 binder,64 signature,36 addition and6 backend rejections plus183 admitted routes. Typed scalar formulas reuse installed proofs; ConversionDefined removes unreachable floating-to-integer fallbacks before stored values.1700 exact cases bind37 families/296 ports; empty/all-masked/setup/observation cases claim no numerical output and require no unused ports. All scalar conformance remains explicitly deferred, not presumed proved.',
 'verification_attribution':'Peer independently rebuilt9 modules with exact object reproduction,46 named clean/7 noncircular checks,9 independent regressions and a484-row admission theorem. It replayed484 frontend and189 backend profiles,921 matching IR artifacts,189 scalar fingerprints,484 actual JIT binder cases,73 CPU wrapper cases,16 compiled exact-i32 boundaries plus signed64 overflow and7 signed-grid parser probes. All296 omissions and20 adapter mutations were rejected. Root reverified3302 producer artifacts,1776 peer artifacts,198 unchanged project sources/objects, exact public statement and numerical counts. This is not represented as a second root rebuild or runtime replay; both reviewer failed attempts and all producer failures remain frozen.',
 'limits':'No GPU launch, live CUDA translation/access measurement, native scalar validation or official comparator pass is established. The explicit ordinary runtime/memory primitive boundary remains trusted. Literal current integration pins, additive authority equality and repository gates are still required; this review alone increments accepted count by zero.'}
V.mkdir()
out={'schema':1,'reviewer':'root','status':'substantively_reviewed_for_finite_scope_integration','task':'add_example','revision':1,'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'producer_receipt_sha256':rh,'peer_review_sha256':hash_file(PEER/'peer-review.json'),'peer_receipt_sha256':hash_file(PEER/'review-receipt.json'),'reviewed_candidate_source_sha256':r['modules'],'public_statement':{k:decl[k] for k in ['file','declaration','statement_sha256']},'semantic_review':semantics,'verification':review['verification'],'verified_artifact_counts':{'producer':len(r['artifacts']),'independent_peer':len(pr['artifacts'])},'current_repository_integration':'pending','native_execution':False,'numerical_conformance_validated':False,'acceptance_increment_before_integration':0}
write_json(V/'review.json',out);shutil.copyfile(Path(__file__),V/'review-package.py')
write_json(V/'review-receipt.json',{'schema':1,'frozen':True,'producer_receipt_sha256':rh,'review_sha256':hash_file(V/'review.json'),'artifacts':{p.name:hash_file(p) for p in V.iterdir() if p.is_file()},'source_acceptance_increment':0})
print(json.dumps({'review_sha256':hash_file(V/'review.json'),'receipt_sha256':hash_file(V/'review-receipt.json'),'status':'approved_for_integration_only'},indent=2))
