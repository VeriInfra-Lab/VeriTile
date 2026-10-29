from pathlib import Path
from datetime import datetime,timezone
import json,sys,shutil,difflib
sys.dont_write_bytecode=True
T=Path('/tmp/veritile-standardization-20261015');R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
P=T/'vector-custom-finite-acceptance-revision1-wip';PEER=T/'vector-custom-revision1-independent-review';V=T/'vector-custom-revision1-root-review'
BASE=T/'vector-finite-acceptance-revision2-wip';FOCUS=T/'vector-pointer-totalization-review1-wip'
sys.path.insert(0,str(R/'bench'));from adapter_common import hash_file,load,write_json,declarations
assert not V.exists()
rh='261d2be00ae6152fbbad9fc6c540ae96d0ff653198e9ae3ea7533878be7de8e1'
assert hash_file(P/'evidence/receipt.json')==rh
assert hash_file(PEER/'peer-review.json')=='ca8752a1d50fb48e02499b6aa8f63a53690c513818720e7a8c92db2f93ef595e'
assert hash_file(PEER/'review-receipt.json')=='a974abed17dba92df7d3cd0b06aab552457a36b0c2548118bb0808a6db907935'
assert hash_file(FOCUS/'receipt.json')=='aa62e23414c2cafe4bc69906f304256e3bdeadc36b9bebe1cb3ebaf97b864d60'
r=load(P/'evidence/receipt.json');pr=load(PEER/'review-receipt.json');review=load(PEER/'peer-review.json')
for folder,receipt in [(P,r),(PEER,pr),(FOCUS,load(FOCUS/'receipt.json'))]:
 for rel,h in receipt['artifacts'].items():assert hash_file(folder/rel)==h,rel
for rel,h in r['modules'].items():assert hash_file(P/'after'/rel)==h,rel
assert not review['blockers'] and review['candidate_and_repository_dependencies_unchanged']
assert review['verification']['modules_rebuilt']==7 and review['verification']['candidate_named_clean_axiom_checks']==44
assert review['verification']['candidate_noncircular_checks']==6
deps=load(PEER/'dependencies-before.json');assert deps==load(PEER/'dependencies-after.json')
for rel,v in deps['project'].items():
 assert hash_file(Path(v['source']))==v['sha256'],rel
 assert hash_file(Path(v['olean']))==v['olean_sha256'],rel
decl=load(P/'evidence/declaration-proposal.json')
assert declarations(P/'after'/decl['file'])[decl['declaration']]['statement_sha256']==decl['statement_sha256']
assert decl['statement_sha256']=='cf664860c1a6ec09372f7f7a4624dc93c1acac42875db6170666b825e7d07a84'
delta=load(P/'evidence/numerical-authority-delta.json')
assert (len(delta['rules']),len(delta['provenance']),sum(len(x['cases']) for x in delta['requirements']))==(37,296,4498)
V.mkdir();diffs={}
for rel in r['modules']:
 oldrel=rel.replace('/vector_addition_custom/','/vector_addition/')
 old=(BASE/'after'/oldrel).read_text();new=(P/'after'/rel).read_text()
 normalized=new.replace('bench.tritonbench_g.vector_addition_custom.','bench.tritonbench_g.vector_addition.').replace('VeriTile.Bench.TritonBenchG.VectorAdditionCustom.Native','VeriTile.Bench.TritonBenchG.VectorAddition.Native')
 dest=V/(Path(rel).stem+'.normalized.diff')
 dest.write_text(''.join(difflib.unified_diff(old.splitlines(True),normalized.splitlines(True),fromfile=oldrel,tofile=rel)))
 diffs[rel]={'path':dest.name,'sha256':hash_file(dest),'vector_revision2_source_sha256':hash_file(BASE/'after'/oldrel),'custom_source_sha256':r['modules'][rel]}
semantics={
 'substantive_review':'Root read original custom_add Python and all seven complete normalized module diffs against the already independently rebuilt and substantively reviewed vector revision2. Root additionally read the complete custom host/execution definitions, public statement, independent peer source review and focused pointer-totalization argument. The five scalar/execution/metadata modules preserve the shared execution semantics with namespace qualification and explanatory comments; NativeArithmetic replaces1024 with16 and proves corresponding bounds; NativeCorrect changes the source host path. No producer-only approval is substituted for this review.',
 'source_host':'The actual wrapper first calls empty_like(a), then c.size(0), sets BLOCK16, constructs ceil-div grid from that first dimension, launches and returns exactly c. Rank-zero failure follows allocation. There is no added CUDA assertion, same-device condition, y-size/shape check or contiguous-input premise. Actual preserve-format output strides, input aliases, offsets and freshness remain explicit. First dimension and numel remain separate, including huge [2^31,0] null-pointer tensors.',
 'integer_domain':'The full nonnegative signed64 first-dimension domain is retained. The annotated runtime argument narrows modulo2^32 after actual signed64 host extraction; pid*16+lane wraps at i32, masks use signed32 and byte addresses use sign extension/scaling/u64 wrap. The original dimension determines grid, with exact signed-grid threshold16*(2^31-1). Compilation and grid parsing precede pointer packing. At first dimension2^31 every mask is false and the actual allocated tensor/memory returns without input-access or numerical assumptions when ordinary setup succeeds. Repeated/negative requests are represented explicitly rather than removed by a no-wrap restriction.',
 'execution_values_and_frames':'All x loads in a tile precede all y loads, followed by source scalar arithmetic and stores. The independent plan uses the initial allocated snapshot; proven freshness/frame preservation relates later loads to that snapshot. The scalar implementation reuses installed finite vector formulas whose exact typed dataflow was independently matched across189 custom frontend routes. Undefined conversions become observations before storing any fallback value. An independent last-store byte formula and pure width-specific packing describe successful writes and all unchanged bytes; exterior allocation frames compose with the host return.',
 'metadata_boundary':'Raw Torch data/storage pointers and actual successful extracted launch pointers are distinct. Mapped allocation offsets and launch-device permissions are ordinary trusted memory metadata; owner equality grants no access. Native-coordinate typed memory is an explicit local byte/load/store boundary, with no arbitrary tagged-carrier roundtrip or final-result oracle. On absent/failed extraction the total pointer metadata is unused and does not assert Result.ok; the focused independent note demonstrates satisfiable ordinary raw-pointer completions. Actual successful extraction must still use the actual mapped pointer. Root agrees this interpretation loses no identified pre-extraction source branch and does not infer native CUDA success.',
 'outcome_limits':'Missing bytes, inaccessible spans, invalid encodings, undefined conversions and conflicting output ownership are mathematical observations only. They claim no native trap, race result, rollback, postfailure memory or launch reachability. Complete grid execution and successful final-byte/frame results remain separate from those observations. The environment controls only fallible ordinary setup, not values or kernel results.',
 'coverage_and_numerics':'The exact22x22 original dtype matrix retains195 binder,64 target-signature,36 addition and6 backend failures plus183 backend routes. All4498 cases include stage/dtype/branch bindings;183 exact2^31 return cases require zero numerical ports.37 scalar families/296 exact port instances remain deferred. Historical five-module/616-port custom work remains uninstalled and unregistered. Existing authority is additive and current integration queue equality remains mandatory.',
 'verification_attribution':'Peer independently rebuilt all7 exact modules, reproducing objects, with44 clean/6 noncircular checks; six extra Lean regressions and484-pair recorded admission equality passed. It replayed498 dtype/ABI profiles and991 IR artifacts,189 scalar fingerprints,22272 C pointer cases,15 AST traces,6 CPU metadata and7 grid-parser probes. All296 omissions and20 adapter mutations failed closed. Root reverified every frozen producer/peer/focused artifact,200 unchanged project source/object pairs, public statement and numerical counts. This is not represented as a second root rebuild or replay. Failed peer comparison/namespace attempts remain frozen.',
 'limits':'No GPU launch, live launch-device accessibility measurement, native scalar validation or official comparator pass is established. The pinned finite source/primitive boundary applies. Repository integration and literal authority pins remain pending; this review increments accepted task count by zero.'}
out={'schema':1,'reviewer':'root','status':'substantively_reviewed_for_finite_scope_integration','task':'vector_addition_custom','revision':1,'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'producer_receipt_sha256':rh,'peer_review_sha256':hash_file(PEER/'peer-review.json'),'peer_receipt_sha256':hash_file(PEER/'review-receipt.json'),'focused_pointer_review_receipt_sha256':hash_file(FOCUS/'receipt.json'),'reviewed_candidate_source_sha256':r['modules'],'public_statement':{k:decl[k] for k in ['file','declaration','statement_sha256']},'semantic_review':semantics,'verification':review['verification'],'verified_artifact_counts':{'producer':len(r['artifacts']),'independent_peer':len(pr['artifacts'])},'normalized_source_diffs':diffs,'current_repository_integration':'pending','native_execution':False,'numerical_conformance_validated':False,'acceptance_increment_before_integration':0}
write_json(V/'review.json',out);shutil.copyfile(Path(__file__),V/'review-package.py')
write_json(V/'review-receipt.json',{'schema':1,'frozen':True,'producer_receipt_sha256':rh,'review_sha256':hash_file(V/'review.json'),'artifacts':{p.name:hash_file(p) for p in V.iterdir() if p.is_file()},'source_acceptance_increment':0})
print(json.dumps({'review_sha256':hash_file(V/'review.json'),'receipt_sha256':hash_file(V/'review-receipt.json'),'status':'approved_for_integration_only'},indent=2))
