from pathlib import Path
from datetime import datetime,timezone
import json,sys,shutil
sys.dont_write_bytecode=True
T=Path('/tmp/veritile-standardization-20261015');R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
P=T/'triton-mul2-finite-acceptance-revision2-wip';PEER=T/'triton-mul2-revision2-independent-review';V=T/'triton-mul2-revision2-root-review'
sys.path.insert(0,str(R/'bench'));from adapter_common import hash_file,load,write_json,declarations
assert not V.exists();V.mkdir()
rh='7644e8a8dadc81452975c71b662ff1ae587bb6ceab8fd07c94e996ce0be75a4f'
assert hash_file(P/'evidence/receipt.json')==rh
assert hash_file(PEER/'peer-review.json')=='387fad3157ec542d422fcef6e0b692610ab0a6856246d904a184c025e4237804'
assert hash_file(PEER/'review-receipt-final.json')=='4dcbdb64715e0480706fb2499e14ccbbc618ba23c492e17386d63f9cba2f30a4'
r=load(P/'evidence/receipt.json');pr=load(PEER/'review-receipt-final.json');review=load(PEER/'peer-review.json')
for rel,h in r['artifacts'].items():assert hash_file(P/rel)==h,rel
for rel,h in pr['artifacts'].items():assert hash_file(PEER/rel)==h,rel
for rel,h in r['modules'].items():assert hash_file(P/r['producer_module_source_paths'][rel])==h,rel
assert not review['blockers'] and review['candidate_and_repository_unchanged']
assert review['verification']['modules_rebuilt']==9 and review['verification']['candidate_named_clean_axiom_checks']==36
assert review['verification']['candidate_noncircular_checks']==7
old=load(P/r['revision_lineage']['prior_receipt'])
unchanged=[p for p,h in r['modules'].items() if old['modules'].get(p)==h]
assert {Path(p).stem for p in unchanged}=={'FiniteScalar','FiniteSource','FiniteGrid'}
deps=load(P/'evidence/dependency-closure.json')
for rel,v in deps['project'].items():assert hash_file(R/rel)==v['sha256'],rel
contract=r['public_contracts'][0]
assert declarations(P/r['producer_module_source_paths'][contract['file']])[contract['declaration']]['statement_sha256']==contract['statement_sha256']
delta=load(P/'evidence/numerical-authority-delta.json')
assert (len(delta['rules']),len(delta['provenance']),sum(len(x['cases']) for x in delta['requirements']))==(4,64,4674)
semantics={
 'substantive_scope':'Root previously read all eight v1 modules and the original two wrappers/kernels. In revision2 root read all six changed/new modules in full, including Memory, Program, Host, PointerBoundary, Refinement and public FiniteCorrect. Scalar, exact original AST interpreter and Grid are byte-identical to v1. The public model covers both zeros_like outplace and exact-input inplace return, all canonical dtypes, arbitrary Python integer block arguments, empty routes and nonnegative signed64 host extents without a narrower size or arithmetic-refinement premise.',
 'ownership_repair':'Allocation owner metadata is not an access predicate. AccessContext checks each actual byte against the current launch device. Peer and mapped host permissions are representable; equal owner does not manufacture accessibility. All accesses are structural obligations or explicit interpreter observations, never numerical callbacks.',
 'pointer_translation_repair':'Runtime.pointers returns extracted ABI pointer and launch-space base metadata, distinct from original raw Torch pointers. One shared launch-context address map resolves loads to canonical backing memory and every successful store retains/checks both device and backing addresses. The pointer contract preserves same-object aliases and conditional nonnull byte mapping. Null pointers impose no unused mapping requirement. Independent final-byte results and exterior frames refer to canonical backing memory, so an in-place translated store is visible to later alias loads while returned Tensor metadata remains original.',
 'integer_and_grid_scope':'Annotated n_elements and every pid/lane index operation use actual i32 narrowing/wrap before signed masks; pointers use u64 arithmetic. N=2^31 masks every lane with no input-access/numerical premise when a launch is reached. Exact host_grid_threshold and the table retain B=1 grid rejection before pointers, while B>=2 supported powers admit this no-access route. Native preparation/compilation/context are fallible primitive boundaries; the grid parser precedes pointer extraction.',
 'execution_and_failure_boundary':'Each complete input tile is loaded before stores. The host schedule is a permutation of every pid exactly once. The independent source plan determines all concrete writes with canonical-byte ownership; pointwise last-write equality includes untouched and exterior bytes. Access/missing-byte/ownership failures are model observations and claim no native trap, race winner, rollback or postfailure memory. Ordinary metadata/allocation, CUDA address translation and actual permissions are documented trusted primitives, not output or loaded-value oracles.',
 'numerical_scope':'Only four storage-width floating multiply families are deferred through64 exact ports in4674 cases. Signed/unsigned and bool bit behavior are concrete. Missing ports and stale pointer/grid evidence are rejected by the exact adapter. Current integrated queue equality remains mandatory before acceptance.',
 'verification':'The independent peer rebuilt all9 exact modules with36 clean-axiom/7 noncircular checks and four extra Lean translated-address/grid regressions; replayed394 pointer/wrapper cases,84 size/block partitions and228 real parser boundaries; reviewed4674 scope cases and rejected64 omitted ports plus15 adapter mutations. Root reverified every frozen producer and final peer artifact, unchanged dependency sources, unchanged three modules and the public statement. These peer tests are not represented as a second root execution.',
 'limits':'No GPU execution, live CUDA accessibility measurement, native scalar validation or official comparator pass is established. The documented finite source boundary and exact current repository integration checks govern acceptance.'}
out={'schema':1,'reviewer':'root','status':'substantively_reviewed_for_finite_scope_integration','task':'triton_mul2','revision':2,'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'producer_receipt_sha256':rh,'peer_review_sha256':hash_file(PEER/'peer-review.json'),'peer_receipt_sha256':hash_file(PEER/'review-receipt-final.json'),'reviewed_candidate_source_sha256':r['modules'],'public_statement':contract,'semantic_review':semantics,'verification':review['verification'],'verified_artifact_counts':{'producer':len(r['artifacts']),'independent_peer':len(pr['artifacts'])},'current_repository_integration':'pending','native_execution':False,'numerical_conformance_validated':False,'acceptance_increment_before_integration':0}
write_json(V/'review.json',out);shutil.copyfile(Path(__file__),V/'review-package.py')
write_json(V/'review-receipt.json',{'schema':1,'frozen':True,'producer_receipt_sha256':rh,'review_sha256':hash_file(V/'review.json'),'artifacts':{p.name:hash_file(p) for p in V.iterdir() if p.is_file()},'source_acceptance_increment':0})
print(json.dumps({'review_sha256':hash_file(V/'review.json'),'receipt_sha256':hash_file(V/'review-receipt.json'),'status':'approved_for_integration_only'},indent=2))
