from pathlib import Path
from datetime import datetime, timezone
import json, sys, shutil
sys.dont_write_bytecode = True
T = Path('/tmp/veritile-standardization-20261015')
R = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
P = T / 'cosine-native-finite-acceptance-revision1-wip'
PEER = T / 'cosine-revision1-independent-review'
V = T / 'cosine-revision1-root-review'
sys.path.insert(0, str(R / 'bench'))
from adapter_common import hash_file, load, write_json, declarations
assert not V.exists()
rh = '9b62ea3b598a6ac164f91333e2fa3e98cb634cf020e83247014687479bc0433a'
ph = '13a7392abd074eec9ec09eeeceefcf5846ec2862e751ea7f227a9e699995a9be'
prh = '968386303a7d39705cfa74ed2d3423439aac4ee9c46b46fc02e459dd0dfe36bb'
assert hash_file(P / 'evidence/receipt.json') == rh
assert hash_file(PEER / 'peer-review.json') == ph
assert hash_file(PEER / 'review-receipt.json') == prh
r = load(P / 'evidence/receipt.json')
pr = load(PEER / 'review-receipt.json')
review = load(PEER / 'peer-review.json')
for folder, receipt in [(P, r), (PEER, pr)]:
    for rel, h in receipt['artifacts'].items():
        assert hash_file(folder / rel) == h, rel
for rel, h in r['modules'].items():
    assert hash_file(P / 'after' / rel) == h, rel
    assert hash_file(P / r['checkpoint_module_source_paths'][rel]) == h, rel
assert not review['blocking_findings']
assert review['producer_receipt_sha256'] == rh
assert review['checks']['modules'] == 9
assert review['checks']['clean_axiom_checks'] == 48
assert review['checks']['noncircular_checks'] == 7
deps = load(PEER / 'dependencies-before.json')
assert deps == load(PEER / 'dependencies-after.json')
for rel, item in deps['project'].items():
    assert hash_file(Path(item['source'])) == item['sha256'], rel
    assert hash_file(Path(item['olean'])) == item['olean_sha256'], rel
for rel, item in deps['external_objects'].items():
    assert hash_file(Path(item['olean'])) == item['sha256'], rel
build = load(PEER / 'independent-build-receipt.json')
assert build['status'] == 'passed' and build['dependencies_unchanged']
assert len(build['modules']) == 9
for item in build['modules']:
    assert item['exit_code'] == 0
    assert item['sha256'] == r['modules'][item['source']]
    assert item['object_sha256'] == r['module_objects'][str(Path(item['source']).with_suffix('.olean'))]
    assert hash_file(PEER / item['log']) == item['log_sha256']
decl = load(P / 'evidence/declaration-proposal.json')
statement = declarations(P / 'after' / decl['file'])[decl['declaration']]['statement_sha256']
assert statement == decl['statement_sha256'] == '1455600d97339fdf59e2297584b93dd0c694771ab2bd3f58326d72ad70855ed8'
delta = load(P / 'evidence/numerical-authority-delta.json')
assert (len(delta['rules']), len(delta['provenance']), len(delta['requirements'][0]['cases'])) == (17, 534, 268)
host = {'CosineNative.HostMath.integer64', 'CosineNative.HostMath.sqrtSignificand'}
host_bindings = {k: p for k, p in delta['provenance'].items() if p['callback_declaration'] in host}
assert len(host_bindings) == 492
allocation_cases = 0
for case in delta['requirements'][0]['cases']:
    callbacks = set(case['required_callbacks'])
    if case['configuration']['outcome'] == 'allocation_error':
        allocation_cases += 1
        assert not callbacks and not case['required_binding_refs']
    else:
        assert host <= callbacks
assert allocation_cases == 22
for port in delta['provenance'].values():
    assert port['status'] == 'explicitly_deferred_unvalidated'
    assert port['validation_evidence'] is None
    assert not port['native_execution']
    pin = port['original_source']
    assert hash_file(R / pin['path']) == pin['sha256']

semantics = {
    'substantive_review': 'Root read the unchanged original Python, all nine candidate Lean modules in full, the authority adapter, producer README, independent source review, public statement and exact host/kernel registration examples. The source AST is lowered and executed; the resulting byte formula is independently defined from initial memory. This is not acceptance by hashes or successful compilation alone.',
    'source_and_precision': 'The five original statements retain dynamic BLOCK, axis0 pid, i32 arithmetic, signed mask, masked load, explicit conversion to FP32, cos32 and masked output store. Source prefix execution is proved equal to closed finite requests; the scalar expression is extracted from instruction3. FP32 uses cos32 directly; BF16 uses bfloatTo32 then cos32 then toBFloat. FP64 also narrows to FP32, as written. All active tile inputs are loaded before scalar/store processing. Other completed type evidence is retained without fresh dtype expansion.',
    'host_and_failures': 'empty_like precedes numel, host integer conversion/sqrt/ceil/next-power, cdiv, ordered JIT/native setup, launch and return of that same allocation. Empty input fails at cdiv(0,0) after allocation and before JIT. There is no invented CUDA, rank, contiguous, same-device or caller-safety assertion. On admitted dense tensors host numel has its nonnegative signed64 range. Unsupported binder/architecture and oversized BLOCK scalar/shape stages remain distinct; compilation rejects huge cosine blocks before any putative grid-parser overflow case.',
    'host_numerics': 'HostMath.integer64 implements nearest-even nonnegative integer-to-binary64 rounding. sqrtSignificand uses exact scaled Nat square root and squared-midpoint comparison; ceilSqrt and Triton nextPower preserve represented-value behavior. At 2^52+1 this differs from replacing host math by ideal Real.sqrt. Native CPython conversion and libm sqrt correspondence are explicitly deferred at FE_TONEAREST, with all246 post-allocation cases carrying both exact host ports, including later rejection and zero cases. The22 allocation failures consume neither. The210943 producer samples and24 peer boundaries are not full conformance proofs.',
    'finite_memory': 'Wrapping i32 offset and signed mask feed sign-extended width-scaled u64 pointers. At2^31 all masks are false; successful setup returns the exact allocated tensor and unchanged bytes without input access assumptions. Torch storage base/data and extracted launch pointers are separate mapped metadata. Unconsumed affine witnesses on failure paths do not assert successful extraction. WordMemory requires exact raw little-endian input/store bits and byte presence; it cannot supply cosine values. No separate native-to-canonical address refinement is asserted.',
    'complete_execution': 'Ordinary empty_like freshness permits the independently defined initial-memory plan for every complete pid permutation. The public result includes actual output metadata, exact last-touch raw byte formula, untouched allocation bytes and exterior/input frame. Access, missing-byte, undefined conversion and ownership conflicts are applicability observations; no native trap, rollback, race winner or postfailure memory is claimed. The finite source theorem remains distinct from a full mathematical Real version.',
    'numerical_authority': '268 exact declaration cases bind17 families and534 instances:492 host and42 kernel bindings. All are explicitly unvalidated. Receipt, root review, original source, module/dependency sources, raw IR, native records, host probe and exact public statement are bound by the additive adapter. The unchanged exact queue gate must still reject omitted or mismatched ports; integration must check whole-queue equality. Deferred conformance does not waive source, memory or host composition.',
    'verification_attribution': 'Peer independently rebuilt9 modules with identical objects,48 clean-axiom,7 noncircular and9 module checks;12 targeted regressions passed. Selective native replay used16 FP32/BF16 frontend profiles and2 offline backend compiles. All534 omission,21 adapter and5 translation mutations were rejected. Root rehashed651 producer artifacts and100 peer artifacts,201 current project source/object dependencies and37 external objects, plus exact statement and host-binding coverage. Root did not perform an additional native replay or Lean rebuild. Producer dependency capture was before final rebuild after iterative checks; peer capture preceded its first independent import/build.',
    'limits': 'No GPU execution, native numerical conformance or current official-comparator pass is established. Existing sources, wider proofs and failed attempts are retained. This review authorizes reviewed finite-scope integration only and adds zero accepted programs by itself. Three-primary-version applicability and Real/FP32/BF16 completion require explicit mapping under the new policy.'
}
assert len(r['artifacts']) == 651 and len(pr['artifacts']) == 100
assert len(deps['project']) == 201 and len(deps['external_objects']) == 37
V.mkdir()
out = {
    'schema': 1, 'reviewer': 'root', 'status': 'substantively_reviewed_for_finite_scope_integration',
    'task': 'cosine_compute', 'revision': 1, 'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
    'producer_receipt_sha256': rh, 'peer_review_sha256': ph, 'peer_receipt_sha256': prh,
    'reviewed_candidate_source_sha256': r['modules'],
    'public_statement': {k: decl[k] for k in ['file', 'declaration', 'statement_sha256']},
    'semantic_review': semantics, 'verification': review['checks'],
    'verified_artifact_counts': {'producer': len(r['artifacts']), 'independent_peer': len(pr['artifacts'])},
    'current_repository_integration': 'pending', 'primary_version_mapping': 'pending',
    'native_execution': False, 'numerical_conformance_validated': False,
    'acceptance_increment_before_integration': 0
}
write_json(V / 'review.json', out)
shutil.copyfile(Path(__file__), V / 'review-package.py')
write_json(V / 'review-receipt.json', {
    'schema': 1, 'frozen': True, 'producer_receipt_sha256': rh,
    'review_sha256': hash_file(V / 'review.json'),
    'artifacts': {p.name: hash_file(p) for p in V.iterdir() if p.is_file()},
    'source_acceptance_increment': 0
})
print(json.dumps({'review_sha256': hash_file(V / 'review.json'), 'receipt_sha256': hash_file(V / 'review-receipt.json'), 'status': 'approved_for_finite_scope_integration_only'}, indent=2))
