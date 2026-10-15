"""Freeze the concrete mixed fp32 extension after actual complete verification."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
integers = root.parent / 'vector-addition-dtype-extension-wip'
old = root.parent / 'vector-addition-wip'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
dump = lambda p, d: p.write_text(json.dumps(d, indent=2) + '\n')
prior = json.loads((integers / 'evidence/scope-review.json').read_text())
profiles = json.loads((root / 'evidence/source-profile/receipt.json').read_text())
selected = {(r['x'], r['y']) for r in profiles['profiles']}
rows = prior['dtype_pairs']
for r in rows:
    if (r['x'], r['y']) in selected:
        r['coverage'] = 'new_mixed_fp32_concrete_software_value_and_ordered_grid_proved'
        r['remaining'] = ['native scalar numerical binding',
                          'native memory/address/error/masked-poison/scheduling refinement']

counts = {
    'canonical_signatures': 17, 'canonical_pairs': 289,
    'semantically_promoted_pairs': 217, 'observed_promotion_errors': 72,
    'prior_floating_logical_pairs': 25, 'prior_integer_bool_logical_pairs': 81,
    'new_mixed_fp32_logical_pairs': 18, 'cumulative_logical_value_and_grid_pairs': 124,
    'promoted_pairs_without_logical_value_execution': 93,
    'remaining_mixed_fp16_bf16_fp64_pairs': 54,
    'remaining_supported_frontend_fp64_float_pairs': 11,
    'remaining_promoted_fnuz_frontend_rejections': 28,
    'native_backend_compilations': 0, 'gpu_executions': 0,
    'whole_original_program_completion_increment': 0,
}
public = {
    'module': 'bench.tritonbench_g.vector_addition.MixedFP32ValueCorrect',
    'declarations': [
        'vector_addition_mixed_fp32_total_correctness',
        'vector_addition_mixed_fp32_defined_correctness',
    ],
    'specification_commands': 2,
    'bit_interpreter_closed_claim': False,
    'software_primitives': ['FP.round .fp32 cfg of exact integer rational', 'FP.add cfg',
                            'IntegerConversion.toSigned/toUnsigned and explicit totalSigned/totalUnsigned policy'],
    'unused_floating_rest': 'The result formula contains no Algebra callback; the generic rest supplies unreachable operations only.',
    'total_contract': 'Every source bit pattern and masked lane has a totalized logical execution; undefined active conversions retain the explicitly supplied policy.',
    'defined_contract': 'Only active integer-output addresses must satisfy the shared finite toward-zero range specification. Any undefined policy yields the same specified memory.',
    'fp32_output_contract': 'No finiteness or conversion-domain condition is required when x is fp32; the shared FP.add covers exceptional encodings.',
    'native_conformance_proved': False,
}
numerical = [
    {'id': 'VA-M-N1', 'status': 'software_concrete_native_binding_deferred',
     'primitive': 'signed/unsigned integer to fp32',
     'detail': 'One FP.round from the exact integer rational. Signed widths are interpreted with BitVec.toInt, unsigned/bool with BitVec.toNat. Explicit fp32 precision wraps the .real carrier; the final carrier-tag cast is identity and adds no numerical rounding. Native rounding mode/format/representability must be bound.'},
    {'id': 'VA-M-N2', 'status': 'software_concrete_native_binding_deferred',
     'primitive': 'FP.add cfg at binary32',
     'detail': 'Source operand order is preserved for both orientations. Native cfg, FTZ/DAZ, rounding and exceptional payload/signaling behavior remain unproved. No commutativity, real exactness or finite-input law is assumed.'},
    {'id': 'VA-M-N3', 'status': 'software_partial_conversion_and_active_domain_independence_proved_native_binding_deferred',
     'primitive': 'toward-zero fp32 to signed8/16/32/64 or unsigned8/16/32/64',
     'detail': 'Uses shared exact rational toSigned/toUnsigned, with finite/range domain. Bool destination uses signed8 fptosi, so 3 remains3 and128 is undefined. Arbitrary fallback is exposed; the active defined-domain theorem proves no fallback affects a store. No fixed result is claimed on NaN/Inf/overflow.'},
]
structural = [
    {'id': 'VA-M-S1', 'gap': '93 promoted dtype pairs remain without logical value/grid proofs',
     'detail': '54 mixed fp16/bf16/fp64 pairs,11 supported fp64 float pairs,28 promoted fnuz rejections. Direct int→fp16 must not be modeled with extra fp32 rounding; fp16→integer needs the actual extf→fp32 route.'},
    {'id': 'VA-M-S2', 'gap': 'Native source/compiler/error/masked-lane refinement',
     'detail': 'The18 raw exact-source TTIR graphs are copied from the frozen289-profile audit with matching hashes. No native backend/GPU execution or formal source extraction theorem is supplied. In particular, totalized software execution and masked policy independence do not prove native poison propagation, traps, sanitizer behavior or compiler availability.'},
    {'id': 'VA-M-S3', 'gap': 'Native storage and finite-width address metadata',
     'detail': 'Logical Int/Nat/fp32 cells, own-width integer normalization, bool i8 reinterpretation and Nat offsets require byte/tag/representability refinement. Shape/stride/device/contiguity guards absent from Python were not invented. Physical overlapping views with different bases/widths/strides remain open.'},
    {'id': 'VA-M-S4', 'gap': 'Scheduling and host/harness composition',
     'detail': 'This batch has the complete ascending logical grid with current-memory reads and equal logical region aliases. Native instruction interleaving, allocator/empty_like/device behavior, typed native wrapper dispatch and Python/Torch harness execution retain previous structural gaps.'},
]
scope = {
    'task': 'vector_addition concrete mixed integer/bool and fp32 extension',
    'standard': prior['standard'], 'source_sha256': prior['source_sha256'],
    'whole_original_program_accepted': False, 'completed_original_program_increment': 0,
    'counts': counts, 'public_contract': public,
    'numerical_assumptions': numerical, 'structural_remaining': structural,
    'dtype_pairs': rows,
    'inherited_obligations': 'All prior floating/integer scope obligations remain except the18 explicit mixed fp32 logical graphs now proved here.',
    'prior_integer_receipt_sha256': sha(integers / 'evidence/receipt.json'),
    'prior_integer_scope_sha256': sha(integers / 'evidence/scope-review.json'),
    'prior_floating_receipt_sha256': sha(old / 'evidence/receipt.json'),
    'source_route_profile_sha256': sha(root / 'evidence/source-profile/receipt.json'),
    'independent_review': 'pending parent integration review',
}
dump(root / 'evidence/scope-review.json', scope)
subprocess.run([sys.executable, str(root / 'freeze.py'), 'after'], check=True)
results = json.loads((root / 'evidence/verification-results.json').read_text())
dependencies = json.loads((root / 'evidence/final-dependencies-after.json').read_text())
assert dependencies['frozen_vector_receipt_sha256'] == 'e93c0e62996286f4b03419dfebb76fe0b2b959b6b6eb2cd1153d7387f73e3d1f'
assert dependencies['frozen_integer_receipt_sha256'] == '85dc08106852c4a3f69516046182954693eb5afd0158064db358c708a70b9100'
artifacts = {}
for p in sorted(root.rglob('*')):
    if not p.is_file() or p.is_symlink() or p.suffix in {'.olean', '.pyc'} or p.name == 'receipt.json' and p.parent == root / 'evidence':
        continue
    artifacts[str(p.relative_to(root))] = sha(p)
receipt = {
    'task': scope['task'], 'package_root': str(root),
    'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
    'whole_original_program_accepted': False, 'original_program_completion_increment': 0,
    'modules': {r['module']: sha(root / r['module']) for r in results},
    'public_contract': public,
    'verification': {
        'module_count': len(results), 'actual_exit_codes': [r['exit_code'] for r in results],
        'axiom_checks': sum((root / r['log']).read_text().count('axiom footprint') for r in results),
        'noncircularity_checks': sum((root / r['log']).read_text().count('non-circular ✓') for r in results),
        'source_and_dependency_hashes_unchanged': True,
        'dependency_file_count': len(dependencies['files']),
        'external_library_build_rechecked': False,
        'results': 'evidence/verification-results.json',
        'escalation_reason': 'restricted sandbox cannot locate existing Lean/Lake; toolchain used without modification',
    },
    'coverage_summary': counts, 'acceptance_limits': structural,
    'source_binding': {'python': 'bench/tritonbench_g/vector_addition/vector_addition.py',
                       'source_sha256': prior['source_sha256'],
                       'route_profile_sha256': scope['source_route_profile_sha256'],
                       'source_native_refinement_proved': False},
    'prior_integer_receipt_sha256': dependencies['frozen_integer_receipt_sha256'],
    'prior_floating_receipt_sha256': dependencies['frozen_vector_receipt_sha256'],
    'artifacts': artifacts,
}
dump(root / 'evidence/receipt.json', receipt)
print(json.dumps({'receipt_sha256': sha(root / 'evidence/receipt.json'),
                  'scope_sha256': sha(root / 'evidence/scope-review.json'),
                  'verification': receipt['verification'], 'counts': counts}))
