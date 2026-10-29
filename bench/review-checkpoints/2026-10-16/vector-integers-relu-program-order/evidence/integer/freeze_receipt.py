"""Record the completed integer/bool batch without changing the prior package."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
old = root.parent / 'vector-addition-wip'
audit = root / 'evidence/extension-audit'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
dump = lambda p, d: p.write_text(json.dumps(d, indent=2) + '\n')
old_scope = json.loads((old / 'evidence/scope-review.json').read_text())
promotion = json.loads((audit / 'promotion-matrix.json').read_text())
frontend = {(r['x'], r['y']): r for r in json.loads((audit / 'frontend-probes.json').read_text())}
ints = {'bool', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64'}
typed_floats = {'float16', 'bfloat16', 'float32', 'float8_e4m3fn', 'float8_e5m2'}
dtype_rows = []
for row in promotion:
    x, y = row['x'], row['y']
    front = frontend[x, y]
    if x in ints and y in ints:
        category = 'new_integer_bool_logical_value_and_ordered_grid_proved'
        remaining = ['source/native memory and finite-width address/error bridges']
    elif x in typed_floats and y in typed_floats:
        category = 'prior_frozen_five_format_logical_value_and_ordered_grid_proved'
        remaining = ['prior frozen scope numerical and structural obligations']
    elif row['status'] == 'promotion_error':
        category = 'promotion_error_observed_not_formally_proved'
        remaining = ['source-bound host/compiler error theorem and architecture/runtime binding']
    elif 'fnuz' in x or 'fnuz' in y:
        category = 'semantically_promoted_but_sm90_frontend_rejected_not_formally_proved'
        remaining = ['distinct fnuz format or exact architecture-dependent source error theorem']
    elif x in ints or y in ints:
        category = 'mixed_float_integer_frontend_route_audited_value_proof_pending'
        remaining = ['direct signed/unsigned integer-to-promoted-float conversion',
                     'promoted floating add and exact store cast route',
                     'defined-domain float-to-integer conversion and unspecified masked lanes']
    else:
        assert x == 'float64' or y == 'float64'
        category = 'fp64_float_frontend_generated_value_proof_pending'
        remaining = ['explicit fp64 typed provenance and binary64 load/add/cast/store graph']
    dtype_rows.append({**row, 'frontend_status': front['status'], 'coverage': category,
                       'remaining': remaining})

counts = {
    'canonical_scalar_signatures': 17,
    'promotion_pairs_audited': len(promotion),
    'promoted_pairs_observed': sum(r['status'] == 'promoted' for r in promotion),
    'promotion_errors_observed': sum(r['status'] == 'promotion_error' for r in promotion),
    'exact_source_frontend_profiles': len(frontend),
    'frontend_generated': sum(r['status'] == 'frontend_generated' for r in frontend.values()),
    'frontend_errors': sum(r['status'] == 'frontend_error' for r in frontend.values()),
    'new_integer_bool_pairs_with_logical_value_and_grid_proof': 81,
    'prior_floating_pairs_with_logical_value_and_grid_proof': 25,
    'cumulative_pairs_with_logical_value_and_grid_proof': 106,
    'promoted_pairs_without_logical_value_execution': 111,
    'remaining_mixed_float_integer_pairs': 72,
    'remaining_supported_frontend_fp64_float_pairs': 11,
    'remaining_promoted_fnuz_frontend_rejections': 28,
    'all_fnuz_pairs_rejected_at_sm90': 64,
    'native_backend_compilations': 0,
    'gpu_executions': 0,
    'whole_original_program_completion_increment': 0,
}
assert counts['promoted_pairs_observed'] == 217
assert counts['promotion_errors_observed'] == 72
assert counts['frontend_generated'] == 189 and counts['frontend_errors'] == 100
categories = Counter(r['coverage'] for r in dtype_rows)
assert sorted(categories.values()) == [11, 25, 28, 72, 72, 81]

public = {
    'module': 'bench.tritonbench_g.vector_addition.IntegerValueCorrect',
    'declaration': 'VeriTile.Bench.TritonBenchG.VectorAddition.IntegerValueCorrect.correct',
    'kind': 'ordinary theorem', 'specification_command': False,
    'bit_interpreter_closed_command': False,
    'specification': 'concrete BitVec casts and modular sum from initial logical memory',
    'floating_callback_assumptions': [],
    'algebra_parameter': 'arbitrary and unused by this integer graph; no binary/cast floating primitive occurs',
    'native_conformance_proved': False,
}
numerical = old_scope['numerical_assumptions'] + [{
    'id': 'VA-N5', 'status': 'registered_for_pending_mixed_value_proofs_not_assumed_by_integer_theorem',
    'primitive': 'direct signed/unsigned integer conversion to fp16/fp32/fp64, promoted add and toward-zero float-to-integer store',
    'requirements': [
        'integer-to-fp16 is direct; do not introduce fp32 intermediate rounding',
        'bf16 mixed with integer promotes to fp32',
        'fp16 arithmetic result extends to fp32 before fptosi/fptoui',
        'bool output is fptosi to signed8; not a nonzero predicate',
        'NaN/Inf/out-of-range conversions need an explicit undefined-result policy and a proved active defined-domain independence theorem',
        'do not require masked inactive values to be finite or claim a specific undefined native result',
    ],
}]
structural = [
    {'id': 'VA-I-S1', 'gap': '111 semantically promoted pairs have no logical value/grid proof',
     'detail': '72 mixed float/integer, 11 supported fp64 float pairs, 28 fnuz promoted pairs rejected by this sm90 frontend; per-pair entries below.'},
    {'id': 'VA-I-S2', 'gap': 'Source/native compiler and error refinement',
     'detail': '289 actual AST-identical-source frontend profiles generated 189 TTIR modules and 100 errors. This is an observation at pinned Triton/sm90/options, not a native backend compilation, universal rejection theorem or source interpreter refinement.'},
    {'id': 'VA-I-S3', 'gap': 'Native finite-width address and debug/sanitizer behavior',
     'detail': 'Model pid/block_start/offsets/n use Nat. Native n_elements is i32 and default debug=false,sanitize_overflow=true remains frozen. TTIR overflow computations do not establish native traps/error behavior. No address overflow guard was added to the source.'},
    {'id': 'VA-I-S4', 'gap': 'Native bytes, views, pointer widths and signed value representability',
     'detail': 'The new logical int/nat cells and BitVec casts are explicit. Signed/unsigned loads require an unproved native byte/tag representability relation; Cell.read mismatch defaults are not native values. Equal logical regions at the same element offset are supported, but physical overlaps with differing element widths, strides or bases are not covered.'},
    {'id': 'VA-I-S5', 'gap': 'Native scheduling',
     'detail': 'This batch proves the complete ascending logical grid with current-memory program reads. Arbitrary native instruction interleavings, sanitizer effects and physical byte races are not established.'},
    {'id': 'VA-I-S6', 'gap': 'Host allocation, dtype dispatch, harness and unsupported dtype domains',
     'detail': 'No changes to prior logical wrapper/harness coverage. Typed dispatch is not composed with native empty_like/allocation/Python/Torch execution. Absent dtype/shape/stride/device guards were not invented. All prior noncanonical dtype/error gaps remain.'},
]
scope = {
    'task': 'vector_addition integer/bool extension and full canonical frontend audit',
    'standard': old_scope['standard'],
    'source_sha256': old_scope['source_sha256'],
    'whole_original_program_accepted': False,
    'completed_original_program_increment': 0,
    'prior_frozen_receipt': {'path': str(old / 'evidence/receipt.json'), 'sha256': sha(old / 'evidence/receipt.json')},
    'prior_frozen_scope': {'path': str(old / 'evidence/scope-review.json'), 'sha256': sha(old / 'evidence/scope-review.json')},
    'counts': counts,
    'new_proved': [
        'all 81 canonical integer/bool ordered-pair promotion/store formats',
        'actual complete eight-statement logical source execution without successful-run premise',
        'shared fixed-width signed/unsigned BitVec casts and promoted-width modular addition',
        'bool i8 load followed by !=0, unsigned i1 arithmetic for bool/bool, destination signed8 storage',
        'closed per-program masked readback/frame and full ascending-grid initial-memory formula',
        'all input loads precede output stores; equal logical x/y/output regions permitted',
    ],
    'frontend_assessment': {
        'fp64': '11 new floating pairs generate frontend IR; typed binary64 value graph remains pending. Generic .real is not fp64 provenance.',
        'fnuz': 'all64 pairs containing fnuz reject at sm90; 28 were semantically promoted and36 overlap promotion errors. Other architectures and source/native error proofs remain open.',
        'mixed': 'all72 routes generated; scalar conversion routes were checked but no mixed whole-program value proof is claimed in this batch.',
        'old_evidence_unchanged': 'Prior13 frontend probes remain in the old frozen package; this new289-profile audit does not overwrite them.',
    },
    'public_contract': public,
    'numerical_assumptions': numerical,
    'structural_remaining': structural,
    'dtype_pairs': dtype_rows,
    'independent_review': 'pending parent integration review',
}
dump(root / 'evidence/scope-review.json', scope)
subprocess.run([sys.executable, str(root / 'freeze.py'), 'after'], check=True)
results = json.loads((root / 'evidence/verification-results.json').read_text())
dependencies = json.loads((root / 'evidence/final-dependencies-after.json').read_text())
assert dependencies['frozen_vector_receipt_sha256'] == 'e93c0e62996286f4b03419dfebb76fe0b2b959b6b6eb2cd1153d7387f73e3d1f'
assert dependencies['frozen_vector_scope_sha256'] == '8e2d77dabddb72fdbea97aac47ac20267246beda6607692b0cad86f8a95e51df'
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
        'escalation_reason': 'sandbox cannot locate existing Lean/Lake; external toolchain used without modification',
    },
    'coverage_summary': counts,
    'source_binding': {
        'python': 'bench/tritonbench_g/vector_addition/vector_addition.py',
        'sha256': old_scope['source_sha256'],
        'exact_source_frontend_ast_identical': True,
        'canonical_promotion': 'IntegerSource.canonical_promotion/canonical_store match frozen DTypeScope rules',
        'source_native_refinement_proved': False,
    },
    'acceptance_limits': structural,
    'prior_frozen_receipt_sha256': dependencies['frozen_vector_receipt_sha256'],
    'prior_frozen_scope_sha256': dependencies['frozen_vector_scope_sha256'],
    'artifacts': artifacts,
}
dump(root / 'evidence/receipt.json', receipt)
print(json.dumps({'receipt_sha256': sha(root / 'evidence/receipt.json'),
                  'scope_sha256': sha(root / 'evidence/scope-review.json'),
                  'verification': receipt['verification'], 'counts': counts}))
