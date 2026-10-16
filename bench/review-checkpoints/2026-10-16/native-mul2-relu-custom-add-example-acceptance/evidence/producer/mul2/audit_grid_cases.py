import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib, importlib.util, json

W = Path(__file__).resolve().parent
R = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sys.path.insert(0, str(R / 'bench'))
load = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
cases = load(W / 'evidence/numerical-authority-delta.json')['requirements'][0]['cases']
bykey = {}
for case in cases:
    c = case['configuration']
    key = (case['dtype_profile']['inputs']['x'], c['mode'], c['block_class'], c['size_class'], c['outcome'])
    assert key not in bykey
    bykey[key] = case
admitted = sorted({k[0] for k in bykey if k[-1] == 'returned_active'})
assert len(admitted) == 13
replay = W / 'evidence/pointer-replay'
so = next(replay.glob('pointer_probe*.so'))
spec = importlib.util.spec_from_file_location('pointer_probe', so)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
M = 2**31 - 1
sizes = [('empty', 0, 1), ('narrow_positive', 1, 2**31),
         ('i32_minimum', 2**31, 2**31 + 1), ('wide', 2**31 + 1, 2**63)]
def block_class(k):
    return {4: 'default', 9: 'harness512', 10: 'harness1024'}.get(k, 'other_valid')

# The Lean theorem proves this split for every n, not merely the endpoints.
# Each full size interval is partitioned into at most two exact integer spans.
partitions = []
native_checks = 0
case_checks = 0
grid_case_keys = set()
for k in range(21):
    b = 2**k
    threshold = M*b
    for size, lo, hi in sizes:
        pieces = []
        for fit, left, right in [(True, lo, min(hi, threshold+1)),
                                 (False, max(lo, threshold+1), hi)]:
            if left >= right:
                continue
            assert lo <= left < right <= hi
            points = sorted({left, right-1, (left+right-1)//2})
            probes = []
            for n in points:
                grid = (n+b-1)//b
                assert (grid <= M) == fit == (n <= threshold)
                try:
                    parsed = native.grid(grid)
                    actual = True
                    assert parsed == grid
                except OverflowError:
                    actual = False
                assert actual == fit
                native_checks += 1
                probes.append({'numel': n, 'grid': grid, 'signed_i32_parser_accepts': actual})
            outcome = 'returned_masked' if size in ['empty', 'i32_minimum'] else 'returned_active'
            for dtype in admitted:
                for mode in ['outplace', 'inplace']:
                    key = (dtype, mode, block_class(k), size, outcome if fit else 'grid_parse_rejected')
                    assert key in bykey, key
                    case_checks += 1
                    if not fit:
                        grid_case_keys.add(key)
                        assert not bykey[key]['required_binding_refs'] and not bykey[key]['required_callbacks']
                    if fit and size in ['empty', 'i32_minimum']:
                        assert not bykey[key]['required_binding_refs'] and not bykey[key]['required_callbacks']
            pieces.append({'lower_inclusive': left, 'upper_exclusive': right,
                           'grid_fits': fit, 'endpoint_and_midpoint_probes': probes})
        assert pieces[0]['lower_inclusive'] == lo and pieces[-1]['upper_exclusive'] == hi
        assert all(a['upper_exclusive'] == z['lower_inclusive'] for a, z in zip(pieces, pieces[1:]))
        partitions.append({'block': b, 'exponent': k, 'block_class': block_class(k),
                           'size_class': size, 'threshold_inclusive': threshold, 'partitions': pieces})

actual_grid_keys = {key for key in bykey if key[-1] == 'grid_parse_rejected'}
assert actual_grid_keys == grid_case_keys, (actual_grid_keys-grid_case_keys, grid_case_keys-actual_grid_keys)
minimum = [c for key, c in bykey.items() if key[3:] == ('i32_minimum', 'grid_parse_rejected')]
assert len(minimum) == 26
assert all(c['configuration']['block_class'] == 'other_valid' and
           not c['required_binding_refs'] and not c['required_callbacks'] for c in minimum)
# Zero and invalid blocks cannot reach native grid parsing. Signature/arch
# rejection precedes frontend block validation for unsupported dtype routes.
first_stage_checks = 0
all_dtypes = sorted({k[0] for k in bykey})
for dtype in all_dtypes:
    for mode in ['outplace', 'inplace']:
        for size, _, _ in sizes:
            for bc in ['zero', 'invalid_integer']:
                outcomes = {k[-1] for k in bykey if k[:4] == (dtype, mode, bc, size)}
                expected = {'allocation_error'} if mode == 'outplace' else set()
                if bc == 'zero':
                    expected.add('division_by_zero')
                else:
                    expected.add('prepare_error')
                    expected.add('architecture_rejected' if dtype.endswith('fnuz') else
                                 'block_frontend_rejected' if dtype in admitted+['fp8e4m3fn', 'fp8e5m2'] else
                                 'signature_rejected')
                assert outcomes == expected, (dtype, mode, bc, size, outcomes, expected)
                first_stage_checks += 1

source = 'bench/tritonbench_g/triton_mul2/Proofs/FinitePointerBoundary.lean'
report = {'status': 'passed', 'revision': 2,
          'method': 'Exact integer interval partition using the compiled Lean host_grid_threshold theorem; actual CPython signed-i32 parser checked at each partition endpoint and midpoint.',
          'lean_theorem': 'TritonMul2Finite.host_grid_threshold',
          'theorem_source': source, 'theorem_source_sha256': sha(W/source),
          'valid_power_of_two_blocks': 21, 'size_classes': 4,
          'full_size_block_partitions': len(partitions), 'native_parser_checks': native_checks,
          'case_path_checks': case_checks, 'first_failure_stage_checks': first_stage_checks,
          'grid_rejection_schema_cases': len(grid_case_keys), 'added_minimum_B1_cases': len(minimum),
          'minimum_size_result': 'B1 rejects before pointer setup; all20 other valid powers fit and remain all-masked with no numerical ports.',
          'limitations': 'Parser source plumbing and symbolic finite source proof; no GPU launch or native scalar validation.',
          'partitions': partitions}
(W/'evidence/grid-case-audit.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k: v for k, v in report.items() if k != 'partitions'}))
