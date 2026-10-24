"""Freeze partial byte execution only from real successful final results."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess
import sys

root = Path(__file__).resolve().parent
repo = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
base = root.parent
words = base / 'relu-integer-word-refinement-wip'
bytes_package = base / 'relu-byte-memory-wip'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
write = lambda p, d: p.write_text(json.dumps(d, indent=2) + '\n')
assert not (root / 'evidence/receipt.json').exists(), 'refuse frozen receipt overwrite'
subprocess.run([sys.executable, '-B', str(root / 'freeze.py'), 'after'], check=True)
sys.path.insert(0, str(repo / 'bench'))
from adapter_common import declarations

rows = read(root / 'evidence/verification-results.json')
history = read(root / 'evidence/verification-attempts.json')
modules = {r['module']: sha(root / r['module']) for r in rows}
logs = [(root / r['log']).read_text() for r in rows]
deps = read(root / 'evidence/final-dependencies-after.json')
failed = [r for r in history if r['exit_code'] != 0]
checks = {'module_count': len(rows), 'actual_exit_codes': [r['exit_code'] for r in rows],
    'seconds': sum(r['seconds'] for r in rows),
    'named_axiom_checks': sum(s.count('axiom footprint') for s in logs),
    'spec_noncircularity_checks': sum(s.count('non-circular ✓') for s in logs),
    'module_axiom_audits': sum((root / p).read_text().count('#auditModuleAxioms') for p in modules),
    'warnings': sum(s.count('warning:') for s in logs),
    'dependency_file_count': len(deps['files']),
    'dependency_breakdown': dict(Counter(v['source_kind'] for v in deps['files'].values())),
    'imported_project_olean_count': sum('olean_sha256' in r for r in deps['files'].values()),
    'inherited_ttir_count': len(deps['audited_ttir_sha256']),
    'source_dependency_and_olean_hashes_unchanged': True,
    'all_sixteen_frozen_relu_source_modules_unchanged': True,
    'actual_lakefile_toml_bound': True, 'external_library_build_rechecked': False,
    'source_whitespace_check': 'All four sources have a final newline and no trailing spaces/tabs.',
    'compiler_results': 'evidence/verification-results.json',
    'all_compile_attempts': 'evidence/verification-attempts.json',
    'attempts_retained': len(history), 'failed_attempts_retained': len(failed),
    'failure_exit_codes': [r['exit_code'] for r in failed],
    'failure_logs_and_source_snapshots_retained': True}
assert checks['actual_exit_codes'] == [0, 0, 0, 0]
assert checks['named_axiom_checks'] == 29 and checks['spec_noncircularity_checks'] == 3
assert checks['module_axiom_audits'] == 4 and checks['warnings'] == 0
assert checks['dependency_file_count'] == 184 and checks['imported_project_olean_count'] == 176
publicfile = 'bench/tritonbench_g/relu_triton_kernel/AccessGridCorrect.lean'
publicname = 'relu_access_grid_correctness'
actual = declarations(root / publicfile)[publicname]
assert actual['kind'] == 'specification'
public = {'file': publicfile, 'parser_name': publicname,
    'declaration': 'VeriTile.Bench.TritonBenchG.ReluTritonKernel.AccessGridCorrect.' + publicname,
    'kind': 'specification', 'statement_sha256': actual['statement_sha256'],
    'model_kind': 'backend_value_contract', 'role': 'parametric_value_contract',
    'dtype_scope_relation': 'selected_explicit_profiles_only',
    'profiles': ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'bool'],
    'unclosed_numeric_callbacks': read(words / 'evidence/receipt.json')['public_contract']['unclosed_numeric_callbacks'],
    'bit_interpreter_closed_command': False, 'native_binding_status': 'not_established',
    'carrier': 'BitVec32 Algebra outputs; source words are assembled through checked8-bit reads.',
    'premises': [
        'All N source words allocated/readable; only first min(N,1024) output words allocated/writable.',
        'A complete permutation of original ceil-grid program ids; no execution/result premise.',
        'Complete tagged LogicalView of the latent initial byte payload; no tail-access claim.',
        'Distinct logical input/output names and nonoverlapping full N-element byte intervals.',
        'B1024, N<2^31 and representable64-bit bases/exclusive input-output endpoints.'],
    'conclusion': 'Actual partial-access ascending and complete-schedule executions succeed; actual old typed grid succeeds; final bytes equal old byteGrid and independent byteSpec, final LogicalView and wordSpec hold, access maps/frame are preserved, finite-address inputs succeed and inherited address certificate holds.'}
remaining = [
    {'id': 'RELU-A-S1', 'gap': 'Native allocation, storage and access permissions',
     'detail': 'The custom partial interface is a total mathematical byte payload plus explicit allocated/readable/writable maps. Native device allocation, memory validity, access rights, little-endian layout, alignment and provenance are not related to those maps. Full-N input readability and first-tile output writability are selected-domain premises, not host-established facts.'},
    {'id': 'RELU-A-S2', 'gap': 'Native source/compiler and error correspondence',
     'detail': 'Every active input byte of every pid now passes a failing-access bind before the store gate. The nonzero-pid missing-byte theorem distinguishes this from the old total memory-effect projection. This custom interpreter still has no stepwise Python/TTIR/compiler bridge. Deterministic first-unavailable-byte errors and retained partial-write states are mathematical semantics, not native traps, poison, sanitizers or observable fault order. Scalar comparisons/conversions are pure total callbacks; nonzero-pid scalar results are not separately evaluated and no native instruction trace follows. Unloaded entries in loadWords/loadedPrefix are zero placeholders used only to complete the mathematical container; original no-other masked-load values are not specified.'},
    {'id': 'RELU-A-S3', 'gap': 'Host-established layout and domains',
     'detail': 'Logical name distinction, numeric interval separation, B1024, N<2^31 and representable endpoints remain theorem premises. Native empty_like freshness, layouts/strides, shifted/overlapping aliases, other sizes/blocks, launch limits and native zero-grid/error behavior remain open. No allocation or permission is required for the unmodified output tail.'},
    {'id': 'RELU-A-S4', 'gap': 'Native instruction interleaving and remaining dtype/module scope',
     'detail': 'All complete permutations of whole programs are proved in the partial-access model. Programs remain atomic relative to each other; native instruction interleavings and races remain open. The same nine finite-word profiles are refined with no new dtype names. Prior fp64/fp8/error families and host/harness/module/compiler target scope remain required.'},
]
counts = {'finite_word_profiles': 9, 'new_dtype_pairs': 0, 'new_logical_profile_names': 0,
    'original_logical_profiles_unchanged': 12, 'new_public_contracts': 1,
    'checked_load_byte_width': 8, 'input_word_width_bits': [8, 16, 32, 64],
    'output_word_width_bits': 32, 'raw_bool_input_bytes': 256,
    'complete_whole_program_permutations': True, 'new_frontend_compilations': 0,
    'native_backend_compilations': 0, 'gpu_executions': 0}
scope = {'task': 'relu_triton_kernel partial byte access and complete program permutations',
    'standard': read(words / 'evidence/scope-review.json')['standard'],
    'whole_original_program_accepted': False, 'completed_original_program_increment': 0,
    'source_sha256': sha(repo / 'bench/tritonbench_g/relu_triton_kernel/relu_triton_kernel.py'),
    'prior_frozen_receipts': deps['frozen_receipts'], 'public_contract': public, 'counts': counts,
    'new_proved': [
        'Allocation and read permission are checked for each byte before little-endian source-word assembly.',
        'Every active lane of every pid is loaded through Except bind before inspecting the pid-zero store gate; inactive masked lanes perform no byte access.',
        'A nonzero pid with an unavailable first active input byte returns loadUnavailable even though the old byteProgram has identity memory effect.',
        'Each output byte requires allocation/write permission; a failed later store returns the exact state after all preceding successful byte writes.',
        'Whole-N readable input and first-min(N,1024)-word writable output suffice for real partial-program success and exact old byte effects. Output-tail permissions are unconstrained.',
        'Actual ascending and arbitrary complete whole-program orders succeed on preceding current memory and match the frozen byteGrid and independent byteSpec.',
        'Allocation and permission maps never change; both total payload and allocated-content views have a complete frame outside the write prefix.',
        'Actual old typed-grid success, wordSpec, final LogicalView and finite-address checked input accesses follow by composition, with the inherited address certificate.'
    ],
    'memory_model': {
        'representation': 'total Nat-addressed8-bit payload plus allocated/readable/writable Bool maps',
        'partial_contents': 'allocated-content representation only; it does not assert read permission',
        'load': 'per-byte allocated/readable check, ordered Except binds and actual little-endian assembly',
        'store': 'per-byte allocated/writable check and update in lane/byte order; failure retains preceding write state',
        'failures': ['loadUnavailable address', 'storeUnavailable address'],
        'all_pid_loads': 'loadWords result is matched before pid-zero gate; nonzero programs can fail',
        'output_tail': 'No allocation/readability/writability constraints on output tail; its latent bytes and allocated-content view remain unchanged.',
        'scalar_callbacks': 'Pure total inherited wordValue callbacks; nonzero-pid scalar results are not separately evaluated. No scalar exception or native instruction trace is supplied.',
        'unloaded_placeholders': 'loadWords/loadedPrefix use zero outside the loaded prefix only as a mathematical container placeholder, never an active-store value or a claim about original no-other masked-load values.',
        'scheduling': 'arbitrary complete permutation of whole programs; no interleaving within/between their instructions'},
    'domain': read(bytes_package / 'evidence/scope-review.json')['domain'],
    'permission_domain': {'input': 'Readable initial inputBase (N*elementBytes encoding)',
        'output': 'Writable initial outputBase (min(N,1024)*4)',
        'no_output_tail_requirement': True, 'no_output_read_permission_requirement': True},
    'alias_policy': 'Distinct logical names and full byte-interval separation remain public representation premises. Access-success/effect-schedule sublemmas need only the stated permission footprints.',
    'structural_remaining': remaining,
    'numerical_followups': read(words / 'evidence/scope-review.json')['numerical_followups'],
    'new_numerical_callbacks': [],
    'suggested_covered_stage': 'partial_byte_access_all_active_loads_complete_program_permutation',
    'source_view_constraint': 'Retain original source views. This custom access interpreter is not an independently extracted source or native compiler simulation.',
    'inherited_route_review': deps['inherited_route_review'],
    'verification': checks, 'independent_review': 'pending root/independent integration review'}
write(root / 'evidence/scope-review.json', scope)
artifacts = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*'))
    if p.is_file() and not p.is_symlink() and p.suffix not in {'.olean', '.pyc'}
    and p != root / 'evidence/receipt.json'}
receipt = {'task': scope['task'], 'package_root': str(root), 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
    'whole_original_program_accepted': False, 'original_program_completion_increment': 0,
    'modules': modules, 'public_contract': public, 'coverage_summary': counts, 'verification': checks,
    'prior_frozen_receipts': deps['frozen_receipts'], 'acceptance_limits': remaining,
    'failed_attempts': failed, 'artifacts': artifacts}
write(root / 'evidence/receipt.json', receipt)
print(json.dumps({'receipt_sha256': sha(root / 'evidence/receipt.json'),
    'scope_sha256': sha(root / 'evidence/scope-review.json'),
    'public_statement_sha256': public['statement_sha256'], 'module_sha256': modules, 'verification': checks}, indent=2))
