"""Freeze the checked-source block/grid package from actual final Lean results."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess
import sys

root = Path(__file__).resolve().parent
repo = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
words = root.parent / 'relu-integer-word-refinement-wip'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
write = lambda p, data: p.write_text(json.dumps(data, indent=2) + '\n')
assert not (root / 'evidence/receipt.json').exists(), 'refuse frozen receipt overwrite'
subprocess.run([sys.executable, '-B', str(root / 'freeze.py'), 'after'], check=True)
sys.path.insert(0, str(repo / 'bench'))
from adapter_common import declarations

rows = read(root / 'evidence/verification-results.json')
history = read(root / 'evidence/verification-attempts.json')
modules = {row['module']: sha(root / row['module']) for row in rows}
logs = [(root / row['log']).read_text() for row in rows]
deps = read(root / 'evidence/final-dependencies-after.json')
failed = [row for row in history if row['exit_code'] != 0]
checks = {
    'module_count': len(rows), 'actual_exit_codes': [row['exit_code'] for row in rows],
    'seconds': sum(row['seconds'] for row in rows),
    'named_axiom_checks': sum(log.count('axiom footprint') for log in logs),
    'spec_noncircularity_checks': sum(log.count('non-circular ✓') for log in logs),
    'module_axiom_audits': sum((root / path).read_text().count('#auditModuleAxioms') for path in modules),
    'warnings': sum(log.count('warning:') for log in logs),
    'dependency_file_count': len(deps['files']),
    'dependency_breakdown': dict(Counter(row['source_kind'] for row in deps['files'].values())),
    'imported_project_olean_count': sum('olean_sha256' in row for row in deps['files'].values()),
    'frozen_predecessor_source_count': len(deps['all_twenty_frozen_source_sha256']),
    'inherited_ttir_count': len(deps['audited_ttir_sha256']),
    'source_dependency_and_olean_hashes_unchanged': True,
    'all_twenty_frozen_relu_source_modules_unchanged': True,
    'actual_lakefile_toml_and_declaration_parser_bound': True,
    'external_library_build_rechecked': False, 'repository_comparator_run': False,
    'source_whitespace_check': 'All six sources have a final newline and no trailing spaces/tabs.',
    'compiler_results': 'evidence/verification-results.json',
    'all_compile_attempts': 'evidence/verification-attempts.json',
    'attempts_retained': len(history), 'failed_attempts_retained': len(failed),
    'failure_exit_codes': [row['exit_code'] for row in failed],
    'failure_logs_and_source_snapshots_retained': True,
    'terminated_attempt_note': 'Attempt0001 was explicitly terminated with SIGTERM after reported errors and prolonged simplification; the subprocess returned -15 and its unmodified source/log are retained.',
}
assert checks['actual_exit_codes'] == [0] * 6
assert checks['named_axiom_checks'] == 40 and checks['spec_noncircularity_checks'] == 5
assert checks['module_axiom_audits'] == 6 and checks['warnings'] == 0
assert checks['frozen_predecessor_source_count'] == 20 and checks['inherited_ttir_count'] == 13

numeric = read(words / 'evidence/receipt.json')['public_contract']['unclosed_numeric_callbacks']
followups = read(words / 'evidence/scope-review.json')['numerical_followups']
profiles = ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'bool']
publics = []
for module, name, role in [
    ('SourceBlockCorrect', 'relu_checked_source_block_correctness', 'single_program_source_coupling'),
    ('SourceGridCorrect', 'relu_checked_source_grid_correctness', 'full_grid_source_coupling')]:
    path = 'bench/tritonbench_g/relu_triton_kernel/' + module + '.lean'
    actual = declarations(root / path)[name]
    assert actual['kind'] == 'specification'
    publics.append({
        'file': path, 'parser_name': name,
        'declaration': 'VeriTile.Bench.TritonBenchG.ReluTritonKernel.' + module + '.' + name,
        'kind': 'specification', 'statement_sha256': actual['statement_sha256'],
        'model_kind': 'backend_value_contract', 'role': 'parametric_value_contract',
        'coverage_role': role, 'dtype_scope_relation': 'selected_explicit_profiles_only',
        'profiles': profiles, 'unclosed_numeric_callbacks': numeric,
        'bit_interpreter_closed_command': False, 'native_binding_status': 'not_established',
        'carrier': 'BitVec32 Algebra; signed/unsigned/bool inputs decoded from checked8-bit memory',
    })
publics[0].update({
    'premises': [
        'B1024; arbitrary Nat N, bases and program id in the mathematical model.',
        'Every active logical input cell is the complete decoded cell of the corresponding initial byte word.',
        'All N input words allocated/readable; only first min(N,1024) fp32 output words allocated/writable.',
        'No execution/result premise; no output-tail or output-readability premise.'
    ],
    'conclusion': 'The checked program succeeds and returns the independent seven-state reference trace; all seven original source step equations and original exec success hold, loaded/result registers are exact, inactive lanes preserve undef, all active results match finite words, final bytes equal an independent pid-zero-prefix specification and access maps/frame are preserved.'
})
publics[1].update({
    'premises': [
        'Complete permutation of the original ceil(N/1024) program ids, including the empty N=0 schedule.',
        'Initial complete input/output LogicalView of the latent byte payload.',
        'Distinct logical input/output region names and nonoverlapping full N-element byte intervals.',
        'All N input words allocated/readable; only first min(N,1024) fp32 output words allocated/writable.'
    ],
    'conclusion': 'Actual checked source programs compose on preceding logical and byte states; one seven-statement source trace and actual result register is retained for every pid; final logical memory is the frozen wordSpec, bytes are the frozen byteGrid/byteSpec, final LogicalView holds, the old partial-access schedule agrees, access maps and the output-tail frame are preserved.'
})

remaining = [
    {'id': 'RELU-C-S1', 'gap': 'Native storage, access and address correspondence',
     'detail': 'The allocation/read/write maps and ordered byte failures remain selected mathematical semantics. Native allocation, layout/strides, little-endian storage, alignment, pointer provenance, access rights, finite-width index/pointer behavior and native fault/poison/trap order are not related to this interpreter. Existing finite-address certificates are preserved but are not a native pointer simulation.'},
    {'id': 'RELU-C-S2', 'gap': 'Python/frontend/native compiler and host/module stages',
     'detail': 'The checked program now composes with all seven exact statements of the frozen manually materialized TypedValue.source, using its real step and exec functions. This closes the earlier custom-load-to-source-block gap and computes every pid result. It is not independent Python extraction, TTIR/LLVM/backend correctness, native empty_like freshness, wrapper/layout validation, launch admission/error or import-time test-harness/module closure.'},
    {'id': 'RELU-C-S3', 'gap': 'Native scheduling and error correspondence',
     'detail': 'Every complete permutation of whole programs is executed against preceding state, with a complete source trace per pid. Programs remain atomic relative to one another. Native instruction interleavings, synchronization, races, concurrency faults and callback evaluation side effects remain outside the model. Structural failures and per-byte access failures are explicit custom errors, not proofs of native observable errors.'},
    {'id': 'RELU-C-S4', 'gap': 'Remaining original dtype and admission/error families',
     'detail': 'This package refines only the same nine integer/bool profiles. Floating fp16/bf16/fp32 byte/source composition, fp64 and fp8 execution/admission/error routes, and noncanonical/complex/quantized/packed dtype/device/layout/module branches remain structural work. The original17-profile audit and12-profile logical materialization are unchanged.'},
]
counts = {
    'finite_word_profiles': 9, 'new_dtype_pairs': 0, 'new_logical_profile_names': 0,
    'original_logical_profiles_unchanged': 12, 'new_public_contracts': 2,
    'source_statements_per_program': 7, 'all_pid_result_registers': True,
    'inactive_load_semantics': 'incoming typed State.undef; no other=0',
    'complete_whole_program_permutations': True,
    'raw_bool_input_bytes': 256, 'new_frontend_compilations': 0,
    'native_backend_compilations': 0, 'gpu_executions': 0,
}
scope = {
    'task': 'relu_triton_kernel checked bytes, exact typed-source block traces and complete full-grid permutations',
    'standard': read(words / 'evidence/scope-review.json')['standard'],
    'whole_original_program_accepted': False, 'completed_original_program_increment': 0,
    'source_sha256': sha(repo / 'bench/tritonbench_g/relu_triton_kernel/relu_triton_kernel.py'),
    'prior_frozen_receipts': deps['frozen_receipts'], 'public_contracts': publics,
    'counts': counts,
    'new_proved': [
        'Exact seven-statement traces reference the unchanged TypedValue.source surfaceBody, with no separately substituted source AST.',
        'Checked active byte reads consume the actual offsets/mask registers and populate the actual original x register; their success is proved equal to the original source load step.',
        'Inactive source lanes retain the incoming typed undef and perform no physical access; the actual comparison/conversion/where result statement executes for every pid before the store gate.',
        'The original source gate is extracted and evaluated; checked stores consume actual result/offsets/mask registers and match the original typed final store state and exact byte effect.',
        'Unavailable first active input bytes fail the checked program before result/gate, for arbitrary pid including nonzero programs.',
        'Checked source programs compose on previous logical and byte state for every complete whole-program permutation, with one complete source/result trace for every pid and zero traces for N=0.',
        'Final grid logical memory equals frozen independent wordSpec; final bytes equal frozen byteGrid/byteSpec; final LogicalView, old partial-access schedule agreement, access maps and output-tail frame hold under separation.'
    ],
    'source_coupling': {
        'source': 'Frozen TypedValue.source encoding.profile; SourceStages.statement selects its seven actual list elements.',
        'changed_interpreter_boundary': 'Statements0..3 and5..6 call actual FP.Structural.step. Statement4 uses a checked byte-load hook proved equal to the original step under the input view. A checked physical store hook runs before original logical statement6; both are coupled through actual gate/result/offsets/mask registers.',
        'original_step_trace': 'Seven exact equations, not just memory-effect equality; original exec is derived from those equations.',
        'scalar_callbacks': 'Pure total inherited fromInt/fromNat/fp32zero callbacks; state/value trace only, not native callback scheduling or side-effect evidence.',
        'schedule': 'Every pid receives the preceding checked program final logical state and byte memory; complete schedules are whole-program permutations, not instruction interleavings.'
    },
    'permission_domain': {'input': 'Readable initial inputBase (N*elementBytes encoding)',
        'output': 'Writable initial outputBase (min(N,1024)*4)',
        'no_output_tail_requirement': True, 'no_output_read_permission_requirement': True},
    'address_domain': 'B1024; Nat addresses/indices for these custom composition contracts. Native finite-width/host correspondence remains structural; predecessor finite-address sources remain unchanged.',
    'alias_policy': 'Single-block contract binds initial active input cells and output effects; full-grid LogicalView requires distinct logical names and nonoverlapping input/output byte intervals.',
    'structural_remaining': remaining, 'numerical_followups': followups, 'new_numerical_callbacks': [],
    'suggested_covered_stage': 'checked_byte_source_seven_statement_traces_complete_program_permutation',
    'source_view_constraint': 'Retain original source views; manual typed materialization plus custom checked access is not an independent native source extraction/compiler simulation.',
    'inherited_route_review': deps['inherited_route_review'], 'verification': checks,
    'independent_review': 'pending root/independent integration review'
}
write(root / 'evidence/scope-review.json', scope)
artifacts = {str(path.relative_to(root)): sha(path) for path in sorted(root.rglob('*'))
    if path.is_file() and not path.is_symlink() and path.suffix not in {'.olean', '.pyc'}
    and path != root / 'evidence/receipt.json'}
receipt = {
    'task': scope['task'], 'package_root': str(root), 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
    'whole_original_program_accepted': False, 'original_program_completion_increment': 0,
    'modules': modules, 'public_contracts': publics, 'coverage_summary': counts, 'verification': checks,
    'prior_frozen_receipts': deps['frozen_receipts'], 'acceptance_limits': remaining,
    'failed_attempts': failed, 'artifacts': artifacts,
}
write(root / 'evidence/receipt.json', receipt)
print(json.dumps({'receipt_sha256': sha(root / 'evidence/receipt.json'),
    'scope_sha256': sha(root / 'evidence/scope-review.json'),
    'public_statement_sha256': {p['parser_name']: p['statement_sha256'] for p in publics},
    'module_sha256': modules, 'verification': checks}, indent=2))
