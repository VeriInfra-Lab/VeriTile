import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.dont_write_bytecode = True
root = Path(__file__).resolve().parent
repo = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sys.path.insert(0, str(repo / 'bench'))
from adapter_common import declarations
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
load = lambda p: json.loads(p.read_text())
results = load(root / 'evidence/verification-results.json')
before = load(root / 'evidence/final-dependencies-before.json')
after = load(root / 'evidence/final-dependencies-after.json')
assert before == after and len(results) == 6
for r in results:
    assert r['exit_code'] == 0
    assert r['source_sha256_before'] == r['source_sha256_after'] == sha(root / r['module'])
    assert r['log_sha256'] == sha(root / r['log'])
    assert not re.search(r'\b(error|sorryAx|DISALLOWED|warning)\b', (root / r['log']).read_text())
source = 'bench/tritonbench_g/relu_triton_kernel/relu_triton_kernel.py'
public_file = 'bench/tritonbench_g/relu_triton_kernel/TypedGridCorrect.lean'
public = declarations(root / public_file)['relu_typed_grid_correctness']
dtype = load(root / 'evidence/dtype-audit/receipt.json')
profiles = load(root / 'evidence/dtype-audit/frontend-profiles.json')
for row in profiles:
    if row['status'] == 'frontend_generated':
        assert sha(root / 'evidence/dtype-audit' / row['ttir']) == row['ttir_sha256']
scope = {
    'task': 'relu_triton_kernel',
    'standard': 'All original declared stages, dtypes and branches. Only explicitly registered numerical correspondence may be deferred; structural omissions remain blocking.',
    'whole_original_program_accepted': False,
    'original_program_completion_increment': 0,
    'original_source': source, 'original_source_sha256': sha(repo / source),
    'stage_coverage': {
        'relu_kernel': 'Complete original legacy program and ascending grid; twelve typed logical input profiles with fp32 output; actual pid0-only write, all pre-gate loads/comparisons, arbitrary tail frame.',
        'relu': 'Fallible device access, allocation, output-numel access, unused num_blocks computation, launch and return in original order. relu_logical_correct composes actual typed-grid success for the selected twelve profiles after preceding primitive outcomes.',
        'test_relu': 'All four original literal lists, explicit fp32/CUDA constructor requests, relu calls and dictionary insertion order. Generic failure propagation stops future cases. Successful key-shape theorem permits failure and is not numerical harness success.',
        'module_initialization': 'result_gold assignment occurs only after successful harness return; failures preserve post-error state. Native Python import/execution remains unbound.',
    },
    'counts': {'canonical_input_profiles_audited': 17, 'frontend_generated': 13, 'frontend_errors': 4,
               'typed_logical_execution_profiles': 12, 'successful_frontend_profiles_without_typed_execution': 1,
               'gpu_runs': 0, 'native_backend_compilations': 0, 'harness_literal_scalar_words_checked': 20},
    'public_contract': {
        'file': public_file,
        'declaration': 'VeriTile.Bench.TritonBenchG.ReluTritonKernel.TypedGridCorrect.relu_typed_grid_correctness',
        'parser_name': 'relu_typed_grid_correctness',
        'kind': public['kind'], 'statement_sha256': public['statement_sha256'],
        'model_kind': 'backend_value_contract', 'role': 'parametric_value_contract',
        'dtype_scope': 'selected_explicit_profiles_only',
        'profiles': ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'bool', 'float16', 'bfloat16', 'float32'],
        'unclosed_numeric_callbacks': [
            'cmp16: ordered comparison of the fp16 input with its zero literal',
            'cmp32: ordered comparison of fp32 input or the bf16-to-fp32 comparison operand with fp32 zero',
            'M.literal none fp16/bf16/fp32 0',
            'M.cast none fp16 fp32 and M.cast none bf16 fp32',
            'M.fromInt none and M.fromNat none producing the integer/bool fp32 conversion result',
        ],
        'defined_identity_carrier_retag': 'backend defines same-format casts and the real-to-fp32 carrier retag of fromInt/fromNat as identity; no law about an arbitrary cast callback is assumed.',
        'software_helper_boundary': 'SoftwareFP32.exact_grid instantiates a concrete word-order comparator and fp32 positive-zero literal, with no DAZ/FTZ preprocessing. Its special-value/harness checks do not instantiate the public arbitrary callbacks or establish native correspondence.',
    },
    'source_view_metadata_proposal': {
        'selected_definitions': {'bench/tritonbench_g/relu_triton_kernel/Kernels.lean': ['relu_kernel']},
        'frontend': 'shared_triton_dsl', 'variant': 'original_source_translation',
        'ordinary_shared_dsl_admission': True,
        'memory_semantics': 'single_program_total_typed_memory',
        'translation_trust_boundary': 'The actual original shared-DSL source is selected. TypedValue manually materializes twelve logical profiles; floating_source_projection proves dtype/cast erasure for fp16/bf16/fp32 only. Integer/bool branches have a separately reviewed explicit decoded-value conversion route. Ordered grid and host composition are separately proved. No generic Python/compiler refinement or SC task-local source admission is claimed.',
    },
    'suggested_covered_stage': 'twelve_profile_complete_pid_zero_only_logical_grid',
    'required_uncovered_original_stages': [
        'fp64_and_remaining_original_dtype_or_error_branches',
        'native_allocation_tensor_metadata_and_physical_memory',
        'native_finite_width_address_and_launch_arithmetic',
        'native_host_wrapper_harness_and_module_execution',
        'native_instruction_scheduling_and_compiler_runtime_outcomes',
    ],
    'source_semantic_details': [
        'The original store is gated by pid == 0. N>1024 leaves the caller output tail untouched relative to post-allocation/metadata memory, not initialized to zero. The four five-element harness cases do not cover all caller lengths.',
        'No dtype, CUDA, contiguity, shape or same-device assertion exists in relu. empty_like requests fp32 and x.device before output.numel determines the launch extent.',
        'No other argument is passed to masked tl.load. Inactive values remain unspecified, even though the AST load padding field is zero. Active-output proofs do not assume finite or zero inactive lanes.',
        'The source uses ordered >= and select, not mathematical max. A selected negative-zero bit pattern is retained; the explicit software word profile maps NaNs to positive zero.',
        'fp16 compares/selects in fp16 and widens the selected result to fp32. bf16 widens x to fp32 for comparison, selects original bf16 x/zero, then widens the selected result. fp32 compares/selects in fp32.',
        'Native small integer comparison extends to int32 where required; mathematical decoded Int/Nat comparisons model the predicate but do not prove native decoding/extension. Signed/unsigned conversion to fp32 uses fromInt/fromNat callbacks.',
        'Bool raw loads use i8!=0 and zero-extension before conversion in observed IR. The logical Bool cell must be related to that native representation; this is structural debt.',
        'Cell.read of a wrong logical tag defaults rather than reinterpreting native bytes. Logical equal-region aliases at identical natural offsets do not establish mixed-width, shifted, strided or physical overlapping views.',
        'Every ascending-grid program executes against actual preceding state. Later programs may load/compare but never store. Native instruction scheduling and race properties remain outside this sequential theorem.',
        'Host Tensor descriptors preserve metadata in callback arguments, but logicalLaunch reads only dtype and region and ignores device/shape/strides/storageOffset. No native metadata consistency theorem is supplied.',
        'Host selectProfile returns unsupportedLogicalProfile for unrepresented inputs, which is an interpreter boundary, not proof of the native compiler error branch. In particular native fp64 frontend generation succeeds.',
        'Native tensor construction is not proved by the twenty CPU struct-encoded literal words; those bind only the declared finite exactly representable harness constants to the software examples.',
    ],
    'structural_remaining': [
        {'id': 'RELU-S1', 'gap': 'Native fp64 value execution is unrepresented despite successful exact-source frontend generation; real is not relabeled as fp64.'},
        {'id': 'RELU-S2', 'gap': 'The four observed fp8 errors need complete source-bound host/compiler semantics. e4nv/e5 fail comparison against int32 zero; fnuz admission failures are scoped to sm90, not all architectures.'},
        {'id': 'RELU-S3', 'gap': 'Original dtype domain beyond the17 representative canonical signatures: shell int1/uint1 aliases, complex, quantized, packed and other dtypes need exact allocation, canonicalization and error behavior.'},
        {'id': 'RELU-S4', 'gap': 'Integer width/byte decoding, bool raw-i8 normalization and tag correspondence are unproved. Decoded Int/Nat/Bool is not native byte or modular-arithmetic certification.'},
        {'id': 'RELU-S5', 'gap': 'Native empty_like dtype/device/layout/freshness, accessor failures, input/output storage bounds and noncontiguous/offset views remain unbound; no missing runtime guards are invented.'},
        {'id': 'RELU-S6', 'gap': 'Nat pid/offset/grid arithmetic requires finite-width native integer/address refinement, overflow/debug/error behavior and the valid block-size domain. Native zero-grid/zero-size behavior is unproved.'},
        {'id': 'RELU-S7', 'gap': 'Native allocation overlap and instruction scheduling/compiler/runtime outcomes remain unproved; sequential logical execution alone is not native concurrency correctness.'},
        {'id': 'RELU-S8', 'gap': 'Native host device/numel access, literal tensor creation, allocation, launch, return, dictionary operations and module execution are not bound to the logical primitives. Full native numerical harness completion is not claimed.'},
    ],
    'numerical_followups': [
        {'id': 'RELU-N1', 'status': 'descriptive_pending_exact_queue_scope_binding',
         'primitive': 'cmp16/cmp32 and fp16/bf16/fp32 zero literals',
         'scope': 'Public twelve-profile logical source after explicit comparison routing',
         'requirements': ['Pin ordered NaN comparison, positive-zero literal, selected signed-zero behavior and target FTZ/DAZ policy.', 'The concrete fp32 helper does not close the public arbitrary callbacks.']},
        {'id': 'RELU-N2', 'status': 'descriptive_pending_exact_queue_scope_binding',
         'primitive': 'fp16/bf16-to-fp32 casts',
         'scope': 'bf16 comparison promotion and post-selection fp16/bf16 output casts',
         'requirements': ['Preserve the source comparison/select/cast order.', 'Bind exceptional values, signed zero and target conversion behavior.']},
        {'id': 'RELU-N3', 'status': 'descriptive_pending_exact_queue_scope_binding',
         'primitive': 'M.fromInt/M.fromNat to fp32',
         'scope': 'The eight integer labels and bool after native width/byte decoding is structurally established',
         'requirements': ['Bind fp32 rounding for signed/unsigned values, including64-bit values.', 'Do not defer width interpretation, bool raw-load normalization or storage domains as numerical assumptions.']},
    ],
    'frontend_audit_scope': dtype,
    'dtype_profiles': profiles,
}
(root / 'evidence/scope-review.json').write_text(json.dumps(scope, indent=2) + '\n')
artifacts = ['README.md', 'verify.py', 'freeze.py', 'source_review.py', 'finalize.py',
             'evidence/scope-review.json', 'evidence/source-review.json',
             'evidence/verification-results.json', 'evidence/final-dependencies-before.json',
             'evidence/final-dependencies-after.json', 'evidence/dtype-audit/receipt.json']
artifacts += [r['log'] for r in results]
receipt = {
    'task': 'relu_triton_kernel original typed gated grid and host scope extension',
    'package_root': str(root), 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
    'whole_original_program_accepted': False, 'original_program_completion_increment': 0,
    'modules': {r['module']: r['source_sha256_after'] for r in results},
    'public_contract': scope['public_contract'],
    'verification': {
        'module_count': 6, 'actual_exit_codes': [r['exit_code'] for r in results],
        'seconds': sum(r['seconds'] for r in results),
        'named_axiom_checks': sum((root/r['log']).read_text().count('axiom footprint') for r in results),
        'spec_noncircularity_checks': sum((root/r['log']).read_text().count('non-circular ✓') for r in results),
        'module_axiom_audits': 6, 'warnings': 0,
        'dependency_file_count': len(before['files']),
        'dependency_breakdown': {k: sum(v['source_kind'] == k for v in before['files'].values()) for k in ['package', 'repository']},
        'source_and_repository_dependencies_unchanged': True, 'head_equality_required': False,
        'actual_lakefile_toml_bound': True, 'external_library_build_rechecked': False,
        'compiler_results': 'evidence/verification-results.json',
    },
    'source': {'file': source, 'sha256': sha(repo/source),
               'legacy_dsl': 'bench/tritonbench_g/relu_triton_kernel/Kernels.lean',
               'legacy_dsl_sha256': sha(repo/'bench/tritonbench_g/relu_triton_kernel/Kernels.lean'),
               'typed_float_erasure': 'TypedValue.floating_source_projection',
               'typed_native_refinement_proved': False},
    'coverage_summary': scope['counts'],
    'acceptance_limits': ['backend_value_contract only', 'twelve decoded logical profiles, not full original dtype/native domain',
                          'full ascending logical grid, not native instruction scheduling',
                          'full logical host/harness control flow with fallible primitives, not native implementation',
                          'source has pid0-only store; arbitrary allocated tail is preserved',
                          'No repository or frozen vector package files changed by this task'],
    'artifacts': {p: sha(root / p) for p in artifacts},
}
(root / 'evidence/receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'receipt': str(root/'evidence/receipt.json'), 'receipt_sha256': sha(root/'evidence/receipt.json'),
                  'scope_sha256': sha(root/'evidence/scope-review.json'),
                  'public_statement_sha256': public['statement_sha256'], 'verification': receipt['verification']}, indent=2))
