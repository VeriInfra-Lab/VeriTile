"""Freeze direct original-source reviews; no repository writes or acceptance."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
OUT = Path('/tmp/veritile-standardization-20261015/root-scope-batch04-five')
CAT = ROOT / 'bench/review-checkpoints/2026-10-16/relu-source-grid-and-exact-numerics/current-source-catalog.json'
sys.path.insert(0, str(ROOT / 'bench'))
from adapter_common import declarations, split_statement

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
catalog = json.loads(CAT.read_text())
ledger = json.loads((ROOT / 'bench/original_program_scope.json').read_text())
base = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
assert base == '25070582c24c7e0f2e597e5fc263c3c5e31794eb'
assert not OUT.exists()


def review(task, preserved, coverage, dtype, branches, numerical, definitions):
    prefix = 'bench/tritonbench_g/' + task + '/'
    row = next(r for r in ledger['tasks'] if r['task'] == task)
    assert row['initial_assessment'] == 'full_scope_not_yet_verified'
    all_files = {rel: h for group in row['input_sha256'].values() for rel, h in group.items()}
    actual_files = {p.relative_to(ROOT).as_posix() for p in (ROOT / prefix).rglob('*')
                    if p.suffix in {'.lean', '.py'}}
    assert set(all_files) == actual_files
    for rel, h in all_files.items():
        assert sha(ROOT / rel) == h, rel
    cat = next(r for r in catalog['tasks'] if r['task'] == task)
    public = []
    for d in cat['declarations']:
        path = ROOT / d['file']
        parsed = declarations(path)[d['declaration']]
        assert parsed['kind'] == 'specification'
        assert parsed['statement_sha256'] == d['statement_sha256']
        public.append({'file': d['file'], 'declaration': d['declaration'],
                       'statement_sha256': d['statement_sha256'],
                       'line': path.read_text().count('\n', 0, parsed['start']) + 1,
                       'sha256': sha(path), 'statement': split_statement(parsed['text'])[0]})
    actual_public = {(rel, name) for rel in all_files if rel.endswith('.lean')
                     for name, d in declarations(ROOT / rel).items() if d['kind'] == 'specification'}
    assert actual_public == {(d['file'], d['declaration']) for d in public}
    assert len(public) == len(cat['declarations'])

    def evidence(file, start, end):
        rel = prefix + file
        lines = (ROOT / rel).read_text().splitlines()
        lo = next(i for i, line in enumerate(lines) if start in line)
        hi = next(i for i in range(lo, len(lines)) if end in lines[i])
        return {'path': rel, 'line': lo + 1, 'end_line': hi + 1,
                'sha256': all_files[rel], 'excerpt': '\n'.join(lines[lo:hi+1])}

    gaps = []
    for key, kind, finding, blocking, refs in definitions:
        gaps.append({'id': task + ':' + key, 'kind': kind, 'finding': finding,
                     'why_blocks_full_original_scope': blocking,
                     'can_defer_as_numeric_assumption': False,
                     'evidence': [evidence(*r) for r in refs]})
    py = prefix + task + '.py'
    functions = [n for n in ast.walk(ast.parse((ROOT / py).read_text())) if isinstance(n, ast.FunctionDef)]
    production = [n for n in functions if not n.name.startswith('test_')]
    assert {n.name for n in production} == set(coverage)
    stages = [{'function': n.name, 'path': py, 'sha256': all_files[py],
               'line': n.lineno, 'end_line': n.end_lineno, 'coverage': coverage[n.name]}
              for n in production]
    return {'schema': 1, 'kind': 'original_scope_gap_review', 'reviewer': 'root', 'task': task,
            'reviewed_at_utc': datetime.now(timezone.utc).isoformat(), 'reviewed_base_commit': base,
            'catalog': {'path': CAT.relative_to(ROOT).as_posix(), 'sha256': sha(CAT)},
            'status': 'structural_gaps_confirmed', 'whole_original_program_accepted': False,
            'completion_increment': 0, 'source_sha256': all_files,
            'all_input_sha256': row['input_sha256'], 'public_declarations_checked': public,
            'existing_proofs_preserved': preserved, 'production_stage_mapping': stages,
            'test_harness_classification': 'The top-level test_* helper only constructs examples and calls production; production does not call it. It remains in the hashed source inventory and is explicitly classified as a harness for this review.',
            'dtype_review': dtype, 'branch_review': branches, 'structural_gaps': gaps,
            'numerical_boundary': numerical,
            'precondition_assessment': 'Per-program bounds, positive reduction width and injective/nonoverlapping output maps are legitimate selected-domain premises, not defects by themselves. Source-domain exclusions and absent host composition are recorded separately.',
            'verification': {'manual_production_functions_read': len(production),
                             'actual_public_statements_reparsed': len(public),
                             'key_source_helpers_read': True, 'new_lean_runs': 0,
                             'new_frontend_runs': 0, 'gpu_runs': 0, 'numerical_deferrals_registered': 0}}


def gap(key, kind, finding, refs):
    return (key, kind, finding,
            'The omitted source behavior needs an executing correspondence and coverage statement; a scalar numerical binding alone cannot supply it.', refs)


records = []
records.append(review('f8_conversion_utils',
    'Both source directions, two fp8 formats and fp16/fp32 reverse inputs are materialized with destination casts. The duplicate forward store is preserved. Four public Real contracts prove complete masked copy execution, finite bounds and tagged-cell frame; two full typed address-commutation equivalences preserve every state/failure with no numerical rewrite rule. TypedExecution supplies nonempty abstract exact-Real witnesses, not native conversion or round-trip identities.',
    {'kernel_f8_to_f16': 'Full typed load and both destination-converting stores; general public value result is precision-erased.',
     'f8_to_f16': 'int8 assertion, CUDA check, byte reinterpretation, allocation, storage-size count and complete launch not composed.',
     'kernel_f16_to_f8': 'Full typed load and destination conversion for fp16/fp32 input and e4/e5 output.',
     'f16_to_f8': 'Input dtype check is represented by two input selectors; host reinterpretation/allocation/storage count/grid/return uncomposed.'},
    'The wrapper explicitly accepts fp16/fp32 reverse input and requires int8 forward storage; these are retained. Its dtypes argument is unguarded, whereas Format has only e4nv and e5. Other supplied reinterpretation formats or precise rejections are not covered.',
    'Forward duplicate stores, reverse single store and tail masks are retained. No invented round-trip identity is assumed. Assertions and host choices remain outside public per-program proofs.',
    'Exact destination conversion callbacks can be registered and deferred once used by a general executing value result. Current abstract witnesses and relative equivalence do not provide that result or the byte reinterpretation relation.',
    [gap('typed_conversion_result', 'typed_value_execution',
         'All four public value contracts erase dtype and conclude the original Real value. TypedExecution proves only a fixed one-active-lane abstract Real witness with identity casts; the two typed public theorems are equivalences. General source conversion values and frame under explicit rounding callbacks are absent.',
         [('Correct.lean', 'def correct_target', 'specification f8_to_f16_correctness'),
          ('Proofs/TypedExecution.lean', 'def toHalfExecution', '      (initial (formatType format)')]),
     gap('format_domain', 'dtype_domain',
         'The source dtypes parameter is passed directly to triton.reinterpret without restricting it to two formats. Format materializes only float8e4nv and float8e5, without exhaustive admission/rejection for the original parameter domain.',
         [('f8_conversion_utils.py', 'def f8_to_f16(', '    return ret'),
          ('Kernels.lean', 'inductive Format', '  deriving DecidableEq'),
          ('Kernels.lean', 'def toHalf', 'def toFP8')]),
     gap('byte_reinterpret_host', 'host_stage_composition',
         'The wrappers reinterpret int8 storage as the supplied fp8 format, allocate output, derive N from underlying storage size while the grid uses tensor numel, launch B1024 and return a tensor. No public proof composes these byte/storage/view relations and the complete grid.',
         [('f8_conversion_utils.py', 'def f8_to_f16(', '    return ret'),
          ('f8_conversion_utils.py', 'def f16_to_f8(', '    return ret')])]))

records.append(review('triton_mul2',
    'Both distinct-output and in-place originals and both candidates have complete Real masked IO proofs, including original input readback, termination, bounds and frame. Two complete selected-fp32 relative equivalences commute only 2*x, preserve states/failure and include evaluated masked lanes in their finite multiplication rule. Symbolic N/B and in-place aliasing are preserved.',
    {'mul2_kernel': 'Complete per-program Real doubling and selected-fp32 candidate equivalence.',
     'mul2_inplace_kernel': 'Complete in-place load-before-store result and frame.',
     'triton_mul2': 'zeros_like allocation, numel, complete ceil grid and returned output not composed.',
     'triton_mul2_inplace': 'Complete grid and returned original tensor not composed.'},
    'Neither wrapper restricts x dtype. Formal source has no typed storage selectors and relative equivalence defaults to fp32. Integer/bool and other floating source routes or precise rejections are not established.',
    'Both separate and in-place programs and all tail masks are explicitly covered. The missing work is full source dtype and wrapper execution, not an omitted in-place branch.',
    'Selected multiplication rules and exact conversion/rounding correspondence may be explicitly deferred; missing storage paths and complete-grid composition cannot.',
    [gap('typed_storage_result', 'dtype_execution_coverage',
         'The public original value contracts use mathematical Real storage and the equivalences select fp32. The original wrapper dtype-dependent 2*x result and output storage paths are not materialized or proved.',
         [('Kernels.lean', 'def mul2_kernel', '  tl.store(out_ptr'),
          ('Correct.lean', 'specification mul2_kernel_correctness', '      ⊨ fun xs i'),
          ('FPEquiv.lean', 'def original', '  domain :=')]),
     gap('both_host_launches', 'host_stage_composition',
         'The separate-output wrapper initializes zeros_like and returns the allocation; the in-place wrapper returns x after every ceil-div block. Public contracts describe one pid and do not compose either complete launch.',
         [('triton_mul2.py', 'def triton_mul2(', '    return output'),
          ('triton_mul2.py', 'def triton_mul2_inplace(', '    return x'),
          ('Correct.lean', 'def mul2InplaceIO', '  mask :=')])]))

records.append(review('rms_norm_triton',
    'Complete row normalization, fp32 input conversion/runtime epsilon, zero padding, sqrt/division and the pre-weight destination cast are preserved across sixteen floating storage/weight pairs. Three public Real contracts include successful flattened execution and frame; typed address commutation preserves states/failure without numerical rules. SourceChecks proves all eleven statements and cast position. The mixed fp16/bf16 nonzero-pid witness uses exact Real arithmetic and identity casts.',
    {'rms_norm_kernel': 'Full typed row graph, Real output and relative equivalence for sixteen storage/weight pairs.',
     'forward': 'RmsNorm.forward computes M/N, next_power_of_two, contiguous copies, allocation and all-row launch; not composed.',
     'rms_norm': 'Autograd apply dispatch to forward is not in public kernel contracts.'},
    'Input/output share storage as required by empty_like; weight is independent. Four storage constructors cover sixteen floating pairs; the original x and weight have no equivalent dtype guard.',
    'Zero padding, masked stores, pre-weight cast and all floating pair promotion choices remain credited. N<=B matches the valid wrapper tile, not a one-block omission. Host shape/copy/dispatch remains separate.',
    'Runtime epsilon, sqrt, ordered arithmetic and conversion conformance may be bound numerically; a concrete witness with identity casts does not supply a general typed value theorem.',
    [gap('general_typed_value', 'typed_value_execution',
         'Public correctness uses typedIO with eraseDType; its source-equivalence theorem is relative. Execution.full_numerical_execution is a concrete exact-Real identity-cast witness. There is no general original value/frame theorem for the rounded sixteen-route graph.',
         [('Correct.lean', 'def typedIO', '    projection := typed_supported'),
          ('Proofs/Execution.lean', 'theorem full_numerical_execution', '      t.mem "Y" 8 =')]),
     gap('full_storage_domain', 'dtype_domain',
         'Storage only enumerates fp32/fp16/bf16/fp64, while forward accepts unrestricted x and weight tensors before contiguous copies and empty_like output. Remaining source-admitted paths or exact rejection behavior are unreviewed.',
         [('Kernels.lean', 'inductive Storage', '  deriving DecidableEq'),
          ('rms_norm_triton.py', '    def forward(', '        y = torch.empty_like(x)')]),
     gap('autograd_shape_grid', 'host_stage_composition',
         'The original computes M and N from normalized_shape, copies x/weight contiguously, allocates y, enters the CUDA device, launches M programs with N row strides and returns through RmsNorm.apply. Public row IO does not compose those stages.',
         [('rms_norm_triton.py', 'class RmsNorm(', '    return RmsNorm.apply')])]))

records.append(review('mean_reduction',
    'The complete streamed accumulation loop is proved for arbitrary N and nonzero BN, with every active row mean, successful execution, access safety and frame. All sixteen four-by-four floating input/output routes materialize fp32 loads, ordered accumulation, reduction/division and output casts. Exact source precision checks and full typed index-commutation equivalence are retained. The R-parametric public uses the cast-free Real projection where rounding Real is identity; it is not a rounded floating accumulation result.',
    {'mean_dim_kernel': 'Complete kernel loop and Real streamed result; typed source families and full relative equivalence.',
     'dim_compress': 'Dimension coercion, stride sorting, permutation and contiguous copy are outside public proofs.',
     'mean_dim': 'dtype default, dimension normalization, shape loop, allocation, grid and keepdim branch uncomposed.'},
    'Original x and dtype override are not restricted to the four floating Storage constructors. Selected 16 routes remain useful and source-precision checked.',
    'Entire for-off loop and masks are covered. Host dims int/list, negative-index normalization, stride-sorted permutation, dtype=None and keepdim branches have no composed contract.',
    'Exact reduction/arithmetic/cast callbacks may be bound and deferred after a general typed execution result exists. Cast-free Real R-parametric correctness cannot itself certify fp32 accumulation.',
    [gap('rounded_stream_result', 'typed_value_execution',
         'The public streamed R contract is the cast-free Real kernel, and mean_typed_real_correct explicitly erases dtype under RoundingModel.triv. The typed public result is only candidate equivalence; no general rounded original stream value/frame contract is present.',
         [('Correct.lean', 'def typedIO', '    projection := typed_supported'),
          ('Correct.lean', 'specification mean_typed_real_correct', '      ⊨[RoundingModel.triv]'),
          ('FPEquiv.lean', 'specification mean_index_equiv', '      program input output Bool.true')]),
     gap('dtype_override_domain', 'dtype_domain',
         'mean_dim accepts x storage and an independent dtype argument, while typedKernel has only four floating input and output constructors. The whole original admission/rejection domain is not represented.',
         [('mean_reduction.py', 'def mean_dim(', '    dtype = x.dtype'),
          ('Kernels.lean', 'inductive Storage', '  deriving DecidableEq')]),
     gap('dimension_host_branches', 'host_stage_composition',
         'dim_compress normalizes int/list dims, sorts reduction axes by stride, permutes and copies; mean_dim normalizes negative axes, computes output shape/N/M, launches fixed 8x8 tiles, and conditionally squeezes. These original stages and branches are absent from public kernel IO.',
         [('mean_reduction.py', 'def dim_compress(', '  return out')])]))

records.append(review('rbe_triton_transform',
    'Both complete rotary outputs, the inlined frequency helper, source pid remapping, pow/sin/cos graph, masks, zero padding and both debug barriers are represented. The Real public includes flat IO and both independent formulas. A separate backend-parametric successful execution/value/frame result covers actual fp16 storage and fp32 computation with explicit numerical callbacks. Odd trailing columns retain separate masks and the source +1 pointer offsets; no repaired stride behavior is substituted.',
    {'get_freq_multi_tokens': 'Frequency helper is fully inlined with symbolic DIM (original 128), theta and token positions.',
     'rbe_triton': 'Complete pair body, fp16 backend value/frame and Real result for natural start positions.',
     'rbe_triton_wrapper': 'Allocation, signed Python pos, full batch/tile launch and returned tensor not composed.'},
    'The public backend source selects fp16 input/output; wrapper empty_like inherits unrestricted x storage. Public Parameters also uses Nat for startToken although Python pos is an unguarded int.',
    'Even/odd mask differences, missing odd-tail zero fill, pid_m/pid_n mapping and source +1 addressing are retained. Negative start-token positions and full host launch remain outside current statements.',
    'Native pow/sin/cos, arithmetic and fp16 conversions can be exactly registered as numerical callbacks for the existing backend theorem. Missing dtype/signed parameter routes and host composition are structural.',
    [gap('storage_domain', 'dtype_domain',
         'FPTyped.source fixes typed .fp16, whereas rbe_triton_wrapper accepts arbitrary x storage and allocates empty_like. Other source dtype routes or exact rejections have no public typed value coverage.',
         [('Proofs/FPTyped.lean', 'def source (p', '    p.startToken p.theta'),
          ('rbe_triton_transform.py', 'def rbe_triton_wrapper(', '    out = torch.empty_like(x)')]),
     gap('signed_start_position', 'source_parameter_domain',
         'The original pos:int is passed unguarded into the frequency numerator. Both public models bind start_token_position/startToken as Nat, excluding negative supplied positions and their signed conversion behavior.',
         [('rbe_triton_transform.py', 'def rbe_triton_wrapper(', '    return out'),
          ('Proofs/FPTyped.lean', 'structure Parameters', '  (frequencyDimension'),
          ('RbeTritonTransform.lean', 'specification rbe_triton_transform_output_summary_general', '    (THETA :')]),
     gap('complete_batch_grid', 'host_stage_composition',
         'The wrapper decomposes x.shape, allocates output, passes tensor strides and launches batch by ceil(M/2)*ceil(K/1024) with theta10000/DIM128. The public results operate at a selected pid and do not compose the complete returned tensor.',
         [('rbe_triton_transform.py', 'def rbe_triton_wrapper(', '    return out'),
          ('FPCorrect.lean', 'specification rbe_triton_backend_correctness', '  FPTyped.correctness')])]))

OUT.mkdir()
for data in records:
    path = OUT / (data['task'] + '.json')
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'path': str(path), 'sha256': sha(path),
                      'publics': len(data['public_declarations_checked']),
                      'gaps': len(data['structural_gaps'])}))
