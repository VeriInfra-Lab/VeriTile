"""Freeze direct original-source reviews; no repository writes or acceptance."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
OUT = Path('/tmp/veritile-standardization-20261015/root-scope-batch03-pair')
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
    assert len(public) == 3

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
    functions = [n for n in ast.parse((ROOT / py).read_text()).body if isinstance(n, ast.FunctionDef)]
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


quant = review('quantize_kv_transform',
    'All three public contracts are retained. Full per-program masked load, abs/max/divide, scale cast, int8 conversion, destination scatter and both stores are represented. The machine theorem executes both fp16 and fp32 input routes with real binary16 scale payloads, fp32 arithmetic, independent encoded maximum and signed toward-zero conversion, active outputs and full frame. A padded nonzero-pid witness and undefined-policy independence on InputDefined inputs remain credited. This is not merely an erased Real proof.',
    {'_fwd_kernel_destindex_copy_quantize_kv': 'Complete kernel body and both outputs; selected typed formats and defined signed-conversion domain.',
     'destindex_copy_quantize_kv': 'Shape checks, next-power-of-two dimensions, tensor strides and full sequence launch are not composed into the public per-program result.'},
    'Storage enumerates fp16/fp32 input, while OutScale is fixed fp16, Out is int8 and DestLoc is Nat. The wrapper supplies all four independently without matching dtype guards. Other admitted routes and precise source rejections require coverage; arbitrary dtypes are not assumed to compile.',
    'Kernel head/dimension masks and zero padding are covered. Wrapper shape assertions and complete grid remain uncomposed. The public typed value theorem requires each active quotient to admit signed8 conversion, so source zero-scale/nonfinite cases need an explicit coverage decision.',
    'Native conformance of represented fp16 casts, fp32 division, max and signed conversion may be registered as exact numerical deferrals. Adding dtype paths, host composition and a result or explicit undefined-behavior relation outside InputDefined is structural work. No numerical rule is registered here.',
    [
      ('independent_storage_domain', 'dtype_and_index_domain',
       'The wrapper independently accepts K, DestLoc, Out and Out_scale; typed execution only supplies fp16/fp32 K, fp16 scale, int8 output and natural destination metadata.',
       'The absent source-admitted dtype/index routes or rejection behavior must be represented before numerical bindings can cover them.',
       [('quantize_kv_transform.py', 'def destindex_copy_quantize_kv(', '    num_warps = 1'),
        ('Proofs/TypedSource.lean', 'inductive Storage where', '  cur_index = tl.program_id(0)')]),
      ('active_conversion_domain', 'value_domain_coverage',
       'The public machine output contract requires InputDefined, meaning every active quotient has a successful finite in-range signed8 conversion. The original divides by a rounded per-row scale without a zero/nonfinite guard. Execution has an explicit total fallback callback, but no public value/frame result covers active inputs outside InputDefined.',
       'A complete result may expose the source undefined-conversion policy as a numerical callback, but simply excluding these original input cases is not a numerical-conformance deferral.',
       [('quantize_kv_transform.py', '    abs_data =', '    q_src_data ='),
        ('Proofs/TypedSource.lean', 'def InputDefined', '  (toSigned 8 (quotient cfg d p s i)).getD 0'),
        ('FPCorrect.lean', 'specification quantize_kv_transform_machine_correctness', '    (hregions :')]),
      ('wrapper_grid', 'host_stage_composition',
       'The wrapper computes sequence length, two next-power-of-two dimensions and all strides, asserts matching head shapes, and launches every destination row. Public contracts execute only the current pid; no composition relates the full in-place destination/scales tensors to that launch.',
       'Single-program readback does not establish all-sequence launch behavior, including the source destination map and both output buffers.',
       [('quantize_kv_transform.py', 'def destindex_copy_quantize_kv(', '    return'),
        ('FPCorrect.lean', 'specification quantize_kv_transform_machine_correctness', '    ∃ t,')])])

adam = review('adam_update_triton',
    'Despite its name, the source implements Lion. Both original and candidate complete per-program Real contracts prove parameter and momentum outputs, masks, bounds, termination and off-write frame. Both stores update their input buffers in source order. The full fp32-relative equivalence preserves failure and aliases and changes only the final momentum addition under a site-local finite add_commute rule; zero and either sign outcome are not removed from the original Real result.',
    {'update_fn_kernel': 'Complete sign/zero-select body and both in-place outputs in Real, plus contextual selected-fp32 candidate equivalence.',
     'update_fn': 'CUDA check, n_elements, both autotune configurations, restore_value behavior and complete grid are not composed.'},
    'Kernel regions have mathematical-real storage with no typed tensor selectors. The equivalence Program defaults to fp32; independent p/grad/exp_avg source dtypes and dtype-dependent arithmetic/store conversions lack an original typed result. The wrapper only checks CUDA placement.',
    'The update != 0 and update > 0 predicates and both stores are represented and proved. B is symbolic, so neither 128 nor 1024 is missing from the per-program body. Host autotune/restore and complete-grid behavior remain outside the public statements.',
    'The contextual add_commute law and eventual exact arithmetic/conversion correspondence can be explicitly bound and deferred. Real output formulas and an fp32 relative equivalence do not supply omitted source dtype paths or host execution. No new numerical registration.',
    [
      ('typed_original_result', 'dtype_execution_coverage',
       'The source accepts independent tensor dtypes, but Kernels.lean has no typed regions and Correct.lean proves Real Lion formulas. The third public contract is selected-fp32 original/candidate equivalence with a local finite addition rule, not a dtype-parametric original value theorem.',
       'Tensor-dependent loads, comparisons, arithmetic precision and both storage conversions need an executing source correspondence before scalar numerical rules can be deferred.',
       [('adam_update_triton.py', 'def update_fn(', '    n_elements = p.numel()'),
        ('Kernels.lean', 'def update_fn_kernel', '  p = tl.load(offset_p_ptr, mask=mask)'),
        ('Correct.lean', 'specification update_fn_kernel_correctness', '         fun i => TiledOptimizer.lionMomentum'),
        ('FPEquiv.lean', 'specification update_commuted_equiv', '      commuted p_ptr')]),
      ('host_grid_autotune', 'host_stage_composition',
       'The source autotunes B128/B1024 with restore_value for p and exp_avg, then update_fn launches ceil(n_elements/B) programs after CUDA assertions. Public adamIO describes one program and does not compose the host and complete in-place update.',
       'The complete launch and source autotune restoration behavior must be related to both final tensors; symbolic per-program B alone does not establish this.',
       [('adam_update_triton.py', '@triton.autotune', "], key = ['n_elements'], restore_value=['p_ptr', 'exp_avg_ptr'])"),
        ('adam_update_triton.py', 'def update_fn(', '        n_elements'),
        ('Correct.lean', 'def adamIO', '  mask := fun pid j')])])

OUT.mkdir()
for data in (quant, adam):
    path = OUT / (data['task'] + '.json')
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'path': str(path), 'sha256': sha(path), 'publics': 3,
                      'gaps': len(data['structural_gaps'])}))
