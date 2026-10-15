"""Integrate eleven independently reviewed sources and retain prior evidence."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
TMP = Path('/tmp/veritile-standardization-20261015')
Q = TMP / 'quantize-integer-stage-wip'
S = TMP / 'token-softmax-mixed-output-wip'
OLD_REL = '2026-10-16/relu-source-grid-and-exact-numerics'
NEW_REL = '2026-10-16/quantization-and-mixed-softmax'
OLD = ROOT / 'bench/review-checkpoints' / OLD_REL
NEW = ROOT / 'bench/review-checkpoints' / NEW_REL
BASE = ROOT / 'bench/review-checkpoints/2026-10-15/original-scope-extensions/metadata-baseline'
HEAD = '25070582c24c7e0f2e597e5fc263c3c5e31794eb'
REVIEW = TMP / 'quant-soft-packages-root-review.json'
sys.path.insert(0, str(ROOT / 'bench'))
from adapter_common import canonical, declarations
from current_value_metadata import value_scope

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
digest = lambda b: hashlib.sha256(b).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
hashes = lambda p: {f.relative_to(p).as_posix(): sha(f) for f in sorted(p.rglob('*')) if f.is_file()}


def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2) + '\n')


assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == HEAD
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)
assert not NEW.exists()
assert sha(REVIEW) == '9ac3e466551df09fb4372b8ad36fc28b1d227aa35be25f126d48005d3273b7af'
independent = read(REVIEW)
assert independent['proof_defects'] == [] and independent['whole_program_completion_increment'] == 0
for path, h in independent['checked_file_sha256'].items():
    assert sha(path) == h, path
qr, sr = read(Q / 'producer-receipt.json'), read(S / 'evidence/receipt.json')
qscope, sscope = read(Q / 'scope.json'), read(S / 'evidence/scope-review.json')
mapping = {
    'QuantizeIntegerStage.Semantics': 'bench.tritonbench_g.quantize_global.Proofs.IntegerSemantics',
    'QuantizeIntegerStage.Global': 'bench.tritonbench_g.quantize_global.IntegerCorrect',
    'QuantizeIntegerStage.Transpose': 'bench.tritonbench_g.quant_transpose_kernel.IntegerCorrect',
    'QuantizeIntegerStage.SourceChecks': 'bench.tritonbench_g.quantize_global.Proofs.IntegerSourceChecks',
}
new_sources, source_transforms = {}, []
for source_module, target_module in mapping.items():
    src = Q / 'src' / (source_module.replace('.', '/') + '.lean')
    target = target_module.replace('.', '/') + '.lean'
    old_text = src.read_text()
    text = ''.join(('import ' + mapping[line[7:].strip()] + '\n')
                   if line.startswith('import ') and line[7:].strip() in mapping else line
                   for line in old_text.splitlines(keepends=True))
    new_sources[target] = text.encode()
    source_transforms.append({'source': str(src), 'source_sha256': sha(src), 'destination': target,
                              'destination_sha256': digest(text.encode()), 'change': 'import module paths only'})
for rel, h in sr['modules'].items():
    assert sha(S / rel) == h
    new_sources[rel] = (S / rel).read_bytes()
assert len(new_sources) == 11 and all(not (ROOT / p).exists() for p in new_sources)
old_hashes, base_hashes = hashes(OLD), hashes(BASE)
assert base_hashes['receipt.json'] == '207d0fc6f6fd80c3dd15b54908e87c4c72a5fc6e7bf5ebac08d9344ed890ddd3'
for rel, h in read(BASE / 'input-sha256.json').items():
    assert sha(ROOT / rel) == h, rel
supplement = copy.deepcopy(read(OLD / 'catalog-supplement.json'))
old_tasks = {t['task'] for t in supplement['tasks']}
tasks = ['quantize_global', 'quant_transpose_kernel', 'token_softmax_bloom', 'token_softmax_llama']
assert not set(tasks) & old_tasks
ledger = read(ROOT / 'bench/original_program_scope.json')
ledger_rows = {t['task']: t for t in ledger['tasks']}
inputs = {Path(path).relative_to(ROOT).as_posix(): h
          for path, h in independent['checked_file_sha256'].items()
          if Path(path).is_relative_to(ROOT) and Path(path).suffix in {'.lean', '.py'} and '/.lake/' not in path}
inputs.update({rel: digest(data) for rel, data in new_sources.items()})
residuals, claims, new_tasks = {}, {}, []
for task in tasks:
    for files in ledger_rows[task]['input_sha256'].values():
        inputs.update(files)
    prefix = 'bench/tritonbench_g/' + task + '/'
    is_quant = task.startswith('quant')
    if is_quant:
        public = next(d for d in qr['selected_public_declarations'] if
                      ('/Global.' in d['file']) == (task == 'quantize_global'))
        rel = prefix + 'IntegerCorrect.lean'
        stage = ('complete_fp32_llrint_int8_kernel' if task == 'quantize_global'
                 else 'fp32_llrint_int8_transposed_tile_stage')
        profile = 'FP32 input and inverse scale, source-ordered software FP32 multiplications, explicit BitVec64 llrint callback, signed int8 output. Natural indices and logical tagged cells; mathematical tile dimensions.'
        remaining = [item for item in qscope['remaining_structural_scope']
                     if not item.startswith('quant_transpose') or task == 'quant_transpose_kernel']
        callbacks = ['round : FP.Value .fp32 -> BitVec 64']
        semantic = independent['quantization']['semantic_review']
        numeric = qscope['explicit_numerical_boundaries']
        view_file = rel
        dtype_cases = [{'input': 'fp32', 'inverse_scale': 'fp32', 'compute': 'fp32', 'llrint': 'int64', 'output': 'int8'}]
    else:
        public = next(d for d in sr['public_declarations'] if d['task'] == task)
        rel = public['file']
        stage = public['covers_stages'][0]
        profile = 'Nine independent fp16/bf16/fp32 input/output storage pairs, fp32 max/sub/exp/sum/div graph, positive tile width, nonzero output token stride and natural metadata/strides; total typed logical memory.'
        remaining = [r['finding'] for r in sscope['tasks'][task]['remaining_structural_obligations']]
        callbacks = [r['callback'] for r in read(S / 'numerical-followup.json')['rules']]
        semantic = independent['mixed_softmax']['semantic_review']
        numeric = ['All nine exact scalar families remain unvalidated and unregistered for these new statements.']
        view_file = 'bench/tritonbench_g/token_softmax_bloom/Proofs/MixedSource.lean'
        dtype_cases = public['selected_storage_pairs']
    residuals[task] = remaining
    contract = {'file': rel, 'declaration': public['declaration'], 'kind': 'specification',
                'statement_sha256': public['statement_sha256'], 'model_kind': 'backend_value_contract',
                'role': 'parametric_value_contract', 'scope_relation': 'full_declared_stage',
                'source_variant': 'original_source_translation', 'covers_stages': [stage],
                'dtype_profile': profile, 'dtype_scope_relation': 'selected_explicit_profiles_only',
                'unclosed_numeric_callbacks': callbacks, 'extra_numeric_law_ids': [],
                'native_binding_status': 'not_established', 'review_basis': semantic,
                'original_scope': {'standard_accepted': False, 'covered': stage,
                    'remaining_structural_obligations': remaining,
                    'remaining_native_numerical_bindings': numeric,
                    'selected_storage_cases': dtype_cases,
                    'numerical_callbacks_validated': False,
                    'exact_queue_binding': 'No new exact rule/statement/case/port registration in this batch.',
                    'whole_program_completion_increment': 0}}
    claims[task] = [contract]
    bound = {p: h for p, h in inputs.items() if p.startswith(prefix)}
    if task == 'token_softmax_llama':
        bound.update({p: h for p, h in inputs.items() if p.startswith('bench/tritonbench_g/token_softmax_bloom/')})
    if task == 'quant_transpose_kernel':
        helper = 'bench/tritonbench_g/quantize_global/Proofs/IntegerSemantics.lean'
        bound[helper] = inputs[helper]
    view = {'id': stage, 'variant': 'original_source_translation', 'frontend': 'shared_triton_dsl',
            'ordinary_shared_dsl_admission': not is_quant,
            'memory_semantics': 'single_program_total_typed_memory',
            'translation_trust_boundary': semantic,
            'original_python': prefix + task + '.py', 'candidate_python': None,
            'selected_definitions': {view_file: ['source']}}
    new_tasks.append({'task': task, 'required_value_stages': [stage], 'source_sha256': bound,
                      'source_views': [view], 'declarations': [contract]})
supplement['tasks'] += new_tasks
assert sum(len(t['declarations']) for t in supplement['tasks']) == 49
workflow = ROOT / '.github/workflows/bench-audit.yml'
workflow_text = workflow.read_text()
assert workflow_text.count(OLD_REL) == 3
state = {'base_commit': HEAD, 'predecessor_file_sha256': old_hashes,
         'metadata_baseline_sha256': base_hashes,
         'original_baseline_input_map': read(BASE / 'input-sha256.json'),
         'scope_ledger_before_sha256': sha(ROOT / 'bench/original_program_scope.json'),
         'scope_markdown_before_sha256': sha(ROOT / 'bench/original_program_scope.md'),
         'workflow_before_sha256': sha(workflow),
         'new_source_sha256': {p: digest(b) for p, b in new_sources.items()},
         'quantization_import_transformations': source_transforms,
         'prepared_by_sha256': sha(__file__), 'whole_program_accepted': 0,
         'new_numerical_rules_registered': 0, 'changed_tasks': tasks}

# Writes start only after all frozen input and old-source checks.
for rel, data in new_sources.items():
    (ROOT / rel).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / rel).write_bytes(data)
for task in new_tasks:
    for public in task['declarations']:
        assert declarations(ROOT / public['file'])[public['declaration']]['statement_sha256'] == public['statement_sha256']
for rel, h in inputs.items():
    assert sha(ROOT / rel) == h, rel
NEW.mkdir(parents=True)
retained, not_copied = {}, {}
packages = [('quantization', Q, {**read(Q / 'artifact-manifest.json')['files'],
                               'artifact-manifest.json': sha(Q / 'artifact-manifest.json'),
                               'producer-receipt.json': sha(Q / 'producer-receipt.json')}),
            ('mixed-softmax', S, {**sr['artifacts'], 'evidence/receipt.json': sha(S / 'evidence/receipt.json')})]
for name, package, files in packages:
    for rel, h in files.items():
        assert sha(package / rel) == h
        if Path(rel).suffix in {'.olean', '.cubin'}:
            not_copied[name + '/' + rel] = h
            continue
        dest = NEW / 'evidence/producer' / name / (rel + '.txt' if rel.endswith('.lean') else rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package / rel, dest)
        retained[dest.relative_to(NEW).as_posix()] = h
for source, name in [(REVIEW, 'package-review.json'), (TMP / 'review_quant_soft_packages.py', 'package-checker.py'), (Path(__file__), 'prepare.py')]:
    dest = NEW / 'evidence' / name
    shutil.copyfile(source, dest)
    retained[dest.relative_to(NEW).as_posix()] = sha(source)
review = {'schema': 1, 'kind': 'quantization_and_mixed_softmax_extension_review', 'reviewer': 'root',
          'status': 'reviewed_partial_execution_stages', 'base_commit': HEAD,
          'reviewed_input_sha256': dict(sorted(inputs.items())),
          'reviewed_value_scope_sha256': {t['task']: digest(canonical(value_scope(t))) for t in new_tasks},
          'reviewed_metadata_corrections_sha256': {}, 'new_public_contracts': claims,
          'semantic_review': {task: claims[task][0]['review_basis'] for task in tasks},
          'whole_stage_residuals': residuals,
          'metadata_baseline': read(OLD / 'root-review.json')['metadata_baseline'],
          'whole_original_program_acceptance': 'not_established; 0/183 accepted',
          'prior_declarations_preserved': 45, 'execution_declarations': 45, 'protocol_declarations': 4,
          'numerical_registration_boundary': 'Existing 29 rules unchanged. Four new contracts require exact numerical bindings.',
          'package_review': {'path': 'evidence/package-review.json', 'sha256': sha(REVIEW)}}
write(NEW / 'root-review.json', review)
for task in new_tasks:
    task['review'] = {'path': NEW_REL + '/root-review.json', 'sha256': sha(NEW / 'root-review.json'),
                      'source_hashes_pointer': ['reviewed_input_sha256']}
write(NEW / 'catalog-supplement.json', supplement)
shutil.copyfile(OLD / 'numerical-followup.json', NEW / 'numerical-followup.json')
write(NEW / 'evidence-index.json', {'schema': 1, 'new_source_sha256': state['new_source_sha256'],
                                   'retained_exact_artifacts': retained,
                                   'compiled_artifacts_not_copied': not_copied})
write(NEW / 'evidence/preparation-state.json', state)
# Preserve the exact original ledger bytes, including its Unicode encoding.
shutil.copyfile(ROOT / 'bench/original_program_scope.json', NEW / 'evidence/ledger-before.json')
workflow.write_text(workflow_text.replace(OLD_REL, NEW_REL))
(NEW / 'README.md').write_text('''# Quantization integer stage and mixed-output token softmax

Eleven modules add four partial execution contracts. The two quantization stages
now execute source-ordered fp32 multiplication, an explicit llrint-to-int64
callback and signed int8 store narrowing, with values and full memory frame.
Global covers the selected kernel; transpose still uses independent tile
coordinates rather than the original grouped pid map. The eight-bit readback
describes values in logical cells, not a checked physical byte-memory model.

Both token softmax variants now execute independent fp16/bf16/fp32 input/output
storage for all nine pairs. The complete fp32 row graph and destination cast
have successful execution, independent initial-state values and off-write frame.
Both public contracts retain complete erased-source equality.

Producer results include eleven successful final compilations, 26 named axiom
checks, one noncircularity check and seven module audits. Two selected quantization
signatures compiled offline to sm90; eighteen exact-original softmax frontend
routes generated connected SSA evidence. No GPU execution or native numerical
refinement is claimed. Root checked 681 frozen file hashes and read all sources.

All prior proofs, 938 catalog declarations, the original metadata baseline,
35 descriptive followups and 29 registered numerical rules remain unchanged.
Supplemental inventory becomes 45 execution and four protocol declarations
across 14 tasks. The four new statements have no exact numerical registration.
Host/grid composition and remaining source dtype routes are still required.

Whole-original-program acceptance remains 0/183, with all 183 pending. At this
integration base the scope ledger contains 85 confirmed gaps and 98 unreviewed
programs. Current integration outcomes will be recorded after actual checks.
''')
assert hashes(OLD) == old_hashes and hashes(BASE) == base_hashes
print(json.dumps({'modules': 11, 'new_publics': 4, 'supplemental_declarations': 49,
                  'checkpoint': str(NEW), 'acceptance_increment': 0}))
