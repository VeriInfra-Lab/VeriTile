"""Integrate the reviewed six-module ReLU package without modifying old evidence."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
TMP = Path('/tmp/veritile-standardization-20261015')
PKG = TMP / 'relu-source-coupled-block-wip'
OLD_REL = '2026-10-16/fp64-admission-relu-access'
NEW_REL = '2026-10-16/relu-source-grid-and-exact-numerics'
OLD = ROOT / 'bench/review-checkpoints' / OLD_REL
NEW = ROOT / 'bench/review-checkpoints' / NEW_REL
BASE = ROOT / 'bench/review-checkpoints/2026-10-15/original-scope-extensions/metadata-baseline'
HEAD = 'ea6512dd1c663d24b2bcbdf0b0c29b89481c5a8a'
sys.path.insert(0, str(ROOT / 'bench'))
from adapter_common import canonical, declarations
from current_value_metadata import value_scope

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def digest(data):
    return hashlib.sha256(data).hexdigest()

def read(p):
    return json.loads(Path(p).read_text())

def write(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2) + '\n')

def hashes(p):
    return {f.relative_to(p).as_posix(): sha(f) for f in sorted(p.rglob('*')) if f.is_file()}

assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == HEAD
assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() == 'fp-examples'
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)
assert not NEW.exists()
assert sha(PKG / 'evidence/receipt.json') == 'ad2051d04165c629830a901f25ccd73652254b7453dd3d04f183861b74856fc7'
assert sha(PKG / 'evidence/scope-review.json') == 'e647c106a1cf7fc3f3e95b8331ee573d344b9359ca80c2ba019c100c60291e5c'
REVIEW = TMP / 'relu-source-coupled-block-independent-review.json'
assert sha(REVIEW) == 'b5fbfe40334546e6683aa2cbc3ad5c33ea9407b7eebc8f56ef018b5191a8bf65'
independent, receipt, scope = read(REVIEW), read(PKG / 'evidence/receipt.json'), read(PKG / 'evidence/scope-review.json')
assert independent['proof_defects'] == [] and independent['original_program_completion_increment'] == 0
for rel, h in receipt['artifacts'].items():
    assert sha(PKG / rel) == h, rel
for rel, h in independent['reviewed_input_sha256'].items():
    path = PKG / rel if rel in receipt['modules'] else ROOT / rel
    assert sha(path) == h, rel
assert all(not (ROOT / rel).exists() for rel in receipt['modules'])

old_hashes = hashes(OLD)
base_hashes = hashes(BASE)
assert base_hashes['receipt.json'] == '207d0fc6f6fd80c3dd15b54908e87c4c72a5fc6e7bf5ebac08d9344ed890ddd3'
assert base_hashes['input-sha256.json'] == 'a898cb04b06112f4d7bd0983560e172d7daf5657a0c6c3517e800ec624ba17b2'
assert base_hashes['inputs.zip'] == 'a732792b3bb9f709795c5a5767120f672aa0b0b18cf4de2f38a8ba7956740c76'
for rel, h in read(BASE / 'input-sha256.json').items():
    assert sha(ROOT / rel) == h, rel
supplement = copy.deepcopy(read(OLD / 'catalog-supplement.json'))
previous = copy.deepcopy(supplement)
relu = next(t for t in supplement['tasks'] if t['task'] == 'relu_triton_kernel')
prior_relu = copy.deepcopy(relu)
assert len(relu['declarations']) == 5
remaining = [r['id'] + ': ' + r['gap'] + '. ' + r['detail'] for r in scope['structural_remaining']]
new_claims = []
for public in receipt['public_contracts']:
    assert declarations(PKG / public['file'])[public['parser_name']]['statement_sha256'] == public['statement_sha256']
    stage = ('checked_byte_source_seven_statement_block_trace' if public['coverage_role'] == 'single_program_source_coupling'
             else 'checked_byte_source_seven_statement_traces_complete_program_permutation')
    contract = {
        'file': public['file'], 'declaration': public['parser_name'], 'kind': 'specification',
        'statement_sha256': public['statement_sha256'], 'model_kind': 'backend_value_contract',
        'role': 'parametric_value_contract', 'scope_relation': 'full_declared_stage',
        'source_variant': 'original_source_translation', 'covers_stages': [stage],
        'dtype_profile': 'The same nine integer/bool finite-word profiles; B1024, mathematical Nat indices and byte addresses, actual seven-statement source traces, pure total numerical callbacks. ' + public['coverage_role'],
        'dtype_scope_relation': 'selected_explicit_profiles_only',
        'unclosed_numeric_callbacks': public['unclosed_numeric_callbacks'], 'extra_numeric_law_ids': [],
        'native_binding_status': 'not_established',
        'review_basis': public['conclusion'],
        'original_scope': {
            'standard_accepted': False, 'covered': public['conclusion'],
            'remaining_structural_obligations': remaining,
            'bit_interpreter_closed_command': False,
            'premises': public['premises'], 'carrier': public['carrier'],
            'source_coupling': scope['source_coupling'],
            'permission_domain': scope['permission_domain'], 'alias_policy': scope['alias_policy'],
            'address_domain': scope['address_domain'],
            'numerical_callbacks_validated': False,
            'exact_queue_binding': 'Not registered by the separate three-declaration exact adapter; requires explicit statement/case/port bindings.',
            'new_dtype_profiles': 0, 'whole_program_completion_increment': 0
        }
    }
    new_claims.append(contract)
relu['declarations'] += new_claims
relu['required_value_stages'] += [s for d in new_claims for s in d['covers_stages']]
assert len(relu['required_value_stages']) == len(set(relu['required_value_stages']))
relu['source_sha256'].update({p: h for p, h in independent['reviewed_input_sha256'].items()
                            if p.startswith('bench/tritonbench_g/relu_triton_kernel/')})
inputs = dict(independent['reviewed_input_sha256'])
inputs.update(relu['source_sha256'])
for rel, h in inputs.items():
    assert sha((PKG if rel in receipt['modules'] else ROOT) / rel) == h
all_declarations = [d for t in supplement['tasks'] for d in t['declarations']]
assert len(all_declarations) == 45
assert sum(d['role'] == 'parametric_value_contract' for d in all_declarations) == 41
assert sum(d['role'] == 'protocol_contract' for d in all_declarations) == 4
assert all(t == next(p for p in previous['tasks'] if p['task'] == t['task'])
           for t in supplement['tasks'] if t['task'] != 'relu_triton_kernel')

state = {
    'base_commit': HEAD, 'predecessor_file_sha256': old_hashes, 'metadata_baseline_sha256': base_hashes,
    'scope_ledger_before_sha256': sha(ROOT / 'bench/original_program_scope.json'),
    'scope_markdown_before_sha256': sha(ROOT / 'bench/original_program_scope.md'),
    'workflow_before_sha256': sha(ROOT / '.github/workflows/bench-audit.yml'),
    'original_baseline_input_map': read(BASE / 'input-sha256.json'),
    'new_source_sha256': receipt['modules'],
    'prepared_by_sha256': sha(__file__), 'whole_program_accepted': 0,
    'queue_rebind_pending_exact_adapter_integration': True
}
workflow = ROOT / '.github/workflows/bench-audit.yml'
workflow_text = workflow.read_text()
assert workflow_text.count(OLD_REL) == 3

# All mutations follow complete input/preimage validation.
for rel, h in receipt['modules'].items():
    (ROOT / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PKG / rel, ROOT / rel)
    assert sha(ROOT / rel) == h
NEW.mkdir(parents=True)
retained = {}
for rel, h in {**receipt['artifacts'], 'evidence/receipt.json': sha(PKG / 'evidence/receipt.json')}.items():
    if rel in receipt['modules']:
        continue
    dest = NEW / 'evidence/producer' / (rel + '.txt' if rel.endswith('.lean') else rel)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PKG / rel, dest)
    assert sha(dest) == h
    retained[dest.relative_to(NEW).as_posix()] = h
for src, rel in [(REVIEW, 'evidence/independent-review.json'),
                 (TMP / 'review_relu_source_coupled.py', 'evidence/independent-checker.py'),
                 (Path(__file__), 'evidence/prepare.py')]:
    shutil.copyfile(src, NEW / rel)
    retained[rel] = sha(src)
review = {
    'schema': 1, 'kind': 'relu_source_grid_extension_review', 'reviewer': 'root',
    'status': 'reviewed_partial_source_and_checked_byte_grid_composition', 'base_commit': HEAD,
    'reviewed_input_sha256': dict(sorted(inputs.items())),
    'reviewed_value_scope_sha256': {'relu_triton_kernel': digest(canonical(value_scope(relu)))},
    'reviewed_metadata_corrections_sha256': {}, 'new_public_contracts': {'relu_triton_kernel': new_claims},
    'semantic_review': independent['semantic_review'],
    'independent_review': {'path': 'evidence/independent-review.json', 'sha256': sha(REVIEW)},
    'whole_stage_residuals': {'relu_triton_kernel': remaining},
    'metadata_baseline': read(OLD / 'root-review.json')['metadata_baseline'],
    'whole_original_program_acceptance': 'not_established;0/183accepted,183pending',
    'native_functional_correctness': 'not_established',
    'prior_declarations_preserved': 43, 'execution_declarations': 41, 'protocol_declarations': 4,
    'numerical_registration_boundary': 'Exact adapter registers three earlier public declarations only. These two new source-coupled declarations remain unregistered.',
    'independent_comparator': 'No current source comparator result inferred from historical pinned replay.'
}
write(NEW / 'root-review.json', review)
relu['review'] = {'path': NEW_REL + '/root-review.json', 'sha256': sha(NEW / 'root-review.json'),
                  'source_hashes_pointer': ['reviewed_input_sha256']}
write(NEW / 'catalog-supplement.json', supplement)
# Keep descriptive followups exact; exact queue registration lives separately.
shutil.copyfile(OLD / 'numerical-followup.json', NEW / 'numerical-followup.json')
write(NEW / 'evidence-index.json', {'schema': 1, 'new_source_sha256': receipt['modules'],
                                  'retained_exact_artifacts': retained,
                                  'compiled_artifacts_copied': False})
write(NEW / 'evidence/preparation-state.json', state)
write(NEW / 'evidence/ledger-before.json', read(ROOT / 'bench/original_program_scope.json'))
workflow.write_text(workflow_text.replace(OLD_REL, NEW_REL))
(NEW / 'README.md').write_text('''# ReLU source/grid composition and exact numerical registration

Six Lean modules add two partial ReLU contracts. The checked byte execution now
composes with all seven actual frozen typed-source statements and retains every
program's result register. Complete permutations of whole programs pass both
logical and byte state between programs and reach the independent word/byte
specifications. Inactive lanes retain incoming undef, including in nonzero pids.

The same nine integer/bool profiles and three pure total callbacks are used.
All N input words need read access; only min(N,1024) fp32 output words need write
access. The grid additionally requires distinct logical names and nonoverlapping
full byte intervals. Floating profiles and the remaining original scope are open.
This is whole-program permutation coverage, not native instruction scheduling.

All 43 earlier supplemental declarations are retained exactly. The cumulative
supplement now contains 41 execution and four protocol declarations over 10 tasks.
The 35 descriptive numerical followups remain unchanged. Separate exact numerical
registration covers only its explicitly bound three previous declarations; the
two new ReLU contracts are not automatically registered or numerically validated.

Producer evidence retains six successful final compilations, 40 named axiom
checks, five noncircularity checks, six module audits and all 23 attempts,
including 10 failures. Root independently checked all six sources, 190 dependency
files, 179 imported project oleans, 20 predecessor sources and 13 TTIR hashes.
No new frontend/native compilation or GPU execution is claimed.

The original metadata baseline and older checkpoints are unchanged. Current
integration results are recorded separately after actual checks. Whole-original
program acceptance remains 0/183; the ledger retains 66 confirmed gaps and 117
programs without a completed scope review. All 183 remain pending.
''')
assert hashes(OLD) == old_hashes and hashes(BASE) == base_hashes
print(json.dumps({'new_modules': 6, 'new_execution_contracts': 2, 'supplemental_declarations': 45,
                  'accepted_original_programs': 0, 'checkpoint': str(NEW)}, indent=2))
