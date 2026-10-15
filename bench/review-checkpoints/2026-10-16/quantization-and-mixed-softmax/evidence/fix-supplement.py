"""Correct task-local catalog bindings, retaining the rejected preparation."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
TMP = Path('/tmp/veritile-standardization-20261015')
BATCH = ROOT / 'bench/review-checkpoints/2026-10-16/quantization-and-mixed-softmax'
FAILURE = BATCH / 'evidence/integration-attempts/01-cross-task-source-bindings'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert not FAILURE.exists()
supp = json.loads((BATCH / 'catalog-supplement.json').read_text())
review = json.loads((BATCH / 'root-review.json').read_text())
changes = []
for row in supp['tasks']:
    if row['task'] not in {'quant_transpose_kernel', 'token_softmax_llama'}:
        continue
    prefix = 'bench/tritonbench_g/' + row['task'] + '/'
    shared = {p:h for p,h in row['source_sha256'].items() if not p.startswith(prefix)}
    assert shared and all(review['reviewed_input_sha256'][p] == h for p,h in shared.items())
    row['source_sha256'] = {p:h for p,h in row['source_sha256'].items() if p.startswith(prefix)}
    row['shared_dependency_sha256'] = shared
    if row['task'] == 'token_softmax_llama':
        view = row['source_views'][0]
        view['selected_definitions'] = {prefix + 'TokenSoftmaxLlama.lean':['token_softmax_surface']}
        view['shared_typed_source'] = {
            'file':'bench/tritonbench_g/token_softmax_bloom/Proofs/MixedSource.lean',
            'definition':'source',
            'shape_bridge':prefix + 'Proofs/MixedShape.lean',
            'public_statement':prefix + 'MixedFPCorrect.lean',
            'boundary':'Task-local erased original source, with the shared mixed-typed execution graph and complete erased-source equality in the public statement.'}
    changes.append({'task':row['task'], 'shared_dependency_sha256':shared})
assert len(changes) == 2
FAILURE.mkdir(parents=True)
for name in ('catalog-supplement.json','root-review.json'):
    shutil.copyfile(BATCH/name, FAILURE/name)
shutil.copyfile(TMP/'quant-soft-catalog.log',FAILURE/'catalog.log')
(BATCH/'catalog-supplement.json').write_text(json.dumps(supp,indent=2)+'\n')
(FAILURE/'correction.json').write_text(json.dumps({
    'observed_catalog_exit_code':1,
    'error':'supplement source outside task',
    'correction':'Only task-local files belong to source_sha256. Shared files remain bound in the existing root review and explicit shared_dependency_sha256. The llama view selects its original task-local source and records the proved shared typed source bridge.',
    'changes':changes, 'proof_sources_changed':0,
    'root_review_sha256_unchanged':sha(BATCH/'root-review.json'),
    'corrected_supplement_sha256':sha(BATCH/'catalog-supplement.json')},indent=2)+'\n')
shutil.copyfile(Path(__file__), BATCH/'evidence/fix-supplement.py')
print(json.dumps({'corrected_tasks':[r['task'] for r in changes], 'proof_sources_changed':0}))
