"""Bind the four partial contracts without altering historical scope reviews."""
import copy
import json
from pathlib import Path
import sys

ROOT=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
BATCH=ROOT/'bench/review-checkpoints/2026-10-16/quantization-and-mixed-softmax'
sys.path.insert(0,str(ROOT/'bench'))
from adapter_common import canonical,hash_file,load,sha,write_json
from check_original_program_scope import task_bindings
from current_source_catalog import validate_catalog

catalog=load(BATCH/'current-source-catalog.json')
validate_catalog(ROOT,catalog)
ledger=load(ROOT/'bench/original_program_scope.json')
before=load(BATCH/'evidence/ledger-before.json')
assert ledger==before,'refuse overwriting changed ledger'
rows={t['task']:t for t in catalog['tasks']}
supp={t['task']:t for t in load(BATCH/'catalog-supplement.json')['tasks']}
review=load(BATCH/'root-review.json')
names=set(review['whole_stage_residuals'])
assert names=={'quantize_global','quant_transpose_kernel','token_softmax_bloom','token_softmax_llama'}
assert len(ledger['tasks'])==183 and all(t['status']=='pending' for t in ledger['tasks'])
changed=[]
for row in ledger['tasks']:
    old=copy.deepcopy(row)
    row['input_sha256']=task_bindings(ROOT,catalog,rows[row['task']])
    if row['task'] in names:
        ext=supp[row['task']]
        row['structural_gaps']=[{'kind':'remaining_original_program_scope','finding':item,'can_defer_as_numeric_assumption':False}
                                for item in review['whole_stage_residuals'][row['task']]]
        row['initial_assessment']='structural_gaps_confirmed'
        row['partial_progress']={
            'status':'reviewed_partial_execution_stages_not_whole_program_acceptance',
            'declarations':[{k:d[k] for k in ('file','declaration','statement_sha256','dtype_profile','covers_stages','model_kind','role','scope_relation')}
                            for d in ext['declarations']],
            'review':ext['review'],'execution_contracts':1,'protocol_contracts':0,'protocols_counted_as_execution':False,
            'previous_structural_gaps':old['structural_gaps'],
            'previous_partial_progress':old.get('partial_progress'),
            'exact_numeric_registration':'No new rule/statement/case/port bindings. Scalar callbacks remain unvalidated and require explicit registration for this new statement.'}
    if row!=old:changed.append(row['task'])
assert set(changed)==names,changed
ledger['catalog_sha256']=sha(canonical(catalog))
ledger['progress_checkpoint']=(BATCH/'README.md').relative_to(ROOT).as_posix()
assert hash_file(ROOT/ledger['deferred_queue']['path'])==ledger['deferred_queue']['sha256']
assert sum(t.get('initial_assessment')=='structural_gaps_confirmed' for t in ledger['tasks'])==85
markdown=ROOT/'bench/original_program_scope.md'
text=markdown.read_text()
assert text.count('`relu-source-grid-and-exact-numerics`')==1
assert text.count('contains 41 execution contracts')==1
assert text.count('four protocol contracts across 10 tasks')==1
text=text.replace('`relu-source-grid-and-exact-numerics`','`quantization-and-mixed-softmax`')
text=text.replace('contains 41 execution contracts','contains 45 execution contracts')
text=text.replace('four protocol contracts across 10 tasks','four protocol contracts across 14 tasks')
write_json(ROOT/'bench/original_program_scope.json',ledger)
markdown.write_text(text)
print(json.dumps({'accepted':0,'pending':183,'confirmed_gaps':85,'unreviewed':98,'changed_tasks':changed,
                  'catalog_sha256':ledger['catalog_sha256']}))
