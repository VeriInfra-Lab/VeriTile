"""Bind reviewed source extension and exact queue, preserving all scope reviews."""
import copy
import json
from pathlib import Path
import sys
ROOT=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
BATCH=ROOT/'bench/review-checkpoints/2026-10-16/relu-source-grid-and-exact-numerics'
sys.path.insert(0,str(ROOT/'bench'))
from adapter_common import canonical,hash_file,load,sha,write_json
from check_original_program_scope import task_bindings
from current_source_catalog import validate_catalog
catalog=load(BATCH/'current-source-catalog.json');validate_catalog(ROOT,catalog)
ledger=load(ROOT/'bench/original_program_scope.json')
before=load(BATCH/'evidence/ledger-before.json')
assert ledger==before,'refuse overwriting changed ledger'
rows={t['task']:t for t in catalog['tasks']}
supp={t['task']:t for t in load(BATCH/'catalog-supplement.json')['tasks']}
review=load(BATCH/'root-review.json')
assert len(ledger['tasks'])==183 and all(t['status']=='pending' for t in ledger['tasks'])
changed=[]
for row in ledger['tasks']:
 old=copy.deepcopy(row)
 row['input_sha256']=task_bindings(ROOT,catalog,rows[row['task']])
 if row['task']=='relu_triton_kernel':
  ext=supp[row['task']]
  row['structural_gaps']=[{'kind':'remaining_original_program_scope','finding':item,'can_defer_as_numeric_assumption':False}
                        for item in review['whole_stage_residuals']['relu_triton_kernel']]
  row['initial_assessment']='structural_gaps_confirmed'
  row['partial_progress']={
   'status':'reviewed_source_byte_composition_not_whole_program_acceptance',
   'declarations':[{k:d[k] for k in ('file','declaration','statement_sha256','dtype_profile','covers_stages','model_kind','role','scope_relation')}
                   for d in ext['declarations']],
   'review':ext['review'],'execution_contracts':7,'protocol_contracts':0,'protocols_counted_as_execution':False,
   'exact_numeric_registration':'The older access-grid declaration has nine N1025/B1024/sm90 cases; the two new source-coupled declarations are not yet registered.'}
 if row!=old:changed.append(row['task'])
assert changed==['relu_triton_kernel'],changed
ledger['catalog_sha256']=sha(canonical(catalog))
ledger['progress_checkpoint']=(BATCH/'README.md').relative_to(ROOT).as_posix()
queue=ledger['deferred_queue'];queue['sha256']=hash_file(ROOT/queue['path'])
assert sum(t.get('initial_assessment')=='structural_gaps_confirmed' for t in ledger['tasks'])==66
write_json(ROOT/'bench/original_program_scope.json',ledger)
print(json.dumps({'accepted':0,'pending':183,'confirmed_gaps':66,'unreviewed':117,'changed_tasks':changed,
                  'catalog_sha256':ledger['catalog_sha256'],'queue_sha256':queue['sha256']}))
