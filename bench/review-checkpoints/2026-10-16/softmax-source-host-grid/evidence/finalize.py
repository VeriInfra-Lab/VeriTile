from pathlib import Path
from datetime import datetime,timezone
import copy,json,re,shutil,subprocess,sys
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');T=Path('/tmp/veritile-standardization-20261015');B=R/'bench/review-checkpoints/2026-10-16/softmax-source-host-grid';OLD=R/'bench/review-checkpoints/2026-10-16/relu-finite-source-acceptance';BASE=R/'bench/review-checkpoints/2026-10-15/original-scope-extensions/metadata-baseline';G=T/'softmax-host-grid-current-global'
sys.path.insert(0,str(R/'bench'))
from adapter_common import canonical,hash_file,load,sha,write_json,module_name
from check_original_program_scope import task_bindings,validate_task

def hashes(p):return {q.relative_to(p).as_posix():hash_file(q) for q in p.rglob('*') if q.is_file()}
def exact(p,h):assert hash_file(p)==h,str(p)
assert not (B/'validation.json').exists();s=load(B/'evidence/preparation-state.json');assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()==s['base_commit']
assert hashes(OLD)==s['predecessor_file_sha256'];assert hashes(BASE)==s['metadata_baseline_sha256']
for p,h in load(BASE/'input-sha256.json').items():exact(R/p,h)
for p,h in s['new_source_sha256'].items():exact(R/p,h)
for p,h in load(B/'root-review.json')['reviewed_input_sha256'].items():exact(R/p,h)
for p,h in load(B/'evidence-index.json')['retained_exact_artifacts'].items():exact(B/p,h)
cat=load(B/'current-source-catalog.json');old=load(OLD/'current-source-catalog.json');supp=load(B/'catalog-supplement.json');osupp=load(OLD/'catalog-supplement.json')
claims=lambda c:{(t['task'],d['file'],d['declaration']):d for t in c['tasks'] for d in t['declarations']}
a,b=claims(old),claims(cat);assert len(a)==951 and len(b)==953 and all(b[k]==v for k,v in a.items())
assert set(cat['registered_modules'])-set(old['registered_modules'])=={module_name(p) for p in s['new_source_sha256']};assert all(cat['registered_modules'][k]==v for k,v in old['registered_modules'].items())
assert sum(len(t['declarations']) for t in supp['tasks'])==60
sr={t['task']:t for t in supp['tasks']};osr={t['task']:t for t in osupp['tasks']};assert set(sr)-set(osr)=={'softmax_flaggems'}
for name,prev in osr.items():assert sr[name]==prev
ledger=load(R/'bench/original_program_scope.json');prior=load(B/'evidence/ledger-before.json');priorRows={t['task']:t for t in prior['tasks']};catRows={t['task']:t for t in cat['tasks']};queue=load(R/ledger['deferred_queue']['path'])
assert ledger['catalog_sha256']==sha(canonical(cat));assert ledger['deferred_queue']==prior['deferred_queue']
for row in ledger['tasks']:
 assert row['input_sha256']==task_bindings(R,cat,catRows[row['task']])
 if row['task'] in ['matrix_transpose','relu_triton_kernel']:
  expected=copy.deepcopy(priorRows[row['task']]);expected['review']=row['review'];expected['accepted_scope_progress']['scope_review']=row['review'];assert row==expected;assert validate_task(R,cat,catRows[row['task']],row,queue)==expected['status']
 elif row['task']!='softmax_flaggems':assert row==priorRows[row['task']]
assert sum(t['status']=='verified' for t in ledger['tasks'])==1 and sum(t['status']=='verified_with_deferred_numerics' for t in ledger['tasks'])==1 and sum(t['status']=='pending' for t in ledger['tasks'])==181
q='bench/tritonbench_g/deferred_numerical_experiments.json';exact(R/q,s['queue_sha256']);assert (R/q).read_bytes()==subprocess.check_output(['git','show',s['base_commit']+':'+q],cwd=R);assert len(queue['rules'])==41
assert not list(B.rglob('*.lean')) and not list(B.rglob('*.olean'))
g=load(G/'receipt.json');assert g['catalog_sha256']==sha(canonical(cat));fail={k for k,v in g['gates'].items() if v['status']!='passed'};assert fail=={'scope_review','comparator_preflight','library_replay'},fail
for name,v in g['gates'].items():
 if 'log' in v:exact(G/v['log'],v['log_sha256'])
 if 'exit_code' in v:assert (v['exit_code']==0)==(v['status']=='passed')
scope=load(G/'scope_review.log');assert scope['counts']=={'verified':1,'verified_with_deferred_numerics':1,'not_accepted':181};assert scope['errors']==['181 original-program scopes remain unaccepted']
for name in ['comparator_preflight','library_replay']:assert 'Official comparator is required' in (G/g['gates'][name]['log']).read_text()
build=T/'softmax-host-grid-integration-build.log';assert re.findall(r'Build completed successfully \(([\d,]+) jobs\)\.',build.read_text())==['7383']
shards=T/'softmax-host-grid-shard-tests.log';assert re.findall(r'^Ran (\d+) tests? in ',shards.read_text(),re.M)==['25'] and shards.read_text().rstrip().endswith('OK')
for p in [G/'receipt.json',G/'value-metadata.json',*G.glob('*.log')]:
 if p.exists():d=B/'evidence/global-gates'/p.name;d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d)
actual={'integration-build':{'exit_code':0,'jobs':7383},'catalog':{'exit_code':0,'public_declarations':953},'rebind-ledger':{'exit_code':0,'verified':1,'verified_with_deferred_numerics':1,'pending':181},'global':{'exit_code':1,'failed_gates':sorted(fail)},'shard-tests':{'exit_code':0,'tests':25}}
for name,row in actual.items():
 p=T/('softmax-host-grid-'+name+'.log');d=B/'evidence/integration'/p.name;d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d);row['log']={'path':d.relative_to(B).as_posix(),'sha256':hash_file(d)}
for p,name in [(T/'rebind_softmax_host_grid_ledger.py','rebind-ledger.py'),(Path(__file__),'finalize.py')]:shutil.copyfile(p,B/'evidence'/name)
write_json(B/'evidence/observed-exits.json',actual)
readme=B/'README.md';readme.write_text(readme.read_text()+'''\n## Observed integration results\n\nThe eight-target build passed (7,383 jobs), establishing active import resolution\nfor all nine new modules. Catalog generation and all 25 sharding tests passed.\nThe unchanged schema2 checker preserves both accepted source scopes after their\ncatalog-only review rebinds: **2 accepted / 183**, with **181 pending** and all\n183 reviewed. Softmax remains partial progress; the 41 numerical rules are intact.\n\nThe actual global audit exited 1. Its only failures are scope_review (181 pending),\ncomparator_preflight and library_replay (official comparator unavailable). Every\nother gate passed. Whole-corpus CI, official replay and GPU correctness are not\nclaimed. No push was retried after the prior authentication failure.\n\nAll 951 previous public records, 58 supplemental contracts, 180 unrelated scope\nrows and the original metadata baseline remain unchanged. The new catalog has\n953 public declarations and 60 supplemental contracts (54 execution, six protocol)\nacross 16 tasks. Old accepted scope source/coverage/numerical bindings are intact.\nThe initial integration preparation failed while looking for an optional previous\nnumerical-followup file; its exact script/log are retained and the corrected\npreparation adds no numerical follow-up or queue changes.\n''')
paths={R/p for p in s['new_source_sha256']}|{R/'.github/workflows/bench-audit.yml',R/'bench/original_program_scope.json',R/'bench/original_program_scope.md',R/q,R/'lakefile.toml',R/'lake-manifest.json',R/'lean-toolchain'}|{p for p in B.rglob('*') if p.is_file()}
v={'schema':1,'kind':'softmax_source_host_grid_integration','recorded_at_utc':datetime.now(timezone.utc).isoformat(),'base_commit':s['base_commit'],'status':'partial_extension_integrated_two_accepted_scopes_comparator_blocked','scope':{'requested':183,'accepted':2,'verified':1,'verified_with_deferred_numerics':1,'pending':181,'reviewed':183,'unreviewed':0},'inventory':{'public_declarations':len(b),'task_modules':cat['summary']['task_modules'],'registered_modules':len(cat['registered_modules']),'supplement_execution':54,'supplement_protocol':6},'checked_input_sha256':{p.relative_to(R).as_posix():hash_file(p) for p in sorted(paths)},'actual_runs':actual,'preservation':{'prior_public_records':951,'prior_supplement_declarations':58,'unrelated_scope_rows':180,'prior_numeric_rules':41,'metadata_baseline_unchanged':s['metadata_baseline_sha256']},'global_gates':{k:v['status'] for k,v in g['gates'].items()},'new_numeric_registrations':0,'acceptance_increment':0,'current_comparator_success':False,'native_execution':False}
write_json(B/'validation.json',v);print(json.dumps({'validation_sha256':hash_file(B/'validation.json'),'scope':v['scope'],'inventory':v['inventory']},indent=2))
