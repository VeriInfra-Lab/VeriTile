from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,json,re,shutil,subprocess,sys
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');T=Path('/tmp/veritile-standardization-20261015');B=R/'bench/review-checkpoints/2026-10-16/vector-typed-host-and-rotary-grid';OLD=R/'bench/review-checkpoints/2026-10-16/matrix-scope-and-relu-fp64';BASE=R/'bench/review-checkpoints/2026-10-15/original-scope-extensions/metadata-baseline';G=T/'vector-rotary-current-global'
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
a,b=claims(old),claims(cat);assert len(a)==945 and len(b)==950 and all(b[k]==v for k,v in a.items())
assert set(cat['registered_modules'])-set(old['registered_modules'])=={module_name(p) for p in s['new_source_sha256']};assert all(cat['registered_modules'][k]==v for k,v in old['registered_modules'].items())
assert sum(len(t['declarations']) for t in supp['tasks'])==57
sr={t['task']:t for t in supp['tasks']};osr={t['task']:t for t in osupp['tasks']}
for name,prev in osr.items():
 if name=='vector_addition':assert sr[name]['declarations'][:len(prev['declarations'])]==prev['declarations']
 else:assert sr[name]==prev
assert set(sr)-set(osr)=={'rotary_emb'}
ledger=load(R/'bench/original_program_scope.json');prior=load(B/'evidence/ledger-before.json');priorRows={t['task']:t for t in prior['tasks']};catRows={t['task']:t for t in cat['tasks']};queue=load(R/ledger['deferred_queue']['path'])
assert ledger['catalog_sha256']==sha(canonical(cat))
for row in ledger['tasks']:
 assert row['input_sha256']==task_bindings(R,cat,catRows[row['task']])
 if row['task']=='matrix_transpose':
  expected=copy.deepcopy(priorRows['matrix_transpose']);expected['review']=row['review'];expected['accepted_scope_progress']['scope_review']=row['review'];assert row==expected;assert validate_task(R,cat,catRows[row['task']],row,queue)=='verified'
 elif row['task'] not in s['changed_tasks']:assert row==priorRows[row['task']]
assert sum(t['status']=='verified' for t in ledger['tasks'])==1 and sum(t['status']=='pending' for t in ledger['tasks'])==182
q='bench/tritonbench_g/deferred_numerical_experiments.json';exact(R/q,s['queue_sha256']);assert (R/q).read_bytes()==subprocess.check_output(['git','show',s['base_commit']+':'+q],cwd=R);assert len(queue['rules'])==29
assert not list(B.rglob('*.lean')) and not list(B.rglob('*.olean'))
g=load(G/'receipt.json');assert g['catalog_sha256']==sha(canonical(cat));fail={k for k,v in g['gates'].items() if v['status']!='passed'};assert fail=={'scope_review','comparator_preflight','library_replay'},fail
for name,v in g['gates'].items():
 if 'log' in v:exact(G/v['log'],v['log_sha256'])
 if 'exit_code' in v:assert (v['exit_code']==0)==(v['status']=='passed')
scope=load(G/'scope_review.log');assert scope['counts']=={'verified':1,'not_accepted':182};assert scope['errors']==['182 original-program scopes remain unaccepted'];assert next(t for t in scope['tasks'] if t['task']=='matrix_transpose')['status']=='verified'
for name in ['comparator_preflight','library_replay']:assert 'Official comparator is required' in (G/g['gates'][name]['log']).read_text()
build=T/'vector-rotary-integration-build.log';assert re.findall(r'Build completed successfully \(([\d,]+) jobs\)\.',build.read_text())==['7369']
shards=T/'vector-rotary-shard-tests.log';assert re.findall(r'^Ran (\d+) tests? in ',shards.read_text(),re.M)==['25'] and shards.read_text().rstrip().endswith('OK')
for p in [G/'receipt.json',G/'value-metadata.json',*G.glob('*.log')]:
 if p.exists():d=B/'evidence/global-gates'/p.name;d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d)
actual={'integration-build':{'exit_code':0,'jobs':7369},'catalog':{'exit_code':0,'public_declarations':950},'rebind-ledger':{'exit_code':0,'scope_verified':1,'pending':182},'global':{'exit_code':1,'failed_gates':sorted(fail)},'shard-tests':{'exit_code':0,'tests':25}}
for name,row in actual.items():
 p=T/('vector-rotary-'+name+'.log');d=B/'evidence/integration'/p.name;d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d);row['log']={'path':d.relative_to(B).as_posix(),'sha256':hash_file(d)}
for p,name in [(T/'rebind_vector_rotary_ledger.py','rebind-ledger.py'),(Path(__file__),'finalize.py')]:shutil.copyfile(p,B/'evidence'/name)
write_json(B/'evidence/observed-exits.json',actual)
readme=B/'README.md';readme.write_text(readme.read_text()+'''\n## Observed integration results\n\nThe eight-target build passed (7,369 jobs), catalog generation passed, and all\n25 sharding tests passed. The unchanged schema2 checker preserves the matrix\nsource acceptance after a catalog-only review rebind. Source-scope progress is\n**1 verified / 183**, with **182 pending** and all 183 reviewed. Vector and rotary\nremain partial progress; no numerical registrations were added.\n\nThe actual global audit exited 1: its only failures are scope_review (182 pending),\ncomparator_preflight and library_replay (official comparator unavailable). Every\nother gate passed. Whole-corpus CI, official replay and GPU correctness are not\nclaimed. No push was retried after the previously observed authentication failure.\n\nAll 945 prior public records, 52 prior supplemental declarations, 180 unrelated\nscope rows, 29 numerical rules and the original metadata baseline are unchanged.\nMatrix coverage, source hashes, assumptions and public statements are unchanged;\nits review catalog binding was refreshed in a separate immutable record. The\ncurrent catalog contains 950 public declarations and the supplement contains 51\nexecution plus six protocol contracts across 15 tasks. Producer proof failures\nand raw evidence are retained, including the disclosed rotary preexisting log\nloss and distinct replay; no reconstructed log is called an original raw log.\n''')
paths={R/p for p in s['new_source_sha256']}|{R/'.github/workflows/bench-audit.yml',R/'bench/original_program_scope.json',R/'bench/original_program_scope.md',R/q,R/'lakefile.toml',R/'lake-manifest.json',R/'lean-toolchain'}|{p for p in B.rglob('*') if p.is_file()}
v={'schema':1,'kind':'vector_typed_host_rotary_grid_integration','recorded_at_utc':datetime.now(timezone.utc).isoformat(),'base_commit':s['base_commit'],'status':'partial_extensions_integrated_one_verified_scope_comparator_blocked','scope':{'requested':183,'verified':1,'pending':182,'reviewed':183,'unreviewed':0},'inventory':{'tasks':cat['summary']['tasks'],'public_declarations':len(b),'task_modules':cat['summary']['task_modules'],'registered_modules':len(cat['registered_modules']),'supplement_execution':51,'supplement_protocol':6},'checked_input_sha256':{p.relative_to(R).as_posix():hash_file(p) for p in sorted(paths)},'actual_runs':actual,'preservation':{'prior_public_statements_and_metadata':945,'prior_supplemental_declarations':52,'unrelated_scope_rows_unchanged':180,'matrix_source_and_coverage_unchanged':True,'registered_numerical_rules':29,'original_metadata_baseline_unchanged':s['metadata_baseline_sha256']},'global_gates':{k:v['status'] for k,v in g['gates'].items()},'current_comparator_success':False,'native_execution':False,'new_exact_numerical_registrations':0}
write_json(B/'validation.json',v);print(json.dumps({'validation_sha256':hash_file(B/'validation.json'),'scope':v['scope'],'inventory':v['inventory']},indent=2))
