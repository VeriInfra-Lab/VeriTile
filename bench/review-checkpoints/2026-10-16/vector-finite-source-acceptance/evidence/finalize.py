from pathlib import Path
from datetime import datetime,timezone
import ast,copy,json,re,shutil,subprocess,sys
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');T=Path('/tmp/veritile-standardization-20261015');B=R/'bench/review-checkpoints/2026-10-16/vector-finite-source-acceptance';OLD=R/'bench/review-checkpoints/2026-10-16/softmax-source-host-grid';BASE=R/'bench/review-checkpoints/2026-10-15/original-scope-extensions/metadata-baseline';G=T/'vector-finite-current-global'
sys.path.insert(0,str(R/'bench'))
from adapter_common import canonical,hash_file,load,sha,write_json,module_name
from check_original_program_scope import task_bindings,validate_task
from exact_numerical_bindings import collect,validate_exact_queue

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
a,b=claims(old),claims(cat);assert len(a)==953 and len(b)==954 and all(b[k]==v for k,v in a.items())
assert set(cat['registered_modules'])-set(old['registered_modules'])=={module_name(p) for p in s['new_source_sha256']};assert all(cat['registered_modules'][k]==v for k,v in old['registered_modules'].items())
assert sum(len(t['declarations']) for t in supp['tasks'])==61
sr={t['task']:t for t in supp['tasks']};osr={t['task']:t for t in osupp['tasks']};assert set(sr)==set(osr)
for name,prev in osr.items():
 if name=='vector_addition':assert sr[name]['declarations'][:len(prev['declarations'])]==prev['declarations']
 else:assert sr[name]==prev
ledger=load(R/'bench/original_program_scope.json');prior=load(B/'evidence/ledger-before.json');priorRows={t['task']:t for t in prior['tasks']};catRows={t['task']:t for t in cat['tasks']};queue=load(R/ledger['deferred_queue']['path'])
assert ledger['catalog_sha256']==sha(canonical(cat));assert ledger['deferred_queue']['sha256']==hash_file(R/ledger['deferred_queue']['path'])
for row in ledger['tasks']:
 assert row['input_sha256']==task_bindings(R,cat,catRows[row['task']])
 if row['task'] in ['matrix_transpose','relu_triton_kernel']:
  expected=copy.deepcopy(priorRows[row['task']]);expected['review']=row['review'];expected['accepted_scope_progress']['scope_review']=row['review'];assert row==expected;assert validate_task(R,cat,catRows[row['task']],row,queue)==priorRows[row['task']]['status']
 elif row['task']=='vector_addition':assert validate_task(R,cat,catRows[row['task']],row,queue)=='verified_with_deferred_numerics'
 else:assert row==priorRows[row['task']]
assert sum(t['status']=='pending' for t in ledger['tasks'])==180
legacy,authority,actualQueue=collect(R);assert actualQueue==queue;assert len(queue['rules'])==84
previous=load(B/'evidence/numerical-queue-before.json');assert queue['rules'][:41]==previous['rules'];assert queue['exact_callback_requirements'][:4]==previous['exact_callback_requirements'];assert all(queue['exact_binding_provenance'][k]==v for k,v in previous['exact_binding_provenance'].items())
assert validate_exact_queue(queue,authority,legacy)=={'status':'passed','exact_bindings':786,'declaration_cases':2999,'native_executions':0,'structural_acceptance':False}
# Reformatting the producer adapter did not change its syntax tree. ReLU's
# existing adapter body is unchanged except for its private function name.
adapter=ast.parse((R/'bench/reviewed_numerical_extensions.py').read_text());functions={n.name:n for n in adapter.body if isinstance(n,ast.FunctionDef)}
producer=next(n for n in ast.parse((B/'evidence/producer/vector/authority_adapter.py').read_text()).body if isinstance(n,ast.FunctionDef));producer.name='_extend_vector_authority';assert ast.dump(producer)==ast.dump(functions[producer.name])
prior_adapter=ast.parse(subprocess.check_output(['git','show',s['base_commit']+':bench/reviewed_numerical_extensions.py'],cwd=R,text=True));prior_function=next(n for n in prior_adapter.body if isinstance(n,ast.FunctionDef));prior_function.name='_extend_relu_authority';assert ast.dump(prior_function)==ast.dump(functions[prior_function.name])
assert not list(B.rglob('*.lean')) and not list(B.rglob('*.olean'))
g=load(G/'receipt.json');assert g['catalog_sha256']==sha(canonical(cat));fail={k for k,v in g['gates'].items() if v['status']!='passed'};assert fail=={'scope_review','comparator_preflight','library_replay'},fail
for name,v in g['gates'].items():
 if 'log' in v:exact(G/v['log'],v['log_sha256'])
 if 'exit_code' in v:assert (v['exit_code']==0)==(v['status']=='passed')
scope=load(G/'scope_review.log');assert scope['counts']=={'verified':1,'verified_with_deferred_numerics':2,'not_accepted':180};assert scope['errors']==['180 original-program scopes remain unaccepted']
for name in ['comparator_preflight','library_replay']:assert 'Official comparator is required' in (G/g['gates'][name]['log']).read_text()
build=T/'vector-finite-integration-build.log';assert re.findall(r'Build completed successfully \(([\d,]+) jobs\)\.',build.read_text())==['7389']
for kind,n in [('other-bench-tests',210),('exact-tests',34),('shard-tests',25)]:
 log=T/('vector-finite-'+kind+'.log');assert re.findall(r'^Ran (\d+) tests? in ',log.read_text(),re.M)==[str(n)] and log.read_text().rstrip().endswith('OK')
for p in [G/'receipt.json',G/'value-metadata.json',*G.glob('*.log')]:
 if p.exists():d=B/'evidence/global-gates'/p.name;d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d)
actual={'integration-build':{'exit_code':0,'jobs':7389},'catalog':{'exit_code':0,'public_declarations':954},'queue':{'exit_code':0,'rules':84},'rebind-ledger':{'exit_code':0,'verified':1,'verified_with_deferred_numerics':2,'pending':180},'global':{'exit_code':1,'failed_gates':sorted(fail)},'other-bench-tests':{'exit_code':0,'tests':210,'selection':'all benchmark tests except the separately run exact_numerical_bindings module'},'exact-tests':{'exit_code':0,'tests':34},'shard-tests':{'exit_code':0,'tests':25}}
for name,row in actual.items():
 p=T/('vector-finite-'+name+'.log');d=B/'evidence/integration'/p.name;d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d);row['log']={'path':d.relative_to(B).as_posix(),'sha256':hash_file(d)}
for p in (T/'vector-finite-integration-attempts').rglob('*'):
 if p.is_file():d=B/'evidence/integration-attempts'/p.relative_to(T/'vector-finite-integration-attempts');d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d)
for p,name in [(T/'rebind_vector_finite_ledger.py','rebind-ledger.py'),(Path(__file__),'finalize.py')]:shutil.copyfile(p,B/'evidence'/name)
write_json(B/'evidence/observed-exits.json',actual)
readme=B/'README.md';readme.write_text(readme.read_text()+'''\n## Observed integration results

The eight-target build passed (7,389 jobs). All 244 benchmark Python tests
passed: 34 exact numerical tests and the remaining 210 tests, run separately
without repeating the exact suite. All 25 sharding tests passed. The unchanged
schema2 checker accepts vector addition with exact numerical deferrals.
Source scope is **3 accepted / 183**: matrix transpose verified, ReLU and vector
addition verified with deferred numerics, and **180 pending**. All 183 are reviewed.

The actual global audit exited 1 with only scope_review (180 pending),
comparator_preflight and library_replay (official comparator unavailable) failing.
Every other gate passed. Whole-corpus CI, official replay and GPU correctness
are not claimed. No push was retried after the prior authentication failure.

The previous 953 public records, 60 supplemental declarations, 180 unrelated
scope rows, 41 numerical rules and original metadata baseline remain unchanged.
The current catalog has 954 public declarations and 61 supplemental contracts
(55 execution, six protocol) across 16 tasks. The numerical queue now has 84
rules, including 43 new reviewed vector families, with 786 exact ports across
2,999 cases in five declarations. Whole-queue equality still rejects changed
or missing sources, reviews, cases and scalar ports. ReLU's existing adapter
body and the reviewed vector adapter syntax tree are preserved. All producer
and integration failures remain byte-preserved.
''')
active=['.github/workflows/bench-audit.yml','bench/original_program_scope.json','bench/original_program_scope.md','bench/exact_numerical_bindings.py','bench/reviewed_numerical_extensions.py','bench/deferred_numerical_experiments.py','bench/tests/test_exact_numerical_bindings.py','bench/tritonbench_g/deferred_numerical_experiments.json','bench/tritonbench_g/deferred_numerical_experiments.md','lakefile.toml','lake-manifest.json','lean-toolchain']
paths={R/p for p in s['new_source_sha256']}|{R/p for p in active}|{p for p in B.rglob('*') if p.is_file()}
v={'schema':1,'kind':'vector_finite_source_acceptance_integration','recorded_at_utc':datetime.now(timezone.utc).isoformat(),'base_commit':s['base_commit'],'status':'finite_vector_source_accepted_with_deferred_numerics_comparator_blocked','scope':{'requested':183,'accepted':3,'verified':1,'verified_with_deferred_numerics':2,'pending':180,'reviewed':183,'unreviewed':0},'inventory':{'public_declarations':len(b),'task_modules':cat['summary']['task_modules'],'registered_modules':len(cat['registered_modules']),'supplement_execution':55,'supplement_protocol':6},'checked_input_sha256':{p.relative_to(R).as_posix():hash_file(p) for p in sorted(paths)},'actual_runs':actual,'preservation':{'prior_public_records':953,'prior_supplement_declarations':60,'unrelated_scope_rows':180,'prior_numeric_rules':41,'metadata_baseline_unchanged':s['metadata_baseline_sha256']},'global_gates':{k:v['status'] for k,v in g['gates'].items()},'new_numeric_families':43,'new_numeric_ports':616,'numerical_conformance_validated':False,'current_comparator_success':False,'native_execution':False}
write_json(B/'validation.json',v);print(json.dumps({'validation_sha256':hash_file(B/'validation.json'),'scope':v['scope'],'inventory':v['inventory']},indent=2))
