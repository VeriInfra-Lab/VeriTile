from pathlib import Path
from datetime import datetime,timezone
import ast,copy,json,re,shutil,subprocess,sys
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');T=Path('/tmp/veritile-standardization-20261015');B=R/'bench/review-checkpoints/2026-10-16/quant-global-finite-source-acceptance';OLD=R/'bench/review-checkpoints/2026-10-16/vector-finite-source-acceptance';BASE=R/'bench/review-checkpoints/2026-10-15/original-scope-extensions/metadata-baseline';G=T/'quant-global-finite-current-global'
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
a,b=claims(old),claims(cat);assert len(a)==954 and len(b)==955 and all(b[k]==v for k,v in a.items())
assert set(cat['registered_modules'])-set(old['registered_modules'])=={module_name(p) for p in s['new_source_sha256']};assert all(cat['registered_modules'][k]==v for k,v in old['registered_modules'].items())
assert sum(len(t['declarations']) for t in supp['tasks'])==62
sr={t['task']:t for t in supp['tasks']};osr={t['task']:t for t in osupp['tasks']};assert set(sr)==set(osr)
for name,prev in osr.items():
 if name=='quantize_global':assert sr[name]['declarations'][:len(prev['declarations'])]==prev['declarations']
 else:assert sr[name]==prev
ledger=load(R/'bench/original_program_scope.json');prior=load(B/'evidence/ledger-before.json');priorRows={t['task']:t for t in prior['tasks']};catRows={t['task']:t for t in cat['tasks']};queue=load(R/ledger['deferred_queue']['path'])
assert ledger['catalog_sha256']==sha(canonical(cat));assert ledger['deferred_queue']['sha256']==hash_file(R/ledger['deferred_queue']['path'])
for row in ledger['tasks']:
 assert row['input_sha256']==task_bindings(R,cat,catRows[row['task']])
 if row['task'] in ['matrix_transpose','relu_triton_kernel','vector_addition']:
  expected=copy.deepcopy(priorRows[row['task']]);expected['review']=row['review'];expected['accepted_scope_progress']['scope_review']=row['review'];assert row==expected;assert validate_task(R,cat,catRows[row['task']],row,queue)==priorRows[row['task']]['status']
 elif row['task']=='quantize_global':assert validate_task(R,cat,catRows[row['task']],row,queue)=='verified_with_deferred_numerics'
 else:assert row==priorRows[row['task']]
assert sum(t['status']=='pending' for t in ledger['tasks'])==179
legacy,authority,actualQueue=collect(R);assert actualQueue==queue;assert len(queue['rules'])==100
previous=load(B/'evidence/numerical-queue-before.json');assert queue['rules'][:84]==previous['rules'];assert queue['exact_callback_requirements'][:5]==previous['exact_callback_requirements'];assert all(queue['exact_binding_provenance'][k]==v for k,v in previous['exact_binding_provenance'].items())
assert validate_exact_queue(queue,authority,legacy)=={'status':'passed','exact_bindings':1206,'declaration_cases':5399,'native_executions':0,'structural_acceptance':False}
# The prior reviewed adapter functions and every validator predicate remain
# unchanged. Only four exact dtype labels and the newly pinned extension are added.
adapter=ast.parse((R/'bench/reviewed_numerical_extensions.py').read_text());functions={n.name:n for n in adapter.body if isinstance(n,ast.FunctionDef)}
prior_adapter=ast.parse(subprocess.check_output(['git','show',s['base_commit']+':bench/reviewed_numerical_extensions.py'],cwd=R,text=True))
for oldfunc in prior_adapter.body:
 if isinstance(oldfunc,ast.FunctionDef) and oldfunc.name!='extend_reviewed_authority':assert ast.dump(oldfunc)==ast.dump(functions[oldfunc.name])
newfunc=next(n for n in ast.parse((T/'quant_global_authority_adapter.py').read_text()).body if isinstance(n,ast.FunctionDef));assert ast.dump(newfunc)==ast.dump(functions[newfunc.name])
oldcheck=ast.parse(subprocess.check_output(['git','show',s['base_commit']+':bench/check_original_program_scope.py'],cwd=R,text=True));newcheck=ast.parse((R/'bench/check_original_program_scope.py').read_text())
def pop_dtype(tree):
 node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='DTYPES' for x in n.targets));tree.body.remove(node);return ast.literal_eval(node.value)
oldtypes=pop_dtype(oldcheck);newtypes=pop_dtype(newcheck);assert newtypes==oldtypes|{'complex32','complex64','complex128','fp8e8m0fnu'};assert ast.dump(oldcheck)==ast.dump(newcheck)
assert not list(B.rglob('*.lean')) and not list(B.rglob('*.olean'))
g=load(G/'receipt.json');assert g['catalog_sha256']==sha(canonical(cat));fail={k for k,v in g['gates'].items() if v['status']!='passed'};assert fail=={'scope_review','comparator_preflight','library_replay'},fail
for name,v in g['gates'].items():
 if 'log' in v:exact(G/v['log'],v['log_sha256'])
 if 'exit_code' in v:assert (v['exit_code']==0)==(v['status']=='passed')
scope=load(G/'scope_review.log');assert scope['counts']=={'verified':1,'verified_with_deferred_numerics':3,'not_accepted':179};assert scope['errors']==['179 original-program scopes remain unaccepted']
for name in ['comparator_preflight','library_replay']:assert 'Official comparator is required' in (G/g['gates'][name]['log']).read_text()
build=T/'quant-global-finite-integration-build.log';assert re.findall(r'Build completed successfully \(([\d,]+) jobs\)\.',build.read_text())==['7400']
for kind,n in [('other-bench-tests',210),('exact-tests',37),('shard-tests',25)]:
 log=T/('quant-global-finite-'+kind+'.log');assert re.findall(r'^Ran (\d+) tests? in ',log.read_text(),re.M)==[str(n)] and log.read_text().rstrip().endswith('OK')
for p in [G/'receipt.json',G/'value-metadata.json',*G.glob('*.log')]:
 if p.exists():d=B/'evidence/global-gates'/p.name;d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d)
actual={'integration-build':{'exit_code':0,'jobs':7400},'catalog':{'exit_code':0,'public_declarations':955},'queue':{'exit_code':0,'rules':100},'rebind-ledger':{'exit_code':0,'verified':1,'verified_with_deferred_numerics':3,'pending':179},'global':{'exit_code':1,'failed_gates':sorted(fail)},'other-bench-tests':{'exit_code':0,'tests':210,'selection':'all benchmark tests except the separately run exact_numerical_bindings module'},'exact-tests':{'exit_code':0,'tests':37},'shard-tests':{'exit_code':0,'tests':25}}
for name,row in actual.items():
 p=T/('quant-global-finite-'+name+'.log');d=B/'evidence/integration'/p.name;d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d);row['log']={'path':d.relative_to(B).as_posix(),'sha256':hash_file(d)}
for p in (T/'quant-global-finite-integration-attempts').rglob('*'):
 if p.is_file():d=B/'evidence/integration-attempts'/p.relative_to(T/'quant-global-finite-integration-attempts');d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d)
for p,name in [(T/'rebind_quant_global_finite_ledger.py','rebind-ledger.py'),(Path(__file__),'finalize.py')]:shutil.copyfile(p,B/'evidence'/name)
shutil.copyfile(T/'quant_global_authority_adapter.py',B/'evidence/authority-adapter.py')
write_json(B/'evidence/observed-exits.json',actual)
readme=B/'README.md';readme.write_text(readme.read_text()+'\n## Observed integration results\n\nThe eight-target build passed (7,400 jobs). All 247 benchmark Python tests\npassed: 37 exact numerical tests and the remaining 210 tests, run separately.\nAll 25 sharding tests passed. The unchanged schema2 predicates accept quantize_global\nwith exact numerical deferrals after adding four exact rejected-dtype labels.\nSource scope is **4 accepted / 183**: matrix transpose verified, ReLU, vector\naddition and quantize_global verified with deferred numerics, and **179 pending**.\nAll 183 original programs have recorded scope assessments.\n\nThe actual global audit exited 1 with only scope_review (179 pending),\ncomparator_preflight and library_replay (official comparator unavailable) failing.\nEvery other gate passed. Whole-corpus CI, official replay and GPU correctness\nare not claimed. No push was retried after the prior authentication failure.\n\nThe previous 954 public records, 61 supplemental declarations, 179 unrelated\nscope rows, 84 numerical rules and original metadata baseline remain unchanged.\nThe current catalog has 955 public declarations and 62 supplemental contracts\n(56 execution, six protocol) across 17 tasks. The queue has 100 rules, with\n1,206 exact ports across 5,399 cases in six declarations. Prior reviewed numerical\nadapter functions are unchanged; strict whole-queue equality remains mandatory.\nThe source-scope checker syntax tree is unchanged except for four dtype labels.\nAll producer, independent replay and failed-attempt evidence is preserved.\n')
active=['.github/workflows/bench-audit.yml','bench/original_program_scope.json','bench/original_program_scope.md','bench/exact_numerical_bindings.py','bench/check_original_program_scope.py','bench/reviewed_numerical_extensions.py','bench/deferred_numerical_experiments.py','bench/tests/test_exact_numerical_bindings.py','bench/tritonbench_g/deferred_numerical_experiments.json','bench/tritonbench_g/deferred_numerical_experiments.md','lakefile.toml','lake-manifest.json','lean-toolchain']
paths={R/p for p in s['new_source_sha256']}|{R/p for p in active}|{p for p in B.rglob('*') if p.is_file()}
v={'schema':1,'kind':'quant_global_finite_source_acceptance_integration','recorded_at_utc':datetime.now(timezone.utc).isoformat(),'base_commit':s['base_commit'],'status':'finite_quant_global_source_accepted_with_deferred_numerics_comparator_blocked','scope':{'requested':183,'accepted':4,'verified':1,'verified_with_deferred_numerics':3,'pending':179,'reviewed':183,'unreviewed':0},'inventory':{'public_declarations':len(b),'task_modules':cat['summary']['task_modules'],'registered_modules':len(cat['registered_modules']),'supplement_execution':56,'supplement_protocol':6},'checked_input_sha256':{p.relative_to(R).as_posix():hash_file(p) for p in sorted(paths)},'actual_runs':actual,'preservation':{'prior_public_records':954,'prior_supplement_declarations':61,'unrelated_scope_rows':179,'prior_numeric_rules':84,'metadata_baseline_unchanged':s['metadata_baseline_sha256']},'global_gates':{k:v['status'] for k,v in g['gates'].items()},'new_numeric_families':16,'new_numeric_ports':420,'numerical_conformance_validated':False,'current_comparator_success':False,'native_execution':False}
write_json(B/'validation.json',v);print(json.dumps({'validation_sha256':hash_file(B/'validation.json'),'scope':v['scope'],'inventory':v['inventory']},indent=2))
