from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import ast,copy,json,re,shutil,subprocess,sys
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');T=Path('/tmp/veritile-standardization-20261015');B=R/'bench/review-checkpoints/2026-10-16/quant-transpose-finite-source-acceptance';OLD=R/'bench/review-checkpoints/2026-10-16/quant-global-finite-source-acceptance';BASE=R/'bench/review-checkpoints/2026-10-15/original-scope-extensions/metadata-baseline';G=T/'quant-transpose-finite-current-global'
sys.path.insert(0,str(R/'bench'));from adapter_common import canonical,hash_file,load,sha,write_json,module_name
from check_original_program_scope import task_bindings,validate_task
from exact_numerical_bindings import collect,validate_exact_queue

def hashes(p):return {q.relative_to(p).as_posix():hash_file(q) for q in p.rglob('*') if q.is_file()}
def exact(p,h):assert hash_file(p)==h,str(p)
def oldtext(p):return subprocess.check_output(['git','show',s['base_commit']+':'+p],cwd=R,text=True)
assert not (B/'validation.json').exists();s=load(B/'evidence/preparation-state.json');assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()==s['base_commit'];assert hashes(OLD)==s['predecessor_file_sha256'];assert hashes(BASE)==s['metadata_baseline_sha256']
for p,h in load(BASE/'input-sha256.json').items():exact(R/p,h)
for p,h in s['new_source_sha256'].items():exact(R/p,h)
for p,h in load(B/'root-review.json')['reviewed_input_sha256'].items():exact(R/p,h)
for p,h in load(B/'evidence-index.json')['retained_exact_artifacts'].items():exact(B/p,h)
cat=load(B/'current-source-catalog.json');old=load(OLD/'current-source-catalog.json');supp=load(B/'catalog-supplement.json');osupp=load(OLD/'catalog-supplement.json')
claims=lambda c:{(t['task'],d['file'],d['declaration']):d for t in c['tasks'] for d in t['declarations']}
a,b=claims(old),claims(cat);assert len(a)==955 and len(b)==956 and all(b[k]==v for k,v in a.items());assert set(cat['registered_modules'])-set(old['registered_modules'])=={module_name(p) for p in s['new_source_sha256']};assert all(cat['registered_modules'][k]==v for k,v in old['registered_modules'].items())
assert sum(len(t['declarations']) for t in supp['tasks'])==63;assert Counter(d['role'] for t in supp['tasks'] for d in t['declarations'])=={'parametric_value_contract':57,'protocol_contract':6};assert sum(bool(t['declarations']) for t in supp['tasks'])==16
sr={t['task']:t for t in supp['tasks']};osr={t['task']:t for t in osupp['tasks']};assert set(sr)==set(osr)
for name,prev in osr.items():
 if name=='quant_transpose_kernel':assert sr[name]['declarations'][:len(prev['declarations'])]==prev['declarations']
 else:assert sr[name]==prev
ledger=load(R/'bench/original_program_scope.json');prior=json.loads(oldtext('bench/original_program_scope.json'));priorRows={t['task']:t for t in prior['tasks']};catRows={t['task']:t for t in cat['tasks']};queue=load(R/ledger['deferred_queue']['path']);assert ledger['catalog_sha256']==sha(canonical(cat));assert ledger['deferred_queue']['sha256']==hash_file(R/ledger['deferred_queue']['path'])
for row in ledger['tasks']:
 assert row['input_sha256']==task_bindings(R,cat,catRows[row['task']])
 if row['task'] in ['matrix_transpose','quantize_global']:
  expected=copy.deepcopy(priorRows[row['task']]);expected['review']=row['review'];expected['accepted_scope_progress']['scope_review']=row['review'];assert row==expected
 elif row['task']!='quant_transpose_kernel':assert row==priorRows[row['task']]
assert Counter(t['status'] for t in ledger['tasks'])=={'pending':180,'verified':1,'verified_with_deferred_numerics':2}
legacy,authority,actualQueue=collect(R);assert actualQueue==queue;previous=json.loads(oldtext('bench/tritonbench_g/deferred_numerical_experiments.json'));assert queue['rules'][:100]==previous['rules'];assert queue['exact_callback_requirements'][:6]==previous['exact_callback_requirements'];assert all(queue['exact_binding_provenance'][k]==v for k,v in previous['exact_binding_provenance'].items());assert len(queue['rules'])==116
assert validate_exact_queue(queue,authority,legacy)=={'status':'passed','exact_bindings':1346,'declaration_cases':6262,'native_executions':0,'structural_acceptance':False}
for p in ['bench/check_original_program_scope.py','bench/exact_numerical_bindings.py','bench/exact_scope_gate.py','bench/deferred_numerical_experiments.py','lakefile.toml','lake-manifest.json','lean-toolchain']:assert (R/p).read_text()==oldtext(p),p
current=ast.parse((R/'bench/reviewed_numerical_extensions.py').read_text());functions={n.name:n for n in current.body if isinstance(n,ast.FunctionDef)}
for f in ast.parse(oldtext('bench/reviewed_numerical_extensions.py')).body:
 if isinstance(f,ast.FunctionDef) and f.name!='extend_reviewed_authority':assert ast.dump(f)==ast.dump(functions[f.name])
producer=(T/'quant-transpose-finite-acceptance-revision1-wip/authority_adapter.py').read_text();expected=producer.replace("    for relative, expected in receipt['artifacts'].items():\n        bind(package_relative / relative, expected)","    for relative, expected in receipt['artifacts'].items():\n        # Historical Lean source is archived with exact bytes, never registered as active code.\n        archived = relative + '.txt' if relative.endswith('.lean') else relative\n        bind(package_relative / archived, expected)");assert expected==(R/'bench/quant_transpose_numerical_extension.py').read_text();assert not list(B.rglob('*.lean')) and not list(B.rglob('*.olean'))
g=load(G/'receipt.json');assert g['catalog_sha256']==sha(canonical(cat));fail={k for k,v in g['gates'].items() if v['status']!='passed'};assert fail=={'scope_review','comparator_preflight','library_replay'},fail
for name,v in g['gates'].items():
 if 'log' in v:exact(G/v['log'],v['log_sha256'])
 if 'exit_code' in v:assert (v['exit_code']==0)==(v['status']=='passed')
scope=load(G/'scope_review.log');assert scope['counts']=={'verified':1,'verified_with_deferred_numerics':2,'not_accepted':180};assert scope['errors']==['180 original-program scopes remain unaccepted']
for name in ['comparator_preflight','library_replay']:assert 'Official comparator is required' in (G/g['gates'][name]['log']).read_text()
build=T/'quant-transpose-finite-integration-build.log';assert re.findall(r'Build completed successfully \(([\d,]+) jobs\)\.',build.read_text())==['7418']
for kind,n in [('other-bench-tests',210),('exact-tests',40),('shard-tests',25)]:
 log=T/('quant-transpose-finite-'+kind+'.log');assert re.findall(r'^Ran (\d+) tests? in ',log.read_text(),re.M)==[str(n)] and log.read_text().rstrip().endswith('OK')
for p in [G/'receipt.json',G/'value-metadata.json',*G.glob('*.log')]:
 if p.exists():d=B/'evidence/global-gates'/p.name;d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d)
actual={'integration-build':{'exit_code':0,'jobs':7418},'catalog':{'exit_code':0,'public_declarations':956},'queue':{'exit_code':0,'rules':116},'rebind-ledger':{'exit_code':0,'verified':1,'verified_with_deferred_numerics':2,'pending':180},'global':{'exit_code':1,'failed_gates':sorted(fail)},'other-bench-tests':{'exit_code':0,'tests':210},'exact-tests':{'exit_code':0,'tests':40},'shard-tests':{'exit_code':0,'tests':25},'prepare':{'exit_code':0},'authority':{'exit_code':0}}
for name,row in actual.items():
 p=T/('quant-transpose-finite-'+name+'.log');d=B/'evidence/integration'/p.name;d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d);row['log']={'path':d.relative_to(B).as_posix(),'sha256':hash_file(d)}
for p in (T/'quant-transpose-finite-integration-attempts').rglob('*'):
 if p.is_file():d=B/'evidence/integration-attempts'/p.relative_to(T/'quant-transpose-finite-integration-attempts');d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d)
for name,dest in [('rebind_quant_transpose_finite_ledger.py','rebind-ledger.py'),('install_quant_transpose_authority.py','install-authority.py'),('finalize_quant_transpose_finite_integration.py','finalize.py')]:shutil.copyfile(T/name,B/'evidence'/dest)
write_json(B/'evidence/observed-exits.json',actual)
# Current prose counts actual nonempty task records; older checkpoint prose remains historical.
f=R/'bench/original_program_scope.md';text=f.read_text();assert 'six protocol contracts across 18 tasks' in text;text=text.replace('six protocol contracts across 18 tasks','six protocol contracts across 16 tasks');text+='\nQuant_transpose_kernel now covers all 22 canonical dense dtype families, both\nsource configurations, host abs/max and reciprocal, finite autotune trials,\nactual wrapped integer masks and byte addresses, and returned byte-memory\nresults and frames. Its 863 cases bind 140 exact ports to 16 deferred numerical\nfamilies. The mathematical transpose formula additionally requires the stated\narithmetic refinement; source-scope acceptance does not establish native GPU\ncorrectness or official comparator success.\n';f.write_text(text)
readme=B/'README.md';readme.write_text(readme.read_text()+'\n## Observed integration results\n\nAll eight build targets passed (7,418 jobs); all250 benchmark Python tests\npassed (40 exact numerical and210 other tests), together with25 sharding tests.\nThe source ledger accepts3/183: matrix transpose, quantize_global and\nquant_transpose_kernel;180 remain pending, including the reopened vector and\nReLU scopes. The previous955 public records and100 numerical rules are preserved.\nThe new catalog has956 public declarations and63 supplemental contracts\n(57 execution,6 protocol) across16 nonempty task records. Earlier prose task\ncounts were arithmetically wrong; no task or contract was removed.\n\nThe actual global audit exited1 only for the180 pending source scopes and the\nunavailable official comparator/preflight replay. Every other gate passed.\nThe queue has116 rules,1,346 exact bindings,6,262 cases and7 declarations;\nall new numerical conformance remains deferred. No validator predicate was\nweakened. Prior reviewed adapter functions are unchanged, and the added adapter\nrequires separate exact producer/root pins. No push was retried.\n')
active=['.github/workflows/bench-audit.yml','bench/original_program_scope.json','bench/original_program_scope.md','bench/reviewed_numerical_extensions.py','bench/quant_transpose_numerical_extension.py','bench/tests/test_exact_numerical_bindings.py','bench/tritonbench_g/deferred_numerical_experiments.json','bench/tritonbench_g/deferred_numerical_experiments.md','bench/check_original_program_scope.py','bench/exact_numerical_bindings.py','bench/exact_scope_gate.py','bench/deferred_numerical_experiments.py','lakefile.toml','lake-manifest.json','lean-toolchain']
paths={R/p for p in s['new_source_sha256']}|{R/p for p in active}|{p for p in B.rglob('*') if p.is_file()}
v={'schema':1,'kind':'quant_transpose_finite_source_acceptance_integration','recorded_at_utc':datetime.now(timezone.utc).isoformat(),'base_commit':s['base_commit'],'status':'finite_quant_transpose_source_accepted_with_deferred_numerics_comparator_blocked','scope':{'requested':183,'accepted':3,'verified':1,'verified_with_deferred_numerics':2,'pending':180,'reviewed':183,'unreviewed':0},'inventory':{'public_declarations':956,'task_modules':cat['summary']['task_modules'],'registered_modules':len(cat['registered_modules']),'supplement_execution':57,'supplement_protocol':6,'supplement_nonempty_tasks':16},'checked_input_sha256':{p.relative_to(R).as_posix():hash_file(p) for p in sorted(paths)},'actual_runs':actual,'preservation':{'prior_public_records':955,'prior_supplement_declarations':62,'unrelated_scope_rows':180,'prior_numeric_rules':100,'metadata_baseline_unchanged':s['metadata_baseline_sha256']},'global_gates':{k:v['status'] for k,v in g['gates'].items()},'new_numeric_families':16,'new_numeric_ports':140,'numerical_conformance_validated':False,'current_comparator_success':False,'native_execution':False};write_json(B/'validation.json',v);print(json.dumps({'validation_sha256':hash_file(B/'validation.json'),'scope':v['scope'],'inventory':v['inventory']},indent=2))
