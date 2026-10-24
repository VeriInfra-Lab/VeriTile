from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import ast,copy,json,re,shutil,subprocess,sys
sys.dont_write_bytecode=True
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');T=Path('/tmp/veritile-standardization-20261015')
B=R/'bench/review-checkpoints/2026-10-16/native-vector-quant-transpose-acceptance'
OLD=R/'bench/review-checkpoints/2026-10-16/quant-global-finite-source-acceptance'
WITHDRAWN=R/'bench/review-checkpoints/2026-10-16/quant-transpose-finite-source-acceptance'
BASE=R/'bench/review-checkpoints/2026-10-15/original-scope-extensions/metadata-baseline';G=T/'native-vector-quant-current-global'
sys.path.insert(0,str(R/'bench'))
from adapter_common import canonical,hash_file,load,sha,write_json,module_name
from check_original_program_scope import task_bindings,validate_task
from exact_numerical_bindings import collect,validate_exact_queue
def hashes(p):return {q.relative_to(p).as_posix():hash_file(q) for q in p.rglob('*') if q.is_file()}
def exact(p,h):assert hash_file(p)==h,str(p)
def oldtext(p):return subprocess.check_output(['git','show',s['base_commit']+':'+p],cwd=R,text=True)
assert not (B/'validation.json').exists();s=load(B/'evidence/preparation-state.json')
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()==s['base_commit']
assert hashes(OLD)==s['predecessor_file_sha256'];assert hashes(BASE)==s['metadata_baseline_sha256'];assert hashes(WITHDRAWN)==s['withdrawn_checkpoint_sha256']
for p,h in load(BASE/'input-sha256.json').items():exact(R/p,h)
for p,h in s['new_source_sha256'].items():exact(R/p,h)
for p,h in load(B/'root-review.json')['reviewed_input_sha256'].items():exact(R/p,h)
for p,h in load(B/'evidence-index.json')['retained_exact_artifacts'].items():exact(B/p,h)
cat=load(B/'current-source-catalog.json');old=load(OLD/'current-source-catalog.json');supp=load(B/'catalog-supplement.json');osupp=load(OLD/'catalog-supplement.json')
claims=lambda c:{(t['task'],d['file'],d['declaration']):d for t in c['tasks'] for d in t['declarations']}
a,b=claims(old),claims(cat);assert len(a)==955 and len(b)==957 and all(b[k]==v for k,v in a.items())
assert set(cat['registered_modules'])-set(old['registered_modules'])=={module_name(p) for p in s['new_source_sha256']}
assert all(cat['registered_modules'][k]==v for k,v in old['registered_modules'].items())
assert len(cat['registered_modules'])==4064 and cat['summary']['task_modules']==3633
assert sum(len(t['declarations']) for t in supp['tasks'])==64
assert Counter(d['role'] for t in supp['tasks'] for d in t['declarations'])=={'parametric_value_contract':58,'protocol_contract':6}
assert sum(bool(t['declarations']) for t in supp['tasks'])==16
sr={t['task']:t for t in supp['tasks']};osr={t['task']:t for t in osupp['tasks']};assert set(sr)==set(osr)
for name,prev in osr.items():
 if name in ['quant_transpose_kernel','vector_addition']:assert sr[name]['declarations'][:len(prev['declarations'])]==prev['declarations']
 else:assert sr[name]==prev
ledger=load(R/'bench/original_program_scope.json');prior=json.loads(oldtext('bench/original_program_scope.json'));priorRows={t['task']:t for t in prior['tasks']};catRows={t['task']:t for t in cat['tasks']};queue=load(R/ledger['deferred_queue']['path'])
assert ledger['catalog_sha256']==sha(canonical(cat));assert ledger['deferred_queue']['sha256']==hash_file(R/ledger['deferred_queue']['path'])
withdrawnRows={t['task']:t for t in load(B/'evidence/inherited-withdrawal-ledger.json')['tasks']}
for row in ledger['tasks']:
 name=row['task'];assert row['input_sha256']==task_bindings(R,cat,catRows[name])
 if name in ['matrix_transpose','quantize_global']:
  expected=copy.deepcopy(priorRows[name]);expected['review']=row['review'];expected['accepted_scope_progress']['scope_review']=row['review'];assert row==expected
 elif name in ['vector_addition','quant_transpose_kernel']:
  assert not row['structural_gaps'];assert validate_task(R,cat,catRows[name],row,queue)=='verified_with_deferred_numerics'
  for key in ['accepted_scope_history','scope_reopening']:
   if key in withdrawnRows[name]:assert row[key]==withdrawnRows[name][key]
 else:assert row==priorRows[name]
assert Counter(t['status'] for t in ledger['tasks'])=={'pending':179,'verified':1,'verified_with_deferred_numerics':3}
legacy,authority,actualQueue=collect(R);assert actualQueue==queue;previous=json.loads(oldtext('bench/tritonbench_g/deferred_numerical_experiments.json'))
assert queue['rules'][:100]==previous['rules'];assert queue['exact_callback_requirements'][:6]==previous['exact_callback_requirements']
assert all(queue['exact_binding_provenance'][k]==v for k,v in previous['exact_binding_provenance'].items());assert len(queue['rules'])==153
assert validate_exact_queue(queue,authority,legacy)=={'status':'passed','exact_bindings':1642,'declaration_cases':11041,'native_executions':0,'structural_acceptance':False}
for p in ['bench/check_original_program_scope.py','bench/exact_numerical_bindings.py','bench/exact_scope_gate.py','bench/deferred_numerical_experiments.py','lakefile.toml','lake-manifest.json','lean-toolchain']:assert (R/p).read_text()==oldtext(p),p
current=ast.parse((R/'bench/reviewed_numerical_extensions.py').read_text());functions={n.name:n for n in current.body if isinstance(n,ast.FunctionDef)}
for f in ast.parse(oldtext('bench/reviewed_numerical_extensions.py')).body:
 if isinstance(f,ast.FunctionDef) and f.name!='extend_reviewed_authority':assert ast.dump(f)==ast.dump(functions[f.name])
adapter=load(B/'evidence/archive-adapter-review.json')
exact(R/'bench/quant_transpose_numerical_extension.py',adapter['quant_current_adapter_sha256']);exact(R/'bench/vector_native_numerical_extension.py',adapter['vector_current_adapter_sha256'])
assert not list(B.rglob('*.lean')) and not list(B.rglob('*.olean')) and not list(B.rglob('*.pyc'))
g=load(G/'receipt.json');assert g['catalog_sha256']==sha(canonical(cat))
fail={k for k,v in g['gates'].items() if v['status']!='passed'};assert fail=={'scope_review','comparator_preflight','library_replay'},fail
for name,v in g['gates'].items():
 if 'log' in v:exact(G/v['log'],v['log_sha256'])
 if 'exit_code' in v:assert (v['exit_code']==0)==(v['status']=='passed')
scope=load(G/'scope_review.log');assert scope['counts']=={'verified':1,'verified_with_deferred_numerics':3,'not_accepted':179}
assert scope['errors']==['179 original-program scopes remain unaccepted']
for name in ['comparator_preflight','library_replay']:assert 'Official comparator is required' in (G/g['gates'][name]['log']).read_text()
build=T/'native-vector-quant-integration-build.log';assert re.findall(r'Build completed successfully \(([\d,]+) jobs\)\.',build.read_text())==['7425']
def testlog(name,n):
 p=T/('native-vector-quant-'+name+'.log');txt=p.read_text()
 assert re.findall(r'^Ran (\d+) tests? in ',txt,re.M)==[str(n)] and txt.rstrip().endswith('OK'),name
testlog('exact-tests',43);testlog('shard-tests',25);testlog('other-bench-tests-recovered',68)
failed=(T/'native-vector-quant-other-bench-tests.log').read_text()
assert re.findall(r'^Ran (\d+) tests? in ',failed,re.M)==['144']
assert failed.count('ModuleNotFoundError: No module named \'test_current_source_adapters\'')==2 and failed.rstrip().endswith('FAILED (errors=2)')
assert failed.count('\nERROR:')==2
for p in [G/'receipt.json',G/'value-metadata.json',*G.glob('*.log')]:
 if p.exists():d=B/'evidence/global-gates'/p.name;d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d)
actual={'integration-build':{'exit_code':0,'jobs':7425},'catalog':{'exit_code':0,'public_declarations':957},'queue':{'exit_code':0,'rules':153},'rebind-ledger':{'exit_code':0,'verified':1,'verified_with_deferred_numerics':3,'pending':179},'global':{'exit_code':1,'failed_gates':sorted(fail)},'other-bench-tests':{'exit_code':1,'passed_tests':142,'loader_errors':2,'reason':'Invocation omitted bench/tests from import search; two test modules never loaded.'},'other-bench-tests-recovered':{'exit_code':0,'tests':68,'reason':'Only the two previously unloaded modules rerun with fixture import path; total210 other tests passed.'},'exact-tests':{'exit_code':0,'tests':43},'shard-tests':{'exit_code':0,'tests':25},'prepare-attempt002':{'exit_code':0},'authority':{'exit_code':0}}
for name,row in actual.items():
 p=T/('native-vector-quant-'+name+'.log');d=B/'evidence/integration'/p.name;d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d);row['log']={'path':d.relative_to(B).as_posix(),'sha256':hash_file(d)}
for name in ['native-vector-quant-prepare.log','prepare_native_vector_quant_integration_attempt001.py']:
 p=T/name;d=B/'evidence/integration-attempts'/name;d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d)
shutil.copyfile(Path(__file__),B/'evidence/finalize.py');write_json(B/'evidence/observed-exits.json',actual)
readme=B/'README.md';readme.write_text(readme.read_text()+'''
## Observed integration results

All eight Lean build targets passed (7,425 jobs). All 253 benchmark Python
tests passed (43 exact numerical and210 other tests), together with25 sharding
tests. The first other-test invocation omitted the fixture import directory;
142 tests passed and two modules failed to load. Their68 tests then passed with
the corrected invocation. The initial failure is retained, not relabelled.

The ledger accepts4/183: matrix_transpose, quantize_global, vector_addition and
quant_transpose_kernel. The remaining179 scopes stay pending. The previous955
public declarations,62 supplemental declarations and100 numerical rules are
preserved. The catalog now has957 public declarations and64 supplemental
contracts (58 execution,6 protocol) across16 nonempty task records.

The global audit exited1 only for179 pending scopes and unavailable official
comparator/preflight replay. All other gates passed. The exact queue has153
rules,1,642 bindings,11,041 cases and8 declarations. New numerical conformance
remains deferred; no native GPU correctness or comparator pass is claimed.
No validation predicate was weakened. The withdrawn quantize-transpose draft
and historical baseline remain byte-identical. No push was retried.
''')
active=['.github/workflows/bench-audit.yml','bench/original_program_scope.json','bench/original_program_scope.md','bench/reviewed_numerical_extensions.py','bench/quant_transpose_numerical_extension.py','bench/vector_native_numerical_extension.py','bench/tests/test_exact_numerical_bindings.py','bench/tritonbench_g/deferred_numerical_experiments.json','bench/tritonbench_g/deferred_numerical_experiments.md','bench/check_original_program_scope.py','bench/exact_numerical_bindings.py','bench/exact_scope_gate.py','bench/deferred_numerical_experiments.py','lakefile.toml','lake-manifest.json','lean-toolchain']
paths={R/p for p in s['new_source_sha256']}|{R/p for p in active}|{p for folder in [B,WITHDRAWN] for p in folder.rglob('*') if p.is_file()}
v={'schema':1,'kind':'native_vector_quant_transpose_source_acceptance_integration','recorded_at_utc':datetime.now(timezone.utc).isoformat(),'base_commit':s['base_commit'],'status':'four_source_scopes_accepted_with_registered_numerical_deferrals_comparator_blocked','scope':{'requested':183,'accepted':4,'verified':1,'verified_with_deferred_numerics':3,'pending':179,'reviewed':183,'unreviewed':0},'inventory':{'public_declarations':957,'task_modules':3633,'registered_modules':4064,'supplement_execution':58,'supplement_protocol':6,'supplement_nonempty_tasks':16},'checked_input_sha256':{p.relative_to(R).as_posix():hash_file(p) for p in sorted(paths)},'actual_runs':actual,'preservation':{'prior_public_records':955,'prior_supplement_declarations':62,'unrelated_scope_rows':179,'prior_numeric_rules':100,'metadata_baseline_unchanged':s['metadata_baseline_sha256'],'withdrawn_quant_transpose_checkpoint_unchanged':True},'global_gates':{k:v['status'] for k,v in g['gates'].items()},'new_numeric_families':53,'new_numeric_ports':436,'numerical_conformance_validated':False,'current_comparator_success':False,'native_execution':False}
write_json(B/'validation.json',v);print(json.dumps({'validation_sha256':hash_file(B/'validation.json'),'scope':v['scope'],'inventory':v['inventory']},indent=2))
