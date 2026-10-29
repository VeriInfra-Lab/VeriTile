"""Integrate an explicit per-declaration review registry, never discover proof status.
Run only after source review and final compilation. The old model audit stays frozen.
"""
from pathlib import Path
from collections import defaultdict
import ast, hashlib, json, re, sys
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
BASE=HERE/'before/bench/tritonbench_g'
REG=HERE/'reviewed_value_registry.json'
sys.path[:0]=[str(ROOT/'bench'),str(ROOT/'scripts')]
from port_layout import proof_files
from check_proof_gap_manifest import collect
from numerical_obligations import statement_map
from spec_sheet import split_decls

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
registry=json.loads(REG.read_text()); tasks={t['task']:t for t in registry['tasks']}
baseline=json.loads((BASE/'numerical_obligations.json').read_text())
original={t['task'] for t in baseline['tasks'] if t['fp_adequacy_status']=='math_contract_only'}
assert set(tasks)==original and len(tasks)==len(registry['tasks']), 'Exact original repair membership required'
assert all(t['accepted_scope_fp_coverage']=='complete_in_declared_profile' for t in tasks.values()), 'Incomplete accepted scope'
assert all(t['native_binding_status']=='not_established' for t in tasks.values()), 'No native promotion authorized by these receipts'
old_cov=json.loads((BASE/'coverage_review.json').read_text())
old_num=json.loads((BASE/'numerical_obligations_review.json').read_text())
locations=json.loads((REG.parent/'reviewed_line_mapping.json').read_text())
line_map={}
for loc in locations:
 assert loc['tag']=='equal', 'Changed evidence anchor needs an explicit new review'
 assert loc['before_sha256']==old_num['input_sha256'][loc['path']]
 assert loc['after_sha256']==sha(ROOT/loc['path']), 'Evidence location snapshot changed'
 line_map[loc['path'],loc['before_line']]=loc['after_line']
def refresh_locations(obj):
 if isinstance(obj,list):
  for x in obj:refresh_locations(x)
 elif isinstance(obj,dict):
  key=(obj.get('path',obj.get('file')),obj.get('line'))
  if key in line_map:obj['line']=line_map[key]
  for x in obj.values():refresh_locations(x)
refresh_locations(old_num['declarations'])
refresh_locations(old_num['obligations'])
refresh_locations(old_num['native_boundaries'])
updates=json.loads((REG.parent/'reviewed_obligation_updates.json').read_text())
replacements={o['id']:o for o in updates['obligations']}
assert replacements.keys() <= {o['id'] for o in old_num['obligations']}
old_num['obligations']=[replacements.get(o['id'],o) for o in old_num['obligations']]
for family, entries in updates['experiment_tasks_replace_family'].items():
 old_num['experiment_tasks']=[t for t in old_num['experiment_tasks'] if t.get('parent_family',t.get('obligation_id'))!=family]+entries

raw=collect(reviewed=False)
rows=defaultdict(list)
for r in raw:rows[r.file].append(r)
old_ports={p['file']:p for p in old_cov['ports']}
levels={'kcache_copy_triton':'specialization','kv_cache_copy':'specialization',
 'masked_select':'precomputed_input_slice','kv_cache_filling':'precomputed_input_slice',
 'rope_transform':'full_value_candidate','rope_backward_transform':'full_value_candidate',
 'token_softmax_bloom':'full_value_candidate','token_softmax_llama':'full_value_candidate',
 'quant_transpose_kernel':'pre_rounding_slice'}
new_models={}
ports=[]
for file,rr in sorted(rows.items()):
 path=ROOT/file; task=path.parent.name
 if file in old_ports:
  port=old_ports[file]
  if port['lean_sha256'] != sha(path):
   changes=json.loads((REG.parent/'reviewed_legacy_changes.json').read_text())
   assert file in changes, f'Original public source changed; independent review required: {file}'
   entry=changes[file]
   assert entry['before_sha256']==port['lean_sha256'] and entry['after_sha256']==sha(path), f'Legacy review hash changed: {file}'
   port=dict(port,lean_sha256=sha(path),reviewed_on='2026-10-14',legacy_compatibility_review=entry)
   # The old statement digest is still checked below. Its mathematical model
   # label is preserved; tactic/legacy projection repairs earn no FP credit.
 else:
  assert path.name=='FPCorrect.lean' and task in tasks, f'Missing explicit new review: {file}'
  t=tasks[task]; models={d['name']:d for d in t['declarations']}
  assert len(models)==len(t['declarations']), f'Duplicate public declaration: {task}'
  assert set(models)=={r.declaration for r in rr}, f'Public names changed: {file}'
  old=[p for p in old_ports.values() if Path(p['file']).parent.name==task]
  assert old
  py=old[0]['python_file'];links={ (f['name'],f['line']):f for p in old for d in p['declarations'] for f in d['python_functions']}
  declarations=[]
  for r in rr:
   model=models[r.declaration]; new_models[file,r.declaration]=(t,model)
   fixed=model['role']=='fixed_nonempty_witness'
   declarations.append(dict(declaration=r.declaration,
    coverage_level='specialization' if fixed else levels.get(task,'specialization'),
    numeric_model=model['numeric_model'],python_functions=list(links.values()),
    python_link_scope='task_level_source_association; declaration/stage correspondence is stated in scope and the bound public proposition',
    scope=('Fixed nonempty rounding witness only; it does not extend parametric scope. ' if fixed else '')+
      t['covered_scope']+' '+t['remaining_scope']+' No native correctness or empirical acceptance is inferred.'))
  port=dict(file=file,python_file=py,python_sha256=sha(ROOT/py),lean_sha256=sha(path),reviewed_on='2026-10-14',declarations=declarations)
 if (path.parent/'Kernels.lean').is_file() or (path.parent/'FPCorrect.lean').is_file():
  port['support_sha256']={str(p.relative_to(ROOT)):sha(p) for p in proof_files(path.parent) if p!=path}
 ports.append(port)
old_cov.update(ports=ports,reviewed_on='2026-10-14')
old_cov['basis']='Explicit old mathematical/relative scope reviews plus new per-operation/bit-value contracts, source-shape bridges, independent initial-state values and frames. Primitive/native refinement remains unestablished. Hashes bind source inspection, not a proof of native behavior.'
# Derive statement digests only after exact new names and model labels match the manual registry.
public=[dict(file=p['file'],**d) for p in ports for d in p['declarations']]
statements=statement_map(public)
old_rows={(r['file'],r['declaration']):r for r in old_num['declarations']}
reviewed=[]
for row in public:
 key=row['file'],row['declaration'];file=key[0]
 if key in old_rows:
  review=old_rows[key]
  current_text=(ROOT/file).read_text()
  current_decl=next(d for d in split_decls(current_text) if d['name']==key[1])
  review['line']=current_text[:current_decl['start']].count('\n')+1
  assert review['statement_sha256']==hashlib.sha256(statements[key].encode()).hexdigest(), f'Old statement changed: {key}'
 else:
  task,model=new_models[key]
  source=ROOT/file;decl=next(d for d in split_decls(source.read_text()) if d['name']==key[1]);line=source.read_text()[:decl['start']].count('\n')+1
  support=proof_files(source.parent)
  evidence=[dict(path=file,line=line,sha256=sha(source),description='Inspected actual public per-operation/bit-value statement; includes successful execution and independent value/frame target.')]
  evidence += [dict(path=str(p.relative_to(ROOT)),sha256=sha(p),description='Task-local source, value, shape or software interpretation supporting the public statement.') for p in support if p!=source]
  for name in ['VeriTile/Triton/Float/Structural.lean','VeriTile/Triton/Float/ScalarBackend.lean','VeriTile/Triton/Float/ScalarOps.lean','VeriTile/Triton/Float/BitValue.lean']:
   evidence.append(dict(path=name,sha256=sha(ROOT/name),description='Reviewed computation and payload semantics; mathematical real evaluation is not used by this new contract.'))
  review=dict(file=file,declaration=key[1],line=line,sha256=sha(source),port=task['task'],numeric_model=model['numeric_model'],
   statement_sha256=hashlib.sha256(statements[key].encode()).hexdigest(),explicit_law_ids=[],model_requirement_ids=[],
   accepted_input_conditions=['Exact ordinary input premises are displayed in the bound public statement.'],
   accepted_stage_conditions=[task['covered_scope'],task['remaining_scope']],oracle_model_conditions=[],
   review_status='reviewed',evidence=evidence,native_semantics_boundary='primitive_and_native_binding_not_established',
   notes='Independent source execution/value/frame theorem, not assumed final output and not candidate=self. Software arithmetic canonicalizes arithmetic NaNs; native payload policy, reduction tree, cast and compiler correspondence remain separate.',
   extra_numeric_ids=[],shared_model_ids=[],review_basis='Explicit inspection of the public statement, actual operation graph and independent initial-memory reference; no additional algebraic rewrite law is used.',
   fp_model_review=dict(kind=model['model_kind'],role=model['role'],review_basis='Concrete FP.Value/bit encoding interpretation.' if model['model_kind']=='bit_interpreter_contract' else 'Precision-tagged per-operation backend value/address specification; opaque native primitives remain explicit.',
    accepted_scope_coverage=task['accepted_scope_fp_coverage'],scope_basis=task['covered_scope']+' '+task['remaining_scope'],evidence=evidence))
 reviewed.append(review)
# Add new files and bind current shared semantics. Historical checkpoint paths are unchanged.
names=set(old_num['input_sha256'])
for directory in ['VeriTile','bench/tests','bench/tritonbench_g']:
 names.update(str(p.relative_to(ROOT)) for p in (ROOT/directory).rglob('*.lean'))
names.update(str(p.relative_to(ROOT)) for p in (ROOT/'experiments/floating_point/value_validation').iterdir() if p.is_file() and p.suffix in {'.py','.lean','.md'})
names.add('bench/fp_value_repairs.py')
names.add('bench/tritonbench_g/numeric_experiment_handoff.md')
for name in ['reviewed_obligation_updates.json','reviewed_legacy_changes.json','reviewed_line_mapping.json','reviewed_value_registry.json','integrate_value_reviews.py','reviewed_input_sha256.json','finalize_reports.py']:
 names.add(str((REG.parent/name).relative_to(ROOT)))
for name in ['coverage_review.json','numerical_obligations_review.json','numerical_obligations.json','fp_proof_gaps.json']:
 names.add(str((BASE/name).relative_to(ROOT)))
for obligation in updates['obligations']:
 for evidence in obligation['evidence']:
  names.add(evidence['path'])

names.add('bench/tritonbench_g/fifth_order_sph_harmonics/Proofs/check_fp_source.py')
names.update(['bench/port_layout.py','bench/coverage_review.py','bench/numerical_obligations.py','bench/acceptance_tritonbench_g.py','bench/inventory_tritonbench_g_completion.py'])
old_num['input_sha256']={name:sha(ROOT/name) for name in sorted(names) if (ROOT/name).is_file()}
# A captured, reviewed input manifest prevents this one-time integration tool
# from silently blessing future proof/primitive changes with fresh hashes.
expected=json.loads((REG.parent/'reviewed_input_sha256.json').read_text())
generated={'bench/tritonbench_g/coverage_review.json','bench/tritonbench_g/fp_proof_gaps.json','bench/tritonbench_g/fp_proof_gaps.md'}
for name, actual in old_num['input_sha256'].items():
 if name not in generated and name != str((REG.parent/'reviewed_input_sha256.json').relative_to(ROOT)):
  assert expected.get(name)==actual, f'Input differs from reviewed source snapshot: {name}'
old_num['input_sha256']['bench/tritonbench_g/coverage_review.json']=hashlib.sha256((json.dumps(old_cov,ensure_ascii=False,indent=2)+'\n').encode()).hexdigest()
# Existing evidence about shared implementations is refreshed only as part of this
# explicit reviewed semantics repair; removed support sources fail here.
def refresh_evidence(obj):
 if isinstance(obj,list):
  for x in obj:refresh_evidence(x)
 elif isinstance(obj,dict):
  path=obj.get('path',obj.get('file'))
  if path and 'sha256' in obj and path in old_num['input_sha256']:
   obj['sha256']=old_num['input_sha256'][path]
  for x in obj.values():refresh_evidence(x)
old_num['declarations']=reviewed
refresh_evidence(old_num['declarations']);refresh_evidence(old_num['obligations']);refresh_evidence(old_num['native_boundaries'])
old_num['method']['source_changes']='New typed value and concrete bit-interpreter contracts plus reviewed frontend precision/storage and primitive repairs. Original mathematical statements and historical model audit preserved; new source/hash/build evidence is separate.'
old_num['method']['nonfp']='Original 629-entry audit retained for unchanged public statements; supplemental FPCorrect declarations have individual inspected operation-graph/model reviews. Absence of rewrite laws does not establish native binding.'
old_num['native_value_validation']={'status':'pending_native_primitive_bindings','guide':'experiments/floating_point/value_validation/README.md','scalar_runner':'experiments/floating_point/value_validation/run_scalar.py','native_execution_performed':False,'universal_native_equality_proved':False,'task_profiles':[{k:t[k] for k in ('task','native_profile','native_binding_status','covered_scope','remaining_scope')} for t in registry['tasks']]}
if '--write' in sys.argv:
 registry['status']='reviewed_value_contracts_native_binding_pending'
 save(ROOT/'bench/tritonbench_g/fp_value_repairs.json',registry)
 old_num['input_sha256']['bench/tritonbench_g/fp_value_repairs.json']=sha(ROOT/'bench/tritonbench_g/fp_value_repairs.json')
 save(ROOT/'bench/tritonbench_g/coverage_review.json',old_cov)
 save(ROOT/'bench/tritonbench_g/numerical_obligations_review.json',old_num)
else:
 save(HERE/'coverage_review.preview.json',old_cov)
 save(HERE/'numerical_obligations_review.preview.json',old_num)
print(json.dumps({'public_declarations':len(public),'new_declarations':len(new_models),'tasks':len(tasks),'written':'--write' in sys.argv}))
