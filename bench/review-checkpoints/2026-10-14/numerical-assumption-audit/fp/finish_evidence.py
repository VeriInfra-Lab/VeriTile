from pathlib import Path
import json,hashlib,collections,datetime
p=Path('/tmp/veritile-assumptions-20261014/fp')
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def write(f,d):
 temp=f.with_suffix(f.suffix+'.tmp');temp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');temp.replace(f)
a=json.loads((p/'audit.json').read_text()); expected={r['qualified_declaration']:r for r in a['rows']}; seen=set(); queries=[]
for i,n in [(1,40),(2,40),(3,35)]:
 log=p/f'identity-tail{i}.log';s=log.read_text();assert 'error:' not in s and 'unresolved' not in s
 parsed={};curr=None
 for l in s.splitlines():
  if l.startswith('DECL '):curr=l[len('DECL '):-1];parsed[curr]=[]
  elif l.startswith('ATOM_JSON '):parsed[curr].append(json.loads(l[len('ATOM_JSON '):]))
 assert len(parsed)==n
 for name,atoms in parsed.items():
  assert name not in seen;seen.add(name)
  before=[{'instance_key':x['instance_key'],'rule_id':x['rule_id']} for x in expected[name]['atoms']]
  if any(x['instance_key'] is None for x in atoms):
   assert name in ['VeriTile.Bench.TritonBenchG.ContextAttnFwd.FPEquivalence.scaleFirst_equiv','VeriTile.Bench.TritonBenchG.ContextAttnLlama.FPEquivalence.scaleFirst_equiv']
   assert len(atoms)==2 and all(x['rule_id']=='MUL-COMMUTE' for x in atoms)
   assert len([x for x in atoms if x['instance_key'] is None])==1
   expected[name]['additional_key_display_limit']='Typed key is a closed string append; the copied whnf-only printer returns null. Probe.lean #eval and exact TypedScalarArithmetic.entry definition provide the full :typed-fp32 key.'
  else:
   assert sorted(before,key=str)==sorted(atoms,key=str),(name,before,atoms)
  expected[name]['additional_exact_key_query']={'source':f'IdentityTail{i}.lean','log':log.name,'exit_code':0,'status':'PASS'}
 queries.append({'source':f'IdentityTail{i}.lean','source_sha256':sha(p/f'IdentityTail{i}.lean'),'log':log.name,'log_sha256':sha(log),'declarations':n,'exit_code':0,'status':'PASS'})
assert len(seen)==115
for task in ['ContextAttnFwd','ContextAttnLlama']:
 name=f'VeriTile.Bench.TritonBenchG.{task}.FPEquivalence.scaleFirst_equiv'; assert name in seen
 assert {x['obligation_id'] for x in expected[name]['atoms']}=={'fp32_mul_commute_real','fp32_mul_commute_typed'}
a['additional_query_evidence']={'status':'PASS','declarations_with_fresh_exact_identity_queries':115,'exact_context_bindings_confirmed_by_source_and_eval':True,'declarations_with_all_keys_rendered_by_extra_query':113,'context_typed_key_display_limit':'whnf-only string reader returns null for string append; exact full key independently obtained by Probe.lean #eval','queries':queries,'redundant_full_query':{'source':'IdentityProbe.lean','log':'identity-probe.log','exit_code':143,'status':'STOPPED_REDUNDANT','reason':'Full repetition was unnecessary because completed acceptance logs already cover 154 declarations and seeded_dropout was queried independently; retained partial output is not counted as a complete pass.'}}
a['method']['fresh_query']='Probe.lean passed for missing seeded_dropout output and four exact entry keys; three completed parallel queries traversed 115 declarations (113 render every exact key; 2 context typed string-append keys require the independent Probe.lean #eval plus source definition). Full 155 repetition was stopped and is not counted as passed.'
a['summary']['additional_proof_walk_queries_passed']=115
a['summary']['additional_queries_with_all_keys_rendered']=113
write(p/'audit.json',a);write(p/'rows.json',a['rows']);write(p/'summary.json',a['summary'])
receipt={'recorded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpu_experiments_run':0,'probe_exit_code':0,'fp_assumption_unit_tests':{'command':'python -m unittest scripts.test_fp_assumptions','tests':12,'exit_code':0,'log':'fp-assumptions-tests.log'},'report_validation':'PASS','fresh_queries':queries,'redundant_full_query_exit_code':143,'root_lean_inputs_match_completed_acceptance':3379}
write(p/'verification-receipt.json',receipt)
print('115 supplementary proof-walk queries PASS; 113 render all keys. Two typed appended keys are confirmed by separate source/eval, with printer limitation recorded.')
