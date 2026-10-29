import hashlib,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
p=Path(__file__).parent;r=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');sys.path.insert(0,str(r/'bench'));import adapter_common as a
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest();catp=r/'bench/review-checkpoints/2026-10-16/relu-source-grid-and-exact-numerics/current-source-catalog.json';cat={x['task']:x for x in json.loads(catp.read_text())['tasks']};parts=[];tasks=[];sources={};allinputs={};evcount=0;public=0;gaps=0;metadata=[]
for f in sorted(p.glob('batch-06-part*-review.json')):
 d=json.loads(f.read_text());assert d['catalog_sha256']==sha(catp);parts.append({'path':f.name,'sha256':sha(f),'tasks':d['task_count'],'public_declarations':d['public_declarations_checked'],'structural_gaps':d['structural_gap_count']})
 metadata.extend(d['catalog_declaration_metadata_discrepancies'])
 for t in d['tasks']:
  assert t['task'] not in tasks;tasks.append(t['task']);gaps+=len(t['structural_gaps']);public+=len(t['public_declarations_checked'])
  for f,h in t['source_sha256'].items():assert sha(r/f)==h;sources[f]=h
  for files in t['all_input_sha256'].values():
   for f,h in files.items():assert sha(r/f)==h;allinputs[f]=h
  expected={(x['file'],x.get('parser_name') or x['declaration'],x['statement_sha256']) for x in cat[t['task']]['declarations']}
  actual={(x['file'],x['parser_name'],x['statement_sha256']) for x in t['public_declarations_checked']};assert actual==expected
  for x in t['public_declarations_checked']:
   z=a.declarations(r/x['file'])[x['parser_name']];assert z['statement_sha256']==x['statement_sha256'];assert a.split_statement(z['text'])[0]==x['statement'];assert sha(r/x['file'])==x['sha256']
  for g in t['structural_gaps']:
   assert g['can_defer_as_numeric_assumption'] is False
   for e in g['evidence']:
    q=r/e['path'];assert sha(q)==e['sha256'];assert '\n'.join(q.read_text().splitlines()[e['line']-1:e['end_line']])==e['excerpt'];evcount+=1
assert len(tasks)==20 and public==138 and gaps==44
out={'schema':1,'kind':'scope_batch_aggregate_receipt','reviewer':'/root/scope_remaining_cohort','reviewed_base_commit':'fc183f5dae5e3f09f549b8151a9a1e4c366beb02','parts':parts,'task_count':len(tasks),'tasks':tasks,'public_declarations_checked':public,'structural_gap_count':gaps,'evidence_ranges_checked':evcount,'reviewed_source_files':len(sources),'bound_task_input_files':len(allinputs),'all_inputs_current_and_unchanged':True,'exact_public_sets_reparsed_and_matched':True,'catalog_metadata_discrepancies':metadata,'whole_original_program_accepted':0,'completion_increment':0,'numerical_deferrals_registered':0,'new_lean_runs':0,'new_frontend_runs':0,'gpu_runs':0,'repo_mutations':False,'catalog':{'path':str(catp.relative_to(r)),'sha256':sha(catp)},'source_sha256':sources,'all_input_sha256':allinputs,'producer_scripts':{f.name:sha(f) for f in p.glob('*.py')}}
f=p/'batch-06-cohort20-aggregate-receipt.json';assert not f.exists();f.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'path':str(f),'sha256':sha(f),'tasks':len(tasks),'public_declarations':public,'gaps':gaps,'evidence_ranges':evcount,'reviewed_source_files':len(sources),'bound_input_files':len(allinputs)}))
