"""Run fixed integration checks and retain actual exits, logs and stable hashes."""
from common import *
from datetime import datetime,timezone
p=arguments(__doc__);p.add_argument('--phase',choices=['generate','verify'],required=True);p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();r,b,old,s=state(a)
run=external_output(r,a.run_dir);require(not run.exists(),'Use a fresh external /tmp run directory')
run.mkdir(parents=True);record={'schema':1,'kind':'actual_integration_run','phase':a.phase,'base_commit':s['base_commit'],'started_at_utc':datetime.now(timezone.utc).isoformat(),'state':'running','commands':[]}
write(run/'run.json',record)
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONPATH']=os.pathsep.join([str(r/'bench'),str(r/'bench/tests'),str(r/'scripts'),env.get('PYTHONPATH','')])
record['pythonpath']=env['PYTHONPATH'];record['python_executable']=sys.executable

def execute(name,command):
 log=run/(name+'.log');row={'name':name,'command':command,'cwd':str(r),'started_at_utc':datetime.now(timezone.utc).isoformat(),'state':'running','log':log.name}
 record['commands'].append(row);write(run/'run.json',record)
 try:
  with log.open('wb') as output:
   process=subprocess.run(command,cwd=r,env=env,stdout=output,stderr=subprocess.STDOUT)
  row.update(exit_code=process.returncode,state='completed')
 except BaseException as error:
  row.update(exit_code=None,state='launch_failed_or_interrupted',exception=type(error).__name__+': '+str(error))
  if not log.exists():log.write_text(str(error)+'\n')
  raise
 finally:
  row['log_sha256']=digest(log);row['finished_at_utc']=datetime.now(timezone.utc).isoformat();write(run/'run.json',record)
 return row

try:
 if a.phase=='generate':
  commands=[('prepare-baseline',[sys.executable,'-B','bench/current_source_catalog.py','prepare-baseline','--package',str(r/BASE),'--output',str(run/'baseline')]),
   ('catalog',[sys.executable,'-B','bench/current_source_catalog.py','--root',str(r),'--baseline',str(run/'baseline'),'--source-map',str(r/SOURCE_MAP),'--legacy-inventory',str(r/LEGACY),'--evidence-root',str(r/'bench/review-checkpoints'),'--review-manifest',str(r/REVIEW),'--supplement',str(b/'catalog-supplement.json.gz'),'--output',str(run/'current-source-catalog.json.gz')]),
   ('queue',[sys.executable,'-B','bench/deferred_numerical_experiments.py','--root',str(r)])]
  for name,command in commands:
   row=execute(name,command)
   if row['exit_code']!=0:raise ValueError('Generation failed: '+name)
   if name=='catalog':shutil.copyfile(run/'current-source-catalog.json.gz',b/'current-source-catalog.json.gz')
  require((r/QUEUE).stat().st_size<LIMIT,'Generated compressed queue exceeds 100 MiB; stop')
  record['generated_sha256']={str(path.relative_to(r)):digest(path) for path in [b/'current-source-catalog.json.gz',r/QUEUE,r/'bench/tritonbench_g/deferred_numerical_experiments.md']}
 else:
  preservation(a)
  record['input_sha256_before']=verification_inputs(a);write(run/'run.json',record)
  # Fixed explicit targets and full benchmark discovery, with fixture import paths.
  commands=[('build',['lake','build',*TARGETS]),
   ('bench-tests',[sys.executable,'-B','-m','unittest','discover','-s','bench/tests','-p','test_*.py','-v']),
   ('shard-tests',[sys.executable,'-B','scripts/test_audit_shards.py']),
   ('global',[sys.executable,'-B','bench/current_source_audit.py','--root',str(r),'--catalog',str(b/'current-source-catalog.json.gz'),'--baseline-package',str(r/BASE),'--source-map',str(r/SOURCE_MAP),'--legacy-inventory',str(r/LEGACY),'--evidence-root',str(r/'bench/review-checkpoints'),'--review-manifest',str(r/REVIEW),'--supplement',str(b/'catalog-supplement.json.gz'),'--scope-ledger',str(r/a.ledger),'--output',str(run/'global')])]
  for name,command in commands:execute(name,command)
  record['input_sha256_after']=verification_inputs(a)
  record['inputs_unchanged']=record['input_sha256_before']==record['input_sha256_after']
  require(record['inputs_unchanged'],'Inputs changed while checks ran')
 record['state']='completed'
except BaseException as error:
 record['state']='failed_or_interrupted';record['exception']=type(error).__name__+': '+str(error)
 raise
finally:
 record['finished_at_utc']=datetime.now(timezone.utc).isoformat();write(run/'run.json',record)
print(json.dumps({'phase':a.phase,'run':str(run),'actual_exits':{row['name']:row['exit_code'] for row in record['commands']},'state':record['state']},indent=2))

sys.exit(0 if all(row['exit_code']==0 for row in record['commands']) else 1)
