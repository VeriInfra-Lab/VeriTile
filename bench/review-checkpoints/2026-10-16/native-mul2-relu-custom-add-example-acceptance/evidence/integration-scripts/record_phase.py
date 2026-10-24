"""Record actual prepare/install/rebind exits, retaining failed attempts."""
from common import *
from datetime import datetime,timezone
p=arguments(__doc__);p.add_argument('--phase',choices=['prepare','install','rebind'],required=True);p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();r,b,old=context(a)
run=a.run_dir.resolve();require(run.is_relative_to(Path('/tmp')) and not run.exists(),'Use a fresh /tmp run directory');run.mkdir(parents=True)
command=[sys.executable,'-B',str(HERE/(a.phase+'.py')),'--root',str(r),'--packages',str(a.packages),'--ledger',a.ledger,'--baseline-ledger',a.baseline_ledger,'--predecessor-catalog',a.predecessor_catalog,'--predecessor-supplement',a.predecessor_supplement,'--predecessor-review',a.predecessor_review]
row={'name':a.phase,'command':command,'cwd':str(r),'state':'running','log':a.phase+'.log'}
record={'schema':1,'kind':'actual_integration_run','phase':a.phase,'base_commit':git(r,'rev-parse','HEAD').decode().strip(),'started_at_utc':datetime.now(timezone.utc).isoformat(),'state':'running','commands':[row]};write(run/'run.json',record)
try:
 with (run/row['log']).open('wb') as output:process=subprocess.run(command,cwd=r,stdout=output,stderr=subprocess.STDOUT)
 row.update(state='completed',exit_code=process.returncode);record['state']='completed' if process.returncode==0 else 'failed_or_interrupted'
except BaseException as error:
 row.update(state='launch_failed_or_interrupted',exit_code=None,exception=type(error).__name__+': '+str(error));record['state']='failed_or_interrupted';raise
finally:
 row['log_sha256']=digest(run/row['log']);record['finished_at_utc']=datetime.now(timezone.utc).isoformat();write(run/'run.json',record)
print(json.dumps(record,indent=2));sys.exit(row['exit_code'])
