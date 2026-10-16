"""Record bounded candidate checks only; never invoke integration mutation phases."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,re,subprocess,sys,time

sys.dont_write_bytecode=True
W=Path(__file__).resolve().parent
assert not (W/'candidate-receipt.json').exists(),'Frozen script candidate must not be overwritten'
A=W/'evidence'/('preparation-checks-'+str(time.time_ns()));A.mkdir(parents=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs={p.name:sha(p) for p in sorted(W.glob('*.py'))}
inputs.update({name:sha(W/name) for name in ['specs.json','future-protected-pins.json','transform-preimages.json','design-origin.json']})
commands=[
 ('preflight',[sys.executable,'-B',str(W/'preflight.py'),'--report',str(A/'preflight.json')]),
 ('lint',[sys.executable,'-B',str(W/'lint_candidates.py'),'--output',str(A/'lint')]),
 ('unit-tests',[sys.executable,'-B','-m','unittest','discover','-s',str(W),'-p','test_candidate.py','-v']),
]
records=[]
for name,command in commands:
 log=A/(name+'.log');start=time.monotonic()
 with log.open('w') as stream:result=subprocess.run(command,cwd=W,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=stream,stderr=subprocess.STDOUT)
 records.append({'name':name,'command':command,'exit_code':result.returncode,'seconds':round(time.monotonic()-start,3),'log':str(log.relative_to(W)),'log_sha256':sha(log)})
 (A/'records.json').write_text(json.dumps(records,indent=2)+'\n')
 if result.returncode:
  print(log.read_text());raise SystemExit(result.returncode)
after={name:sha(W/name) for name in inputs};assert inputs==after,'Candidate input changed during checks'
text=(A/'unit-tests.log').read_text();assert re.findall(r'^Ran (\d+) tests in ',text,re.M)==['7'] and text.rstrip().endswith('OK')
report={'schema':1,'status':'passed_for_script_preparation_only','recorded_at_utc':datetime.now(timezone.utc).isoformat(),'commands':records,'input_sha256_before':inputs,'input_sha256_after':after,'test_count':7,'frozen_input_preflight':str((A/'preflight.json').relative_to(W)),'lint_report':str((A/'lint/report.json').relative_to(W)),'future_base_commit':None,'runtime_plan_executed':False,'repository_modified':False,'frozen_package_source_or_receipt_modified':False,'large_dependency_copies':0,'new_dtype_matrix':False,'Lean_or_GPU_executed':False,'legacy_acceptance_increment':0,'primary_completion_increment':0}
target=W/'evidence/final-checks.json';assert not target.exists();target.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'tests':7,'records':str(target),'sha256':sha(target)}))
