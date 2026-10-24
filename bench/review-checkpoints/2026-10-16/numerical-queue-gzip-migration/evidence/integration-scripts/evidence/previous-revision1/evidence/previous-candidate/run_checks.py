from pathlib import Path
import subprocess,sys,os,json,hashlib,time
W=Path(__file__).resolve().parent;E=W/'evidence';R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');stamp=time.time_ns();A=E/'attempts'/str(stamp);A.mkdir(parents=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for p in [W/'audit_candidate.py',W/'integration.py',W/'test_integration.py']:(A/(p.name+'.txt')).write_bytes(p.read_bytes())
commands=[('candidate-audit',[sys.executable,'-B',str(W/'audit_candidate.py')]),('integration-tests',[sys.executable,'-B','-m','unittest','discover','-s',str(W),'-p','test_integration.py','-v']),('consumer-tests',[sys.executable,'-B','-m','unittest','discover','-s',str(W/'after/bench/tests'),'-p','test_numerical_queue*.py','-v'])]
records=[];env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(W/'after/bench')+':'+str(R/'bench'))
for name,cmd in commands:
 start=time.monotonic();log=A/(name+'.log')
 with log.open('w') as f:r=subprocess.run(cmd,cwd=W,env=env,stdout=f,stderr=subprocess.STDOUT)
 record={'name':name,'command':cmd,'exit_code':r.returncode,'seconds':round(time.monotonic()-start,3),'log':str(log.relative_to(W)),'log_sha256':sha(log)};records.append(record);(A/'records.json').write_text(json.dumps(records,indent=2)+'\n');print(log.read_text())
 if r.returncode:raise SystemExit(r.returncode)
out={'status':'passed','commands':records,'unit_tests':22,'live_migration_run':False,'base_commit_supplied':False,'acceptance_increment':0};(E/'checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':'passed','unit_tests':22,'attempt':str(A)}))
