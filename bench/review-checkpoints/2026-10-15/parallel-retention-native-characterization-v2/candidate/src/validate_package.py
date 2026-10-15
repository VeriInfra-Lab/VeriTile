#!/usr/bin/env python3
"""Revalidate existing actual captures and run regression tests; no redundant compilation."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,time
sys.dont_write_bytecode=True
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);here=Path(__file__).resolve().parent;root=here.parent
    # An in-progress producer may finish; this never fabricates its status.
    for _ in range(300):
        if (root/'offline-v1/report.json').exists():break
        time.sleep(1)
    assert (root/'offline-v1/report.json').exists(),'offline producer has no terminal report'
    inputs={str(q):sha(q) for q in [*here.glob('*.py'),root/'offline-v1/report.json',root/'gpu-attempt-v1/report.json',root/'reference-v1/reference.json']}
    cmds=[('verify-offline',[sys.executable,'-B',str(here/'verify_capture.py'),str(root/'offline-v1')]),('verify-gpu',[sys.executable,'-B',str(here/'verify_capture.py'),str(root/'gpu-attempt-v1')]),('tests',[sys.executable,'-B',str(here/'test_runner.py')])]
    records=[];code=0
    for name,cmd in cmds:
        log=out/(name+'.log');before={p:sha(Path(p)) for p in inputs}
        with log.open('w') as f:x=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        after={p:sha(Path(p)) for p in inputs};changed=[p for p,h in inputs.items() if before[p]!=h or after[p]!=h]
        records.append({'name':name,'command':cmd,'exit_code':x.returncode,'log':log.name,'log_sha256':sha(log),'input_sha256':inputs,'changed_inputs':changed})
        (out/'running.json').write_text(json.dumps(records,indent=2)+'\n')
        if x.returncode or changed:code=1;break
    offline=json.loads((root/'offline-v1/report.json').read_text());gpu=json.loads((root/'gpu-attempt-v1/report.json').read_text())
    r={'schema':1,'status':'validated_capture_tooling' if code==0 else 'validation_failed','records':records,'exit_code':code,
      'offline_status':offline['status'],'offline_producer_exit_code':offline['exit_code'],'offline_records':len(offline['records']),
      'native_status':gpu['status'],'native_producer_exit_code':gpu['exit_code'],'native_execution':gpu['native_execution'],'native_launches_completed':gpu['native_launches_completed'],
      'numeric_acceptance':False,'numeric_comparison':'NOT_ESTABLISHED','note':'These process records validate the already produced, fully bound compiler captures; they do not claim the captures were recompiled during this check. CPU synthetic tests are not native evidence.'}
    (out/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='records'}));return code
if __name__=='__main__':raise SystemExit(main())
