#!/usr/bin/env python3
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys
sys.dont_write_bytecode=True
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);here=Path(__file__).resolve().parent
    bindings={str(q):sha(q) for q in [*here.glob('*.py'),a.inputs/'source-bindings.json']}
    cmds=[('prepare',[sys.executable,'-B',str(here/'run.py'),'prepare','--inputs',str(a.inputs.resolve()),'--output',str(out/'reference')],[0]),
      ('offline',[sys.executable,'-B',str(here/'run.py'),'compile','--reference',str(out/'reference'),'--output',str(out/'offline'),'--arch','80','--arch','90'],[0]),
      ('gpu-attempt',[sys.executable,'-B',str(here/'run.py'),'run','--reference',str(out/'reference'),'--output',str(out/'gpu-attempt')],[0,1,2]),
      ('verify-offline',[sys.executable,'-B',str(here/'verify_capture.py'),str(out/'offline')],[0]),
      ('verify-gpu',[sys.executable,'-B',str(here/'verify_capture.py'),str(out/'gpu-attempt')],[0]),
      ('tests',[sys.executable,'-B',str(here/'test_runner.py')],[0])]
    env=os.environ.copy();env.update(RBE_TEST_REFERENCE=str(out/'reference'),RBE_TEST_OFFLINE=str(out/'offline'),RBE_TEST_GPU=str(out/'gpu-attempt'))
    records=[];code=0
    for name,cmd,allowed in cmds:
        log=out/(name+'.log');before={p:sha(Path(p)) for p in bindings}
        with log.open('w') as f:x=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
        after={p:sha(Path(p)) for p in bindings};changed=[p for p,h in bindings.items() if before[p]!=h or after[p]!=h]
        records.append({'name':name,'command':cmd,'exit_code':x.returncode,'log':log.name,'log_sha256':sha(log),'input_sha256':bindings,'changed_inputs':changed})
        (out/'running-validation.json').write_text(json.dumps(records,indent=2)+'\n')
        if x.returncode not in allowed or changed:code=1;break
    gp=out/'gpu-attempt/report.json';gpu=json.loads(gp.read_text()) if gp.exists() else {}
    r={'schema':1,'status':'validated_capture_tooling' if code==0 else 'validation_failed','records':records,'exit_code':code,
      'native_status':gpu.get('status','NOT_RUN'),'native_exit_code':gpu.get('exit_code'),'native_execution':gpu.get('native_execution',False),
      'numeric_acceptance':False,'numeric_comparison':'NOT_ESTABLISHED','note':'Tool tests include CPU-only simulations, not native evidence. Frame checks and complete output capture do not supply missing numeric reference.'}
    (out/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='records'}));return code
if __name__=='__main__':raise SystemExit(main())
