#!/usr/bin/env python3
"""Run CPU reference, offline compiles, actual GPU attempt and regression suite in fresh directories."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,subprocess,sys
sys.dont_write_bytecode=True

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--proof',type=Path,required=True);p.add_argument('--root-replay',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);here=Path(__file__).resolve().parent
    inputs={str(q):sha(q) for q in [*here.glob('*.py'),a.proof/'receipt.json',a.root_replay]}
    cmds=[('prepare',[sys.executable,'-B',str(here/'run.py'),'prepare','--proof-package',str(a.proof.resolve()),'--root-replay',str(a.root_replay.resolve()),'--output',str(out/'reference-final')],[0]),
     ('offline',[sys.executable,'-B',str(here/'run.py'),'compile','--reference',str(out/'reference-final'),'--output',str(out/'offline-final'),'--arch','80','--arch','90','--profile-fusion-off','compiled_separate_ascending_fma','--profile-fusion-on','compiled_fused_ascending_fma'],[0]),
     ('gpu-attempt',[sys.executable,'-B',str(here/'run.py'),'run','--reference',str(out/'reference-final'),'--output',str(out/'gpu-attempt-final'),'--profile-fusion-off','compiled_separate_ascending_fma','--profile-fusion-on','compiled_fused_ascending_fma'],[0,1,2]),
     ('verify-offline',[sys.executable,'-B',str(here/'verify_capture.py'),str(out/'offline-final')],[0]),
     ('verify-gpu-capture',[sys.executable,'-B',str(here/'verify_capture.py'),str(out/'gpu-attempt-final')],[0]),
     ('tests',[sys.executable,'-B',str(here/'test_runner.py')],[0])]
    env=os.environ.copy();env.update(DEQUANT_TEST_REFERENCE=str(out/'reference-final'),DEQUANT_TEST_OFFLINE=str(out/'offline-final'),DEQUANT_TEST_NO_GPU=str(out/'gpu-attempt-final'))
    records=[];code=0
    for name,cmd,allowed in cmds:
        log=out/(name+'.log');before={s:sha(Path(s)) for s in inputs}
        with log.open('w') as f:x=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
        after={s:sha(Path(s)) for s in inputs};changed=[s for s in inputs if before[s]!=inputs[s] or after[s]!=inputs[s]]
        records.append({'name':name,'command':cmd,'exit_code':x.returncode,'log':log.name,'log_sha256':sha(log),'input_sha256':inputs,'changed_inputs':changed})
        (out/'running-validation.json').write_text(json.dumps(records,indent=2)+'\n')
        if x.returncode not in allowed or changed:code=1;break
    gpu=json.loads((out/'gpu-attempt-final/report.json').read_text()) if (out/'gpu-attempt-final/report.json').exists() else None
    receipt={'schema':1,'status':'validated_characterization_tooling' if code==0 else 'validation_failed','records':records,
      'native_status':gpu['status'] if gpu else 'NOT_RUN','native_exit_code':gpu['exit_code'] if gpu else None,
      'native_execution':gpu['native_execution'] if gpu else False,'numeric_acceptance':False,'universal_native_equality_proved':False,
      'exit_code':code,'note':'Tool validation can succeed with native NOT_RUN; process results and native status remain separate. Synthetic test fixtures never count as GPU records.'}
    (out/'validation-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='records'}));return code
if __name__=='__main__':raise SystemExit(main())
