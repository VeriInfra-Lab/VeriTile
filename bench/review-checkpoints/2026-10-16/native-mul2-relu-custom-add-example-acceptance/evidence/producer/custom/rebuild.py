from pathlib import Path
import json,subprocess,sys
W=Path(__file__).resolve().parent
order=['Proofs/NativeArithmetic.lean','Proofs/NativeScalar.lean','Proofs/NativeMemory.lean','Proofs/NativeProgram.lean','Proofs/NativeGrid.lean','Proofs/NativeAdmission.lean','NativeCorrect.lean']
records=[]
for rel in order:
 rel='bench/tritonbench_g/vector_addition_custom/'+rel
 r=subprocess.run([sys.executable,str(W/'compile.py'),rel],capture_output=True,text=True)
 print(r.stdout,flush=True);print(r.stderr,flush=True)
 records.append({'module':rel,'exit_code':r.returncode})
 if r.returncode:break
(W/'evidence/final-rebuild-run.json').write_text(json.dumps(records,indent=2)+'\n')
raise SystemExit(int(len(records)!=7 or any(r['exit_code'] for r in records)))
