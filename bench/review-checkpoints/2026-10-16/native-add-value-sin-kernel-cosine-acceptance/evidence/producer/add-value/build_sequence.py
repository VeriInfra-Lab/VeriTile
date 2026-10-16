from pathlib import Path
import subprocess,sys
W=Path(__file__).resolve().parent
mods=sys.argv[1:] or ['Proofs/NativeMemory.lean','Proofs/NativeProgram.lean','Proofs/NativeGrid.lean','Proofs/NativeAdmission.lean','NativeCorrect.lean']
for m in mods:
 r=subprocess.run([sys.executable,str(W/'compile.py'),'bench/tritonbench_g/add_value/'+m])
 if r.returncode:raise SystemExit(r.returncode)
