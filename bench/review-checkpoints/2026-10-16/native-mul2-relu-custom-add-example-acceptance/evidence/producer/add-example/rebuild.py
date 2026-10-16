from pathlib import Path
import json,subprocess,sys
W=Path(__file__).resolve().parent
MODULES=['bench/tritonbench_g/add_example/Proofs/Finite'+n+'.lean' for n in ['Indices','Scalar','Source','Physical','Program','Grid','Admission','Host']]+['bench/tritonbench_g/add_example/FiniteCorrect.lean']
records=[]
for module in MODULES:
 result=subprocess.run([sys.executable,str(W/'compile.py'),module],cwd=W)
 if result.returncode:sys.exit(result.returncode)
 latest=json.loads((W/'evidence'/(Path(module).stem+'-latest.json')).read_text())
 records.append({'module':module,'record':latest['log'].replace('.log','.json')})
(W/'evidence/final-build.json').write_text(json.dumps(records,indent=2)+'\n')
