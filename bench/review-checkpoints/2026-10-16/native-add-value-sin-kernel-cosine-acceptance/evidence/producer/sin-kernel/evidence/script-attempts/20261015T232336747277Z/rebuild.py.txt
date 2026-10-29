from pathlib import Path
import json,subprocess,sys
sys.dont_write_bytecode=True
from propose import W,MODULES
rows=[]
for mod in MODULES:
 before=set((W/'evidence/lean-attempts').glob('*.json'))
 r=subprocess.run([sys.executable,str(W/'compile.py'),mod],capture_output=True,text=True)
 added=set((W/'evidence/lean-attempts').glob('*.json'))-before;assert len(added)==1
 row={'module':mod,'record':str(next(iter(added)).relative_to(W))};rows.append(row)
 print(r.stdout,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
(W/'evidence/final-build.json').write_text(json.dumps(rows,indent=2)+'\n')
