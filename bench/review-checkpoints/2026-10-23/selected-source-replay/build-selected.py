import json,subprocess,time,sys
from pathlib import Path
prefix=sys.argv[1] if len(sys.argv)>1 else 'selected-build'
ledger=json.loads(Path('bench/tritonbench_g/task_progress.json').read_text())
modules={task['task']:[p.removesuffix('.lean').replace('/','.') for p in task['kernel_revision']['public_entries']] for task in ledger['tasks']}
command=['lake','build',*sorted({m for ms in modules.values() for m in ms})]
source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
started=time.monotonic()
try:
 with open(prefix+'.log','w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
 result={'exit_code':r.returncode}
except KeyboardInterrupt:
 result={'exit_code':130,'interrupted':True}
result.update(command=command,source_commit=source,cwd=str(Path.cwd()),elapsed_seconds=time.monotonic()-started)
Path(prefix+'-result.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['exit_code'],flush=True)
