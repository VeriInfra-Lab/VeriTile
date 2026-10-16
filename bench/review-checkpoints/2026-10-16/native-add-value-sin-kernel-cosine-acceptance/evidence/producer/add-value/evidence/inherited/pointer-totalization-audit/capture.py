from pathlib import Path
import hashlib,json,subprocess,sys
W=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
packages={'vector_revision2':Path('/tmp/veritile-standardization-20261015/vector-finite-acceptance-revision2-wip'),'custom_revision1':Path('/tmp/veritile-standardization-20261015/vector-custom-finite-acceptance-revision1-wip')}
inputs={}
for key,p in packages.items():
 task='vector_addition' if key.startswith('vector') else 'vector_addition_custom'
 paths=[p/'evidence/receipt.json',p/'after/bench/tritonbench_g'/task/'NativeCorrect.lean',p/'after/bench/tritonbench_g'/task/'Proofs/NativeArithmetic.lean',p/'after/bench/tritonbench_g'/task/'Proofs/NativeMemory.lean']
 inputs[key]={str(f):sha(f) for f in paths}
 (W/(key+'-NativeCorrect.lean.txt')).write_bytes(paths[1].read_bytes())
(W/'inputs-before.json').write_text(json.dumps(inputs,indent=2)+'\n')
with (W/'cpu-console.log').open('w') as out:
 result=subprocess.run([sys.executable,str(W/'witness.py')],cwd=W,stdout=out,stderr=subprocess.STDOUT)
assert result.returncode==0,(W/'cpu-console.log').read_text()
records=json.loads((W/'cpu-console.log').read_text());assert len(records['witnesses'])==5
(W/'cpu-results.json').write_text(json.dumps(records,indent=2)+'\n')
for group in inputs.values():
 for f,h in group.items():assert sha(f)==h,f
(W/'inputs-after.json').write_text(json.dumps({'all_unchanged':True,'inputs':inputs},indent=2)+'\n')
print(json.dumps({'status':'passed','witnesses':5,'GPU_executed':False,'inputs_unchanged':True}))
