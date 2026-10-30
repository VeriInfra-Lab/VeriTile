"""Current source/SM90 scalar-routing binding; no compiler or GPU invocation."""
from pathlib import Path
import sys,hashlib,json
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'checks'))
from resolve import resolve_pair
source=ROOT/'parallel_attention_repaired.py'
artifacts=Path('/tmp/parallel-attention-lowering')
rows=[]
for forward in sorted(artifacts.glob('actual-binder-*-forward/receipt.json')):
    backward=forward.parent.with_name(forward.parent.name.replace('-forward','-backward'))/'receipt.json'
    result=resolve_pair(forward,backward,source)
    rows.append(result)
    print(forward.parent.name,result['status'],flush=True)
assert len(rows)==8
(ROOT/'lowering-binding.json').write_text(json.dumps({'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'profiles':rows,
    'GPU_execution':False,'native_numerical_conformance':False,'status':'PASS'},indent=2)+'\n')
