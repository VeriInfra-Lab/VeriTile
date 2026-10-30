"""Seal completed compiler controls and explicitly cancelled/unattempted work."""
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
OUT=ROOT/'.veritile-cache/task-deliveries/chunk_gla_simple_min_dot_tile'
NATIVE=OUT/'native-admission'
def pin(p):return {'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
cases=json.loads((NATIVE/'progress.json').read_text())
assert len(cases)==12 and all(c['status']=='compile_success' for c in cases)
partial=[]
for folder in (NATIVE/'cache').iterdir():
    if (folder/'chunk_simple_gla_fwd_kernel_o_repaired.json').exists():continue
    ir=folder/'chunk_simple_gla_fwd_kernel_o_repaired.ttir'
    if not ir.exists():continue
    assert '512x512' in ir.read_text()
    for ext in ('source','ttir','ttgir','llir','ptx'):
        old=folder/f'chunk_simple_gla_fwd_kernel_o_repaired.{ext}'
        if old.exists():
            new=NATIVE/f'B1-H1-T1-K1-V1-BT512.partial.{ext}'
            new.write_bytes(old.read_bytes());partial.append(pin(new))
assert partial
cases.append({'planned_shape':{'B':1,'H':1,'T':1,'K':1,'V':1,'BT':512},
 'status':'cancelled_not_completed','stage':'ptxas after emitted PTX; no cubin/metadata completion',
 'artifacts':partial,'cancellation':pin(OUT/'probe-cancellation.json')})
remaining=[(1,1,1,1024,1,1),(0,1,17,16,1,1),(7,65,33,32,2,3),(65,1,17,16,1,1),
           (1,1,1,16,0,1),(1,1,1,16,1,0),(1,1,0,16,1,1),(1,0,1,16,1,1)]
for K,V,T,BT,B,H in remaining:
    cases.append({'planned_shape':dict(B=B,H=H,T=T,K=K,V=V,BT=BT),'status':'not_started_after_cancellation'})
result={'source':pin(HERE/'chunk_gla_simple_min_dot_tile.py'),
 'checker':pin(HERE/'check_kernel_admission.py'),'finalizer':pin(Path(__file__).resolve()),
 'triton_version':'3.7.1','target':'cuda sm80 warp32','cases':cases,
 'completed_compile_successes':12,'cancelled_partial_compiles':1,'unattempted':8,
 'GPU_execution':False,
 'boundary':'Actual wrapper assertion AST, generated native binder and signature-order packing; exact JIT signature/body with only host autotune decorator omitted. CUDA ownership/allocation/submission mocked; no GPU execution. BT512 resource-only continuation was explicitly cancelled; BT1024 and all later queued profiles were not started. Partial IR/PTX is not a successful compilation or launch. Separate whole-kernel interpreter/contract controls cover K0, empty outputs, tails and multiple heads.'}
(NATIVE/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'result':pin(NATIVE/'result.json'),'compile_success':12,'cancelled':1,'not_started':8,'partial_artifacts':len(partial)},indent=2))
