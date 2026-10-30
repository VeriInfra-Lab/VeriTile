"""Candidate AST/regression/boundary validation; capture is not GPU execution."""
import ast,contextlib,hashlib,importlib.util,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
OLD=HERE.with_name('flash_decode2_phi')/'flash_decode2_phi_repaired.py'
NEW=HERE/'flash_decode2_phi_block_checked.py'
old,new=ast.parse(OLD.read_text()),ast.parse(NEW.read_text())
guard=ast.parse('if BLOCK_DMODEL > 1048576:\n    raise ValueError("lane block exceeds Triton arange maximum 1048576")').body[0]
host=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='flash_decode_stage2')
positions=[i for i,n in enumerate(host.body) if ast.dump(n)==ast.dump(guard)]
assert len(positions)==1
idx=positions[0]
assert ast.dump(host.body[idx-1])==ast.dump(ast.parse('if batch == 0 or head_num == 0:\n    return').body[0])
del host.body[idx]
assert ast.dump(old)==ast.dump(new)
# Run the 42 original AST-interpreter checks over the candidate, replacing
# only their source path. The original checker and evidence stay untouched.
checker=OLD.with_name('check_contract.py').read_text()
needle='source=HERE.parent/task/(task+"_repaired.py")'
assert checker.count(needle)==1
checker=checker.replace(needle,'source=Path('+repr(str(NEW))+')')
exec(compile(checker,str(OLD.with_name('check_contract.py')),'exec'),{'__file__':str(NEW),'__name__':'__main__'})
import torch
class CudaFacade(torch.Tensor):
    @property
    def is_cuda(self):return True
f=lambda t:t.as_subclass(CudaFacade)
torch.cuda.device=lambda _:contextlib.nullcontext()
spec=importlib.util.spec_from_file_location('phi_block_candidate',NEW)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class Capture:
    def __init__(self):self.calls=[]
    def __getitem__(self,grid):
        def call(*args,**kw):self.calls.append((grid,args,kw))
        return call
capture=Capture();mod._fwd_kernel_flash_decode_stage2=capture
records=[]
for B,H,D,accept,launch in [(1,1,2**20,True,True),(1,1,2**20+1,False,False),(1,1,2**21,False,False),(0,1,2**20+1,True,False),(1,0,2**20+1,True,False),(0,1,2**30,True,False)]:
    a=[f(torch.zeros(B,H,1,D)),f(torch.zeros(B,H,1)),f(torch.ones(B,dtype=torch.int32)),f(torch.full((B,H,D),7.0)),1]
    before=a[3].clone();n=len(capture.calls)
    try:result=mod.flash_decode_stage2(*a)
    except ValueError as e:
        assert not accept and str(e)=='lane block exceeds Triton arange maximum 1048576'
    else:assert accept and result is None
    assert len(capture.calls)==n+int(launch) and torch.equal(a[3],before)
    if launch:assert capture.calls[-1][0]==(B,H) and capture.calls[-1][2]['BLOCK_DMODEL']==2**20
    records.append({'batch':B,'heads':H,'dimension':D,'accepted':accept,'launch_captured':launch,'output_unchanged':True})
print(json.dumps({'candidate_sha256':hashlib.sha256(NEW.read_bytes()).hexdigest(),'sole_guard_AST_change':True,'kernel_AST_identical':True,'boundary_cases':records,'gpu_execution':False,'scope':'Original source AST-interpreter regression, actual wrapper with CPU CUDA-admission facade, launch capture only for large boundaries'},sort_keys=True))
