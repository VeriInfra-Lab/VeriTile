"""Actual candidate guard/interpreter checks; no GPU execution claim."""
import ast, contextlib, hashlib, importlib.util, json, os, runpy
from pathlib import Path
os.environ['TRITON_INTERPRET']='1'
import torch
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
OLD=HERE.with_name('token_attn_mistral')/'token_attn_mistral_repaired.py'
NEW=HERE/'token_attn_mistral_grid_y_checked.py'
old,new=ast.parse(OLD.read_text()),ast.parse(NEW.read_text())
old_kernel=next(x for x in old.body if isinstance(x,ast.FunctionDef) and x.name=='_fwd_kernel_token_att2')
new_kernel=next(x for x in new.body if isinstance(x,ast.FunctionDef) and x.name=='_fwd_kernel_token_att2')
assert ast.dump(old_kernel)==ast.dump(new_kernel)
check=next(x for x in new.body if isinstance(x,ast.FunctionDef) and x.name=='_checked_arguments')
guard=ast.parse('if head > 65535:\n    raise ValueError("query heads exceed the CUDA grid y limit 65535")').body[0]
positions=[i for i,x in enumerate(check.body) if ast.dump(x)==ast.dump(guard)]
assert len(positions)==1
pos=positions[0]
assert ast.dump(check.body[pos-1])==ast.dump(ast.parse('if not batch:\n    return False').body[0])
del check.body[pos]
assert ast.dump(old)==ast.dump(new), 'There must be exactly one added guard and no other AST changes'
# Execute the original complete test logic over the candidate, changing only its
# import path. Keep the original test source and frozen evidence untouched.
checker=OLD.with_name('check_contract.py').read_text()
needle="p=Path(__file__).with_name(name+'_repaired.py')"
assert checker.count(needle)==1
checker=checker.replace(needle,'p=Path('+repr(str(NEW))+')')
exec(compile(checker,str(OLD.with_name('check_contract.py')),'exec'),{'__file__':str(NEW),'__name__':'candidate_checks'})
class CudaFacade(torch.Tensor):
    @property
    def is_cuda(self): return True
f=lambda x:x.as_subclass(CudaFacade)
torch.cuda.device=lambda _:contextlib.nullcontext()
spec=importlib.util.spec_from_file_location('grid_y_candidate',NEW)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class Capture:
    def __init__(self):self.calls=[]
    def __getitem__(self,grid):
        def call(*args,**kw):self.calls.append((grid,args,kw))
        return call
capture=Capture();mod._fwd_kernel_token_att2=capture
records=[]
for B,H,accepted,launch in [(1,65535,True,True),(1,65536,False,False),(1,65537,False,False),(0,65536,True,False),(0,100000,True,False)]:
    a=[f(torch.empty(H,0)),f(torch.empty(0,1,16)),f(torch.full((B,H,16),7.0)),f(torch.empty(1,0,dtype=torch.int32)),f(torch.zeros(B,dtype=torch.int32)),f(torch.full((B,),-2147483648,dtype=torch.int32)),f(torch.zeros(B,dtype=torch.int32)),f(torch.zeros(B,dtype=torch.int32)),f(torch.zeros(B,dtype=torch.int32)),0]
    before=a[2].clone();count=len(capture.calls)
    try:
        result=mod.token_att_fwd2(*a)
    except ValueError as e:
        assert not accepted and str(e)=='query heads exceed the CUDA grid y limit 65535'
    else:
        assert accepted and result is None
    assert len(capture.calls)==count+int(launch)
    if launch:assert capture.calls[-1][0]==(B,H)
    assert torch.equal(a[2],before)
    records.append(dict(batch=B,heads=H,accepted=accepted,launch_captured=launch,output_unchanged=True))
print(json.dumps({'candidate_sha256':hashlib.sha256(NEW.read_bytes()).hexdigest(),'kernel_AST_identical':True,'only_change':'guard head > 65535 after empty-batch return','boundary_cases':records,'backend':'CPU CUDA facade and launch capture, plus actual Triton CPU interpreter; no GPU execution'},sort_keys=True))
