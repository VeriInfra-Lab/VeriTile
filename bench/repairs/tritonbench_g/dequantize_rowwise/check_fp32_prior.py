"""Run actual dequantization source AST on CPU; optional prior FP32 scope."""
import ast
import contextlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'dequantize_rowwise_fp32_prior.py'
spec=importlib.util.spec_from_file_location('add_helpers',HERE.parent/'add_example/check_contract.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
class Tensor(helpers.TensorFacade):
    @property
    def ndim(self):return self.data.ndim

def function(name):
    return next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)

def main():
    pid,calls=[0],[]
    class Pointer:
        def __init__(self,tensor,offset=0):self.tensor,self.offset=tensor,offset
        def __add__(self,offset):return Pointer(self.tensor,offset)
    def load(ptr,mask=None):
        storage=helpers.raw_storage(ptr.tensor)
        if mask is None:return storage[ptr.tensor.storage_offset()+int(ptr.offset)]
        result=torch.zeros_like(ptr.offset,dtype=torch.float32)
        result[mask]=storage[ptr.tensor.storage_offset()+ptr.offset[mask]]
        return result
    def store(ptr,value,mask):ptr.tensor.reshape(-1)[ptr.offset[mask]]=value[mask].to(torch.float16)
    tl=SimpleNamespace(int64=torch.int64,program_id=lambda axis:torch.tensor(pid[0],dtype=torch.int32),
        arange=lambda a,b:torch.arange(a,b,dtype=torch.int32),load=load,store=store)
    node=function('_dequantize_rowwise_repaired');node.decorator_list=[]
    for a in node.args.args:a.annotation=None
    kns={'tl':tl}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE),'exec'),kns)
    class Launch:
        def __getitem__(self,grid):
            def run(x,s,o,inv,n,BLOCK_SIZE,P2):
                calls.append('launch')
                # Runtime Python scalar enters the source at compiled FP32 precision.
                inv=torch.tensor(inv,dtype=torch.float32)
                for r in reversed(range(grid({})[0])):
                    pid[0]=r
                    kns['_dequantize_rowwise_repaired'](Pointer(x.data),Pointer(s.data),Pointer(o.data),inv,n,BLOCK_SIZE,P2)
            return run
    def allocate(shape,device,dtype):
        calls.append('allocation');return Tensor(torch.empty(shape,dtype=dtype),str(device))
    fake=SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,float16=torch.float16,
        empty=allocate,cuda=SimpleNamespace(device=lambda _:contextlib.nullcontext()))
    ns={'torch':fake,'triton':SimpleNamespace(next_power_of_2=lambda x:1<<(x-1).bit_length()),'_dequantize_rowwise_repaired':Launch()}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function('_check'),function('dequantize_rowwise_repaired')],type_ignores=[])),str(SOURCE),'exec'),ns)
    guards=[ast.unparse(n.test) for n in function('_check').body if isinstance(n,ast.If)]
    assert guards==['not isinstance(x, torch.Tensor) or not isinstance(state_x, torch.Tensor)',
        'x.layout != torch.strided or state_x.layout != torch.strided','x.dtype != torch.float32 or state_x.dtype != torch.float32',
        'not x.is_cuda or not state_x.is_cuda','x.device != state_x.device','x.ndim != 2 or state_x.ndim != 1',
        'not x.is_contiguous() or not state_x.is_contiguous() or x.is_neg() or state_x.is_neg()',
        'x.shape[1] > 2 ** 20','x.shape[0] >= 2 ** 31','x.numel() > 0 and x.shape[0] > state_x.numel()',
        'x.numel() > 0 and (x.data_ptr() % 4 != 0 or state_x.data_ptr() % 4 != 0)']
    assignment={n.targets[0].id:ast.unparse(n.value) for n in function('_dequantize_rowwise_repaired').body if isinstance(n,ast.Assign)}
    assert assignment['pid']=='tl.program_id(axis=0).to(tl.int64)'
    assert assignment['arange']=='tl.arange(0, P2).to(tl.int64)'
    assert assignment['output']=='max_val * x * inv_127'
    assert not any(isinstance(n,ast.Attribute) and n.attr in ('clone','to','resolve_neg','contiguous','copy_') for n in ast.walk(function('dequantize_rowwise_repaired')))
    assert '.Proofs.' not in (HERE/'Proofs/FP32Prior/Kernels.lean').read_text()
    accepted=0
    for r,c in ((0,0),(0,7),(3,0),(1,1),(2,3),(3,17),(2,129),(2,513)):
        for scale in (0.,-2.,3.):
            backing=torch.linspace(-127,127,r*c+8)
            x=Tensor(backing[3:3+r*c].reshape(r,c))
            sb=torch.full((r+7,),scale)
            s=Tensor(sb[2:2+r])
            bx,bs=backing.clone(),sb.clone()
            calls.clear();out=ns['dequantize_rowwise_repaired'](x,s)
            expected=((s.data[:,None]*x.data)*torch.tensor(1/127,dtype=torch.float32)).to(torch.float16)
            torch.testing.assert_close(out.data,expected,rtol=0,atol=0,equal_nan=True)
            torch.testing.assert_close(backing,bx,rtol=0,atol=0)
            torch.testing.assert_close(sb,bs,rtol=0,atol=0)
            assert out.device==x.device and out.shape==x.shape and out.dtype==torch.float16
            assert calls==['allocation']+(['launch'] if r*c else [])
            accepted+=1
    # Read-only source aliases, negative/zero scales, NaN/Inf, signed zero and overflow.
    backing=torch.tensor([0.,-0.,float('inf'),-float('inf'),float('nan'),torch.finfo(torch.float32).max,1e-38,-1e-38])
    x,s=Tensor(backing.reshape(2,4),'cuda:1'),Tensor(backing[:2],'cuda:1')
    before=backing.clone();out=ns['dequantize_rowwise_repaired'](x,s)
    torch.testing.assert_close(out.data,((s.data[:,None]*x.data)*torch.tensor(1/127)).half(),rtol=0,atol=0,equal_nan=True)
    torch.testing.assert_close(backing,before,rtol=0,atol=0,equal_nan=True)
    goodx,goods=Tensor(torch.ones(2,3)),Tensor(torch.ones(2))
    bad=[([],goods),(goodx,[]),(Tensor(torch.eye(2).to_sparse()),goods),
        (Tensor(torch.ones(2,3,dtype=torch.int8)),goods),(goodx,Tensor(torch.ones(2,dtype=torch.float64))),
        (Tensor(torch.ones(2,3),'cpu',False),goods),(goodx,Tensor(torch.ones(2),'cuda:1')),
        (Tensor(torch.ones(3)),goods),(goodx,Tensor(torch.ones(2,1))),
        (Tensor(torch.ones(3,2).T),goods),(goodx,Tensor(torch.ones(4)[::2])),
        (Tensor(torch._neg_view(torch.ones(2,3))),goods),(goodx,Tensor(torch._neg_view(torch.ones(2)))),
        (Tensor(torch.empty(0,2**20+1)),goods),(Tensor(torch.empty(2**31,0)),goods),(goodx,Tensor(torch.empty(1)))]
    bad.extend((Tensor(helpers.unaligned_tensor(i).reshape(1,4)),Tensor(torch.ones(1))) for i in (1,2,3))
    bad.extend((goodx,Tensor(helpers.unaligned_tensor(i))) for i in (1,2,3))
    for x,s in bad:
        calls.clear()
        try:ns['dequantize_rowwise_repaired'](x,s)
        except (TypeError,ValueError):pass
        else:raise AssertionError('invalid inputs admitted')
        assert calls==[]
    calls.clear()
    out=ns['dequantize_rowwise_repaired'](Tensor(torch.empty(3,0)),Tensor(torch.empty(0)))
    assert out.numel()==0 and calls==['allocation']
    # Exact host scalar bits and widened padded endpoint, without huge allocations.
    assert torch.tensor(1/127,dtype=torch.float32).view(torch.int32).item()==1006699012
    assert (2**31-2)*2**20+(2**20-1)<2**51
    print(f'PASS: {accepted+2} complete CPU AST output/frame cases; {len(bad)} guard rejections; FP32 inputs to FP16 output, empty scale skip, aliases, tails and runtime scalar bits.')
    print('CUDA mocked; no GPU conformance claim.')
if __name__=='__main__':main()
