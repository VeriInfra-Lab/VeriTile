"""Run actual dequantization source AST on CPU; selected INT8/FP32/FP16 route."""
import ast
import contextlib
import copy
import re
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'dequantize_rowwise_repaired.py'
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
        storage=torch.empty(0,dtype=ptr.tensor.dtype).set_(ptr.tensor.untyped_storage(),0,(ptr.tensor.untyped_storage().nbytes()//ptr.tensor.element_size(),),(1,))
        if mask is None:return storage[ptr.tensor.storage_offset()+int(ptr.offset)]
        result=torch.zeros_like(ptr.offset,dtype=ptr.tensor.dtype)
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
    fake=SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,float16=torch.float16,int8=torch.int8,
        empty=allocate,cuda=SimpleNamespace(device=lambda _:contextlib.nullcontext()))
    ns={'torch':fake,'triton':SimpleNamespace(next_power_of_2=lambda x:1<<(x-1).bit_length()),'_dequantize_rowwise_repaired':Launch()}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function('_check'),function('dequantize_rowwise_repaired')],type_ignores=[])),str(SOURCE),'exec'),ns)
    guards=[ast.unparse(n.test) for n in function('_check').body if isinstance(n,ast.If)]
    assert guards==['not isinstance(x, torch.Tensor) or not isinstance(state_x, torch.Tensor)',
        'x.layout != torch.strided or state_x.layout != torch.strided','x.dtype != torch.int8 or state_x.dtype != torch.float32',
        'not x.is_cuda or not state_x.is_cuda','x.device != state_x.device','x.ndim != 2 or state_x.ndim != 1',
        'not x.is_contiguous() or not state_x.is_contiguous() or x.is_neg() or state_x.is_neg()',
        'x.shape[1] > 2 ** 20','x.shape[0] >= 2 ** 31','x.numel() > 0 and x.shape[0] > state_x.numel()',
        'x.numel() > 0 and state_x.data_ptr() % 4 != 0']
    # Bind the checked Python guard list to the actual Lean check function.
    lean=(HERE/'Contract.lean').read_text()
    checks=lean.split('def checkTensorInputs',1)[1].split('def checkInputs',1)[0]
    assert re.findall(r'\bif (.*?) then',checks)==[
        'x.layout != .strided || s.layout != .strided',
        'x.dtype != .other "int8" || s.dtype != .floating .fp32',
        '!x.isCuda || !s.isCuda','x.device != s.device',
        'x.shape.length != 2 || s.shape.length != 1',
        '!x.isContiguous || !s.isContiguous || x.isNeg || s.isNeg',
        'columns x>2^20','rows x≥2^31','x.numel>0 && rows x>s.numel',
        'x.numel>0 && s.dataPtr.toNat%4 != 0']
    assert 'dtype : x.dtype=.other "int8" ∧ s.dtype=.floating .fp32' in lean
    assert 'alignment : 0<x.numel → s.dataPtr.toNat%4=0' in lean
    # Actual JIT matches original source except for explicit signed64 widening.
    original=HERE.parents[2]/'tritonbench_g/dequantize_rowwise/dequantize_rowwise.py'
    old=next(n for n in ast.parse(original.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_dequantize_rowwise')
    new=copy.deepcopy(function('_dequantize_rowwise_repaired'));new.name=old.name
    class EraseWidening(ast.NodeTransformer):
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if isinstance(node.func,ast.Attribute) and node.func.attr=='to' and len(node.args)==1 and ast.unparse(node.args[0])=='tl.int64':
                return node.func.value
            return node
    new=EraseWidening().visit(new)
    assert ast.dump(old)==ast.dump(new)
    assignment={n.targets[0].id:ast.unparse(n.value) for n in function('_dequantize_rowwise_repaired').body if isinstance(n,ast.Assign)}
    assert assignment['pid']=='tl.program_id(axis=0).to(tl.int64)'
    assert assignment['arange']=='tl.arange(0, P2).to(tl.int64)'
    assert assignment['output']=='max_val * x * inv_127'
    assert not any(isinstance(n,ast.Attribute) and n.attr in ('clone','to','resolve_neg','contiguous','copy_') for n in ast.walk(function('dequantize_rowwise_repaired')))
    assert '.Proofs.' not in (HERE/'Kernels.lean').read_text()
    accepted=0
    for r,c in ((0,0),(0,7),(3,0),(1,1),(2,3),(3,17),(2,129),(2,513)):
        for scale in (0.,-2.,3.):
            backing=((torch.arange(r*c+8)%256)-128).to(torch.int8)
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
    # Different dtypes may share the same read-only bytes, including partial overlap.
    backing=torch.tensor([1.,-2.,3.,4.])
    x,s=Tensor(backing.view(torch.int8).reshape(2,8),'cuda:1'),Tensor(backing[:2],'cuda:1')
    before=backing.clone();out=ns['dequantize_rowwise_repaired'](x,s)
    torch.testing.assert_close(out.data,((s.data[:,None]*x.data)*torch.tensor(1/127)).half(),rtol=0,atol=0,equal_nan=True)
    torch.testing.assert_close(backing,before,rtol=0,atol=0,equal_nan=True)
    special=torch.tensor([0.,-0.,float('inf'),-float('inf'),float('nan'),torch.finfo(torch.float32).max,1e-38,-1e-38])
    weights=torch.tensor([-128,-1,0,1,127],dtype=torch.int8).repeat(len(special),1)
    out=ns['dequantize_rowwise_repaired'](Tensor(weights),Tensor(special))
    torch.testing.assert_close(out.data,((special[:,None]*weights)*torch.tensor(1/127)).half(),rtol=0,atol=0,equal_nan=True)
    goodx,goods=Tensor(torch.ones(2,3,dtype=torch.int8)),Tensor(torch.ones(2))
    bad=[([],goods),(goodx,[]),(Tensor(torch.eye(2,dtype=torch.int8).to_sparse()),goods),
        (Tensor(torch.ones(2,3,dtype=torch.float32)),goods),(goodx,Tensor(torch.ones(2,dtype=torch.float64))),
        (Tensor(torch.ones(2,3,dtype=torch.int8),'cpu',False),goods),(goodx,Tensor(torch.ones(2),'cuda:1')),
        (Tensor(torch.ones(3,dtype=torch.int8)),goods),(goodx,Tensor(torch.ones(2,1))),
        (Tensor(torch.ones(3,2,dtype=torch.int8).T),goods),(goodx,Tensor(torch.ones(4)[::2])),
        (Tensor(torch._neg_view(torch.ones(2,3,dtype=torch.int8))),goods),(goodx,Tensor(torch._neg_view(torch.ones(2)))),
        (Tensor(torch.empty(0,2**20+1,dtype=torch.int8)),goods),(Tensor(torch.empty(2**31,0,dtype=torch.int8)),goods),(goodx,Tensor(torch.empty(1)))]
    bad.extend((goodx,Tensor(helpers.unaligned_tensor(i))) for i in (1,2,3))
    for x,s in bad:
        calls.clear()
        try:ns['dequantize_rowwise_repaired'](x,s)
        except (TypeError,ValueError):pass
        else:raise AssertionError('invalid inputs admitted')
        assert calls==[]
    calls.clear()
    out=ns['dequantize_rowwise_repaired'](Tensor(torch.empty(3,0,dtype=torch.int8)),Tensor(torch.empty(0)))
    assert out.numel()==0 and calls==['allocation']
    # Concrete original-source defects: logical and pointer order differ.
    transposed=torch.arange(6,dtype=torch.int8).reshape(3,2).T
    physical=torch.empty(0,dtype=torch.int8).set_(transposed.untyped_storage(),0,(6,),(1,))
    assert not torch.equal(physical[:3],transposed[0])
    negative=torch._neg_view(torch.tensor([1],dtype=torch.int8))
    physical=torch.empty(0,dtype=torch.int8).set_(negative.untyped_storage(),0,(1,),(1,))
    assert physical.item()==1 and negative.item()==-1
    assert (torch.tensor(2**20,dtype=torch.int32)*2**12).item()==0
    assert (torch.tensor(2**20,dtype=torch.int64)*2**12).item()==2**32
    # Exact host scalar bits and widened padded endpoint, without huge allocations.
    assert torch.tensor(1/127,dtype=torch.float32).view(torch.int32).item()==1006699012
    assert (2**31-2)*2**20+(2**20-1)<2**51
    print(f'PASS: {accepted+3} complete CPU AST output/frame cases; {len(bad)} guard rejections; INT8 weights, FP32 scales, FP16 output, empty scale skip, aliases, tails and runtime scalar bits.')
    print('CUDA mocked; no GPU execution or native conformance claim. No numerical transformation assumption.')
if __name__=='__main__':main()
