"""Targeted source/guard/host checks; CUDA and autotuning are mocked explicitly."""
import ast
import contextlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch

HERE = Path(__file__).resolve().parent.parent
SOURCE = HERE / 'masked_select_repaired.py'
ORIGINAL = HERE.parents[2] / 'tritonbench_g/masked_select/masked_select.py'

def functions(path):
    return {n.name: n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}

class Tensor:
    def __init__(self, data, device='cuda:0'):
        self.data, self.device = data, torch.device(device)
    @property
    def shape(self): return self.data.shape
    @property
    def layout(self): return self.data.layout
    @property
    def dtype(self): return self.data.dtype
    @property
    def is_cuda(self): return self.device.type == 'cuda'
    def is_neg(self): return self.data.is_neg()
    def numel(self): return self.data.numel()
    def data_ptr(self): return self.data.data_ptr()
    def resolve_neg(self): return Tensor(self.data.resolve_neg(), str(self.device))
    def contiguous(self): return Tensor(self.data.contiguous(), str(self.device))
    def ravel(self): return Tensor(self.data.ravel(), str(self.device))
    def cumsum(self, axis): return Tensor(self.data.cumsum(axis), str(self.device))
    def __getitem__(self, key): return Tensor(self.data[key], str(self.device))
    def item(self): return self.data.item()

class Pointer:
    def __init__(self, tensor, offsets=None): self.tensor, self.offsets = tensor, offsets
    def __add__(self, offsets): return Pointer(self.tensor, offsets)

def raw(t):
    return torch.empty(0, dtype=t.dtype).set_(t.untyped_storage(), 0,
        (t.untyped_storage().nbytes() // t.element_size(),), (1,))

def main():
    original, revised = functions(ORIGINAL), functions(SOURCE)
    for name in ['broadcastable', 'cfggen', 'masked_select_kernel']:
        assert ast.dump(original[name]) == ast.dump(revised[name]), name
    assert '.Proofs.' not in (HERE / 'Kernels.lean').read_text()
    assert '.Proofs.' not in (ORIGINAL.parent / 'Kernels.lean').read_text()
    spec=importlib.util.spec_from_file_location('masked_revision', SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    events=[]
    def broadcast(a,b):
        events.append('broadcast')
        return tuple(Tensor(x,str(a.device)) for x in torch.broadcast_tensors(a.data,b.data))
    def empty(n,dtype,device):
        events.append('allocation');return Tensor(torch.empty(n,dtype=dtype),str(device))
    module.torch=SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,bool=torch.bool,
        broadcast_shapes=torch.broadcast_shapes,broadcast_tensors=broadcast,empty=empty,
        cuda=SimpleNamespace(device=lambda _:contextlib.nullcontext()))
    pid=[0]
    def load(p,mask,other=0.):
        v=torch.full(p.offsets.shape,other,dtype=p.tensor.dtype)
        v[mask]=raw(p.tensor)[p.tensor.storage_offset()+p.offsets[mask]]
        return v
    def store(p,v,mask): raw(p.tensor)[p.tensor.storage_offset()+p.offsets[mask]]=v[mask]
    k=revised['masked_select_kernel'];k.decorator_list=[]
    for a in k.args.args:a.annotation=None
    # Installed CPU interpretation cannot use vector Python `and`; the native
    # source/frontend retains elementwise conjunction. This adaptation is explicit.
    class VectorAnd(ast.NodeTransformer):
        def visit_BoolOp(self,node):
            assert isinstance(node.op,ast.And) and len(node.values)==2
            return ast.copy_location(ast.BinOp(node.values[0],ast.BitAnd(),node.values[1]),node)
    k=VectorAnd().visit(k)
    ns={'tl':SimpleNamespace(program_id=lambda axis:pid[0],arange=torch.arange,
        load=load,store=store,int1=torch.bool)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[k],type_ignores=[])),str(SOURCE),'exec'),ns)
    class Launch:
        def __getitem__(self,grid):
            def run(x,mask,prefix,out,n):
                events.append('launch')
                source_values = raw(x.data)[x.data.storage_offset():x.data.storage_offset()+n]
                expected = source_values[mask.data.reshape(-1)]
                input_words = raw(x.data).view(torch.int32).clone()
                mask_before, prefix_before = mask.data.clone(), prefix.data.clone()
                for width in (256,512,1024,2048,4096):
                    # Each real source configuration must independently fill
                    # the entire output; previous trials cannot hide missing writes.
                    out.data.view(torch.int32).fill_(0x3f765432)
                    for p in range(grid({'BLOCK_SIZE':width})[0]):
                        pid[0]=p;ns['masked_select_kernel'](Pointer(x.data),Pointer(mask.data),
                            Pointer(prefix.data),Pointer(out.data),n,width)
                    assert torch.equal(out.data.view(torch.int32), expected.view(torch.int32))
                    assert torch.equal(raw(x.data).view(torch.int32), input_words)
                    assert torch.equal(mask.data, mask_before) and torch.equal(prefix.data, prefix_before)

            return run
    module.masked_select_kernel=Launch()
    original_ns={'torch':module.torch,'triton':module.triton,'logging':module.logging,
        'broadcastable':module.broadcastable,'masked_select_kernel':module.masked_select_kernel}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[original['masked_select']],type_ignores=[])),
        str(ORIGINAL),'exec'),original_ns)
    negative=torch._neg_view(torch.arange(1.,6.))
    original_out=original_ns['masked_select'](Tensor(negative),Tensor(torch.ones(5,dtype=torch.bool))).data
    assert negative.is_contiguous() and negative.contiguous().is_neg()
    assert negative.data_ptr()==negative.contiguous().data_ptr()
    assert original_out.tolist()==[1.,2.,3.,4.,5.]
    assert negative.tolist()==[-1.,-2.,-3.,-4.,-5.]
    print('Original negative-view witness: raw output',original_out.tolist(),'logical target',negative.tolist())
    backing=torch.arange(20,dtype=torch.float32)
    cases=[(torch.tensor(2.),torch.tensor(True)),
        (backing[::2],torch.arange(10)%2==0),
        (torch._neg_view(backing[2:8]),torch.tensor([True,False,True,True,False,True])),
        (torch.arange(6,dtype=torch.float32).reshape(2,3),torch.tensor([[True],[False]])),
        (backing,torch.zeros(20,dtype=torch.bool)),
        (torch.arange(1025,dtype=torch.float32),torch.ones(1025,dtype=torch.bool)),
        (torch.tensor([0x80000000,0x7f800001,0x7fc00001,0x7f800000],dtype=torch.int64).to(torch.int32).view(torch.float32),torch.ones(4,dtype=torch.bool))]
    for x,m in cases:
        events.clear();before=raw(x).clone();result=module.masked_select(Tensor(x),Tensor(m))
        xx,mm=torch.broadcast_tensors(x,m);expected=xx.resolve_neg()[mm]
        assert torch.equal(result.data.view(torch.int32),expected.view(torch.int32))
        assert torch.equal(raw(x).view(torch.int32),before.view(torch.int32))
        assert events==['broadcast','allocation','launch']
    valid=Tensor(torch.ones(3));mask=Tensor(torch.ones(3,dtype=torch.bool))
    bad=[([],mask),(Tensor(torch.eye(3).to_sparse()),mask),(Tensor(torch.ones(3,dtype=torch.float16)),mask),
        (valid,Tensor(torch.ones(3))),(Tensor(torch.ones(3),'cpu'),mask),(valid,Tensor(mask.data,'cuda:1')),
        (valid,Tensor(torch.ones(2,dtype=torch.bool))),
        (Tensor(torch.empty(0)),Tensor(torch.empty(0,dtype=torch.bool))),
        (Tensor(torch.empty(1<<31,device='meta')),Tensor(torch.empty(1<<31,dtype=torch.bool,device='meta')))]
    for x,m in bad:
        events.clear()
        try:module.masked_select(x,m)
        except (TypeError,ValueError):pass
        else:raise AssertionError('unsupported input admitted')
        assert not events
    storage=torch.arange(20,dtype=torch.uint8).untyped_storage()[1:17]
    unaligned=torch.empty(0,dtype=torch.float32).set_(storage,0,(4,),(1,))
    events.clear()
    try:module.masked_select(Tensor(unaligned),Tensor(torch.ones(4,dtype=torch.bool)))
    except ValueError as e:assert 'aligned' in str(e)
    else:raise AssertionError('unaligned prepared input admitted')
    assert events==['broadcast']
    print('7 raw-word output/frame cases, 10 rejection cases, source identity and explicit CPU BoolOp adaptation passed.')
    print('No GPU execution or numerical conformance claim.')
if __name__=='__main__':main()
