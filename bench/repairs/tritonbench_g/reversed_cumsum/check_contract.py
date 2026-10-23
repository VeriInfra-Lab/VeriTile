"""Source guards and bounded CPU replay of the actual repaired kernel AST."""
import ast
import contextlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'reversed_cumsum_repaired.py'
ORIGINAL = HERE.parents[2] / 'tritonbench_g/reversed_cumsum/reversed_cumsum.py'

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

shared = load(HERE.parent / 'add_example/check_contract.py', 'vector_test_storage')
function, raw_storage = shared.function, shared.raw_storage

class Tensor(shared.TensorFacade):
    @property
    def ndim(self): return self.data.ndim

class Pointer:
    dtype = SimpleNamespace(element_ty=torch.float32)
    def __init__(self, tensor, offsets=0): self.tensor, self.offsets = tensor, offsets
    def __add__(self, offsets): return Pointer(self.tensor, self.offsets+offsets)

class Scalar(int):
    def to(self, dtype): assert dtype==torch.int64; return int(self)

def source_checks():
    kernel = function(SOURCE, 'chunk_global_reversed_cumsum_vector_kernel')
    old = function(ORIGINAL, kernel.name)
    assert list(map(ast.dump,kernel.decorator_list)) == list(map(ast.dump,old.decorator_list))
    def assignment(nodes,name):
        return next(n for n in nodes if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name)
    for name in ('o_i','m_s','b_z'):
        assert ast.dump(assignment(kernel.body,name))==ast.dump(assignment(old.body,name))
    loop = next(n for n in kernel.body if isinstance(n,ast.For))
    old_loop = next(n for n in old.body if isinstance(n,ast.For))
    assert ast.dump(assignment(loop.body,'b_c'))==ast.dump(assignment(old_loop.body,'b_c'))
    assert ast.dump(loop.body[-1])==ast.dump(old_loop.body[-1]), 'post-store carry order changed'
    text = SOURCE.read_text()
    for fragment in ['tl.program_id(0).to(tl.int64)', 'tl.program_id(1).to(tl.int64)',
        'range(tl.cast(tl.cdiv(T, BT), tl.int64) - 1, -1, -1)',
        'tl.where(active_t, lane_t, 0)', 'tl.where(active_s, lane_s, 0)',
        'tl.load(p_s, mask=active, other=0).to(tl.float32)', 'T * S, S, 1,']:
        assert fragment in text
    assert 'resolve_neg' not in text and '.contiguous()' not in text
    wrapper = function(SOURCE,'chunk_global_reversed_cumsum_vector')
    expected = ['not isinstance(s, torch.Tensor)','s.layout != torch.strided','s.dtype != torch.float32',
        'dtype is not None and dtype != torch.float32','s.ndim != 4','not s.is_cuda',
        'not s.is_contiguous()','s.is_neg()',
        's.numel() > 0 and (s.shape[0] * s.shape[1] >= 2 ** 31 or (s.shape[3] + 31) // 32 >= 2 ** 31)',
        's.numel() > 0 and s.data_ptr() % 4 != 0','z.numel() == 0']
    assert [ast.unparse(n.test) for n in wrapper.body if isinstance(n,ast.If)]==expected
    lean=(HERE/'Contract.lean').read_text()
    for fragment in ['s.shape.length != 4','!s.isContiguous','s.isNeg',
        'rows s >= 2^31 || featureBlocks s >= 2^31','s.dataPtr.toNat % 4 != 0']:
        assert fragment in lean
    assert '.Proofs.' not in (HERE/'Kernels.lean').read_text()
    assert 'FPEquiv' not in (HERE/'Correct.lean').read_text()
    assert 'Correct' not in (HERE/'FPEquiv.lean').read_text()
    assert 'dotAcc' in (HERE/'Proofs/NativeGraph.lean').read_text()

def interpreter(name='chunk_global_reversed_cumsum_vector_kernel'):
    pid=[0,0]
    def bounds(p):
        offsets=torch.as_tensor(p.offsets)
        assert bool((offsets>=0).all()) and bool((offsets<p.tensor.numel()).all()), 'even inactive coordinates must be clamped'
        return offsets
    def load_word(p,mask,other):
        offsets=bounds(p)
        value=torch.full_like(offsets,other,dtype=torch.float32)
        value[mask]=raw_storage(p.tensor)[p.tensor.storage_offset()+offsets[mask]]
        return value
    def store_word(p,value,mask):
        offsets=bounds(p)
        raw_storage(p.tensor)[p.tensor.storage_offset()+offsets[mask]]=value[mask]
    node=function(SOURCE,name)
    node.decorator_list=[]
    for arg in node.args.args: arg.annotation=None
    ns={'tl':SimpleNamespace(program_id=lambda ax:Scalar(pid[ax]),arange=torch.arange,
        int64=torch.int64,float32=torch.float32,where=torch.where,zeros=torch.zeros,
        cast=lambda x,dtype:int(x),cdiv=lambda x,y:(x+y-1)//y,load=load_word,store=store_word,
        dot=lambda x,y,allow_tf32:torch.matmul(x.to(torch.float32),y),sum=torch.sum)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE),'exec'),ns)
    return ns[name],pid

def checks():
    module=load(SOURCE,'vector_contract_review')
    kernel,pid=interpreter()
    calls=[];backings=[];launches=[0]
    def empty_like(x,dtype):
        # Output has guard words and a nonzero offset; singleton strides may
        # differ while active coordinates remain contiguous.
        backing=torch.full((x.numel()+16,),-12345.,dtype=dtype)
        output=backing[8:8+x.numel()].reshape(x.shape)
        backings.append(backing)
        calls.append(('allocation',x))
        return Tensor(output,str(x.device))
    class Launch:
        def __getitem__(self,grid):
            def run(x,out,row_stride,time_stride,feature_stride,**kw):
                assert row_stride==kw['T']*kw['S'] and time_stride==kw['S'] and feature_stride==1
                assert grid==((kw['S']+31)//32,x.shape[0]*x.shape[1]) and kw['BS']==32
                calls.append(('launch',x,out))
                # Every original tuning option consumes the previous actual output.
                for bt in (16,32,64):
                    for warps in (2,4,8):
                        for row in reversed(range(grid[1])):
                            for feature in reversed(range(grid[0])):
                                pid[:]=[feature,row]
                                kernel(Pointer(x.data),Pointer(out.data),row_stride,time_stride,feature_stride,
                                    kw['T'],kw['S'],bt,32)
                                launches[0]+=1
            return run
    module.torch=SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,
        empty_like=empty_like,cuda=SimpleNamespace(device=lambda _:contextlib.nullcontext()))
    module.chunk_global_reversed_cumsum_vector_kernel=Launch()
    bad=[([],None,TypeError),(Tensor(torch.eye(2).to_sparse()),None,ValueError),
        (Tensor(torch.ones(1,1,1,1,dtype=torch.float64)),None,TypeError),
        (Tensor(torch.ones(1,1,1,1)),torch.float64,TypeError),(Tensor(torch.ones(2,3)),None,ValueError),
        (Tensor(torch.ones(1,1,2,3),'cpu',False),None,ValueError),
        (Tensor(torch.ones(1,1,2,3).transpose(2,3)),None,ValueError),
        (Tensor(torch.ones(1,1,2,1).expand(1,1,2,3)),None,ValueError),
        (Tensor(torch._neg_view(torch.ones(1,1,2,3))),None,ValueError),
        (Tensor(shared.unaligned_tensor(1).reshape(1,1,1,4)),None,ValueError)]
    class MetaTensor(Tensor):
        def data_ptr(self): return 4
    bad += [(MetaTensor(torch.empty(shape,device='meta')),None,ValueError)
        for shape in ((2**31,1,1,1),(1,1,1,2**36))]
    for x,dtype,error in bad:
        calls.clear()
        try: module.chunk_global_reversed_cumsum_vector(x,dtype=dtype)
        except error:pass
        else:raise AssertionError('unsupported input admitted')
        assert calls==[]
    good=[torch.arange(2*1*17*33,dtype=torch.float32).reshape(2,1,17,33).as_strided((2,1,17,33),(561,999999,33,1)),
        torch.arange(2*3*17*33+8,dtype=torch.float32)[8:].reshape(2,3,17,33),
        torch.empty(0,1,17,33),torch.empty(2,0,17,33),torch.empty(2,3,0,33),torch.empty(2,3,17,0),
        shared.unaligned_tensor(1,empty=True).reshape(1,1,0,1)]
    good += [torch.arange(t*s,dtype=torch.float32).reshape(1,1,t,s)%7 for t,s in
        ((1,1),(15,31),(16,32),(17,33),(31,1),(32,32),(33,35),(63,31),(64,32),(65,33))]
    for tensor in good:
        calls.clear();backings.clear()
        x=Tensor(tensor);before=raw_storage(tensor).clone()
        out=module.chunk_global_reversed_cumsum_vector(x,dtype=torch.float32)
        expected=tensor.flip(2).cumsum(2).flip(2)
        torch.testing.assert_close(out.data,expected,rtol=0,atol=0)
        assert out.shape==tensor.shape and out.dtype==torch.float32 and out.device==x.device
        assert bool((backings[0][:8]==-12345).all()) and bool((backings[0][-8:]==-12345).all())
        torch.testing.assert_close(raw_storage(tensor),before,rtol=0,atol=0)
        assert len(calls)==(2 if tensor.numel() else 1)
        if tensor.numel(): assert calls[1][1] is x and calls[1][2] is out
    # Original singleton metadata uses a stride with no active head variation.
    singleton=good[0]
    assert singleton.is_contiguous() and singleton.stride(1)>singleton.numel()
    # Original block-load padding is unspecified; a permitted nonzero tail
    # changes an active suffix. The repair explicitly zeroes that tail.
    tri=torch.triu(torch.ones(16,16));padded=torch.ones(16,1);padded[0]=2
    assert (tri@padded)[0,0].item()!=2
    assert (tri@torch.cat((torch.tensor([[2.]]),torch.zeros(15,1))))[0,0].item()==2
    return len(bad),len(good),launches[0]

if __name__=='__main__':
    source_checks()
    rejected,admitted,launches=checks()
    print(f'Passed source/guards, {rejected} rejections, {admitted} admissions, {launches} program executions over all nine configs.')
    print('CPU storage AST replay and original singleton/padding counterexamples; no native GPU numerical claim.')
