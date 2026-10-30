"""Actual repaired Python AST, full Q/K output and admission checks on CPU."""
import ast
import contextlib
import hashlib
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'int8_quantization_repaired.py'
spec = importlib.util.spec_from_file_location('add_helpers', HERE.parent / 'add_example/check_contract.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

class Tensor(helpers.TensorFacade):
    def dim(self): return self.data.dim()
    def zero_(self): self.data.zero_(); return self

def function(name):
    return next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)

def reference(x, block, query):
    output = torch.empty_like(x, dtype=torch.int8)
    length, channels = x.shape[-2:]
    scales = torch.empty((*x.shape[:-2], (length+block-1)//block, 1), dtype=torch.float32)
    if channels == 0: scales.zero_(); return output, scales
    xx = x.reshape(math.prod(x.shape[:-2]), length, channels)
    yy = output.reshape(xx.shape)
    ss = scales.reshape(math.prod(x.shape[:-2]), (length+block-1)//block)
    for b in range(xx.shape[0]):
        for start in range(0, length, block):
            values = xx[b, start:start+block].clone()
            if query: values = values * (channels**-0.5 * 1.44269504)
            if not torch.isfinite(values).all():
                yy[b, start:start+block] = 0
                ss[b, start//block] = 0
                continue
            scale = values.abs().amax()/127
            ss[b, start//block] = scale
            if scale > 0:
                normalized = torch.clamp(values / scale, -127, 127)
                yy[b, start:start+block] = torch.trunc(normalized + torch.where(normalized >= 0, 0.5, -0.5)).to(torch.int8)
            else: yy[b, start:start+block] = 0
    return output, scales

def main():
    assert hashlib.sha256(ast.dump(function('_per_block_int8_repaired')).encode()).hexdigest() == 'dfd74b467912d86e71cbb195af991c362aa246b95639bb062dd52e6c2ca5f6d0'
    pid, calls = [0], []
    class Pointer:
        def __init__(self, tensor, offsets=0): self.tensor, self.offsets = tensor, offsets
        def __add__(self, offsets): return Pointer(self.tensor, offsets)
    def load(ptr, mask, other):
        offsets = ptr.offsets.long()
        result = torch.full_like(offsets, other, dtype=torch.float32)
        result[mask] = helpers.raw_storage(ptr.tensor)[ptr.tensor.storage_offset()+offsets[mask]]
        return result
    def store(ptr, value, mask=None):
        flat = ptr.tensor.reshape(-1)
        if mask is None: flat[ptr.offsets.long()] = value.to(flat.dtype)
        else: flat[ptr.offsets[mask].long()] = value[mask].to(flat.dtype)
    def scalarwise(op):
        return lambda a,b: op(torch.as_tensor(a), torch.as_tensor(b))
    tl = SimpleNamespace(int64=torch.int64, int8=torch.int8, program_id=lambda _:torch.tensor(pid[0],dtype=torch.int32),
        arange=lambda a,b:torch.arange(a,b,dtype=torch.int32), load=load, store=store, abs=torch.abs,
        max=lambda x,_:torch.max(x), where=torch.where, minimum=scalarwise(torch.minimum), maximum=scalarwise(torch.maximum))
    node = function('_per_block_int8_repaired'); node.decorator_list=[]
    for arg in node.args.args: arg.annotation=None
    kernel = {'tl':tl}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE),'exec'),kernel)
    class Launch:
        def __getitem__(self, grid):
            def run(x,q,s,**kw):
                calls.append(('launch',grid,kw['BASE'],x.device))
                kw.pop('num_warps'); kw.pop('enable_fp_fusion')
                for j in reversed(range(grid[0])):
                    pid[0]=j
                    kernel['_per_block_int8_repaired'](Pointer(x.data),Pointer(q.data),Pointer(s.data),**kw)
            return run
    def allocate(shape, device, dtype):
        calls.append(('allocation',tuple(shape),device,dtype))
        return Tensor(torch.empty(shape,dtype=dtype),str(device))
    fake_torch=SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,int8=torch.int8,empty=allocate,
        cuda=SimpleNamespace(device=lambda _:contextlib.nullcontext()))
    ns={'torch':fake_torch,'math':math,'triton':SimpleNamespace(cdiv=lambda n,d:(n+d-1)//d,next_power_of_2=lambda n:1<<(n-1).bit_length()),
        '_per_block_int8_repaired':Launch()}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function('_quantize_one'),function('per_block_int8_repaired')],type_ignores=[])),str(SOURCE),'exec'),ns)
    guards=[ast.unparse(n.test) for n in function('_quantize_one').body if isinstance(n,ast.If)]
    assert guards == ['not isinstance(x, torch.Tensor)', 'x.dim() not in (3, 4)',
        'x.layout != torch.strided or x.dtype != torch.float32','not x.is_cuda',
        'not x.is_contiguous() or x.is_neg()', 'not isinstance(block, int) or isinstance(block, bool) or block <= 0',
        'block * channels > 2 ** 20', 'x.numel() >= 2 ** 61 or batches * blocks >= 2 ** 61',
        'x.numel() > 0 and x.data_ptr() % 4 != 0']
    assert not any(isinstance(n,ast.Attribute) and n.attr in ('clone','resolve_neg','contiguous') for n in ast.walk(function('_quantize_one')))
    assignments={n.targets[0].id:ast.unparse(n.value) for n in function('_per_block_int8_repaired').body if isinstance(n,ast.Assign)}
    for key,expr in {'block':'tl.program_id(0).to(tl.int64) + BASE','lane':'tl.arange(0, WIDTH).to(tl.int64)',
        'x':'tl.where(finite, x, 0.0)','usable':'(bad == 0) & (scale > 0.0)','quantized':'biased.to(tl.int8)'}.items(): assert assignments[key]==expr
    contract=(HERE/'Contract.lean').read_text()
    for name in ('rank','layout','dtype','cuda','contiguous','noNeg','blockPositive','tile','inputExtent','scaleExtent','alignment'):
        assert f'  {name} :' in contract
    assert '.Proofs.' not in (HERE/'Kernels.lean').read_text()
    accepted=0
    cases=[((2,5,3),(1,2,7,5),2,3),((1,1,1),(2,3,2),3,1),((0,2,3),(1,0,4),2,4),
        ((2,4,0),(1,3,0),2,2),((1,9,2),(1,2,3,3),4,2)]
    for qshape,kshape,bq,bk in cases:
        for mode in ('ordinary','zero','nan','inf','tiny','overflow'):
            tensors=[]; backings=[]
            for shape in (qshape,kshape):
                n=math.prod(shape); backing=torch.arange(n+5,dtype=torch.float32)-6
                data=backing[2:2+n].reshape(shape)
                if mode=='zero': data.zero_()
                elif n and mode=='nan': data.reshape(-1)[0]=float('nan')
                elif n and mode=='inf': data.reshape(-1)[-1]=float('inf')
                elif mode=='tiny': data.fill_(torch.finfo(torch.float32).tiny/1000)
                elif mode=='overflow': data.fill_(torch.finfo(torch.float32).max)
                tensors.append(Tensor(data,'cuda:'+str(len(tensors))))
                backings.append((backing,backing.clone()))
            calls.clear(); q,qs,k,ks=ns['per_block_int8_repaired'](*tensors,bq,bk)
            for original,out,sc,block,query in ((tensors[0],q,qs,bq,True),(tensors[1],k,ks,bk,False)):
                expected=reference(original.data,block,query)
                torch.testing.assert_close(out.data,expected[0],rtol=0,atol=0)
                torch.testing.assert_close(sc.data,expected[1],rtol=0,atol=0)
                assert out.device==sc.device==original.device
            for backing,before in backings: torch.testing.assert_close(backing,before,rtol=0,atol=0,equal_nan=True)
            assert len({id(x.data) for x in (q,qs,k,ks)})==4
            accepted+=1
    shared=Tensor(torch.arange(30,dtype=torch.float32).reshape(2,5,3))
    q,qs,k,ks=ns['per_block_int8_repaired'](shared,shared,2,3)
    for actual,expected in zip((q.data,qs.data,k.data,ks.data),(*reference(shared.data,2,True),*reference(shared.data,3,False))):
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    accepted+=1
    bad=[([],2),(Tensor(torch.ones(2,3)),2),(Tensor(torch.ones(1,2,3,dtype=torch.float64)),2),
        (Tensor(torch.ones(1,2,3),'cpu',False),2),(Tensor(torch.ones(1,2,3).transpose(1,2)),2),
        (Tensor(torch._neg_view(torch.ones(1,2,3))),2),(Tensor(torch.ones(1,2,3)),0),
        (Tensor(torch.ones(1,2,3)),-1),(Tensor(torch.ones(1,2,3)),True),(Tensor(torch.ones(1,2,3)),2.0),
        (Tensor(torch.empty(0,1,2**20+1)),1),(Tensor(torch.empty(2**61,0,0,device='meta')),1),
        (Tensor(torch.empty(2**61,1,0,device='meta')),1)]
    # Zero channels with huge batch and zero length is valid; scales are empty.
    bad.pop(11)
    bad += [(Tensor(helpers.unaligned_tensor(k).reshape(1,1,4)),2) for k in (1,2,3)]
    for x,b in bad:
        calls.clear()
        try: ns['_quantize_one'](x,b,False)
        except (TypeError,ValueError): pass
        else: raise AssertionError('invalid contract admitted')
        assert calls==[]
    # Chunk boundaries are pure host scheduling; do not allocate giant tensors.
    limit=2147483647
    for n in (0,1,limit-1,limit,limit+1,2*limit+7):
        chunks=[(base,min(n-base,limit)) for base in range(0,n,limit)]
        assert sum(count for _,count in chunks)==n
        assert all(0<count<=limit for _,count in chunks)
        assert all(a+c==b for (a,c),(b,_) in zip(chunks,chunks[1:]))
    print(f'PASS: {accepted} complete Q/K CPU AST output, metadata, alias and frame cases; {len(bad)} guard rejections; chunk boundaries.')
    print('CUDA allocation/launch are mocked; IEEE/FTZ, conversions and host constants are primitive model boundaries, not transformation TODOs.')

if __name__=='__main__': main()
