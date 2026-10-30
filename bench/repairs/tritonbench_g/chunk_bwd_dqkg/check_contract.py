"""Retained source AST on CPU storage; CUDA launches and arithmetic are mocked."""
import ast
import hashlib
import importlib.util
import math
from pathlib import Path
import struct
from types import SimpleNamespace
import torch

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'chunk_bwd_dqkg_repaired.py'
NAME = 'chunk_simple_gla_bwd_kernel_dqkg'
spec = importlib.util.spec_from_file_location('chunk_helpers', HERE.parent / 'bmm_chunk_fwd/check_contract.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
function, raw, backing_bytes, Pointer = helpers.function, helpers.raw, helpers.backing_bytes, helpers.Pointer


class Tensor(helpers.Tensor):
    def __init__(self, data, **metadata):
        super().__init__(data)
        self.metadata = metadata
        self.fills = []
    def __getattribute__(self, name):
        if name not in {'metadata', '__dict__', '__class__'}:
            metadata = object.__getattribute__(self, '__dict__').get('metadata', {})
            if name in metadata:
                return metadata[name]
        return super().__getattribute__(name)
    def stride(self, dim=None):
        return self.data.stride() if dim is None else self.data.stride(dim)
    def is_conj(self):
        return self.data.is_conj()
    def fill_(self, value):
        self.fills.append(value)
        self.data.fill_(value)
        return self


def canonical(shape):
    result, n = [], 1
    for d in reversed(shape):
        result.append(n)
        n *= max(d, 1)
    return tuple(reversed(result))


def admitted(args):
    do, q, k, v, g, h, dh, scale = args
    inputs = [q, k, v, g, do, h, dh]
    if not all(isinstance(x, Tensor) for x in inputs):
        return False
    if k.ndim != 4 or v.ndim != 4:
        return False
    B, H, T, K = k.shape
    V = v.shape[-1]
    if min(K, V) <= 0:
        return False
    NT = (T+63)//64
    shapes = [(B,H,T,K), (B,H,T,K), (B,H,T,V), (B,H,T), (B,H,T,V), (B,H,NT*K,V), (B,H,NT*K,V)]
    for x, shape in zip(inputs, shapes):
        if tuple(x.shape) != shape or tuple(x.stride()) != canonical(shape): return False
        if x.dtype != torch.float32 or x.layout != torch.strided: return False
        if x.device != q.device or not x.is_cuda: return False
        if not x.is_contiguous() or x.is_conj() or x.is_neg(): return False
        if x.numel() and x.data_ptr()%4: return False
    if not isinstance(scale, (float, int)) or isinstance(scale, bool): return False
    try:
        scale32 = struct.unpack('f', struct.pack('f', float(scale)))[0]
    except (OverflowError, ValueError):
        return False
    if not math.isfinite(scale32): return False
    if B and H and T:
        for i, x in enumerate(inputs):
            for y in inputs[i+1:]:
                if not (x.data_ptr()+4*x.numel() <= y.data_ptr() or y.data_ptr()+4*y.numel() <= x.data_ptr()):
                    return False
        width = lambda n: 16 if n <= 16 else 32 if n <= 32 else 64
        BK, BV = width(K), width(V)
        NK = (K+BK-1)//BK
        envelope = max(B*H*T*K, B*H*T*V, B*H*NT*K*V, NK*B*H*T) + (64+BK+BV+1)*max(T,K,V,NT*K,K*V,1)
        if envelope >= 2**31 or NK >= 2**31 or NT > 65535 or B*H > 65535: return False
        if any(x.data_ptr()+4*envelope >= 2**64 for x in inputs): return False
    return True


def interpreter():
    pid, dims = [0,0,0], [0,0,0]
    def block(base, shape, strides, offsets, block_shape, order):
        return SimpleNamespace(base=base, shape=shape, strides=strides, offsets=offsets,
            block_shape=block_shape, dtype=base.dtype)
    def indices(p, boundary_check):
        if isinstance(p, Pointer):
            off = torch.as_tensor(p.offset, dtype=torch.int64)
            return p, off, torch.ones_like(off, dtype=torch.bool)
        off = torch.full(p.block_shape, int(p.base.offset), dtype=torch.int64)
        mask = torch.ones(p.block_shape, dtype=torch.bool)
        for axis, (size, stride, start, n) in enumerate(zip(p.shape,p.strides,p.offsets,p.block_shape)):
            shape = [1]*len(p.shape)
            shape[axis] = n
            lane = (torch.arange(n, dtype=torch.int64)+int(start)).reshape(shape)
            off += lane*stride
            if axis in boundary_check:
                mask &= (lane>=0)&(lane<size)
        return p.base, off, mask
    def load(p, boundary_check=(), padding_option=None):
        base, off, mask = indices(p, boundary_check)
        if not isinstance(p, Pointer): assert padding_option == 'zero'
        selected = off[mask]+base.data.storage_offset()
        store = raw(base.data)
        assert bool(((selected>=0)&(selected<len(store))).all()), 'active input OOB'
        out = torch.zeros(off.shape, dtype=base.data.dtype)
        out[mask] = store[selected]
        return out
    def store(p, values, boundary_check=()):
        base, off, mask = indices(p, boundary_check)
        selected = off[mask]+base.data.storage_offset()
        mem = raw(base.data)
        assert bool(((selected>=0)&(selected<len(mem))).all()), 'active output OOB'
        mem[selected] = torch.as_tensor(values).expand(off.shape)[mask]
    node = function(SOURCE, NAME)
    node.decorator_list = []
    for arg in node.args.args: arg.annotation = None
    tl = SimpleNamespace(program_id=lambda axis: pid[axis], num_programs=lambda axis: dims[axis],
        arange=lambda a,b: torch.arange(a,b,dtype=torch.int32), make_block_ptr=block, load=load, store=store,
        cdiv=lambda a,b:(a+b-1)//b, zeros=lambda shape,dtype:torch.zeros(shape,dtype=dtype),
        float32=torch.float32, dot=torch.matmul, trans=lambda x:x.T, exp=torch.exp,
        sum=lambda x,axis=None:torch.sum(x) if axis is None else torch.sum(x,dim=axis), where=torch.where)
    ns = {'tl': tl}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE),'exec'),ns)
    return ns[NAME], pid, dims


def expected(args):
    do, q, k, v, g, h, dh, scale = args
    do,q,k,v,g,h,dh = [x.data.double() for x in (do,q,k,v,g,h,dh)]
    B,H,T,K = q.shape
    V = v.shape[-1]
    BK = 16 if K<=16 else 32 if K<=32 else 64
    NK = (K+BK-1)//BK
    scale = struct.unpack('f',struct.pack('f',float(scale)))[0]
    dq,dk,dg = torch.empty_like(q),torch.empty_like(k),torch.empty((NK,B,H,T),dtype=torch.float64)
    for b in range(B):
        for hd in range(H):
            for t0 in range(0,T,64):
                end = min(t0+64,T)
                gate = g[b,hd,t0:end]
                last = gate[-1]
                values, dout = v[b,hd,t0:end],do[b,hd,t0:end]
                ds = (dout@values.T)*scale*torch.exp(gate[:,None]-gate[None,:])
                ds = ds.tril()
                for tile, k0 in enumerate(range(0,K,BK)):
                    k1 = min(k0+BK,K)
                    hist = h[b,hd,(t0//64)*K+k0:(t0//64)*K+k1]
                    dhist = dh[b,hd,(t0//64)*K+k0:(t0//64)*K+k1]
                    query, key = q[b,hd,t0:end,k0:k1],k[b,hd,t0:end,k0:k1]
                    qcarry = (dout@hist.T)*torch.exp(gate)[:,None]*scale
                    kcarry = (values@dhist.T)*torch.exp(last-gate)[:,None]
                    correction = (hist*dhist).sum()*torch.exp(last)+(kcarry*key).sum()
                    qr, kr = qcarry+ds@key,kcarry+ds.T@query
                    dg[tile,b,hd,t0:end] = (query*qr-key*kr).sum(dim=1)
                    dg[tile,b,hd,end-1] += correction
                    dq[b,hd,t0:end,k0:k1],dk[b,hd,t0:end,k0:k1] = qr,kr
    return dq,dk,dg


def source_binding():
    digest = hashlib.sha256(ast.dump(function(SOURCE,NAME),include_attributes=False).encode()).hexdigest()
    assert digest == '6f8ef806bb0f23d8659e69b71df5b1cae4639c80f5f5054072b04b8950a8f706'
    for name in ('_validate_inputs','chunk_bwd_dqkg_fn'):
        assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in
            {'clone','contiguous','copy_','resolve_neg','resolve_conj','to'} for n in ast.walk(function(SOURCE,name)))
    loads = [n for n in ast.walk(function(SOURCE,NAME)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='load']
    assert sum(any(k.arg=='padding_option' and k.value.value=='zero' for k in n.keywords) for n in loads)==7


def main():
    source_binding()
    spec = importlib.util.spec_from_file_location('direct_chunk', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel,pid,dims = interpreter()
    calls,allocations = [],[]
    selected,trial_trace = [4],[[]]
    bad_output_pointer = [False]
    class Launch:
        def __getitem__(self, grid):
            def run(*args, **kw):
                for warps in trial_trace[0]+[selected[0]]:
                    calls.append((warps,tuple(grid),args[:10]))
                    dims[:] = grid
                    ptrs = [Pointer(x.data) for x in args[:10]]
                    for head in reversed(range(grid[2])):
                        for time in range(grid[1]):
                            for key in reversed(range(grid[0])):
                                pid[:] = key,time,head
                                kernel(*ptrs,*args[10:],*(kw[n] for n in ('T','K','V','BT','BK','BV','NT')))
            return run
    def empty(*shape, dtype, device):
        if len(shape)==1 and isinstance(shape[0],(list,tuple,torch.Size)): shape=tuple(shape[0])
        count = math.prod(shape)
        backing = torch.full((count+9,), -777.,dtype=dtype)
        result = Tensor(backing[4:4+count].reshape(shape))
        if bad_output_pointer[0]: result.metadata['data_ptr']=lambda:2**64-4
        allocations.append((result,backing_bytes(result.data).clone()))
        return result
    module.torch = SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,
        empty=empty,empty_like=lambda x:empty(x.shape,dtype=x.dtype,device=x.device))
    module.chunk_simple_gla_bwd_kernel_dqkg = Launch()
    run = module.chunk_bwd_dqkg_fn
    def inputs(shape, nonzero_gate=False, shared=False):
        B,H,T,K,V=shape
        shapes=[(B,H,T,V),(B,H,T,K),(B,H,T,K),(B,H,T,V),(B,H,T),
            (B,H,((T+63)//64)*K,V),(B,H,((T+63)//64)*K,V)]
        arena=torch.empty(sum(math.prod(s)+7 for s in shapes),dtype=torch.float32) if shared else None
        result,offset=[],0
        for index,s in enumerate(shapes):
            count=math.prod(s)
            backing=arena[offset:offset+count+7] if shared else torch.empty(count+7,dtype=torch.float32)
            backing.copy_((torch.arange(count+7)%5-2)/8)
            data=backing[3:3+count].reshape(s)
            if index==4: data.fill_(0.125 if nonzero_gate else 0.)
            result.append(Tensor(data));offset+=count+7
        return result+[0.5]
    cases=[(1,1,1,1,1),(1,2,65,17,19),(2,2,3,33,65),(1,1,67,65,3),
        (1,1,64,16,16),(1,1,2,32,32),(1,1,1,64,64),(0,2,65,17,19),
        (2,0,65,17,19),(2,2,0,17,19),(1,1,3,5,7),(1,1,65,17,19)]
    for index,shape in enumerate(cases):
        args=inputs(shape,nonzero_gate=index==10,shared=index==11)
        if index==6: args[-1]=-0.0
        assert admitted(args)
        reference=expected(args)
        for selected[0] in (4,8):
            for trial_trace[0] in ([],[4,8]):
                snapshots=[backing_bytes(x.data).clone() for x in args[:-1]]
                old_calls,old_alloc=len(calls),len(allocations)
                result=run(*args)
                assert tuple(result)==tuple(x[0] for x in allocations[old_alloc:])
                assert result[2].fills==[-1e9]
                for out,want in zip(result,reference):
                    torch.testing.assert_close(out.data.double(),want,rtol=3e-6,atol=3e-6)
                    assert out.is_contiguous() and out.dtype==torch.float32 and out.device==args[1].device
                    assert tuple(out.stride())==canonical(out.shape)
                for x,before in zip(args[:-1],snapshots): assert torch.equal(backing_bytes(x.data),before)
                for out,before in allocations[old_alloc:]:
                    start,end=out.data.storage_offset()*4,(out.data.storage_offset()+out.numel())*4
                    after=backing_bytes(out.data)
                    assert torch.equal(after[:start],before[:start]) and torch.equal(after[end:],before[end:])
                active=bool(shape[0] and shape[1] and shape[2])
                assert len(calls)-old_calls == (len(trial_trace[0])+1 if active else 0)
                for _,_,passed in calls[old_calls:]:
                    want=[args[1],args[2],args[3],args[5],args[4],args[0],args[6],*result]
                    assert all(x is y for x,y in zip(passed,want))
    rejections=0
    def reject(args, post_allocation=False):
        nonlocal rejections
        assert admitted(args)==post_allocation
        old_calls,old_alloc=len(calls),len(allocations)
        before=[backing_bytes(x.data).clone() for x in args[:-1] if isinstance(x,Tensor)]
        try: run(*args)
        except ValueError: pass
        else: raise AssertionError('invalid contract accepted')
        assert len(calls)==old_calls
        assert len(allocations)-old_alloc==(3 if post_allocation else 0)
        for x,b in zip((x for x in args[:-1] if isinstance(x,Tensor)),before): assert torch.equal(backing_bytes(x.data),b)
        rejections+=1
    for index in range(7):
        for field,value in [('dtype',torch.float16),('layout',torch.sparse_coo),('is_cuda',False),
            ('device',torch.device('cuda:1')),('is_contiguous',lambda:False),('is_conj',lambda:True),
            ('is_neg',lambda:True),('data_ptr',lambda:1),('stride',lambda dim=None:(0,0,0,0))]:
            args=inputs((1,2,3,5,7));args[index].metadata[field]=value;reject(args)
        args=inputs((1,2,3,5,7));args[index]=None;reject(args)
        args=inputs((1,2,3,5,7));args[index].metadata.update(shape=(1,),ndim=1);reject(args)
    for scale in (True,None,'1',float('inf'),float('nan'),1e100,10**1000):
        args=inputs((1,1,3,5,7));args[-1]=scale;reject(args)
    args=inputs((1,1,3,5,7));args[2]=args[1];reject(args)
    args=inputs((1,1,3,5,7));args[5]=Tensor(torch.zeros((1,1,4,7)));reject(args)
    args=inputs((1,1,3,5,7));args[2].metadata['ndim']=3;reject(args)
    args=inputs((1,1,3,5,7));args[3].metadata['ndim']=3;reject(args)
    args=inputs((1,1,3,5,7));args[0].metadata['data_ptr']=lambda:2**64-4;reject(args)
    for huge in [(1,65536,1,1,1),(1,1,65536*64,1,1),(1,1,1,2**25,1)]:
        args=inputs((1,1,1,1,1));B,H,T,K,V=huge
        shapes=[(B,H,T,V),(B,H,T,K),(B,H,T,K),(B,H,T,V),(B,H,T),
            (B,H,((T+63)//64)*K,V),(B,H,((T+63)//64)*K,V)]
        for i,(x,s) in enumerate(zip(args[:-1],shapes)):
            x.metadata.update(shape=s,ndim=len(s),stride=lambda dim=None,s=s:canonical(s) if dim is None else canonical(s)[dim],
                numel=lambda s=s:math.prod(s),data_ptr=lambda i=i:4096+i*2**40)
        reject(args)
    bad_output_pointer[0]=True
    reject(inputs((1,1,3,5,7)),post_allocation=True)
    print(f'{len(cases)} admissions × both warp choices × empty/nonempty tuner traces; {rejections} rejections passed.')
    print('Complete dQ/dK/per-key-tile dG targets, direct input/return identities, fill and backing-byte frames passed.')
    print('Accepted arithmetic AST retained; CPU arithmetic and CUDA launch mocked; no GPU numerical claim.')


if __name__=='__main__': main()
