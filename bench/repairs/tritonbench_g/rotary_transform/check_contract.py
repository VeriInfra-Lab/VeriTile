"""Actual transform JIT/host replay with physical CPU storage and mocked CUDA."""
import ast
import contextlib
import importlib.util
import itertools
from pathlib import Path
from types import SimpleNamespace
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('fixtures',HERE.parent/'swiglu_fwd/check_contract.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)

class Tensor(f.Tensor):
    def stride(self,axis=None):return self.data.stride() if axis is None else self.data.stride(axis)
    def is_contiguous(self):return self.data.is_contiguous()
    def is_conj(self):return self.data.is_conj()
    def numel(self):return self.data.numel()
    def element_size(self):return self.data.element_size()
    def data_ptr(self):return self.data.data_ptr()
    def wrap(self,value):return self if value is self.data else Tensor(value,self.device,self.is_cuda)
    def contiguous(self):return self.wrap(self.data.contiguous())
    def __getitem__(self,index):return self.wrap(self.data[index])
    def copy_(self,other):self.data.copy_(other.data);return self
    def to(self,*args,**kwargs):return self.data.to(*args,**kwargs)


def storage(t):
    return torch.empty(0,dtype=t.dtype).set_(t.untyped_storage(),0,
        (t.untyped_storage().nbytes()//t.element_size(),),(1,))


def host(path):
    ids,events=[0,0,0],[]
    def load(ptr,mask=None,other=0):
        off=torch.as_tensor(ptr.offset)
        if mask is None:return storage(ptr.tensor)[ptr.tensor.storage_offset()+off]
        off,mask=torch.broadcast_tensors(off,mask)
        value=torch.full(off.shape,other,dtype=ptr.tensor.dtype)
        value[mask]=storage(ptr.tensor)[ptr.tensor.storage_offset()+off[mask]]
        return value
    def store(ptr,value,mask):
        off,mask,value=torch.broadcast_tensors(torch.as_tensor(ptr.offset),mask,value)
        storage(ptr.tensor)[ptr.tensor.storage_offset()+off[mask]]=value[mask]
    fn=f.function(path,'rotary_kernel');fn.decorator_list=[]
    for a in fn.args.args:a.annotation=None
    argnames=[a.arg for a in fn.args.args]
    ns={'tl':SimpleNamespace(program_id=lambda axis:ids[axis],arange=torch.arange,
        load=load,store=store,float32=torch.float32,where=torch.where)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(path),'exec'),ns)
    kernel=ns['rotary_kernel']
    class Launch:
        def __getitem__(self,grid):
            def run(*args,**kwargs):
                params=dict(zip(argnames,args));params.update(kwargs)
                dims=grid(params) if callable(grid) else grid
                events.append(('launch',dims))
                args=[f.Pointer(x.data) if isinstance(x,Tensor) else x for x in args]
                for row,batch,head in itertools.product(*(reversed(range(n)) for n in dims)):
                    ids[:]=[row,batch,head];kernel(*args,**kwargs)
            return run
    @contextlib.contextmanager
    def device(d):events.append(('device',d));yield
    def empty_like(t):events.append(('allocation',tuple(t.shape)));return Tensor(torch.empty_like(t.data),t.device)
    ns.update(torch=SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,
        int32=torch.int32,int64=torch.int64,empty_like=empty_like,cuda=SimpleNamespace(device=device)),
        triton=SimpleNamespace(cdiv=lambda n,b:(n+b-1)//b,next_power_of_2=lambda n:1<<(n-1).bit_length()),
        rotary_kernel=Launch())
    body=[]
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,ast.FunctionDef) and node.name in ('_storage_span','_validate_inputs','_check_launch_pointers','apply_rotary'):
            node.returns=None
            for a in node.args.args:a.annotation=None
            body.append(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(path),'exec'),ns)
    return ns['apply_rotary'],events


def reference(x,cos,sin,offsets,cu,interleaved,conjugate):
    result=x.clone(); half=cos.shape[1]
    seqs=[(x[b],result[b]) for b in range(x.shape[0])] if cu is None else [
        (x[a:b],result[a:b]) for a,b in zip(cu,cu[1:])]
    for b,(source,target) in enumerate(seqs):
        shift=offsets if isinstance(offsets,int) else offsets[b]
        for r in range(source.shape[0]):
            c=cos[r+shift] if r+shift<cos.shape[0] else torch.ones(half)
            s=sin[r+shift] if r+shift<sin.shape[0] else torch.zeros(half)
            if conjugate:s=-s
            a=source[r,:,0:2*half:2] if interleaved else source[r,:,:half]
            z=source[r,:,1:2*half:2] if interleaved else source[r,:,half:2*half]
            if interleaved:
                target[r,:,0:2*half:2]=a*c-z*s;target[r,:,1:2*half:2]=z*c+a*s
            else:
                target[r,:,:half]=a*c-z*s;target[r,:,half:2*half]=a*s+z*c
    return result


def check_task(name):
    directory=HERE.parent/name
    source=directory/(name+'_repaired.py')
    original=HERE.parents[2]/'tritonbench_g'/name/(name+'.py')
    assert ast.dump(f.function(source,'rotary_kernel'))==ast.dump(f.function(original,'rotary_kernel'))
    body=f.function(source,'apply_rotary')
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and
        isinstance(n.func.value,ast.Name) and n.func.value.id=='cu_seqlens' and n.func.attr=='contiguous'
        for n in ast.walk(body))
    assert '.Proofs.' not in (directory/'Kernels.lean').read_text()
    apply,events=host(source)
    paths=0
    for varlen,offset_tensor,interleaved,inplace,conjugate in itertools.product((False,True),repeat=5):
        for dtype in (torch.int32,torch.int64):
            cu=[0,2,2,5] if varlen else None
            shape=(5,2,6) if varlen else (3,5,2,6)
            x=Tensor(torch.arange(torch.empty(shape).numel()).float().reshape(shape)%7-3)
            c=Tensor(torch.arange(12).float().reshape(6,2)%3-1)
            s=Tensor(torch.arange(12).float().reshape(6,2)%5-2)
            shifts=[0,7,4] if offset_tensor else 1
            offsets=Tensor(torch.tensor(shifts,dtype=dtype)) if offset_tensor else shifts
            cumulative=Tensor(torch.tensor(cu,dtype=dtype)) if varlen else None
            before=x.data.clone();c0=c.data.clone();s0=s.data.clone()
            expected=reference(before,c.data,s.data,shifts,cu,interleaved,conjugate)
            events.clear();out=apply(x,c,s,offsets,cumulative,3 if varlen else None,interleaved,inplace,conjugate)
            assert (out is x)==inplace
            assert torch.equal(out.data,expected)
            if not inplace:assert torch.equal(x.data,before)
            assert torch.equal(c.data,c0) and torch.equal(s.data,s0)
            assert sum(e[0]=='launch' for e in events)==1
            paths+=1
    for shape,half in [((0,3,1,6),3),((2,0,1,6),3),((2,3,1,6),0),
                       ((1,2,1,34),17),((1,2,1,130),65),((1,2,1,256),128)]:
        x=Tensor(torch.ones(shape));c=Tensor(torch.full((3,half),2.));s=Tensor(torch.ones(3,half))
        for inplace in (False,True):
            before=x.data.clone();events.clear();out=apply(x,c,s,inplace=inplace)
            assert torch.equal(out.data,reference(before,c.data,s.data,0,None,False,False))
            assert sum(e[0]=='launch' for e in events)==bool(x.numel()*half);paths+=1
    if name=='rotary_transform':
        for interleaved in (False,True):
            x=Tensor(torch.ones(2,3,1,514));c=Tensor(torch.full((1,514),2.)[:,::2]);s=Tensor(torch.ones(1,514)[:,::2])
            offsets=Tensor(torch.tensor([0,99,2,99],dtype=torch.int64)[::2])
            expected=reference(x.data,c.data,s.data,[0,2],None,interleaved,True)
            out=apply(x,c,s,offsets,interleaved=interleaved,conjugate=True)
            assert torch.equal(out.data,expected);paths+=1
    base=[Tensor(torch.ones(2,3,1,6)),Tensor(torch.ones(4,2)),Tensor(torch.ones(4,2))]
    cases=[]
    def bad(i,value):
        args=base.copy();args[i]=value;cases.append((args,{}))
    bad(0,[]);bad(0,Tensor(torch.ones(2,3,1,6,dtype=torch.float64)));bad(0,Tensor(torch._neg_view(base[0].data)))
    bad(0,Tensor(base[0].data,'cpu',False));bad(1,base[2]);bad(0,Tensor(torch.ones(2,3,6,1).transpose(2,3)))
    for kwargs in ({'seqlen_offsets':-1},{'interleaved':1},{'seqlen_offsets':Tensor(torch.tensor([-1,0]))}):cases.append((base,kwargs))
    packed=Tensor(torch.ones(5,1,6))
    for cu in ([0,3,2,5],[1,2,5],[0,2,4]):
        cases.append(([packed,*base[1:]],{'cu_seqlens':Tensor(torch.tensor(cu)),'max_seqlen':3}))
    cu=Tensor(torch.tensor([0,99,2,99,5],dtype=torch.int64)[::2])
    cases.append(([packed,*base[1:]],{'cu_seqlens':cu,'max_seqlen':3}))
    for remainder in (1,2,3):
        bytes_=torch.zeros(150,dtype=torch.uint8).untyped_storage()[remainder:remainder+144]
        bad(0,Tensor(torch.empty(0).set_(bytes_,0,(2,3,1,6),(18,6,6,1))))
    for args,kwargs in cases:
        events.clear()
        try:apply(*args,**kwargs)
        except (TypeError,ValueError):pass
        else:raise AssertionError((name,'unsupported input admitted',kwargs))
        assert not any(e[0]=='launch' for e in events)
    if name=='rotary_transform':
        old,_=host(original)
        x=Tensor(torch.tensor([[[[1.,2.]],[[3.,4.]]]]))
        c=Tensor(torch.full((2,1),2.));s=Tensor(torch.ones(2,1))
        expected=reference(x.data,c.data,s.data,0,None,False,False)
        assert not torch.equal(old(x,c,s).data,expected)
    else:
        old,_=host(original)
        x=Tensor(torch.tensor([[[1.,2.]],[[3.,4.]]]))
        c=Tensor(torch.zeros(1,1));s=Tensor(torch.ones(1,1))
        cu=Tensor(torch.tensor([0,0,1,0,2],dtype=torch.int32)[::2])
        expected=reference(x.data,c.data,s.data,0,[0,1,2],False,False)
        assert not torch.equal(old(x,c,s,cu_seqlens=cu,max_seqlen=1).data,expected)
    print(f'{name}: unchanged complete JIT; {paths} full source/host paths, {len(cases)} rejections.')
    print('CPU storage replay only; CUDA and native floating-point conformance are not tested.')

if __name__=='__main__':check_task('rotary_transform')
