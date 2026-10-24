"""Retained kernel/helper source on CPU; CUDA allocation, context and launch mocked."""
import ast
import contextlib
import hashlib
import importlib.util
import math
from pathlib import Path
import struct
from types import SimpleNamespace
import torch

torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'matmul_dequant_int4_repaired.py'
ORIGINAL=HERE.parents[2]/'tritonbench_g/matmul_dequant_int4/matmul_dequant_int4.py'
spec=importlib.util.spec_from_file_location('int4_helpers',HERE.parent/'bmm_chunk_fwd/check_contract.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
function,raw,backing_bytes,Pointer=helpers.function,helpers.raw,helpers.backing_bytes,helpers.Pointer


def unbox(x): return x.data if isinstance(x,Tensor) else x

class Tensor:
    def __init__(self,data,device='cuda:0',**metadata):
        self.data=data;self.device=torch.device(device);self.metadata=metadata
    def __getattribute__(self,name):
        if name not in {'metadata','__dict__','__class__'}:
            meta=object.__getattribute__(self,'__dict__').get('metadata',{})
            if name in meta:return meta[name]
        return object.__getattribute__(self,name)
    @property
    def shape(self):return self.data.shape
    @property
    def ndim(self):return self.data.ndim
    @property
    def dtype(self):return self.data.dtype
    @property
    def layout(self):return self.data.layout
    @property
    def is_cuda(self):return self.device.type=='cuda'
    @property
    def requires_grad(self):return self.data.requires_grad
    def stride(self,dim=None):return self.data.stride() if dim is None else self.data.stride(dim)
    def numel(self):return self.data.numel()
    def element_size(self):return self.data.element_size()
    def storage_offset(self):return self.data.storage_offset()
    def is_contiguous(self):return self.data.is_contiguous()
    def is_neg(self):return self.data.is_neg()
    def is_conj(self):return self.data.is_conj()
    def data_ptr(self):return self.data.data_ptr()
    def untyped_storage(self):return self.data.untyped_storage()
    def wrap(self,data):return Tensor(data,self.device)
    def transpose(self,a,b):return self.wrap(self.data.transpose(a,b))
    def contiguous(self):return self.wrap(self.data.contiguous())
    def view(self,*shape):return self.wrap(self.data.view(*shape))
    def cuda(self,rank):
        assert rank==self.device.index
        return self
    def to(self,arg):
        if isinstance(arg,torch.dtype):return self.wrap(self.data.to(arg))
        return Tensor(self.data.clone() if torch.device(arg)!=self.device else self.data,arg)
    def amax(self,*args,**kw):return self.wrap(self.data.amax(*args,**kw))
    def amin(self,*args,**kw):return self.wrap(self.data.amin(*args,**kw))
    def all(self):return self.data.all()
    def round(self):return self.wrap(self.data.round())
    def clamp(self,*args):return self.wrap(self.data.clamp(*args))
    def __neg__(self):return self.wrap(-self.data)
    def __add__(self,x):return self.wrap(self.data+unbox(x))
    def __sub__(self,x):return self.wrap(self.data-unbox(x))
    def __mul__(self,x):return self.wrap(self.data*unbox(x))
    def __truediv__(self,x):return self.wrap(self.data/unbox(x))
    def __lt__(self,x):return self.wrap(self.data<unbox(x))
    def __gt__(self,x):return self.wrap(self.data>unbox(x))
    def __lshift__(self,x):return self.wrap(self.data<<x)
    def __getitem__(self,index):return self.wrap(self.data[index])
    def __setitem__(self,index,value):self.data[index]=unbox(value)
    def __iadd__(self,x):self.data+=unbox(x);return self


def interpreter(name):
    pid=[0,0,0]
    def load(p,mask=None,other=0.):
        off=torch.as_tensor(p.offset)+p.data.storage_offset()
        mask=torch.ones_like(off,dtype=torch.bool) if mask is None else torch.broadcast_to(mask,off.shape)
        indices=off[mask].long();storage=raw(p.data)
        assert bool(((indices>=0)&(indices<len(storage))).all()),'active load OOB'
        out=torch.full(off.shape,other,dtype=p.data.dtype);out[mask]=storage[indices]
        return out
    def store(p,values,mask=None):
        off=torch.as_tensor(p.offset)+p.data.storage_offset()
        mask=torch.ones_like(off,dtype=torch.bool) if mask is None else torch.broadcast_to(mask,off.shape)
        indices=off[mask].long();storage=raw(p.data)
        assert bool(((indices>=0)&(indices<len(storage))).all()),'active store OOB'
        storage[indices]=torch.as_tensor(values,dtype=p.data.dtype).expand(off.shape)[mask]
    node=function(SOURCE,name);node.decorator_list=[]
    for arg in node.args.args:arg.annotation=None
    tl=SimpleNamespace(program_id=lambda axis=0:pid[axis],cdiv=lambda a,b:(a+b-1)//b,
        arange=lambda a,b:torch.arange(a,b,dtype=torch.int32),load=load,store=store,float32=torch.float32,
        zeros=lambda shape,dtype:torch.zeros(shape,dtype=dtype),dot=lambda a,b:torch.matmul(a.float(),b.float()))
    ns={'tl':tl};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE),'exec'),ns)
    return ns[name],pid


def configs(name):
    node=function(SOURCE,name)
    entries=node.decorator_list[0].keywords[0].value.elts
    return [ast.literal_eval(x.args[0]) for x in entries]


def packed_values(k,n,group,dtype):
    digits=(torch.arange(k*n).reshape(k,n)%16).to(torch.int64)
    zeros=(torch.arange((k//group)*n).reshape(k//group,n)%16).to(torch.int64)
    scale=((torch.arange((k//group)*n)%3+1).reshape(k//group,n)/8).to(dtype)
    q=torch.zeros((k//8,n),dtype=torch.int64)
    z=torch.zeros((k//group,n//8),dtype=torch.int64)
    for i in range(8):q+=digits[i::8]<<(4*i);z+=zeros[:,i::8]<<(4*i)
    return [Tensor(q.to(torch.int32)),Tensor(scale),Tensor(z.to(torch.int32))],(digits-zeros.repeat_interleave(group,dim=0)).to(dtype)*scale.repeat_interleave(group,dim=0)


def quant_reference(weight,group):
    k,n=weight.shape;fmt='e' if weight.dtype==torch.float16 else 'f'
    cast=lambda x:struct.unpack(fmt,struct.pack(fmt,float(x)))[0]
    q=[[0]*n for _ in range(k//8)];z=[[0]*(n//8) for _ in range(k//group)];s=[[0.]*n for _ in range(k//group)]
    for col in range(n):
        for g in range(k//group):
            xs=[float(weight[g*group+i,col]) for i in range(group)]
            lo,hi=min(0.,*xs),max(0.,*xs)
            scale=cast(cast(hi-lo)/15)
            assert math.isfinite(scale) and scale>0
            zp=min(15,max(0,round(cast(cast(-lo)/scale))))
            s[g][col]=scale;z[g][col//8]+=zp<<(4*(col%8))
            for i,x in enumerate(xs):
                digit=min(15,max(0,round(cast(cast(x/scale)+cast(zp)))))
                row=g*group+i;q[row//8][col]+=digit<<(4*(row%8))
    return torch.tensor(q,dtype=torch.int64).to(torch.int32),torch.tensor(s,dtype=weight.dtype),torch.tensor(z,dtype=torch.int64).to(torch.int32)


def main():
    for name in ('matmul4_kernel','dequantize_kernel'):
        assert ast.dump(function(SOURCE,name),include_attributes=False)==ast.dump(function(ORIGINAL,name),include_attributes=False)
    spec=importlib.util.spec_from_file_location('direct_int4',SOURCE);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    dequant,pid=interpreter('dequantize_kernel');fused,fpid=interpreter('matmul4_kernel')
    ds,fs=configs('dequantize_kernel'),configs('matmul4_kernel')
    assert len(ds)==8 and len(fs)==17
    allocations,calls,mm_calls=[],[],[]
    choice,trial_trace,current=[0],[[]],[3]
    def factory(kind,*shape,dtype=None,device=None):
        if len(shape)==1 and isinstance(shape[0],(tuple,list,torch.Size)):shape=tuple(shape[0])
        dtype=dtype or torch.get_default_dtype();count=math.prod(shape)
        storage=torch.full((count+7,),-77,dtype=dtype)
        data=storage[:count].reshape(shape)
        if kind!='empty':data.fill_(1 if kind=='ones' else 0)
        out=Tensor(data,device or 'cpu');allocations.append((out,backing_bytes(storage).clone()));return out
    @contextlib.contextmanager
    def device(target):
        previous=current[0];current[0]=torch.device(target).index
        try:yield
        finally:current[0]=previous
    class Launch:
        def __getitem__(self,grid):
            def run(b,s,z,out,*args):
                for ci in trial_trace[0]+[choice[0]]:
                    dims=grid(ds[ci]);calls.append((ci,b,s,z,out,current[0]))
                    for x in reversed(range(dims[0])):
                        for y in range(dims[1]):
                            pid[:2]=[x,y]
                            dequant(*(Pointer(t.data) for t in (b,s,z,out)),*args,ds[ci]['BLOCK_SIZE_K'],ds[ci]['BLOCK_SIZE_N'])
            return run
    def mm(a,b,out):
        mm_calls.append((a,b,out,current[0]));out.data.copy_(torch.mm(a.data,b.data));return out
    module.torch=SimpleNamespace(Tensor=Tensor,float16=torch.float16,float32=torch.float32,int32=torch.int32,
        strided=torch.strided,device=torch.device,cuda=SimpleNamespace(device=device),
        ones=lambda *s,**kw:factory('ones',*s,**kw),zeros=lambda *s,**kw:factory('zeros',*s,**kw),
        empty=lambda *s,**kw:factory('empty',*s,**kw),mm=mm,
        isfinite=lambda x:Tensor(torch.isfinite(x.data),x.device),
        where=lambda condition,a,b:Tensor(torch.where(condition.data,unbox(a),unbox(b)),condition.device))
    module.dequantize_kernel=Launch()
    admitted=0
    for dtype in (torch.float16,torch.float32):
        for k,n,g in [(8,8,8),(24,8,8),(24,72,8),(128,256,64)]:
            args,want=packed_values(k,n,g,dtype)
            for choice[0] in range(8):
                trial_trace[0]=[0,7] if choice[0]%2 else []
                before=[backing_bytes(x.data).clone() for x in args]
                old=len(calls)
                out=module.dequantize_int4(*args,'cuda:0',dtype,g)
                assert out is allocations[-1][0]
                assert torch.equal(out.data.view(torch.int16 if dtype==torch.float16 else torch.int32),want.view(torch.int16 if dtype==torch.float16 else torch.int32))
                assert current[0]==3 and all(c[-1]==0 for c in calls[old:])
                for x,b in zip(args,before):assert torch.equal(backing_bytes(x.data),b)
                assert torch.equal(raw(out.data)[out.numel():],torch.full((7,),-77,dtype=dtype))
                admitted+=1
    for supplied in (False,True):
        args,b=packed_values(24,8,8,torch.float16);a=Tensor((torch.arange(3*24).reshape(3,24)%5-2).half()/8)
        out=Tensor(torch.full((3,8),-33.,dtype=torch.float16)) if supplied else None
        choice[0]=6;trial_trace[0]=[1,4]
        result=module.matmul_dequantize_int4_s1(a,*args,8,out)
        if supplied:assert result is out
        else:assert result is allocations[-2][0]
        assert torch.equal(result.data,torch.mm(a.data,b))
        assert mm_calls[-1][0] is a and mm_calls[-1][2] is result and current[0]==3
        admitted+=1
    quant_cases=0
    for dtype in (torch.float16,torch.float32):
        for default in (torch.float32,torch.float64):
            torch.set_default_dtype(default)
            for k,n,g in [(8,8,8),(24,8,8),(16,16,16),(16,24,8),(128,256,32)]:
                weight=((torch.arange(k*n).reshape(k,n)%17-8)/8).to(dtype)
                tensor=Tensor(weight);before=backing_bytes(weight).clone();want=quant_reference(weight,g)
                q,s,z,returned_group=module.quantize_int4(tensor,g,0)
                assert returned_group==g
                for actual,expected in zip((q,s,z),want):assert torch.equal(actual.data,expected)
                assert torch.equal(backing_bytes(weight),before)
                module._packed(q,s,z,g,dtype)
                for actual in (q,s,z):
                    rows,cols=actual.shape
                    expected=(1,rows) if rows==1 or cols==1 else (cols,1)
                    assert actual.stride()==expected
                quant_cases+=1
    torch.set_default_dtype(torch.float32)
    fused_cases=0
    for no_groups in (False,True):
        m,k,n,g=3,256,256,(256 if no_groups else 128)
        args,b=packed_values(k,n,g,torch.float16)
        a=Tensor((torch.arange(m*k).reshape(m,k)%5-2).half()/8)
        for config in fs:
            out=Tensor(torch.full((m,n),-99.,dtype=torch.float16))
            for program in range(((m+config['BLOCK_SIZE_M']-1)//config['BLOCK_SIZE_M'])*((n+config['BLOCK_SIZE_N']-1)//config['BLOCK_SIZE_N'])):
                fpid[0]=program
                fused(Pointer(a.data),Pointer(args[0].data),Pointer(out.data),Pointer(args[1].data),Pointer(args[2].data),
                    m,n,k,*a.stride(),*args[0].stride(),*out.stride(),*args[1].stride(),*args[2].stride(),g,no_groups,
                    *(config[x] for x in ('BLOCK_SIZE_M','BLOCK_SIZE_N','BLOCK_SIZE_K','GROUP_SIZE_M')))
            assert torch.equal(out.data,torch.mm(a.data,b))
            fused_cases+=1
    rejected=0
    def reject(call):
        nonlocal rejected
        old=len(calls)
        try:call()
        except (ValueError,AssertionError):pass
        else:raise AssertionError('invalid admission accepted')
        assert len(calls)==old and current[0]==3
        rejected+=1
    for index in range(3):
        for field,value in [('dtype',torch.float64),('ndim',1),('is_cuda',False),('layout',torch.sparse_coo),
            ('is_contiguous',lambda:False),('storage_offset',lambda:1),('is_conj',lambda:True),
            ('is_neg',lambda:True),('requires_grad',True),('data_ptr',lambda:1),('stride',lambda dim=None:(999,1))]:
            args,_=packed_values(24,16,8,torch.float16);args[index].metadata[field]=value
            reject(lambda args=args:module.dequantize_int4(*args,'cuda:0',torch.float16,8))
    for group in (0,-1,True,7,100):
        args,_=packed_values(24,16,8,torch.float16);reject(lambda group=group:module.dequantize_int4(*args,'cuda:0',torch.float16,group))
    for target in ('cpu','cuda','cuda:1'):
        args,_=packed_values(24,16,8,torch.float16);reject(lambda target=target:module.dequantize_int4(*args,target,torch.float16,8))
    for dtype in (torch.float16,torch.float32):
        for value in (0.,float('inf'),float('nan')):
            weight=Tensor(torch.full((8,8),value,dtype=dtype));reject(lambda:module.quantize_int4(weight,8,0))
    weight=Tensor(torch.ones((8,8),dtype=torch.float16));reject(lambda:module.quantize_int4(weight,8,1))
    args,_=packed_values(24,16,8,torch.float16);args[2]=args[0];reject(lambda:module.dequantize_int4(*args,'cuda:0',torch.float16,8))
    for index in range(3):
        for kind in ('capacity','storage_end','formed_pointer'):
            args,_=packed_values(24,16,8,torch.float16)
            tensor=args[index]
            original=tensor.untyped_storage()
            base=original.data_ptr() if kind=='capacity' else (2**64-32 if kind=='storage_end' else 2**64-2**32)
            size=1 if kind=='capacity' else original.nbytes()
            tensor.metadata['untyped_storage']=lambda base=base,size=size:SimpleNamespace(data_ptr=lambda:base,nbytes=lambda:size)
            tensor.metadata['data_ptr']=lambda base=base:base
            reject(lambda args=args:module.dequantize_int4(*args,'cuda:0',torch.float16,8))
    for field,value in [('dtype',torch.float32),('device',torch.device('cuda:1')),
                        ('stride',lambda dim=None:(999,1)),('requires_grad',True)]:
        args,_=packed_values(24,24,8,torch.float16)
        a=Tensor(torch.ones((3,24),dtype=torch.float16));out=Tensor(torch.empty((3,24),dtype=torch.float16));out.metadata[field]=value
        reject(lambda:module.matmul_dequantize_int4_s1(a,*args,8,out))
    args,_=packed_values(24,24,8,torch.float16);a=Tensor(torch.ones((3,24),dtype=torch.float16))
    reject(lambda:module.matmul_dequantize_int4_s1(a,*args,8,a))
    for rank in (True,-1):
        weight=Tensor(torch.ones((8,8),dtype=torch.float16))
        reject(lambda rank=rank:module.quantize_int4(weight,8,rank))
    def fail_mm(*args,**kwargs):raise RuntimeError('injected library failure')
    module.torch.mm=fail_mm
    try:module.matmul_dequantize_int4_s1(a,*args,8)
    except RuntimeError:pass
    else:raise AssertionError('injected failure was swallowed')
    assert current[0]==3
    module.torch.mm=mm
    print(f'{admitted} dequant/s1 admissions, {quant_cases} quantizer cases, {fused_cases} fused config/branch cases, {rejected} rejections passed.')
    print('Both retained kernel ASTs, complete output targets, original helper operations, direct identities, layouts, device restoration and byte frames passed.')
    print('CPU arithmetic and CUDA allocation/launch mocked; no GPU numerical claim.')


if __name__=='__main__':main()
