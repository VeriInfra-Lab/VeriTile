"""Check original-source binding, guards and CPU source execution; no CUDA run."""
import ast
from pathlib import Path
from types import SimpleNamespace
import torch

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
SOURCE = HERE/'quant_transpose_kernel_repaired.py'
ORIGINAL = HERE.parents[2]/'tritonbench_g/quant_transpose_kernel/quant_transpose_kernel.py'

class Tensor:
    def __init__(self,data, runtime=None):
        self.data, self.runtime = data,runtime
        self.shape,self.ndim,self.dtype,self.layout = tuple(data.shape),data.ndim,data.dtype,torch.strided
        self.device,self.is_cuda,self.negative = SimpleNamespace(index=0),True,data.is_neg()
        self.strides=data.stride()
    def numel(self): return self.shape[0]*self.shape[1]
    def data_ptr(self): return getattr(self,"pointer",self.data.data_ptr())
    def is_neg(self): return self.negative
    def stride(self,axis=None): return self.strides if axis is None else self.strides[axis]
    def size(self,axis): return self.shape[axis]
    def abs(self):
        self.runtime.events.append('abs')
        if self.runtime.failure=='abs': raise RuntimeError('abs')
        return Tensor(self.data.abs(),self.runtime)
    def max(self):
        self.runtime.events.append('max')
        return Tensor(self.data.max(),self.runtime)
    def unsqueeze(self,axis):
        self.runtime.events.append('unsqueeze')
        result=Tensor(self.data.unsqueeze(axis),self.runtime)
        self.runtime.maximum=result
        return result
    def __rtruediv__(self,scalar):
        self.runtime.events.append('reciprocal')
        result=Tensor(scalar/self.data,self.runtime)
        self.runtime.inverse=result
        return result

class Pointer:
    def __init__(self,tensor,offset=0): self.tensor,self.offset=tensor,offset
    def __add__(self,offset): return Pointer(self.tensor,self.offset+offset)
    def storage(self):
        data=self.tensor.data
        size=data.untyped_storage().nbytes()//data.element_size()
        return torch.as_strided(data,(size,),(1,),storage_offset=0)
    def indices(self): return self.offset+self.tensor.data.storage_offset()

class Language:
    constexpr=object
    extra=SimpleNamespace(cuda=SimpleNamespace(libdevice=SimpleNamespace(llrint=lambda x:torch.round(x).to(torch.int64))))
    def __init__(self): self.pid=0
    def program_id(self,axis): assert axis==0; return self.pid
    @staticmethod
    def arange(start,end): return torch.arange(start,end,dtype=torch.int64)
    @staticmethod
    def load(pointer,mask=None):
        if mask is None: return pointer.storage()[pointer.indices()]
        result=torch.zeros(mask.shape,dtype=pointer.tensor.dtype)
        result[mask]=pointer.storage()[pointer.indices()[mask]]
        return result
    @staticmethod
    def store(pointer,values,mask):
        pointer.storage()[pointer.indices()[mask]]=values[mask].to(pointer.tensor.dtype)

class Runtime:
    Tensor=Tensor
    float32,int8,strided=torch.float32,torch.int8,torch.strided
    cuda=SimpleNamespace(current_device=lambda:0)
    def __init__(self,body,language,execute,failure):
        self.body,self.language,self.execute,self.failure=body,language,execute,failure
        self.events=[]
    def empty(self,*shape,device,dtype):
        self.events.append('allocation')
        assert device=='cuda' and dtype==torch.int8
        if self.failure=='allocation': raise RuntimeError('allocation')
        self.output=Tensor(torch.empty(shape,dtype=dtype),self)
        return self.output
    def __getitem__(self,grid): self.grid=grid; return self.launch
    def launch(self,a,inverse,out,sam,san,sbn,sbm,m,n):
        self.events.append('launch')
        assert inverse is self.inverse and out is self.output
        assert out.shape==(n,m) and (sam,san,sbn,sbm)==(*a.stride(),*out.stride())
        metadata={'BLOCK_M':128,'BLOCK_N':128,'GROUP_M':8}
        expected=((m+127)//128)*((n+127)//128)
        assert self.grid(metadata)==(expected,)
        if self.failure=='launch': raise RuntimeError('launch')
        if self.execute:
            for _ in range(2):  # The duplicate configurations retain the same writes.
                for pid in range(expected):
                    self.language.pid=pid
                    self.body(Pointer(a),Pointer(inverse),Pointer(out),sam,san,sbn,sbm,m,n,128,128,8)

def instantiate(execute=False,failure=None):
    language=Language()
    body=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_quantize_global_transpose')
    body.decorator_list=[]
    kernel_ns={'tl':language}
    exec(compile(ast.Module([body],[]),str(SOURCE),'exec'),kernel_ns)
    runtime=Runtime(kernel_ns[body.name],language,execute,failure)
    ns={'torch':runtime,'triton':SimpleNamespace(cdiv=lambda a,b:(a+b-1)//b),'_quantize_global_transpose':runtime}
    nodes=[n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='quantize_global_transpose' or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_unchecked_quantize_global_transpose' for t in n.targets)]
    exec(compile(ast.Module(nodes,[]),str(SOURCE),'exec'),ns)
    return ns['quantize_global_transpose'],runtime

def make(shape):
    return ((torch.arange(shape[0]*shape[1],dtype=torch.float32)%257)-128).reshape(shape)

def main():
    old=ast.parse(ORIGINAL.read_text().split('################################################################')[0]).body
    new=ast.parse(SOURCE.read_text()).body
    assert [ast.dump(n) for n in old]==[ast.dump(n) for n in new[:len(old)]]
    assert ast.dump(new[-1].body[-1].value)==ast.dump(ast.parse('_unchecked_quantize_global_transpose(input)',mode='eval').body)
    inputs=[make(s) for s in [(1,1),(3,5),(127,129),(128,256),(256,128),(512,256),(256,512),(1281,129)]]
    inputs += [make((9,7)).t(),make((13,17))[2:9,1:12],make((1,7)).expand(5,7)]
    for data in inputs:
        call,rt=instantiate(execute=True)
        a=Tensor(data,rt)
        out,maximum=call(a)
        expected=torch.round(127.0*(data*(1.0/data.abs().max()))).to(torch.int64).to(torch.int8).t().contiguous()
        assert torch.equal(out.data,expected)
        assert out is rt.output and maximum is rt.maximum and maximum is not rt.inverse
        assert torch.equal(maximum.data,data.abs().max().unsqueeze(0))
        assert rt.events==['abs','max','unsqueeze','reciprocal','allocation','launch']
    rejects=[]
    rejects.append((None,TypeError))
    for attr,value,error in [('pointer',4097,ValueError),('layout',torch.sparse_coo,ValueError),('ndim',1,ValueError),('dtype',torch.float16,TypeError),('is_cuda',False,ValueError),('device',SimpleNamespace(index=1),ValueError),('negative',True,ValueError),('strides',(-1,1),ValueError),('strides',(2,2),ValueError),('strides',(2**31,1),ValueError),('strides',(2**30,1),ValueError),('shape',(0,3),ValueError),('shape',(3,0),ValueError),('shape',(2**31,3),ValueError)]:
        a=Tensor(make((3,5))); setattr(a,attr,value); rejects.append((a,error))
    for a,error in rejects:
        call,rt=instantiate()
        if a is not None: a.runtime=rt
        try: call(a)
        except error: pass
        else: raise AssertionError('unsupported input was accepted')
        assert rt.events==[]
    for failure in ['abs','allocation','launch']:
        call,rt=instantiate(failure=failure)
        try: call(Tensor(make((3,5)),rt))
        except RuntimeError as exc: assert str(exc)==failure
        else: raise AssertionError('runtime failure swallowed')
    # The adapter keeps numerical undefinedness explicit; it does not invent a value.
    for data in [torch.zeros((1,1)),torch.full((1,1),float('nan')),torch.full((1,1),float('inf'))]:
        call,rt=instantiate(execute=False)
        call(Tensor(data,rt))
        assert rt.events[-1]=='launch'
    print('PASS: unchanged source/configuration/wrapper AST; 11 CPU source executions including rotated partial groups, strided views and broadcasts; 15 pre-host rejections; 3 propagated failures; numerical exceptional inputs forwarded explicitly. No GPU execution.')

if __name__=='__main__': main()
