"""Actual source AST on CPU backing storage; CUDA contexts and launches mocked."""
import ast
import contextlib
import hashlib
import importlib.util
from pathlib import Path
import re
from types import SimpleNamespace
import torch

torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'bmm_chunk_fwd_repaired.py'
ORIGINAL=HERE.parents[2]/'tritonbench_g/bmm_chunk_fwd/bmm_chunk_fwd.py'
CONFIGS=[(128,256,64,3,8),(64,256,32,4,4),(128,128,32,4,4),(128,64,32,4,4),(64,128,32,4,4),(128,32,32,4,4),(64,32,32,5,2),(32,64,32,5,2),(64,64,32,4,2)]
spec=importlib.util.spec_from_file_location('bmm_helpers',HERE.parent/'destindex_copy_kv1/check_contract.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
function,raw,backing_bytes=helpers.function,helpers.raw,helpers.backing_bytes
class Pointer(helpers.Pointer):
    @property
    def dtype(self):return SimpleNamespace(element_ty=self.data.dtype)
    def __add__(self,value):return Pointer(self.data,self.offset+value)
class Tensor(helpers.Tensor):
    def element_size(self):return self.data.element_size()
    def stride(self,dim):return self.data.stride(dim)
    def contiguous(self):raise AssertionError('admitted direct tensor unexpectedly copied')

def interpreter(path,name):
    pid=[0,0,0]
    def load(p,mask=None,other=0.):
        offs=torch.as_tensor(p.offset)+p.data.storage_offset()
        mask=torch.ones_like(offs,dtype=torch.bool) if mask is None else torch.broadcast_to(mask,offs.shape)
        indices=offs[mask].long();storage=raw(p.data)
        assert bool(((indices>=0)&(indices<len(storage))).all()),'input OOB'
        value=torch.full(offs.shape,other,dtype=p.data.dtype);value[mask]=storage[indices]
        return value
    def store(p,value,mask=None):
        offs=torch.as_tensor(p.offset)+p.data.storage_offset()
        mask=torch.ones_like(offs,dtype=torch.bool) if mask is None else torch.broadcast_to(mask,offs.shape)
        indices=offs[mask].long();storage=raw(p.data)
        assert bool(((indices>=0)&(indices<len(storage))).all()),'output OOB'
        storage[indices]=torch.as_tensor(value,dtype=p.data.dtype).expand(offs.shape)[mask]
    node=function(path,name);node.decorator_list=[]
    for arg in node.args.args:arg.annotation=None
    tl=SimpleNamespace(program_id=lambda axis:torch.tensor(pid[axis],dtype=torch.int32),
        arange=lambda a,b:torch.arange(a,b,dtype=torch.int32),load=load,store=store,
        cast=lambda x,dtype:torch.as_tensor(x).to(dtype),int64=torch.int64,float32=torch.float32,
        cdiv=lambda a,b:(a+b-1)//b,zeros=lambda shape,dtype:torch.zeros(shape,dtype=dtype),
        dot=torch.matmul,where=torch.where)
    ns={'tl':tl};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),ns)
    return ns[name],pid

def source_binding():
    # The accepted widened arithmetic JIT is pinned independently of this wrapper.
    digest=hashlib.sha256(ast.dump(function(SOURCE,'_bmm_chunk_fwd_kernel_repaired'),include_attributes=False).encode()).hexdigest()
    assert digest=='a5aae42341d16291a7492ebd7fd616091698f88bac14c1556ca590ef05cd6580'
    wrapper=function(SOURCE,'_bmm_chunk_fwd_repaired')
    guards=[n for n in wrapper.body if isinstance(n,ast.If) and any(isinstance(x,ast.Raise) for x in n.body)]
    assert len(guards)==17
    lean=(HERE/'Contract.lean').read_text().split('def checkInputs',1)[1].split('def checkArguments',1)[0]
    assert re.findall(r'\.error \.(\w+)',lean)==['layout','dtype','output','cuda','device','shapes','sequence','contiguous','strides','negative','chunk','causal','extent','grid','alignment','overlap']
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in {'clone','resolve_neg','copy_'} for n in ast.walk(wrapper))

def main():
    source_binding()
    spec=importlib.util.spec_from_file_location('bmm_direct',SOURCE);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    kernel,pids=interpreter(SOURCE,'_bmm_chunk_fwd_kernel_repaired')
    clear,clear_pids=interpreter(SOURCE,'_clear_causal_upper')
    calls=[];allocations=[];choice=[0];trials=[False]
    class Launch:
        def __getitem__(self,grid):
            def run(a,b,out,seq,*args,**kw):
                config_ids=[0,6,choice[0]] if trials[0] else [choice[0]]
                for ci in config_ids:
                    m,n,k,_,_=CONFIGS[ci];dims=grid({'BLOCK_SIZE_M':m,'BLOCK_SIZE_N':n})
                    calls.append(('bmm',ci,dims,a,b,out,seq))
                    for y in range(dims[1]):
                        for z in reversed(range(dims[2])):
                            for x in reversed(range(dims[0])):
                                pids[:]=[x,y,z]
                                kernel(Pointer(a.data),Pointer(b.data),Pointer(out.data),None if seq is None else Pointer(seq.data),*args,kw['HAS_SEQ_IDX'],m,n,k)
            return run
    class Clear:
        def __getitem__(self,grid):
            def run(out,chunk,heads,**kw):
                calls.append(('clear',grid,out))
                for y in range(grid[1]):
                    for z in range(grid[2]):
                        for x in range(grid[0]):
                            clear_pids[:]=[x,y,z];clear(Pointer(out.data),chunk,heads,kw['BLOCK_M'],kw['BLOCK_N'])
            return run
    def empty(shape,device,dtype):
        count=1
        for x in shape:count*=x
        storage=torch.full((count+9,),-777.,dtype=dtype)
        out=Tensor(storage[4:4+count].reshape(shape));allocations.append(out);return out
    module._bmm_chunk_fwd_kernel_repaired=Launch();module._clear_causal_upper=Clear()
    module.torch=SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,int64=torch.int64,
        empty=empty,cuda=SimpleNamespace(device=lambda _:contextlib.nullcontext()))
    module.tl=SimpleNamespace(float32=torch.float32)
    run=module._bmm_chunk_fwd_repaired
    cases=[((1,5,3),3,False,False),((2,5,2,5),3,True,True),((1,7,1),4,False,True),
           ((1,3,0),2,False,True),((0,4,3),3,False,True),((2,0,3),2,True,False),((1,4,0,3),3,False,True),
           ((1,129,1),129,False,True),((1,1,1),1,True,True)]
    for ci,(shape,chunk,has_seq,causal) in enumerate(cases):
        count=1
        for x in shape:count*=x
        a=Tensor((torch.arange(count+5,dtype=torch.float32)[2:2+count]%5-2).reshape(shape))
        b=Tensor((torch.arange(count+7,dtype=torch.float32)[3:3+count]%7-3).reshape(shape))
        if ci == 3:
            a=Tensor(a.data.as_strided(shape,(57,19,1)))
            b=Tensor(b.data.as_strided(shape,(69,23,1)))
        seq=Tensor((torch.arange(shape[0]*shape[1],dtype=torch.int64)%3-2).reshape(shape[:2])) if has_seq else None
        before=[backing_bytes(t.data).clone() for t in (a,b,seq) if t is not None]
        choice[0]=ci%9;trials[0]=causal
        old=len(calls);out=run(a,b,chunk,seq,causal,torch.float32 if ci%2 else None)
        assert out is allocations[-1]
        expected=torch.zeros_like(out.data)
        groups=shape[2] if len(shape)==4 else 1
        for batch in range(shape[0]):
            for ch in range((shape[1]+chunk-1)//chunk):
                for head in range(groups):
                    for row in range(chunk):
                        for col in range(chunk):
                            r=ch*chunk+row;c=ch*chunk+col
                            if r>=shape[1] or c>=shape[1] or causal and row<col:continue
                            if seq is not None and seq.data[batch,r]!=seq.data[batch,c]:continue
                            ix=(batch,ch,head,row,col) if len(shape)==4 else (batch,ch,row,col)
                            ar=a.data[batch,r,head] if len(shape)==4 else a.data[batch,r]
                            bc=b.data[batch,c,head] if len(shape)==4 else b.data[batch,c]
                            expected[ix]=(ar*bc).sum()
        assert torch.equal(out.data,expected),(ci,out.data,expected)
        assert all(torch.equal(snapshot,backing_bytes(t.data)) for snapshot,t in zip(before,[t for t in (a,b,seq) if t is not None]))
        backing=backing_bytes(out.data).view(torch.float32)
        assert torch.equal(backing[:4],torch.full((4,),-777.)) and torch.equal(backing[-5:],torch.full((5,),-777.))
        assert (len(calls)>old)==bool(out.numel())
        for entry in calls[old:]:
            if entry[0]=='bmm':assert entry[3] is a and entry[4] is b and entry[5] is out and entry[6] is seq
    rejected=0
    def fresh():return [Tensor(torch.ones(2,5,3)),Tensor(torch.ones(2,5,3)),3,Tensor(torch.ones(2,5,dtype=torch.int64)),True,None]
    def reject(args):
        nonlocal rejected
        nc,na=len(calls),len(allocations)
        before=[backing_bytes(x.data).clone() for x in args if isinstance(x,Tensor)]
        try:run(*args)
        except (TypeError,ValueError):pass
        else:raise AssertionError('invalid input admitted')
        assert (nc,na)==(len(calls),len(allocations))
        assert all(torch.equal(s,backing_bytes(x.data)) for s,x in zip(before,[x for x in args if isinstance(x,Tensor)]))
        rejected+=1
    for i in (0,1,3):
        for mutation in ('object','layout','dtype','cpu','device','contiguous','strides','negative','extent','alignment'):
            args=fresh();x=args[i]
            if mutation=='object':args[i]=object()
            elif mutation=='layout':
                class Sparse(Tensor):
                    @property
                    def layout(self):return torch.sparse_coo
                args[i]=Sparse(x.data)
            elif mutation=='dtype':args[i]=Tensor(x.data.to(torch.float64))
            elif mutation=='cpu':x.is_cuda=False
            elif mutation=='device':x.device=torch.device('cuda:1')
            elif mutation=='contiguous':x.is_contiguous=lambda:False
            elif mutation=='strides':x.stride=lambda dim:17
            elif mutation=='negative':x.is_neg=lambda:True
            elif mutation=='extent':x.numel=lambda:1<<63
            else:x.data_ptr=lambda:3
            reject(args)
    for chunk in (0,-1,True,1.5):args=fresh();args[2]=chunk;reject(args)
    for value in (1,None,'yes'):args=fresh();args[4]=value;reject(args)
    args=fresh();args[5]=torch.float16;reject(args)
    args=fresh();args[0]=Tensor(torch.ones(2,5));reject(args)
    args=fresh();args[1]=Tensor(torch.ones(1,5,3));reject(args)
    args=fresh();args[3]=Tensor(torch.ones(2,5,1,dtype=torch.int64));reject(args)
    for i,j in ((0,1),(0,3),(1,3)):
        args=fresh();ptr=args[i].data_ptr();args[j].data_ptr=lambda ptr=ptr:ptr;reject(args)
    # Actual PyTorch contiguous singleton-axis variants are deliberately outside
    # the literal-stride profile; is_contiguous alone does not pin these scalars.
    args=fresh();args[0]=Tensor(torch.ones(1,5,1).as_strided((1,5,1),(5,1,17)));args[1]=Tensor(torch.ones(1,5,1));args[3]=None;reject(args)
    args=fresh();args[0]=Tensor(torch.ones(1,5,2,1).as_strided((1,5,2,1),(10,2,1,17)));args[1]=Tensor(torch.ones(1,5,2,1));args[3]=None;reject(args)
    args=fresh();args[0]=Tensor(torch.ones(1,5,3));args[1]=Tensor(torch.ones(1,5,3));args[3]=Tensor(torch.ones(1,5,dtype=torch.int64).as_strided((1,5),(19,1)));reject(args)
    # Metadata-only large-shape controls never allocate or launch.
    class Shape(Tensor):
        def __init__(self,data,shape):super().__init__(data);self.shape_value=shape
        @property
        def shape(self):return self.shape_value
        @property
        def ndim(self):return len(self.shape_value)
        def stride(self,dim):
            dim %= len(self.shape_value)
            result=1
            for size in self.shape_value[dim+1:]:result*=size
            return result
    for shape,chunk in [((65536,1,1),1),((1,65536,1),1),((1,1,1),1<<23),((1,1,1),1<<32)]:
        args=fresh();args[0]=Shape(args[0].data,shape);args[1]=Shape(args[1].data,shape);args[2]=chunk;args[3]=None;reject(args)
    # Original causal behavior: an upper element in a surviving tile is nonzero;
    # a skipped cell retains the chosen invocation's initial value.
    original,pid=interpreter(ORIGINAL,'_bmm_chunk_fwd_kernel')
    data=torch.ones(1,129,1);output=torch.full((1,1,129,129),-777.)
    args=(129,129,1,1,129,1,0,1,129,1,0,1,129*129,129*129,0,129,1,0,0,True,torch.float32,False,64,32,32)
    for x in range(3*5):pid[:]=[x,0,0];original(Pointer(data),Pointer(data.clone()),Pointer(output),None,*args)
    assert output[0,0,0,1]==1 and output[0,0,0,100]==-777
    print(f'{len(cases)} admissions, {rejected} rejections, complete causal results, identity/byte frames and two original causal witnesses passed.')
    print('Accepted arithmetic AST retained; CPU operations/launches mocked; no GPU numerical conformance claim.')

if __name__=='__main__':main()
