"""Source/CPU storage and deterministic Philox controls; no native GPU claims."""
import ast
import contextlib
import ctypes
import importlib.util
import struct
from pathlib import Path
from types import SimpleNamespace
import torch

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'uniform_sampling_repaired.py'
ORIGINAL=HERE.parents[2]/'tritonbench_g/uniform_sampling/uniform_sampling.py'
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
shared=load(HERE.parent/'add_example/check_contract.py','uniform_storage_helpers')
function,raw_storage=shared.function,shared.raw_storage

class Normalize(ast.NodeTransformer):
    def visit_Call(self,node):
        self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and node.func.attr=='to' and len(node.args)==1 and ast.unparse(node.args[0]) in ('tl.int64','tl.uint32'):
            return node.func.value
        if ast.unparse(node.func)=='tl.minimum':return node.args[0]
        if ast.unparse(node.func)=='tl.fma':return ast.BinOp(left=ast.BinOp(left=node.args[0],op=ast.Mult(),right=node.args[1]),op=ast.Add(),right=node.args[2])
        return node
    def visit_Name(self,node):
        if node.id=='pid':return ast.parse('tl.program_id(0)',mode='eval').body
        if node.id=='lane':return ast.parse('tl.arange(0, BLOCK)',mode='eval').body
        return node

class InterpreterCasts(ast.NodeTransformer):
    def visit_Call(self,node):
        self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and node.func.attr=='to' and len(node.args)==1 and ast.unparse(node.args[0]) in ('tl.int64','tl.uint32'):
            return node.func.value
        return node

def source_checks():
    source=function(SOURCE,'uniform_sampling_repaired_kernel')
    candidate=function(SOURCE,'uniform_sampling_index_commuted_kernel')
    candidate.name=source.name
    index=next(n for n in candidate.body if isinstance(n,ast.Assign) and
               isinstance(n.targets[0],ast.Name) and n.targets[0].id=='i4')
    assert ast.unparse(index.value)=='BLOCK * pid + lane'
    index.value=ast.parse('pid * BLOCK + lane',mode='eval').body
    assert ast.dump(candidate)==ast.dump(source)
    new=function(SOURCE,'uniform_sampling_repaired_kernel');old=function(ORIGINAL,'uniform_kernel')
    new.body=[node for node in new.body if not (isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('pid','lane'))]
    new=Normalize().visit(new);old=Normalize().visit(old)
    new.name=old.name;new.decorator_list=[];old.decorator_list=[]
    assert ast.dump(new)==ast.dump(old)
    def helper(path):
        node=next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='uint_to_uniform_float')
        node.body=[n for n in node.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))]
        return ast.dump(node)
    assert helper(SOURCE)==helper(ORIGINAL)
    text=SOURCE.read_text()
    assert '_validate_unique_view' not in text and 'resolve_neg' not in text
    assert 'counter_count = (n + 4 * block - 1) // (4 * block) * block' in text
    assert 'seed, start = reservation.reserve(selected, self.device.index, block, counter_count)' in text
    assert '.Proofs.' not in (HERE/'Kernels.lean').read_text()
    assert 'FPEquiv' not in (HERE/'Correct.lean').read_text()
    assert 'Correct' not in (HERE/'FPEquiv.lean').read_text()
    contract=(HERE/'Contract.lean').read_text()
    for term in ('output.isContiguous','output.isNeg','output.dataPtr.toNat%4','output.numel>=2^63','generatorCheck','lowerIsFloat','upperIsFloat'):
        assert term in contract
    cpp=(HERE/'reserve_philox.cpp').read_text()
    for term in ('generator.mutex()','generator.current_seed()','generator.get_offset()','maximum - offset','maximum - start','generator.set_offset(start + counter_count)'):
        assert term in cpp

class Tensor(shared.TensorFacade):
    def __getitem__(self,index):return Tensor(self.data[index],str(self.device))
    def reshape(self,shape):return Tensor(self.data.reshape(shape),str(self.device))
    def copy_(self,other):self.data.copy_(other.data);return self
class Generator:
    def __init__(self,seed=1,offset=0,device='cuda:0'):
        self.seed,self.offset,self.device=seed,offset,torch.device(device)
class Scalar(int):
    def to(self,dtype):return int(self)
class Pointer:
    def __init__(self,tensor,offset=0):self.tensor,self.offset=tensor,offset
    def __add__(self,offset):return Pointer(self.tensor,self.offset+offset)

def philox(seed,c0,c1):
    mask=2**32-1;c=[c0&mask,c1&mask,0,0];key=[seed&mask,(seed>>32)&mask]
    for _ in range(10):
        p0=0xD2511F53*c[0];p1=0xCD9E8D57*c[2]
        c=[((p1>>32)^c[1]^key[0])&mask,p1&mask,((p0>>32)^c[3]^key[1])&mask,p0&mask]
        key=[(key[0]+0x9E3779B9)&mask,(key[1]+0xBB67AE85)&mask]
    return c
SCALE=struct.unpack('<f',struct.pack('<I',0x2fffffff))[0]
def uniform(words):
    words=torch.as_tensor(words,dtype=torch.int64)
    folded=torch.where(words>=2**31,(2**32-1)-words,words)
    return folded.to(torch.float32)*SCALE


def paired_kernel_checks():
    """Both complete displayed kernels, with a shared CPU fused primitive.

    The check compares every memory byte and guard window. It is independent
    of wrapper allocation/reservation and does not assert CUDA conformance.
    """
    fmaf=ctypes.CDLL(None).fmaf
    fmaf.argtypes=(ctypes.c_float,ctypes.c_float,ctypes.c_float)
    fmaf.restype=ctypes.c_float
    pid=[0]
    def fused(x,y,z):
        return torch.tensor([fmaf(float(a),float(y),float(z)) for a in x],dtype=torch.float32)
    def store(pointer,value,mask,**kwargs):
        assert bool((pointer.offset>=0).all()) and bool((pointer.offset<pointer.tensor.numel()).all())
        raw_storage(pointer.tensor)[pointer.tensor.storage_offset()+pointer.offset[mask]]=value[mask]
    def tuples(seed,low,high,zero0,zero1):
        assert bool((zero0==0).all()) and bool((zero1==0).all())
        values=torch.tensor([philox(seed,int(c),high) for c in low],dtype=torch.int64)
        return tuple(values[:,i] for i in range(4))
    namespace={'uint_to_uniform_float':uniform,'tl':SimpleNamespace(
        int64=torch.int64,uint32=torch.int64,program_id=lambda _:Scalar(pid[0]),
        arange=torch.arange,philox=tuples,minimum=lambda x,y:torch.minimum(x,torch.tensor(y)),
        store=store,fma=fused)}
    names=('uniform_sampling_repaired_kernel','uniform_sampling_index_commuted_kernel')
    for name in names:
        node=function(SOURCE,name);node.decorator_list=[]
        node=InterpreterCasts().visit(node)
        for arg in node.args.args:arg.annotation=None
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE),'exec'),namespace)
    bounds=[(0.,1.),(-0.,0.),(-2.,3.),(float('inf'),float('inf')),
            (float('-inf'),float('inf')),(float('nan'),1.),
            (torch.finfo(torch.float32).tiny,torch.finfo(torch.float32).max)]
    seeds=(-2**63,-1,0,2**63-1)
    cases=executions=0
    for n in (1,511,512,513,1024,1025,4095,4096,4097):
        block=512 if n<=512 else 1024
        for seed in seeds:
            for lower,upper in bounds:
                lower=torch.tensor(lower,dtype=torch.float32)
                upper=torch.tensor(upper,dtype=torch.float32)
                initial=torch.arange(n+8,dtype=torch.int32).bitwise_xor_(0x7fc00000).view(torch.float32)
                memories=[initial.clone(),initial.clone()]
                # A launch starts at a B-aligned low counter. This one is
                # adjacent to a low-word carry whenever only one block runs.
                offset=-block if n<=4*block else -2*block
                for p in reversed(range((n+4*block-1)//(4*block))):
                    pid[0]=p
                    for name,memory in zip(names,memories):
                        namespace[name](Pointer(memory[4:4+n]),n,Scalar(seed),Scalar(offset),lower,upper,block)
                        executions+=1
                assert torch.equal(memories[0].view(torch.uint8),memories[1].view(torch.uint8))
                for memory in memories:
                    assert torch.equal(memory[:4].view(torch.uint8),initial[:4].view(torch.uint8))
                    assert torch.equal(memory[-4:].view(torch.uint8),initial[-4:].view(torch.uint8))
                cases+=1
    return cases,executions

def checks():
    module=load(SOURCE,'uniform_contract_review');pid=[0];programs=[0];events=[];scratch=[];all_counters=[]
    defaults=[Generator(seed=2**64-1),Generator(seed=7,device='cuda:1')]
    def reserve(generator,device,block,count):
        events.append(('reserve',generator,block,count))
        assert generator.device.index==device and block in (512,1024) and count>0 and count%block==0
        start=generator.offset+(-generator.offset)%block
        if start+count>2**64-1:raise OverflowError('Philox offset exhausted')
        generator.offset=start+count
        return generator.seed,start
    def empty(shape,device,dtype):
        n=shape[0];backing=torch.full((n+8,),-98765.,dtype=dtype);scratch.append(backing)
        events.append('allocate');return Tensor(backing[4:4+n],str(device))
    def store(pointer,value,mask,**kwargs):
        assert bool((pointer.offset>=0).all()) and bool((pointer.offset<pointer.tensor.numel()).all())
        raw_storage(pointer.tensor)[pointer.tensor.storage_offset()+pointer.offset[mask]]=value[mask]
    def tuples(seed,low,high,zero0,zero1):
        assert bool((zero0==0).all()) and bool((zero1==0).all())
        all_counters.extend((seed&((1<<64)-1),int(high),int(c)) for c in low)
        values=torch.tensor([philox(seed,int(c),high) for c in low],dtype=torch.int64)
        return tuple(values[:,i] for i in range(4))
    node=function(SOURCE,'uniform_sampling_repaired_kernel');node.decorator_list=[]
    node=InterpreterCasts().visit(node)
    for arg in node.args.args:arg.annotation=None
    namespace={'uint_to_uniform_float':uniform,'tl':SimpleNamespace(int64=torch.int64,uint32=torch.int64,
        program_id=lambda _:Scalar(pid[0]),arange=torch.arange,philox=tuples,
        minimum=lambda x,y:torch.minimum(x,torch.tensor(y)),store=store,
        fma=lambda x,y,z:(x.double()*torch.tensor(y,dtype=torch.float32).double()+torch.tensor(z,dtype=torch.float32).double()).float())}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE),'exec'),namespace)
    class Launch:
        def __getitem__(self,grid):
            def run(out,n,seed,offset,lower,upper,**meta):
                events.append(('launch',grid,meta['BLOCK'],meta['num_warps']))
                assert grid[0]<2**31 and meta['enable_fp_fusion'] is True
                for p in reversed(range(grid[0])):
                    pid[0]=p;programs[0]+=1
                    namespace['uniform_sampling_repaired_kernel'](Pointer(out.data),n,Scalar(seed),Scalar(offset),lower,upper,meta['BLOCK'])
            return run
    module.torch=SimpleNamespace(Tensor=Tensor,Generator=Generator,strided=torch.strided,float32=torch.float32,empty=empty,
        cuda=SimpleNamespace(default_generators=defaults,device=lambda _:contextlib.nullcontext()))
    module._reservation_module=lambda:SimpleNamespace(reserve=reserve)
    module.uniform_sampling_repaired_kernel=Launch()
    admitted=0
    for shape,initial,supplied in [((),3,False),((0,),0,False),((2,0,3),11,True),((1,),0,False),
        ((511,),0,False),((512,),17,True),((513,),2**32-1024,True),((1024,),0,False),
        ((1025,),2**32-1024,True),((4095,),1,False),((4096,),0,False),((4097,),2**32-1024,True),((2,4097),5,False)]:
        n=1
        for dim in shape:n*=dim
        data=torch.full((n+8,),-12345.);view=Tensor(data[4:4+n].reshape(shape));selected=Generator(123,initial) if supplied else defaults[0]
        selected.offset=initial;other=(defaults[1].seed,defaults[1].offset)
        events.clear();scratch.clear();all_counters.clear()
        returned=module.uniform_sampling_repaired(view,-2.,3.,generator=selected if supplied else None)
        assert returned is view and (defaults[1].seed,defaults[1].offset)==other
        assert bool((data[:4]==-12345).all()) and bool((data[-4:]==-12345).all())
        if n==0:
            assert events==[] and selected.offset==initial
        else:
            block=module.heur_block({'N':n});start=initial+(-initial)%block
            count=(n+4*block-1)//(4*block)*block
            assert selected.offset==start+count and events[0]=='allocate'
            assert len(all_counters)==count and len(set(all_counters))==count
            words=[philox(selected.seed,(start+(i//(4*block))*block+i%block)%(2**32),
                (start+(i//(4*block))*block+i%block)//(2**32))[(i//block)%4] for i in range(n)]
            expected=(uniform(words).double()*5.-2.).float()
            torch.testing.assert_close(view.data.flatten(),expected,rtol=0,atol=0)
            assert all(bool((b[:4]==-98765).all()) and bool((b[-4:]==-98765).all()) for b in scratch)
            assert all(event[3]==module.heur_num_warps({'N':n}) for event in events if isinstance(event,tuple) and event[0]=='launch')
        admitted+=1
    x=Tensor(torch.empty(4));rejected=0
    bad=[([],{}),(Tensor(torch.empty(4,dtype=torch.float64)),{}),(Tensor(torch.empty(4),'cpu',False),{}),
        (Tensor(torch.sparse_coo_tensor([[0]],[1.],(4,),check_invariants=True)),{}),
        (Tensor(torch.empty(2,3).t()),{}),(Tensor(torch._neg_view(torch.empty(4))),{}),
        (Tensor(shared.unaligned_tensor(1)),{}),(x,{'generator':object()}),(x,{'generator':defaults[1]}),
        (x,{'from_':0}),(x,{'to':1})]
    class Huge(Tensor):
        @property
        def shape(self):return (2**63,)
        def numel(self):return 2**63
        def data_ptr(self):return 4
    bad.append((Huge(torch.empty(0)),{}))
    for tensor,kwargs in bad:
        before=(defaults[0].seed,defaults[0].offset);events.clear()
        try:module.uniform_sampling_repaired(tensor,**kwargs)
        except (TypeError,ValueError):pass
        else:raise AssertionError('unsupported input admitted')
        assert events==[] and before==(defaults[0].seed,defaults[0].offset);rejected+=1
    exhausted=Generator(1,2**64-512);before=x.data.clone();events.clear()
    try:module.uniform_sampling_repaired(x,generator=exhausted)
    except OverflowError:pass
    else:raise AssertionError('offset exhaustion accepted')
    assert exhausted.offset==2**64-512;torch.testing.assert_close(x.data,before,equal_nan=True)
    # Seed/counter output convention is the published all-zero Philox4x32-10 vector.
    assert philox(0,0,0)==[0x6627e8d5,0xe169c58d,0xbc57ac4c,0x9b00dbd8]
    # The original reservation rounded ceil(N/4), but evaluated a whole block.
    assert ((1+3)//4+3)//4*4==4 and module.heur_block({'N':1})==512
    return rejected,admitted,programs[0]

if __name__=='__main__':
    source_checks();rejected,admitted,programs=checks()
    pairs,executions=paired_kernel_checks()
    print(f'Passed source binding, {rejected} guard rejections, {admitted} admissions, {programs} program executions.')
    print(f'Passed {pairs} paired complete kernel cases, {executions} program executions, exact memory bytes and exterior frames.')
    print('CPU storage/Philox replay with mocked CUDA and reservation API; no native numerical or C++ conformance claim.')
