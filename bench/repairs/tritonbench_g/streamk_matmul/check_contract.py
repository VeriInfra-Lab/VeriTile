"""Actual checked wrapper and three Triton kernels, CPU interpreter only."""
import ast
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
os.environ['TRITON_INTERPRET']='1'
os.environ.setdefault('TRITON_CACHE_DIR','/tmp/veritile-streamk-interpreter-cache')
sys.dont_write_bytecode=True
import torch
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'Whole001/streamk_matmul_repaired.py'
ORIGINAL=HERE.parents[2]/'tritonbench_g/streamk_matmul/streamk_matmul.py'
assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()=='794c7aa68254a0cc1caf974b5ed806ce8c884e9f7c9c2c36d68c67d5b654a4f3'
# Kernels are the accepted Whole001 source, pinned independently of guards.
KERNEL_AST_SHA256 = {'swizzle_tile': 'c05e4453573be1fb4f2c6e77b32924cb2a1cb82befb16cddc2504ffe467a6b2b', 'linear_tile': 'b02e9deddf55a9dcecf2727bf2a348020275ddf62f085d717e7aa64c7580b8e2', 'mac_loop': '77c23ce44226388a131e3fa3376c639b594450edd04efce264ba76d99a491816', 'first_wave': 'c3671c9d6ec587f5b9a912f844f5701cd835d6081a33c1db1660d41a8689e4c8', 'full_tiles': '10004a8fdd823f5afd9344f2e0c91027fa8d0408f0bcbd20c02451bcafa12b2f', 'reduce_streamk': 'a6abb8f0cb2f365f9b13ab273328f797e6243ec3fbc3cab0c3c773989b2e65a6'}
for node in ast.parse(SOURCE.read_text()).body:
    if isinstance(node, ast.FunctionDef) and node.name in KERNEL_AST_SHA256:
        assert hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()==KERNEL_AST_SHA256[node.name]
spec=importlib.util.spec_from_file_location('streamk_checked',SOURCE)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class Facade:
    def __init__(self,data,**options):
        self.data=data;self.device=torch.device('cuda:0');self.is_cuda=True
        self.__dict__.update(options)
    @property
    def shape(self):return getattr(self,'shape_override',self.data.shape)
    @property
    def ndim(self):return len(self.shape)
    @property
    def dtype(self):return self.data.dtype
    @property
    def layout(self):return self.data.layout
    def is_contiguous(self):return self.data.is_contiguous()
    def is_neg(self):return getattr(self,'neg_override',False)
    def numel(self):return math.prod(self.shape)
    def data_ptr(self):
        assert self.numel()>0
        return getattr(self,'pointer_override',self.data.data_ptr())

launches=[];allocations=[];rejecting=False
class Launcher:
    def __init__(self,kernel,name):self.kernel=kernel;self.name=name
    def __getitem__(self,grid):
        def launch(*args,**kwargs):
            assert not rejecting, "rejection control unexpectedly reached launch"
            assert all(x>0 for x in grid)
            launches.append((self.name,grid))
            converted=tuple(x.data if isinstance(x,Facade) else x for x in args)
            return self.kernel[grid](*converted,**kwargs)
        return launch
for name in ('first_wave','reduce_streamk','full_tiles'):
    setattr(module,name,Launcher(getattr(module,name),name))
def allocate(shape,device,dtype):
    backing=torch.full((math.prod(shape)+8,),-1234,dtype=dtype)
    tensor=Facade(backing[4:-4].view(shape),device=device)
    allocations.append((tensor,backing))
    return tensor
module.torch=SimpleNamespace(Tensor=Facade,strided=torch.strided,float32=torch.float32,
    float16=torch.float16,bfloat16=torch.bfloat16,empty=allocate)
base=dict(total_programs_streamk=3,BLK_M=32,BLK_N=32,BLK_K=32,two_tiles=True,num_stages=3,num_warps=4)
a=Facade(torch.ones(2,3));b=Facade(torch.ones(3,4))
rejected=0

def reject(a,b,**changes):
    global rejected,rejecting
    rejecting=True
    before=(len(allocations),len(launches))
    try: module.matmul._call(a,b,**(base|changes))
    except (TypeError,ValueError):pass
    else:raise AssertionError(('admitted',changes))
    rejecting=False
    assert (len(allocations),len(launches))==before
    rejected+=1
for ix in (0,1):
    target=(a,b)[ix]
    for bad in (None,Facade(target.data.half()),Facade(target.data.to_sparse()),
        Facade(target.data.reshape(-1)),Facade(target.data.T),Facade(target.data,is_cuda=False),
        Facade(target.data,device=torch.device('cuda:1')),Facade(target.data,neg_override=True),
        Facade(target.data,pointer_override=1),Facade(target.data,pointer_override=2**64-4)):
        args=[a,b];args[ix]=bad;reject(*args)
for changes in (dict(total_programs_streamk=True),dict(BLK_M=False),dict(BLK_N=0),dict(BLK_K=4),
                dict(BLK_M=3),dict(BLK_M=2**21),dict(num_stages=0),dict(num_warps=3),dict(two_tiles=1),
                dict(total_programs_streamk=2**31)):
    reject(a,b,**changes)
reject(a,Facade(torch.ones(5,4)))
reject(Facade(a.data,shape_override=(2**31,3)),b)
reject(Facade(a.data,shape_override=(1,2**31-1)),Facade(b.data,shape_override=(2**31-1,1)))
reject(Facade(a.data,shape_override=(2**31-64,2**31-64)),Facade(b.data,shape_override=(2**31-64,1)))
reject(Facade(a.data,shape_override=(32*2**16,0)),Facade(b.data,shape_override=(0,32*2**16)))
# Scratch byte envelope independently exceeds i64 while all i32 controls fit.
reject(a,b,total_programs_streamk=2**30,BLK_M=2**20,BLK_N=2**20)

trials=0
profiles=[(16,128,128,32),(32,64,64,16),(8,32,32,32),(4,256,256,32)]
for P,BM,BN,BK in profiles:
    for two in (False,True):
        for M,K,N in ((3,7,5),(BM+3,BK*3+1,BN+5),(0,7,5),(3,7,0),(3,0,5)):
            left=torch.randint(-1,2,(M,K)).float()/4
            right=torch.randint(-1,2,(K,N)).float()/4
            aa,bb=Facade(left),Facade(right)
            beforeA,beforeB=left.clone(),right.clone()
            start_alloc,start_launch=len(allocations),len(launches)
            out=module.matmul._call(aa,bb,P,BM,BN,BK,two,3,4)
            assert out is allocations[start_alloc][0]
            assert torch.equal(out.data,left@right),(P,BM,BN,BK,two,M,K,N)
            assert torch.equal(left,beforeA) and torch.equal(right,beforeB)
            for _,storage in allocations[start_alloc:]:
                assert torch.all(storage[:4]==-1234) and torch.all(storage[-4:]==-1234)
            seen=[x[0] for x in launches[start_launch:]]
            assert seen in ([],['full_tiles'],['first_wave','reduce_streamk'],['first_wave','reduce_streamk','full_tiles'])
            if M==0 or N==0:assert seen==[] and len(allocations)-start_alloc==1
            trials+=1
# Split chunks, P<=0, more programs than work, two-tile route, aliases and singleton strides.
for P in (-5,0,3,200):
    for two in (False,True):
        M,K,N=101,97,67
        left=torch.randint(-1,2,(M,K)).float()/4;right=torch.randint(-1,2,(K,N)).float()/4
        out=module.matmul._call(Facade(left),Facade(right),P,32,32,32,two,3,4)
        assert torch.equal(out.data,left@right);trials+=1
shared=torch.eye(33)
out=module.matmul._call(Facade(shared),Facade(shared),3,32,32,32,True,3,4)
assert torch.equal(out.data,shared);trials+=1
left=torch.as_strided(torch.ones(3),(1,3),(2**40,1))
out=module.matmul._call(Facade(left),b,3,32,32,32,True,3,4)
assert torch.equal(out.data,torch.full((1,4),3.));trials+=1
print(dict(rejections=rejected,interpreted_calls=trials,kernel_launches=len(launches),native_gpu_execution=False))
