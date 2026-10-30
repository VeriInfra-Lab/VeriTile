"""CPU execution of the actual candidate AST. No GPU or large allocation.
The adapter preserves signed lane widths and sparse element addresses; dot uses
CPU FP32 matmul and is not a TF32/tensor-core numerical conformance test.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, types
from pathlib import Path
import torch

torch.set_num_threads(1)
P=Path(__file__).resolve().parents[1]
R=P.parents[3]
SOURCE=P/'iv_dependent_matmul_i64.py'
ORIGINAL=R/'bench/tritonbench_g/iv_dependent_matmul/iv_dependent_matmul.py'
PIN='dfd772c19ff1b007b3e6c474fa984a460d0e135593cc3d0f05d279c8746997d7'
assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()==PIN
MODES=['pre_load','post_load','post_pre_mixed','post_load_two_iters','post_load_three_iters']

def extract(path,name):
    t=ast.parse(path.read_text())
    f=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name)
    f.decorator_list=[]
    return f

class EraseWidening(ast.NodeTransformer):
    def visit_Assign(self,n):
        if len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            name=n.targets[0].id
            if name=='k64' or (isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute)
              and n.value.func.attr=='cast' and isinstance(n.value.args[0],ast.Name)
              and n.value.args[0].id==name): return None
        return self.generic_visit(n)
    def visit_Call(self,n):
        n=self.generic_visit(n)
        if isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='tl' and n.func.attr=='cast':
            assert ast.unparse(n.args[1])=='tl.int64'
            return n.args[0]
        return n
    def visit_Name(self,n):
        if n.id=='k64': n.id='k'
        if n.id.endswith('_i64'): n.id=n.id[:-4]
        return n
    def visit_FunctionDef(self,n):
        if n.name.endswith('_i64'): n.name=n.name[:-4]
        return self.generic_visit(n)

for name in ['iv_dependent_matmul_kernel']:
    original=extract(ORIGINAL,name)
    erased=EraseWidening().visit(extract(SOURCE,name+'_i64'))
    assert ast.dump(original)==ast.dump(erased),name

class Pointer:
    def __init__(self,region,offset=0,tensor=None,extent=None,stats=None):
        self.region=region; self.offset=torch.as_tensor(offset,dtype=torch.int64)
        self.tensor=tensor; self.extent=tensor.numel() if tensor is not None else extent
        self.stats=stats if stats is not None else {'loads':0,'writes':[],'largest_load':-1}
    def __add__(self,offset):
        return Pointer(self.region,self.offset+torch.as_tensor(offset,dtype=torch.int64),self.tensor,self.extent,self.stats)
    __radd__=__add__

class Language:
    constexpr=object
    float32,float16,int64=torch.float32,torch.float16,torch.int64
    pid=0
    @staticmethod
    def cast(x,dtype): return torch.as_tensor(x).to(dtype)
    def program_id(self,axis):
        assert axis==0
        return torch.tensor(self.pid,dtype=torch.int32)
    @staticmethod
    def arange(start,stop): return torch.arange(start,stop,dtype=torch.int32)
    @staticmethod
    def cdiv(n,d): return (n+d-1)//d
    @staticmethod
    def zeros(shape,dtype): return torch.zeros(shape,dtype=dtype)
    @staticmethod
    def load(p,mask,other):
        offsets,mask=torch.broadcast_tensors(p.offset,mask)
        indices=offsets[mask]
        assert torch.all((indices>=0)&(indices<p.extent)),(p.region,indices)
        p.stats['loads']+=indices.numel()
        if indices.numel(): p.stats['largest_load']=max(p.stats['largest_load'],indices.max().item())
        v=torch.full(offsets.shape,other,dtype=torch.float32)
        v[mask]=p.tensor.reshape(-1)[indices] if p.tensor is not None else 1/16
        return v
    @staticmethod
    def dot(a,b): return a@b
    @staticmethod
    def store(p,v,mask):
        offsets,mask=torch.broadcast_tensors(p.offset,mask)
        indices=offsets[mask]
        assert torch.all((indices>=0)&(indices<p.extent)),(p.region,indices)
        p.stats['writes'].extend(indices.tolist())
        p.stats['values']=v[mask].to(torch.float32).tolist()
        if p.tensor is not None: p.tensor.reshape(-1)[indices]=v[mask].to(p.tensor.dtype)

tl=Language();ns={'tl':tl}
exec(compile(ast.Module(body=[extract(SOURCE,'iv_dependent_matmul_kernel_i64')],type_ignores=[]),str(SOURCE),'exec'),ns)
kernel=ns['iv_dependent_matmul_kernel_i64']

class Factory:
    device=torch.device
    float32=torch.float32
    get_default_dtype=staticmethod(torch.get_default_dtype)
    def rand(self,shape,device):
        assert device.type=='cuda'
        v=((torch.arange(shape[0]*shape[1],dtype=torch.float32)*7+len(self.inputs)*3)%16/16).reshape(shape)
        self.inputs.append(v);return v
    def empty(self,shape,device):
        self.backing=torch.full((shape[0]*shape[1]+2,),7.25,dtype=torch.float32)
        self.output=self.backing[1:-1].reshape(shape);return self.output

factory=Factory()
class Launcher:
    def __getitem__(self,grid):
        def launch(a,b,c,*args,**meta):
            self.grid=grid(meta);self.meta=meta;self.ptrs=[Pointer(n,tensor=t) for n,t in zip('ABC',(a,b,c))]
            # Native specialization passes the non-constexpr small scalar arguments as i32.
            args=[torch.tensor(v,dtype=torch.int32) for v in args]
            for pid in range(self.grid[0]):
                tl.pid=pid
                kernel(*self.ptrs,*args,**{k:v for k,v in meta.items() if k!='num_stages'})
        return launch
launcher=Launcher();ns.update(torch=factory,triton=types.SimpleNamespace(cdiv=tl.cdiv),iv_dependent_matmul_kernel_i64=launcher)
exec(compile(ast.Module(body=[extract(SOURCE,'iv_dependent_matmul_wrapper_i64')],type_ignores=[]),str(SOURCE),'exec'),ns)
records=[]
for M,N in [(35,37),(1,3),(0,37),(35,0)]:
 for K in [0,1,31,32,33,65]:
  for mode in MODES:
    factory.inputs=[]
    result=ns['iv_dependent_matmul_wrapper_i64'](M,K,N,32,32,32,type=mode,device='cuda')
    A,B=factory.inputs
    expected=(A.double()@B.double()).half().float()
    assert torch.equal(result,expected)
    assert result is factory.output and result.dtype==torch.float32
    assert factory.backing[0]==7.25 and factory.backing[-1]==7.25
    if M*N:
        assert sorted(launcher.ptrs[2].stats['writes'])==list(range(M*N))
        assert launcher.meta['num_stages']==(4 if mode=='post_load_three_iters' else 3)
    else:
        launcher.grid=(0,)
    records.append({'shape':[M,K,N],'selector':mode,'grid':list(launcher.grid),'each_output_once':True,'frame_guards_preserved':True,'source_casts_preserved':True})

sparse=[]
# Both sides of the old 2^31 boundary, plus input-A and staged-B products.
for M,K,N,pid in [(65536,32,32768,2096128),(65537,32,32768,2097152),
                   (65537,32768,1,2048),(1,65537,32768,0)]:
 for mode in MODES:
    ptrs=[Pointer(n,extent=e) for n,e in zip('ABC',(M*K,K*N,M*N))]
    tl.pid=pid
    args=[torch.tensor(v,dtype=torch.int32) for v in (M,N,K,max(K,1),1,max(N,1),1,max(N,1),1)]
    kernel(*ptrs,*args,32,32,32,mode)
    pn=(N+31)//32; row=pid//pn*32;col=pid%pn*32
    expected=[i*N+j for i in range(row,min(row+32,M)) for j in range(col,min(col+32,N))]
    assert ptrs[2].stats['writes']==expected
    expected_value=torch.tensor(K/256,dtype=torch.float32).half().float().item()
    assert all(v==expected_value for v in ptrs[2].stats['values'])
    sparse.append({'shape':[M,K,N],'pid':pid,'selector':mode,'first_store':expected[0],
       'largest_A_load':ptrs[0].stats['largest_load'],'largest_B_load':ptrs[1].stats['largest_load'],
       'all_active_accesses_bounded':True,'correct_sparse_value_after_half':True})

metadata=[]
for shape in [(65537,32),(32,32768),(65537,32768)]:
    t=torch.empty(shape,device='meta')
    metadata.append({'shape':list(shape),'stride':list(t.stride()),'dtype':str(t.dtype),'bytes':t.numel()*t.element_size()})
result={'candidate_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'original_sha256':PIN,
 'kernel_AST_equal_after_erasing_only_i64_widening_and_renaming':True,
 'native_gpu_execution':False,'large_allocation':False,'method':'actual candidate AST; CPU signed-width tensor lanes; sparse pointers for large cases; CPU matmul dot',
 'metadata':metadata,'small_host_cases':records,'sparse_boundary_cases':sparse}
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'small_cases':len(records),'sparse_cases':len(sparse),'ast_binding':True,'native_gpu_execution':False}))
