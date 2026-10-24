"""Replay the unchanged original kernel's positive-K path using sparse pointers.
Integer lanes use signed int32, as independently confirmed by compiler IR.
No multi-gigabyte allocation or GPU execution is performed.
"""
import ast, hashlib, json, types
from pathlib import Path
import torch

source=Path(__file__).resolve().parents[1]/'iv_dependent_matmul.py'
tree=ast.parse(source.read_text())
fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='iv_dependent_matmul_kernel')
fn.decorator_list=[]
class Pointer:
    def __init__(self, region, offset=0):
        self.region=region;self.offset=torch.as_tensor(offset,dtype=torch.int64)
    def __add__(self, x): return Pointer(self.region,self.offset+torch.as_tensor(x,dtype=torch.int64))
    __radd__=__add__
class Language:
    constexpr=object
    float32,float16=torch.float32,torch.float16
    @staticmethod
    def program_id(axis):
        assert axis==0
        return torch.tensor(2097152,dtype=torch.int32)
    @staticmethod
    def arange(start,stop): return torch.arange(start,stop,dtype=torch.int32)
    @staticmethod
    def cdiv(n,d): return (n+d-1)//d
    @staticmethod
    def zeros(shape,dtype): return torch.zeros(shape,dtype=dtype)
    @staticmethod
    def load(p,mask,other):
        offsets,mask=torch.broadcast_tensors(p.offset,mask)
        extent={'A':65537*32,'B':32*32768}[p.region]
        assert torch.all((offsets[mask]>=0)&(offsets[mask]<extent))
        # Address failure is independent of values. These finite samples lie
        # in the self-generated torch.rand [0,1) input domain.
        return torch.where(mask,(offsets%16).float()/16,torch.tensor(other,dtype=torch.float32))
    @staticmethod
    def dot(a,b): return a@b
    def store(self,p,v,mask):
        assert p.region=='C'
        offsets,mask=torch.broadcast_tensors(p.offset,mask)
        self.offsets=offsets[mask].tolist()
        assert torch.all(torch.isfinite(v[mask]))
tl=Language();ns={'tl':tl}
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
records=[]
for mode in ['pre_load','post_load','post_pre_mixed','post_load_two_iters','post_load_three_iters']:
    ns['iv_dependent_matmul_kernel'](Pointer('A'),Pointer('B'),Pointer('C'),65537,32768,32,
      32,1,32768,1,32768,1,32,32,32,mode)
    assert tl.offsets==list(range(-2147483648,-2147483648+32))
    records.append({'selector':mode,'active_store_count':len(tl.offsets),'first_signed_element_offset':tl.offsets[0],
                    'last_signed_element_offset':tl.offsets[-1],'first_byte_offset':4*tl.offsets[0],
                    'all_active_addresses_before_C_allocation':True})
metadata=[]
for shape in [(65537,32),(32,32768),(65537,32768)]:
    t=torch.empty(shape,device='meta')
    metadata.append({'shape':list(shape),'strides':list(t.stride()),'dtype':str(t.dtype),'bytes':t.numel()*t.element_size()})
print(json.dumps({'status':'source_defect_pending_repair','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'native_gpu_executed':False,'large_allocations_performed':False,'method':'unchanged kernel AST, int32 scalar lanes, signed pointer extension, sparse storage',
 'M':65537,'N':32768,'K':32,'blocks':[32,32,32],'grid':2098176,'pid':2097152,
 'metadata':metadata,'records':records},indent=2))
