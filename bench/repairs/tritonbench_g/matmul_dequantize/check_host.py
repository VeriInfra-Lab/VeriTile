from pathlib import Path
import ast,copy,hashlib,importlib.util,json,os,sys,tempfile
sys.dont_write_bytecode=True
D=Path(tempfile.mkdtemp(prefix='veritile-dequant-control-'));(D/'evidence').mkdir()
os.environ['TRITON_INTERPRET']='1';os.environ['TRITON_CACHE_DIR']=str(D/'evidence/interpreter-cache')
import torch,triton
p=Path(__file__).resolve().with_name('matmul_dequantize_repaired.py')
spec=importlib.util.spec_from_file_location('mdq_cpu',p);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
class CudaView(torch.Tensor):
 @property
 def is_cuda(self):return True
 @property
 def device(self):return torch.device('cuda',0)
for name in ('empty','ones','zeros','full'):
 original=getattr(torch,name)
 def factory(*args,_original=original,**kwargs):
  if str(kwargs.get('device','cpu')).startswith('cuda'):
   kwargs['device']='cpu';return _original(*args,**kwargs).as_subclass(CudaView)
  return _original(*args,**kwargs)
 setattr(torch,name,factory)
original=p.parents[4]/'bench/tritonbench_g/matmul_dequantize/matmul_dequantize.py'
source_tree=ast.parse(original.read_text());new_tree=ast.parse(p.read_text())
for name in ('matmul4_kernel','matmul_kernel','dequantize_kernel'):
 before=next(n for n in source_tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
 after=next(n for n in new_tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
 if name=='matmul_kernel':
  stripped=copy.deepcopy(after);loads=0
  for node in ast.walk(stripped):
   if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='load':
    ptr=node.args[0].id
    mask='offs_k[None, :] + k * BLOCK_SIZE_K * SPLIT_K < K' if ptr=='a_ptrs' else 'offs_k[:, None] + k * BLOCK_SIZE_K * SPLIT_K < K'
    assert len(node.keywords)==2 and node.keywords[0].arg=='mask' and node.keywords[1].arg=='other'
    assert ast.dump(node.keywords[0].value)==ast.dump(ast.parse(mask,mode='eval').body)
    assert isinstance(node.keywords[1].value,ast.Constant) and node.keywords[1].value.value==0.0
    node.keywords=[];loads+=1
  assert loads==4 and ast.dump(before)==ast.dump(stripped)
 else:assert ast.dump(before)==ast.dump(after),name
launches=[]
class Selected:
 def __init__(self,tuner,route):self.fn=tuner.fn;self.config=tuner.configs[0];self.route=route
 def __getitem__(self,grid):
  def run(*args,**kwargs):
   cfg=self.config;meta={**cfg.kwargs,**kwargs};shape=grid(meta) if callable(grid) else grid
   if self.route=='dequant':Q,S,Z,C,K,N,G=args[:7];BM=0;BK=meta['BLOCK_SIZE_K'];BN=meta['BLOCK_SIZE_N'];M=0
   else:A,Q,C,S,Z,M,N,K=args[:8];G=args[-2] if self.route=='gptq' else args[-1];BM=meta['BLOCK_SIZE_M'];BN=meta['BLOCK_SIZE_N'];BK=meta['BLOCK_SIZE_K']
   count=0
   def at(t,off):
    nonlocal count
    pos=t.storage_offset()+off;assert 0<=pos<t.untyped_storage().nbytes()//t.element_size(),(self.route,pos,t.shape);count+=1
   if self.route=='dequant':
    for pk in range(shape[0]):
     for pn in range(shape[1]):
      for k in range(pk*BK,min((pk+1)*BK,K)):
       for n in range(pn*BN,min((pn+1)*BN,N)):
        at(Q,k//8*Q.stride(0)+n*Q.stride(1));at(S,k//G*S.stride(0)+n*S.stride(1));at(Z,k//G*Z.stride(0)+n//8*Z.stride(1));at(C,k*C.stride(0)+n*C.stride(1))
   else:
    gm=meta['GROUP_SIZE_M'];nm=triton.cdiv(M,BM);nn=triton.cdiv(N,BN)
    for pid in range(shape[0]):
     first=pid//(gm*nn)*gm;size=min(nm-first,gm);pm=first+pid%size;pn=(pid%(gm*nn))//size
     for block in range(triton.cdiv(K,BK)):
      for e in range(BK):
       k=block*BK+e
       if self.route=='s2' and k>=K:continue
       for rr in range(BM):
        row=pm*BM+rr
        if self.route=='s2':row%=M
        if row<M:at(A,row*A.stride(0)+k*A.stride(1))
       for cc in range(BN):
        n=pn*BN+cc
        if self.route=='s2':n%=N
        at(Q,k//8*Q.stride(0)+n*Q.stride(1));at(S,k//G*S.stride(0)+n*S.stride(1));at(Z,k//G*Z.stride(0)+n//8*Z.stride(1))
     for row in range(pm*BM,min((pm+1)*BM,M)):
      for n in range(pn*BN,min((pn+1)*BN,N)):at(C,row*C.stride(0)+n*C.stride(1))
   launches.append({'route':self.route,'grid':list(shape),'M':M,'N':N,'K':K,'group':G,'BM':BM,'BN':BN,'BK':BK,'checked_accesses':count,'unsafe':False})
   return self.fn[shape](*args,**meta,num_warps=cfg.num_warps,num_stages=cfg.num_stages)
  return run
first=Selected(m.matmul4_kernel,'gptq');second=Selected(m.matmul_kernel,'s2');dequant=Selected(m.dequantize_kernel,'dequant')
first_configs=m.matmul4_kernel.configs
m.matmul4_kernel=first;m.matmul_kernel=second;m.dequantize_kernel=dequant
snapshots=[];output_frames=[]
def raw(t):
 storage=t.untyped_storage()
 return torch.empty(0,dtype=torch.uint8).set_(storage,0,(storage.nbytes(),),(1,))
def supplied_out(M,N):
 base=torch.full((M*N+17,),23,dtype=torch.float16)
 out=base[7:7+M*N].reshape(M,N).as_subclass(CudaView)
 output_frames.append((out,raw(out).clone()))
 return out

def inputs(M,N,K,G,dtype):
 A=(torch.arange(M*K).reshape(M,K)%2).to(dtype)
 q=(torch.arange(K)[:,None]+3*torch.arange(N)[None,:])%16
 z=(torch.arange(K//G)[:,None]+torch.arange(N)[None,:])%16
 Q=sum(q[i::8].to(torch.int64)<<(4*i) for i in range(8)).to(torch.int32)
 Z=sum(z[:,i::8].to(torch.int64)<<(4*i) for i in range(8)).to(torch.int32)
 S=(torch.arange(K//G)[:,None]%2+1).expand(K//G,N).to(dtype)/8
 D=(q-z.repeat_interleave(G,0)).to(dtype)*S.repeat_interleave(G,0)
 snapshots.extend((v,raw(v).clone()) for v in (A,Q,S,Z))
 return tuple(v.as_subclass(CudaView) for v in (A,Q,S,Z,D))
cases=[]
for cfg in first_configs:
 first.config=cfg
 for G,supplied in [(32,False),(64,True)]:
  A,Q,S,Z,DQ=inputs(65,256,64,G,torch.float16);out=supplied_out(65,256) if supplied else None
  ret=m.matmul_dequantize_int4_gptq(A,Q,S,Z,G,out);actual=out if supplied else ret
  expected=(A.float()@DQ.float()).half();torch.testing.assert_close(actual,expected,rtol=0,atol=0)
  assert (ret is None) if supplied else (ret.dtype==torch.float16)
  cases.append({'route':'gptq','group':G,'supplied':supplied,'BM':cfg.kwargs['BLOCK_SIZE_M'],'exact':True})
for M,N,K,G,supplied in [(64,128,40,8,False),(65,136,40,8,True),(577,136,64,8,False)]:
 A,Q,S,Z,DQ=inputs(M,N,K,G,torch.float32);out=supplied_out(M,N) if supplied else None
 ret=m.matmul_dequantize_int4_s2(A,Q,S,Z,G,out);expected=(A.half().float()@DQ.half().float()).half()
 torch.testing.assert_close(ret,expected,rtol=0,atol=0)
 assert ret.dtype==torch.float16 and (not supplied or ret is out)
 cases.append({'route':'s2','M':M,'N':N,'K':K,'group':G,'supplied':supplied,'exact':True})
for dtype in [torch.float32,torch.float16]:
 A,Q,S,Z,DQ=inputs(3,24,40,8,dtype);ret=m.dequantize_int4(Q,S,Z,'cuda:0',dtype,8)
 torch.testing.assert_close(ret,DQ,rtol=0,atol=0);assert ret.dtype==dtype
 cases.append({'route':'dequant','dtype':str(dtype),'K':40,'N':24,'exact':True})
for supplied in [False,True]:
 A,Q,S,Z,DQ=inputs(3,24,40,8,torch.float16);out=supplied_out(3,24) if supplied else None
 ret=m.matmul_dequantize_int4_s1(A,Q,S,Z,8,out);torch.testing.assert_close(ret,A@DQ,rtol=0,atol=0)
 assert ret.dtype==torch.float16 and (not supplied or ret is out)
 cases.append({'route':'s1','supplied':supplied,'torch_mm_CPU_public_API':True,'exact':True})
for t,before in snapshots:assert torch.equal(raw(t),before)
for t,before in output_frames:
 lo=2*t.storage_offset();hi=lo+2*t.numel()
 assert torch.equal(raw(t)[:lo],before[:lo]) and torch.equal(raw(t)[hi:],before[hi:])
rejected=0;admitted=0
def reject(fn):
 global rejected
 try:fn()
 except (ValueError,TypeError):rejected+=1
 else:raise AssertionError('unsupported argument admitted')
A,Q,S,Z,DQ=inputs(2,256,64,32,torch.float16)
for bad in (0,-1,True,32.0,3):reject(lambda bad=bad:m._check('gptq',A,Q,S,Z,bad))
reject(lambda:m._check('gptq',A.float(),Q,S,Z,32))
reject(lambda:m._check('gptq',A,Q.long(),S,Z,32))
reject(lambda:m._check('gptq',A,Q,S.float(),Z,32))
reject(lambda:m._check('gptq',A,Q,S,Z.long(),32))
reject(lambda:m._check('gptq',A,Q[:,:128],S[:,:128],Z[:,:16],32))
reject(lambda:m._check('gptq',A,Q,S[:,:128],Z,32))
reject(lambda:m._check('gptq',A,Q,S,Z[:,:16],32))
reject(lambda:m._check('gptq',A[:,::2],Q,S,Z,32))
reject(lambda:m._check('gptq',A,Q,S,Z,32,output=S))
reject(lambda:m._check('gptq',A,Q,S,Z,32,output=torch.empty((2,256),device='cuda',dtype=torch.float32)))
reject(lambda:m._check('gptq',A,Q,S,Z,32,output=torch.empty((3,256),device='cuda',dtype=torch.float16)))
class Negative(CudaView):
 def is_neg(self):return True
class Conjugate(CudaView):
 def is_conj(self):return True
reject(lambda:m._check('gptq',A.as_subclass(Negative),Q,S,Z,32))
reject(lambda:m._check('gptq',A.as_subclass(Conjugate),Q,S,Z,32))
class WrongPointer(CudaView):
 def data_ptr(self):return super().data_ptr()+1
reject(lambda:m._check('gptq',A.as_subclass(WrongPointer),Q,S,Z,32))
for dtype in (torch.bfloat16,torch.float64,torch.int32):reject(lambda dtype=dtype:m.dequantize_int4(Q,S,Z,'cuda:0',dtype,32))
reject(lambda:m.dequantize_int4(Q,S,Z,'cpu',torch.float16,32))
reject(lambda:m._check('s2',A,Q,S,Z,32))
reject(lambda:m._source_indices('gptq',1<<31,256,64,32,(1<<40,)*6))
reject(lambda:m._source_indices('dequant16',1,1<<30,8,8,(1<<40,)*6))
for route in ('gptq','s1'):
 m._check(route,A,Q,S,Z,32);admitted+=1
Af,Qf,Sf,Zf,_=inputs(2,24,40,8,torch.float32)
m._check('s2',Af,Qf,Sf,Zf,8);admitted+=1
m._check('dequant32',None,Qf,Sf,Zf,8,device='cuda:0',dtype=torch.float32);admitted+=1
m._check('dequant16',None,Q,S,Z,32,device='cuda:0',dtype=torch.float16);admitted+=1
print('GUARDS',admitted,'admissions',rejected,'rejections',flush=True)
result={'schema':1,'status':'PASS_BOUNDED_ACTUAL_SOURCE_CPU_CONTROLS','candidate_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':cases,'launches':launches,'source_launches':len(launches),'unsafe_launches':0,'caller_byte_frames':True,'output_canaries':True,'guard_admissions':admitted,'guard_rejections':rejected,'source_AST_review':True,'native_gpu_execution':False,'native_numerical_conformance':False,'torch':torch.__version__,'triton':triton.__version__}
(D/'evidence/cpu-controls.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['cases','launches']}))

print("Evidence:",D/"evidence/cpu-controls.json")
