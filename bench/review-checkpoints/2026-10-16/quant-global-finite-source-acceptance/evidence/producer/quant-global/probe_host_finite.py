from pathlib import Path
import ast,hashlib,json,math,sys
sys.dont_write_bytecode=True
root=Path(__file__).resolve().parent
out=root/'host-finite-probes';out.mkdir(exist_ok=True)
source=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile/bench/tritonbench_g/quantize_global/quantize_global.py')
tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='quantize_global')
dtypes=['bool','uint8','int8','int16','int32','int64','uint16','uint32','uint64','float16','bfloat16','float32','float64','complex32','complex64','complex128','float8_e4m3fn','float8_e5m2','float8_e4m3fnuz','float8_e5m2fnuz','float8_e8m0fnu','float4_e2m1fn_x2']
real={'complex32':'float16','complex64':'float32','complex128':'float64'}
integer={'bool','uint8','int8','int16','int32','int64'}
base=integer|{'float16','bfloat16','float32','float64'}|set(real)
fp8={'float8_e4m3fn','float8_e5m2','float8_e4m3fnuz','float8_e5m2fnuz'}
events=[];fault=None;block=1024;launches=[]
class Failure(Exception):pass
def event(stage,**detail):
 events.append({'stage':stage,**detail})
 if fault==stage:raise Failure('primitive:'+stage)
class Tensor:
 def __init__(self,name,shape,dtype,cuda,values):self.name,self.shape,self.dtype,self.is_cuda,self.values=name,tuple(shape),dtype,cuda,list(values)
 def abs(self):
  event('abs',input=self.name,dtype=self.dtype)
  allowed=(base if self.is_cuda else (base-{'bool'})|fp8)
  if self.dtype not in allowed:raise Failure('unsupportedAbs')
  return Tensor('abs',self.shape,real.get(self.dtype,self.dtype),self.is_cuda,[abs(v) for v in self.values])
 def max(self):
  event('maximum',input=self.name,dtype=self.dtype)
  if not self.values:raise Failure('emptyMaximum')
  if self.dtype in fp8:raise Failure('unsupportedMaximum')
  return Tensor('max',(),self.dtype,self.is_cuda,[max(self.values)])
 def unsqueeze(self,axis):
  event('unsqueeze',input=self.name,axis=axis)
  assert axis==0 and self.shape==()
  return Tensor('absmax',(1,),self.dtype,self.is_cuda,self.values)
 def __rtruediv__(self,value):
  event('reciprocal',input=self.name,value=value,dtype=self.dtype)
  assert value==1.0 and self.name=='absmax'
  return Tensor('inverse',(1,),'float32' if self.dtype in integer else self.dtype,self.is_cuda,[1.0/self.values[0]])
 def numel(self):event('numel',input=self.name);return math.prod(self.shape)
class Torch:
 Tensor=Tensor;int8='int8'
 @staticmethod
 def empty(*shape,device,dtype):
  events.append({'stage':'outputArguments','shape':shape,'device':device,'dtype':dtype})
  assert device=='cuda' and dtype=='int8'
  if not shape:raise Failure('missingSize')
  event('outputAllocation')
  return Tensor('output',shape,dtype,True,[])
class Triton:
 @staticmethod
 def cdiv(n,b):return (n+b-1)//b
class Launcher:
 def __getitem__(self,grid):
  def call(x,inverse,output,n):
   event('signature',dtype=x.dtype)
   if x.dtype in real:raise Failure('unsupportedSignature')
   event('compilation',dtype=x.dtype)
   if x.dtype in {'float16','bfloat16'}:raise Failure('unsupportedLlrint')
   event('launch',n=n,block=block,grid=grid({'BLOCK_SIZE':block}))
   assert inverse.name=='inverse' and output.name=='output' and inverse.shape==(1,)
   assert n==math.prod(x.shape) and grid({'BLOCK_SIZE':block})==((n+block-1)//block,)
   launches.append((x,inverse,output,n))
  return call
ns={'torch':Torch,'triton':Triton,'_quantize_global':Launcher()}
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
records=[]
for dtype in dtypes:
 for cuda in [False,True]:
  for shape in [(0,),(),(3,),(2049,)]:
   for block in [1024,2048]:
    events.clear();launches.clear();fault=None
    values=[1+abs(i%5-2) for i in range(math.prod(shape))]
    x=Tensor('input',shape,dtype,cuda,values)
    try:
     output,maximum=ns['quantize_global'](x);outcome='success'
     assert output is launches[0][2] and maximum.name=='absmax' and maximum.shape==(1,)
     assert maximum is not launches[0][1] and maximum.dtype==real.get(dtype,dtype)
     assert launches[0][1].dtype==('float32' if maximum.dtype in integer else maximum.dtype)
    except Failure as e:outcome=str(e)
    except AssertionError:
     assert not cuda and not launches
     outcome='cudaAssertion'
    stages=[e['stage'] for e in events]
    # The source chain itself determines reachability; independently verify it
    # is always an exact prefix of the documented source call sequence.
    canonical=['abs','maximum','unsqueeze','reciprocal','outputArguments','outputAllocation','numel','signature','compilation','launch']
    assert stages==canonical[:len(stages)],(dtype,cuda,shape,stages)
    if outcome=='success':assert dtype in integer|{'float32','float64'} and cuda and shape
    elif outcome=='unsupportedAbs':assert stages==['abs']
    elif outcome in ['emptyMaximum','unsupportedMaximum']:assert stages[-1]=='maximum'
    elif outcome=='missingSize':assert stages[-1]=='outputArguments'
    elif outcome=='cudaAssertion':assert stages[-1]=='outputAllocation'
    elif outcome=='unsupportedSignature':assert stages[-1]=='signature'
    elif outcome=='unsupportedLlrint':assert stages[-1]=='compilation'
    records.append({'dtype':dtype,'cuda':cuda,'shape':shape,'block':block,'outcome':outcome,'events':list(events)})
# Resource/primitive failures at each reachable stage, before later source calls.
for fault in ['abs','maximum','unsqueeze','reciprocal','outputAllocation','signature','compilation','launch']:
 events.clear();launches.clear();block=1024
 try:ns['quantize_global'](Tensor('input',(3,),'float32',True,[1,2,3]));raise AssertionError('no failure')
 except Failure as e:assert str(e)=='primitive:'+fault
 assert events[-1]['stage']==fault and not launches
 records.append({'injected_failure':fault,'events':list(events)})
receipt={'original_source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'wrapper_ast_sha256':hashlib.sha256(ast.dump(fn).encode()).hexdigest(),'count':len(records),'all_passed':True,'records':records,'boundary':'Unchanged original host AST with explicitly instrumented trusted tensor/allocator/launcher primitives. Dtype dispatch rules independently sourced from exact installed Torch CUDA/CPU audit; not GPU numerical execution.'}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(len(records),'unchanged original host dtype/outcome wiring probes passed')
