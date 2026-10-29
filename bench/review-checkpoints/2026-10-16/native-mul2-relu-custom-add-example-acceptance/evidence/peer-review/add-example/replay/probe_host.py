from pathlib import Path
import datetime,hashlib,importlib.util,json,sys,ctypes
sys.dont_write_bytecode=True
W=Path(__file__).resolve().parent;stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');O=W/'evidence/host-attempts'/stamp;O.mkdir(parents=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
(O/'probe.py.txt').write_bytes(Path(__file__).read_bytes())
import torch
spec=importlib.util.spec_from_file_location('exact_host_'+stamp,W/'evidence/source-prefix.py');mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
class Recorder:
 def __getitem__(self,grid):
  def launch(x,y,out,n,b):calls.append({'grid':grid,'n':n,'block':b,'x':meta(x),'y':meta(y),'output':meta(out),'storage_bytes':torch.empty(0,dtype=torch.uint8).set_(out.untyped_storage(),0,(out.untyped_storage().nbytes(),),(1,)).tolist()})
  return launch
mod.add_kernel=Recorder();calls=[]
def meta(x):return {'shape':list(x.shape),'strides':list(x.stride()),'numel':x.numel(),'dtype':str(x.dtype),'storage_offset':x.storage_offset(),'data_ptr':x.data_ptr(),'storage_base':x.untyped_storage().data_ptr(),'storage_bytes':x.untyped_storage().nbytes(),'is_contiguous':x.is_contiguous(),'device':str(x.device)}
names=['bool','int8','uint8','int16','uint16','int32','uint32','int64','uint64','float16','bfloat16','float32','float64','float8_e4m3fn','float8_e5m2','float8_e4m3fnuz','float8_e5m2fnuz','complex32','complex64','complex128','float8_e8m0fnu','float4_e2m1fn_x2']
records=[]
for name in names:
 for size in [0,1,5]:
  row={'dtype':name,'size':size}
  try:
   x=torch.empty(size,dtype=getattr(torch,name));y=torch.empty(size,dtype=getattr(torch,name));out=mod.add_wrapper(x,y);row.update(status='returned',x=meta(x),output=meta(out),launch=calls[-1])
  except Exception as e:row.update(status='error',error_type=type(e).__name__,error=str(e),launch_count=len(calls))
  records.append(row)
views={'scalar':torch.ones(()),'transpose':torch.ones(2,3).T,'sliced':torch.ones(30)[3:23:2],'broadcast':torch.ones(1).expand(5),'empty_stride':torch.empty_strided((0,3),(7,2)),'empty_transpose':torch.empty(3,0).T,'offset':torch.ones(10)[3:7]}
for name,x in views.items():
 row={'case':name,'x':meta(x)}
 try:
  out=mod.add_wrapper(x,torch.ones(1));row.update(status='returned',output=meta(out),launch=calls[-1])
 except Exception as e:row.update(status='error',error_type=type(e).__name__,error=str(e))
 records.append(row)
# CPU equivalent of the exact LP64 launcher extraction; no GPU driver call.
abi=[]
for n in [0,1,2,2**31-1,2**31,2**31+1,2**32-1,2**32,2**33-4,2**63-1]:abi.append({'python_n':n,'c_long_bytes':ctypes.sizeof(ctypes.c_long),'PyLong_AsLong_value':ctypes.c_long(n).value,'int32_cast':ctypes.c_int32(ctypes.c_long(n).value).value,'grid':(n+3)//4})
for name,path in [('jit.py','/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py'),('driver.c','/usr/local/lib/python3.12/site-packages/triton/backends/nvidia/driver.c')]:
 p=Path(path);(O/name).write_bytes(p.read_bytes())
(O/'records.json').write_text(json.dumps(records,indent=2)+'\n');(O/'abi.json').write_text(json.dumps(abi,indent=2)+'\n')
receipt={'torch':torch.__version__,'source_prefix_sha256':sha(W/'evidence/source-prefix.py'),'cpu_metadata_only':True,'gpu_executed':False,'actual_gpu_launch':False,'fake_launch_records_only':True,'case_count':len(records),'artifacts':{p.name:sha(p) for p in O.iterdir() if p.is_file()}}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'receipt':str(O/'receipt.json'),'sha256':sha(O/'receipt.json'),'cases':len(records),'errors':[r for r in records if r['status']=='error']},indent=2))
