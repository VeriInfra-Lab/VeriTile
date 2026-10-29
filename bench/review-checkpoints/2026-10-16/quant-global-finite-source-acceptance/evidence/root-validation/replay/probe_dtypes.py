from pathlib import Path
import ast,hashlib,json,sys,traceback
sys.dont_write_bytecode=True
import torch
out=Path(__file__).resolve().parent/'dtype-probes'
out.mkdir(exist_ok=True)
source=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile/bench/tritonbench_g/quantize_global/quantize_global.py')
dtypes=['bool','uint8','int8','int16','int32','int64','uint16','uint32','uint64','float16','bfloat16','float32','float64','complex32','complex64','complex128','float8_e4m3fn','float8_e5m2','float8_e4m3fnuz','float8_e5m2fnuz','float8_e8m0fnu','float4_e2m1fn_x2']
records=[]
for name in dtypes:
 if not hasattr(torch,name): continue
 for shape in ([3],[0],[]):
  rec={'dtype':name,'shape':shape,'device':'cpu','steps':[]}
  try:
   stage='input_construction'
   d=getattr(torch,name)
   x=torch.tensor([-2,0,1] if shape==[3] else ([] if shape==[0] else 1)).to(d)
   for stage,fn in [('abs',lambda:x.abs()),('max',lambda:y.max()),('unsqueeze',lambda:y.unsqueeze(0)),('reciprocal',lambda:1.0/y)]:
    y=fn()
    rec['steps'].append({'stage':stage,'dtype':str(y.dtype),'shape':list(y.shape),'values':repr(y)})
   rec['status']='host_primitives_success'
  except Exception as e:
   rec.update(status='rejected',stage=stage,error_type=type(e).__name__,error=str(e))
  records.append(rec)
result={'torch_version':torch.__version__,'cuda_available':torch.cuda.is_available(),'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'scope':'Real CPU scalar/tensor host primitives; CUDA dispatch remains a separate obligation. No GPU launches.','records':records}
p=out/'receipt.json';p.write_text(json.dumps(result,indent=2)+'\n')
for rec in records:
 if rec['shape']==[3]:print(rec['dtype'],rec['status'],rec.get('stage',''),[(s['stage'],s['dtype']) for s in rec['steps']],rec.get('error',''))
