from pathlib import Path
import sys,ast,json,hashlib,importlib.util,collections,ctypes,inspect
sys.dont_write_bytecode=True
W=Path(__file__).resolve().parent;D=W/'frontend-matrix-001';D.mkdir()
import torch,triton,triton.runtime.jit as jit
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton._C.libtriton import ir
prefix=next((W/'replay/evidence/native-attempts').glob('*/exact_prefix.py'))
spec=importlib.util.spec_from_file_location('independent_relu_frontend',prefix);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
target=GPUTarget('cuda',90,32);backend=CUDABackend(target);options=backend.parse_options({});context=ir.context();ir.load_dialects(context);backend.load_dialects(context)
names=['bool','int8','uint8','int16','uint16','int32','uint32','int64','uint64','float16','bfloat16','float32','float64','float8_e4m3fn','float8_e5m2','float8_e4m3fnuz','float8_e5m2fnuz','complex32','complex64','complex128','float8_e8m0fnu','float4_e2m1fn_x2']
ns=[0,1,2**31-1,2**31,2**32-1,2**32,(2**31-1)*1024,(2**31-1)*1024+1,2**63-1,2**63,2**64-1,2**64]
rows=[];checks=[]
parser=ctypes.pythonapi.PyArg_ParseTuple;parser.restype=ctypes.c_int
for n in ns:
 grid=(n+1023)//1024;v=ctypes.c_int()
 try:
  ok=parser(ctypes.py_object((grid,)),ctypes.c_char_p(b'i'),ctypes.byref(v));assert ok==1 and v.value==grid;parsed=True
 except OverflowError:parsed=False
 assert parsed==(grid<=2**31-1)
 checks.append({'N':n,'grid':grid,'CPython_signed_int_parse':parsed})
for dt in names:
 for n in ns:
  name=dt+'-'+str(n);row={'dtype':dt,'N':n,'grid':(n+1023)//1024}
  try:
   sig={'x_ptr':'*'+jit.type_canonicalisation_dict[dt],'out_ptr':'*fp32'}
   source=ASTSource(m.relu_kernel,sig,constexprs={'N':n,'block_size':1024})
   module=source.make_ir(target,options,backend.get_codegen_implementation(options),backend.get_module_map(),context)
   raw=str(module);path=D/(name+'.raw.ttir');path.write_text(raw);row.update(status='generated',raw_ttir=path.name)
   # The exact original N comparison follows the frontend's finite type promotion.
   predicate='slt' if n<2**31 else 'ult' if n<2**32 else 'slt' if n<2**63 else 'ult'
   assert 'arith.cmpi '+predicate in raw,(dt,n,predicate)
   if n>=2**32:assert 'arith.extsi' in raw
  except Exception as e:
   error=str(e);tail=error.splitlines()[-1]
   if isinstance(e,KeyError):stage='missing_signature_map'
   elif 'not supported in this architecture' in tail:stage='unsupported_signature'
   elif 'Nonrepresentable integer' in tail:stage='nonrepresentable_constexpr'
   elif 'unexpected type fp8' in tail:stage='ordered_comparison_rejected'
   else:raise
   row.update(status=stage,error_type=type(e).__name__,error=error)
  if dt in names[17:]:expected='missing_signature_map'
  elif dt in names[15:17]:expected='unsupported_signature'
  elif n>=2**64:expected='nonrepresentable_constexpr'
  elif dt in names[13:15]:expected='ordered_comparison_rejected'
  else:expected='generated'
  assert row['status']==expected,(dt,n,row)
  rows.append(row)
(D/'records.json').write_text(json.dumps(rows,indent=2)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result={'status':'passed','frontend_cases':len(rows),'dtypes':len(names),'N_boundaries':ns,'outcomes':dict(collections.Counter(r['status'] for r in rows)),'runtime_grid_parse_checks':checks,'all_dtype_failure_precedence_checked':True,'frontend_mask_width_and_signedness_checked':True,'gpu_executed':False,'native_numerical_validation':False,'artifacts':{str(p.relative_to(W)):sha(p) for p in D.iterdir() if p.is_file()}}
(W/'frontend-matrix-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['artifacts','runtime_grid_parse_checks']}))
