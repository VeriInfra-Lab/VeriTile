"""Pinned launch argument parser, without CUDA context or GPU execution."""
from pathlib import Path
import ctypes,hashlib,json,subprocess,sysconfig
W=Path(__file__).resolve().parent;O=W/'driver-parser-001';O.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=Path('/tmp/veritile-standardization-20261015/add-example-finite-acceptance-wip/evidence/runtime-boundary/driver.c');text=source.read_text();body=text[text.index('static PyObject *launchKernel('):]
start=body.index('  // Parse the arguments.');end=body.index('  // launch entry hook.')
parser=body[start:end];assert parser.index('PyArg_ParseTuple')<len(parser)
assert body.index('PyArg_ParseTuple')<body.index('launchHook(')<body.index('extractor.extract(params')
(O/'driver-order-excerpt.txt').write_text(body[:body.index('  // Add scratch objects.')])
c='#include <Python.h>\n#include <stdint.h>\nint pointer_attempted=0;\nPyObject *parse_then_failure(PyObject *args) {\n  pointer_attempted=0;\n'+parser+'''  pointer_attempted=1;
  PyBuffer_Release(&signature);
  PyErr_SetString(PyExc_ValueError,"instrumented downstream pointer-packing failure");
  return NULL;
}
'''
(O/'parser.c').write_text(c);command=['cc','-shared','-fPIC','-O2','-I'+sysconfig.get_path('include'),str(O/'parser.c'),'-o',str(O/'parser.so')]
r=subprocess.run(command,capture_output=True,text=True);(O/'build.log').write_text(r.stdout+r.stderr);assert r.returncode==0
lib=ctypes.PyDLL(str(O/'parser.so'));fn=lib.parse_then_failure;fn.argtypes=[ctypes.py_object];fn.restype=ctypes.py_object;counter=ctypes.c_int.in_dll(lib,'pointer_attempted');rows=[]
for gx in [0,1,2**27,2**31-1,2**31,2**31+1,2**63-1]:
 args=(gx,1,1,0,0,False,False,(4,1,0),None,None,None,None,None,(),b'',())
 try:fn(args)
 except Exception as e:
  expected='ValueError' if gx<=2**31-1 else 'OverflowError'
  assert type(e).__name__==expected
  assert counter.value==int(gx<=2**31-1)
  rows.append({'grid_x':gx,'error_type':type(e).__name__,'error':str(e),'downstream_pointer_stage_reached':bool(counter.value)})
 else:raise AssertionError('expected parser or instrumented pointer error')
(O/'records.json').write_text(json.dumps(rows,indent=2)+'\n')
result={'status':'passed','driver_source_sha256':sha(source),'exact_parser_slice_sha256':hashlib.sha256(parser.encode()).hexdigest(),'source_order':['ensureCudaContext','PyArg_ParseTuple_signed_grid','launch_enter_hook','argument_flattening','kernel_argument_extractors','scratch_extractors','CUDA_launch','exit_hook'],'actual_parser_tests':len(rows),'counterexample':{'host_numel':2**33,'grid_x':2**31,'pointer_stage_reached':False,'first_error':'OverflowError'},'boundary':'Exact installed C parser and source ordering, downstream failure instrumentation; no CUDA context, pointer query or GPU launch performed','build_command':command,'artifacts':{p.name:sha(p) for p in O.iterdir() if p.is_file()}}
(O/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='artifacts'},indent=2))
