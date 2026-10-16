from pathlib import Path
import hashlib,json,subprocess,sysconfig,importlib.util
W=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
src=Path('/usr/local/lib/python3.12/site-packages/triton/backends/nvidia/driver.c')
txt=src.read_text();start=txt.index('static PyObject *data_ptr_str = NULL;');end=txt.index('\nbool extractI8',start)
extract=txt[start:end];i32=txt[txt.index('bool extractI32'):txt.index('\nbool extractI64')]
header=Path('/usr/local/cuda/include/cuda.h');jit=Path('/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py')
for name,p in [('driver.c.txt',src),('cuda.h.txt',header),('jit.py.txt',jit)]: (W/name).write_bytes(p.read_bytes())
code='''#include <Python.h>
#include <stdint.h>
#include <stdbool.h>
typedef uint64_t CUdeviceptr;
typedef int CUresult;
#define CUDA_ERROR_INVALID_VALUE 1
#define CU_POINTER_ATTRIBUTE_DEVICE_POINTER 3
static CUdeviceptr requested_mapping;
static int cuda_calls;
static CUresult cuPointerGetAttribute(CUdeviceptr *out, int attr, CUdeviceptr input) {
 cuda_calls++; *out=requested_mapping; return 0;
}
static bool gpuAssert(CUresult s, const char *file, int line) {return s==0;}
'''+extract+'\n'+i32+'''
static PyObject *probe(PyObject *self, PyObject *args) {
 PyObject *obj; unsigned long long mapped;
 if (!PyArg_ParseTuple(args,"OK", &obj,&mapped)) return NULL;
 if (!data_ptr_str) data_ptr_str=PyUnicode_InternFromString("data_ptr");
 requested_mapping=mapped;cuda_calls=0;CUdeviceptr out=0;
 bool ok=extractPointer(&out,obj);
 if (!ok || PyErr_Occurred()) return NULL;
 return Py_BuildValue("Ki",out,cuda_calls);
}
static PyObject *grid(PyObject *self, PyObject *args) {
 int n; if(!PyArg_ParseTuple(args,"i",&n)) return NULL;return PyLong_FromLong(n);
}
static PyObject *narrow(PyObject *self, PyObject *obj) {
 int32_t n;if(!extractI32(&n,obj))return NULL;return PyLong_FromLong(n);
}
static PyMethodDef methods[]={{"pointer",probe,METH_VARARGS,""},{"grid",grid,METH_VARARGS,""},{"narrow",narrow,METH_O,""},{NULL,NULL,0,NULL}};
static struct PyModuleDef mod={PyModuleDef_HEAD_INIT,"pointer_probe",NULL,-1,methods};
PyMODINIT_FUNC PyInit_pointer_probe(void){return PyModule_Create(&mod);}
'''
c=W/'pointer_probe.c';c.write_text(code);so=W/('pointer_probe'+sysconfig.get_config_var('EXT_SUFFIX'))
cmd=['cc','-shared','-fPIC','-O0','-I'+sysconfig.get_path('include'),str(c),'-o',str(so)]
r=subprocess.run(cmd,capture_output=True,text=True);(W/'pointer-build-001.log').write_text(r.stdout+r.stderr)
assert r.returncode==0,r.stderr
spec=importlib.util.spec_from_file_location('pointer_probe',so);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Tensor:
 def __init__(self,p):self.p=p
 def data_ptr(self):return self.p
 @property
 def device(self):raise AssertionError('driver must not inspect Tensor.device')
rows=[]
for source,mapped in [(0,999),(0x1000,0x1000),(0x1000,0x7000)]:
 out,calls=m.pointer(Tensor(source),mapped);assert out==(0 if source==0 else mapped);assert calls==int(source!=0)
 rows.append({'data_ptr':source,'CUDA_returned_device_pointer':mapped,'driver_argument_pointer':out,'attribute_calls':calls})
gridrows=[]
for n in [2**31-1,2**31,2**32+1,2**63-1]:
 narrow=m.narrow(n);assert narrow==((n+2**31)%2**32-2**31)
 try:g=m.grid(n);status='accepted'
 except OverflowError:g=None;status='OverflowError'
 assert (status=='accepted')==(n<2**31)
 gridrows.append({'n':n,'native_extractI32':narrow,'PyArg_i_grid_status':status,'grid':g})
report={'status':'passed','scope':'Exact installed extractPointer and extractI32 compiled with an explicit CUDA attribute stub; CPython signed grid parser actual. This is source-plumbing evidence, not a GPU or CUDA accessibility experiment.','gpu_launches':0,'cuda_attribute_is_stub':True,'driver_sha256':sha(src),'cuda_header_sha256':sha(header),'jit_sha256':sha(jit),'exact_extractPointer_sha256':hashlib.sha256(extract.encode()).hexdigest(),'compile':cmd,'rows':rows,'size_grid_rows':gridrows,'finding':'Tensor device ownership is never inspected; translated CUDA pointer replaces original data_ptr. Current candidate requires same-device ownership and discards translation.'}
(W/'pointer-probe.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
