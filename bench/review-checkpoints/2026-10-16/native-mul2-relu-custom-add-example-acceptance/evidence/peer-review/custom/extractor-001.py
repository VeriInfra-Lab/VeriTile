from pathlib import Path
import ctypes,hashlib,json,subprocess,sysconfig
W=Path(__file__).resolve().parent;P=Path('/tmp/veritile-standardization-20261015/vector-custom-finite-acceptance-revision1-wip');D=W/'driver-extractor-001';D.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
driver=P/'evidence/inherited/annotated-i32-scope-review/driver.c.txt';text=driver.read_text();start=text.index('bool extractI32(');end=text.index('\nbool extractI64(',start);body=text[start:end]
source=D/'extract_i32.c';source.write_text('#include <Python.h>\n#include <stdint.h>\n#include <stdbool.h>\n'+body+'\n')
cmd=['cc','-shared','-fPIC','-O2','-I'+sysconfig.get_path('include'),str(source),'-o',str(D/'extract_i32.so')]
r=subprocess.run(cmd,text=True,capture_output=True);(D/'build.log').write_text(r.stdout+r.stderr);assert r.returncode==0
lib=ctypes.PyDLL(str(D/'extract_i32.so'));fn=lib.extractI32;fn.argtypes=[ctypes.c_void_p,ctypes.py_object];fn.restype=ctypes.c_bool
ns=[0,1,15,16,17,2**31-1,2**31,2**31+1,2**32-1,2**32,2**32+1,(2**31-1)*16,(2**31-1)*16+1,2**63-1]
rows=[]
for n in ns:
 word=ctypes.c_int32();assert fn(ctypes.byref(word),n);expected=((n+2**31)%2**32)-2**31;assert word.value==expected;rows.append({'host_dimension':n,'actual_i32':word.value})
try:fn(ctypes.byref(ctypes.c_int32()),2**63)
except OverflowError:outside='OverflowError'
else:raise AssertionError('outside signed64 accepted')
result={'status':'passed','source_driver_sha256':sha(driver),'exact_function_sha256':hashlib.sha256(body.encode()).hexdigest(),'command':cmd,'records':rows,'outside_declared_dimension_2pow63':outside,'GPU_launches':0,'artifacts':{p.name:sha(p) for p in D.iterdir() if p.is_file()}}
(W/'driver-extractor-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['records','artifacts']}))
