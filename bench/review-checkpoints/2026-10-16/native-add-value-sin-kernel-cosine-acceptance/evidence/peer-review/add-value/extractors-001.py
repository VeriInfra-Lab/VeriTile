from pathlib import Path
import ctypes,hashlib,json,subprocess,sysconfig
W=Path(__file__).resolve().parent;O=W/'exact-extractors';O.mkdir()
driver=W/'replay/evidence/inherited/annotated-i32-scope-review/driver.c.txt'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
text=driver.read_text();bodies={}
for name in ['extractFP32','extractI32']:
 a=text.index('bool '+name+'(');b=text.index('\n}',a)+2;bodies[name]=text[a:b]
src=O/'extract.c';src.write_text('#include <Python.h>\n#include <stdint.h>\n#include <stdbool.h>\n'+'\n'.join(bodies.values())+'\n')
cmd=['cc','-shared','-fPIC','-O2','-I'+sysconfig.get_path('include'),str(src),'-o',str(O/'extract.so')]
r=subprocess.run(cmd,text=True,capture_output=True);(O/'build.log').write_text(r.stdout+r.stderr);assert r.returncode==0
lib=ctypes.PyDLL(str(O/'extract.so'))
for name in bodies:
 fn=getattr(lib,name);fn.argtypes=[ctypes.c_void_p,ctypes.py_object];fn.restype=ctypes.c_bool
word=ctypes.c_uint32();assert lib.extractFP32(ctypes.byref(word),10) and word.value==0x41200000
rows=[]
for n in [0,1,1023,1024,1025,2**31-1,2**31,2**31+1,2**32-1,2**32,2**32+1,2**31+2**32,(2**31-1)*1024-1,(2**31-1)*1024,(2**31-1)*1024+1,2**63-1]:
 v=ctypes.c_int32();assert lib.extractI32(ctypes.byref(v),n)
 assert v.value==((n+2**31)%2**32)-2**31
 rows.append({'host_n':n,'i32':v.value})
try:lib.extractI32(ctypes.byref(ctypes.c_int32()),2**63)
except OverflowError:overflow=True
else:raise AssertionError('2pow63 admitted as signed long')
report={'status':'passed','runtime_Python_argument':10,'runtime_fp32_bits':word.value,'i32_profiles':rows,'signed64_overflow_rejected':overflow,'driver_sha256':sha(driver),'exact_extractor_hashes':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in bodies.items()},'command':cmd,'GPU_launches':0,'artifacts':{p.name:sha(p) for p in O.iterdir() if p.is_file()}}
(O/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
