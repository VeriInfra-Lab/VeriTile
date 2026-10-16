import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,ctypes,hashlib,json,subprocess,sysconfig
W=Path(__file__).resolve().parent;R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');E=W/'evidence'
sys.path.insert(0,str(R/'bench'))
from adapter_common import imports
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();load=lambda p:json.loads(Path(p).read_text())
def save(n,d):(E/n).write_text(json.dumps(d,indent=2)+'\n')
modules={str(p.relative_to(W/'after')):sha(p) for p in sorted((W/'after').rglob('*.lean'))};seen=set();project={};external=set()
def visit(rel):
 if rel in seen:return
 seen.add(rel);p=W/'after'/rel if rel in modules else R/rel
 if not p.is_file():external.add(rel);return
 if rel not in modules:
  obj=(W/'build-overlay'/rel).with_suffix('.olean').resolve();assert obj.is_file(),(rel,obj)
  project[rel]={'sha256':sha(p),'object':str(obj),'object_sha256':sha(obj)}
 for m in imports(p.read_text()):visit(m.replace('.','/')+'.lean')
for rel in modules:visit(rel)
assert not any('vector_addition_custom' in p or '/add_value/Proofs/Finite' in p for p in project)
save('dependencies-before.json',{'capture_phase':'before final rebuild; after iterative development checks','project':dict(sorted(project.items())),'external_imports':sorted(external),'new_modules':modules})
source=R/'bench/tritonbench_g/add_value/add_value.py';tree=ast.parse(source.read_text());fs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
V=W.parent/'vector-finite-acceptance-revision2-wip';pointer=load(V/'evidence/source-review.json')['pointer_extractor']
save('source-review.json',{'original_source_sha256':sha(source),'statements':{n:[ast.unparse(s) for s in fs[n].body] for n in ['puzzle1_kernel','puzzle1']},'source_order':['empty_like','short_circuit_x_and_output_CUDA_assertion','output.numel','BLOCK1024_original_grid','context','signature_and_compilation','signed_grid_parse','pointer_and_scalar_argument_packing','launch','return_same_allocated_tensor'],'pointer_extractor':pointer,'all_extents_bound_i32':True,'host_extent_upper_exclusive':2**63,'source_return_at_two31':'add_value_two31_return: exact allocated tensor, unchanged memory; ordinary successful setup and original CUDA assertion only; no active access/value/no-wrap premise','runtime_argument':{'python':10,'type':'fp32','bits':0x41200000,'constexpr':False,'second_input_pointer':False},'undefined_conversions':'Option-valued scalar source and independent formula; None observed before output materialization; no arbitrary fallback','trusted_memory':'Typed finite input words, pure width-specific store packing, byte presence/congruence/disjoint/frame laws. No arbitrary carrier readback law.','metadata':'Actual storage/data/extracted pointers distinguished; buffer owner separate from launch-device permissions; output preserve-format strides and freshness unconditional.','test_harness':'Non-JIT test_puzzle has four production-wrapper calls, no production caller; import-time result_gold remains harness.','historical_candidate_registration':False,'GPU_launches':0,'large_allocations':0})
# Exact installed fp32 argument extraction for the actual Python integer10.
O=E/'fp32-argument';O.mkdir(exist_ok=True);driver=E/'inherited/annotated-i32-scope-review/driver.c.txt';text=driver.read_text();start=text.index('bool extractFP32(');end=text.index('\n}',start)+2;body=text[start:end]
(O/'extract.c').write_text('#include <Python.h>\n#include <stdint.h>\n#include <stdbool.h>\n'+body+'\n');cmd=['cc','-shared','-fPIC','-O2','-I'+sysconfig.get_path('include'),str(O/'extract.c'),'-o',str(O/'extract.so')]
r=subprocess.run(cmd,capture_output=True,text=True);(O/'build.log').write_text(r.stdout+r.stderr);assert r.returncode==0
lib=ctypes.PyDLL(str(O/'extract.so'));lib.extractFP32.argtypes=[ctypes.c_void_p,ctypes.py_object];lib.extractFP32.restype=ctypes.c_bool;word=ctypes.c_uint32();assert lib.extractFP32(ctypes.byref(word),10) and word.value==0x41200000
save('fp32-argument/receipt.json',{'status':'passed','driver_source_sha256':sha(driver),'exact_extractor_sha256':hashlib.sha256(body.encode()).hexdigest(),'actual_Python_argument':10,'runtime_fp32_bits':word.value,'constexpr':False,'command':cmd,'artifacts':{p.name:sha(p) for p in O.iterdir() if p.is_file()}})
print(json.dumps({'project_dependencies':len(project),'external_imports':len(external),'modules':len(modules),'runtime_fp32_bits':word.value}))
