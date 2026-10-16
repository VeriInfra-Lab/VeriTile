import sys
sys.dont_write_bytecode=True
from pathlib import Path
from collections import Counter
import ast,hashlib,json,shutil
W=Path(__file__).resolve().parent;R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sys.path.insert(0,str(R/'bench'))
from adapter_common import imports
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
def save(name,value):(W/'evidence'/name).write_text(json.dumps(value,indent=2)+'\n')
BASE='bench/tritonbench_g/triton_mul2/'
names=['FiniteScalar','FiniteMemory','FiniteProgram','FiniteSource','FiniteGrid','FiniteHost','FinitePointerBoundary','FiniteRefinement']
modules=[BASE+'Proofs/'+n+'.lean' for n in names]+[BASE+'FiniteCorrect.lean']
for rel in modules:
 p=W/'after'/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(W/rel,p)
 p=W/(rel+'.txt');p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(W/rel,p)
save('staging-map.json',{m:m for m in modules})
before=load(W/'evidence/dependencies-before.json')
for name,e in before['project'].items():
 assert sha(e['source'])==e['sha256'],name
 assert sha(e['olean'])==e['olean_sha256'],name
for name,e in before['external_objects'].items():assert sha(e['olean'])==e['sha256'],name
for name,h in before['input_sha256'].items():assert sha(name)==h,name
seen=set();project={};external=set()
def visit(rel):
 if rel in seen:return
 seen.add(rel);p=W/rel if rel in modules else R/rel
 if not p.is_file():external.add(rel);return
 if rel not in modules:
  e=before['project'][rel[:-5].replace('/','.')]
  project[rel]={'sha256':sha(p),'olean':e['olean'],'olean_sha256':e['olean_sha256']}
 for name in imports(p.read_text()):visit(name.replace('.','/')+'.lean')
for m in modules:visit(m)
save('dependency-closure.json',{'project':dict(sorted(project.items())),'external_imports':sorted(external),'before_snapshot_sha256':sha(W/'evidence/dependencies-before.json'),'new_modules':{m:sha(W/m) for m in modules}})
native=W/'prior-v1/native-attempts/20261015T214958399467Z';receipt=load(native/'receipt.json')
assert sha(native/'receipt.json')=='c69fad9462a13be7e06236e6eaa1cd26a7e960d4169a467b3dffc167f7103fe3'
for p,h in receipt['artifacts'].items():assert sha(native/p)==h,p
assert receipt['source_sha256']==sha(R/BASE/'triton_mul2.py') and receipt['production_ast_unchanged']
counts=Counter(r['result']['status'] for r in receipt['records']);assert counts=={'compiled':30,'rejected':30,'frontend_passed':6}
tree=ast.parse((R/BASE/'triton_mul2.py').read_text());functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
production=['mul2_kernel','mul2_inplace_kernel','triton_mul2','triton_mul2_inplace']
source_refs={}
for name,path in {
 'native-driver':'/usr/local/lib/python3.12/site-packages/triton/backends/nvidia/driver.c',
 'frontend-semantic':'/usr/local/lib/python3.12/site-packages/triton/language/semantic.py',
 'frontend-utils':'/usr/local/lib/python3.12/site-packages/triton/_utils.py',
 'python-cdiv':'/usr/local/lib/python3.12/site-packages/triton/__init__.py',
 'torch-metadata':'/usr/local/lib/python3.12/site-packages/torch/include/c10/core/TensorImpl.h',
 }.items():
 p=Path(path);dest=W/'evidence'/(name+'.txt');shutil.copyfile(p,dest)
 source_refs[name]={'source':path,'sha256':sha(p),'snapshot':str(dest.relative_to(W))}
save('source-review.json',{'source':BASE+'triton_mul2.py','source_sha256':sha(R/BASE/'triton_mul2.py'),
 'production_functions':production,'statements':{name:[ast.unparse(s) for s in functions[name].body] for name in production},
 'source_harness':'test_mul contains four ordinary wrapper calls, B1024/B512, no production callers; result_gold initializes the test harness at import',
 'native_receipt':str((native/'receipt.json').relative_to(W)),'native_receipt_sha256':sha(native/'receipt.json'),'native_counts':dict(counts),
 'canonical_dtypes':22,'admitted_dtypes':13,'first_signature_rejects':5,'first_architecture_rejects':2,'backend_fp8_rejects':2,
 'source_native_dependencies':source_refs,'debug':False,'instrumentation_mode':'','gpu_executions':0})
save('domain-review.json',{
 'task':'triton_mul2','full_finite_domain':'Both original wrappers/kernels, all22 canonical Torch storage dtype routes, arbitrary-rank ordinary strided Tensor metadata including rank0 and empty views, integer BLOCK_SIZE argument. Every Python integer block either fails cdiv/frontend or is a power of two <=2^20. The default16 and harness512/1024 remain explicit.',
 'large_numel':'All actual Torch numel below2^63. Native extractI32 casts PyLong_AsLong to int32, so n>=2^31 is not an ABI rejection. pid*B+lane and n are i32, comparison signed, offsets sign-extend before width scaling and modulo2^64 pointer addition. Actual grid parse requires signed-i32 grid-x. The n=2^31 full launch is proved all-masked without any access/value premise.',
 'integer_and_bool':'Signed/unsigned multiplication modulo storage width; bool load rawbyte!=0, multiply as i32, store raw0/2. All bit patterns total; no float-to-integer conversion exists on an admitted route.',
 'floating':'Only fp16,bf16,fp32,fp64 storage-width multiplication is deferred. Literal2 IEEE words and typed load/store paths are structural. fp8e4/e5 frontend multiply produces LLVM error, so these are backend rejection routes.',
 'memory':'Optional canonical backing-byte memory with shared launch-context device-address translation, actual ABI pointers from setup, little-endian assembly/splitting, full tile load before stores, independent per-byte last-write result, dynamic actual translated spans and launch-context byte read/write checks, missing-byte observations and byte ownership observations. No logical-stride elementwise claim. Wrapped successful accesses and masked padded offsets are retained.',
 'scheduling':'RuntimeContract requires permutation of every launched pid exactly once. Grid reference reads last-write pointwise memory between programs, supporting in-place updates without input/output separation premises. Repeated programs are not admitted native schedules.',
 'host_order':'Outplace zeros_like, input.numel, Python cdiv, runtime preparation, dtype binder/architecture, block frontend, backend compile/context, grid parse, pointer extraction, positive native launch initiation, source grid, return allocated output. Inplace omits allocation and returns the exact input Tensor. Zero-grid still runs preparation/frontend/grid/pointer setup, skips positive native launch initiation, and makes no byte access.',
 'metadata':'ZerosLike requires fresh object, same shape/count/dtype/device, observed preserve-format strides, offset0, exact zero allocation bytes, finite extent, fresh disjoint storage, and allocation exterior frame. Tensor actual empty data_ptr remains0 independently from backing base/storage_offset. Host result preserves exact in-place identity and all outplace metadata unconditionally.',
 'trusted_boundary':'Ordinary Torch metadata/zeros_like allocation and stride selection; current device/stream preparation; resource-dependent backend/context setup; pointer extraction; positive CUDA launch initiation. Each is fallible and cannot supply source result values. Native compiler/MMU correctness and floating primitive correspondence are not claimed proved.',
 'observational_failures':'Input/output access, missing-byte and ownership observations do not assert native traps, native error messages, race winner, rollback or post-failure memory.',
 'pointer_boundary_revision':'Runtime.pointers returns PointerSetup with actual ABI pointers, translated allocation bases and AccessContext. ReadableByte/writableByte and backingByte are launch-device indexed and independent of tensor owner. Null raw data_ptr preserves null ABI pointer. PointerBinding connects every translated span byte to original canonical backing. Source reads actual device aliases; every store records and proves device-address/backing-byte resolution. Setup supplies no values or final memory.',
 'no_numerical_escape':'Source masks, pointer wrap, all-masked launches, actual load/store effects, dtype rejection order, allocation/metadata and undefined-conversion inventory are structural obligations; none is queued as numerical validation.',
 'revision':2,'prior_review_status':'v1 remains blocked; this is an additive repair proposal for all3 recorded blockers','original_acceptance_increment':0,'requires':'Independent substantive root review pinned to immutable receipt before integration.'})
print(json.dumps({'modules':len(modules),'project_dependencies':len(project),'native_counts':dict(counts)}))
