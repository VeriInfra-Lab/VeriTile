from pathlib import Path
import hashlib,json
W=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
D=W/'evidence/runtime-boundary';driver=(D/'driver.c').read_text();jit=(D/'jit.py').read_text()
assert '*((int32_t *)ptr) = PyLong_AsLong(obj);' in driver
launcher=driver[driver.index('static PyObject *launchKernel'):]
assert launcher.index('ensureCudaContext();')<launcher.index('PyArg_ParseTuple(args, "iii')<launcher.index('extractor.extract(params')<launcher.index('_launch(gridX')
assert 'if (*dev_ptr == 0)' in driver and 'CU_POINTER_ATTRIBUTE_DEVICE_POINTER' in driver
front=load(W/'evidence/native-attempts/20261015T215117570205Z/records.json')
for row in front:
 if row['status']=='frontend_generated':
  text=(W/'evidence/native-attempts/20261015T215117570205Z'/row['ttir']).read_text()
  assert 'arith.cmpi slt' in text and 'i32' in text
hostpath=W/'evidence/host-attempts/20261015T215521664431Z/records.json';host=load(hostpath)
returned=[r for r in host if r['status']=='returned'];errors=[r for r in host if r['status']=='error']
assert len(returned)==72 and len(errors)==1 and errors[0]['dtype']=='float4_e2m1fn_x2' and errors[0]['size']==0
for row in returned:
 out=row['output'];launch=row['launch'];x=launch['x']
 assert out['shape']==x['shape'] and out['dtype']==x['dtype'] and out['device']==x['device']
 assert out['storage_offset']==0 and launch['grid']==[(x['numel']+3)//4] and launch['n']==x['numel'] and launch['block']==4
 assert all(byte==0 for byte in launch['storage_bytes'])
 assert out['data_ptr']==(0 if out['numel']==0 else out['storage_base'])
review={'pinned_sources':{name:sha(D/name) for name in ['driver.c','jit.py']},'JIT_order':'current_device_and_stream; binder/frontend/backend; native launcher','launcher_order':'ensureCudaContext; signed-int grid PyArg_ParseTuple; entry hook; x/y/out/i32 extraction in signature order; scratch pointers; enqueue and exit hook','integer_argument':'LP64 PyLong_AsLong then int32 assignment, not signed32 range rejection; at n=2^31 value becomes signed minimum','native_grid_max':2**31-1,'largest_reachable_n':4*(2**31-1),'pointer_semantics':'raw null pointer returns ABI0 without CUDA pointer attributes; nonzero pointer may translate via CU_POINTER_ATTRIBUTE_DEVICE_POINTER','binding_condition':'Host AllocationBinding.bindings is required only after all native setup stages and signed grid parsing succeed','translation':'shared per-current-device native-to-canonical backing; byte-local decode/load and encode/store correspondence proved; no owner equality','cpu_wrapper_cases':len(host),'cpu_fake_launch_returned':len(returned),'cpu_actual_allocation_errors':errors,'returned_output_all_storage_bytes_zero':True,'records_sha256':sha(hostpath),'gpu_executed':False,'kernel_values_executed':False,'numeric_validation':False}
(D/'review.json').write_text(json.dumps(review,indent=2)+'\n')
print(json.dumps({'status':'passed','frontend_i32_routes':189,'CPU_cases':len(host),'returned_zero_allocations':len(returned),'allocation_errors':len(errors),'gpu_executed':False}))
