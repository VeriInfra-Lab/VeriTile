"""Actual original/candidate AST, explicit CPU storage facade, diagnostic dot."""
from pathlib import Path
import ast
import copy
import hashlib
import inspect
import json
import os
import types
from contextlib import contextmanager
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import torch
torch.set_num_threads(1)
Q = Path(__file__).resolve().parent
REPAIR = Q/'matmul_persistent_triton_repaired.py'
ORIGINAL = Q.parents[2]/'tritonbench_g/matmul_persistent_triton/matmul_persistent_triton.py'
FP16, FP8 = 'fp16', 'e4m3fn'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def quantize(values, dtype):
    array = np.asarray(values, dtype=np.float32)
    if dtype == FP16: return array.astype(np.float16).astype(np.float32)
    if dtype == FP8:
        return torch.from_numpy(array.copy()).to(torch.float8_e4m3fn).float().numpy()
    return array

class Lanes(np.ndarray):
    def __new__(cls, value, dtype=None): return np.asarray(value, dtype=dtype).view(cls)
    def to(self, dtype):
        return Lanes(quantize(self, dtype)) if dtype in (FP16, FP8) else Lanes(self.astype(dtype))

class Tensor:
    def __init__(self, storage, shape, strides, dtype, offset=0, device='cuda:1'):
        self.storage, self.shape, self.strides, self.dtype = storage, shape, strides, dtype
        self.offset, self.device = offset, device
    ndim, layout, is_cuda = 2, 'strided', True
    negative, conjugate = False, False
    def stride(self, axis=None): return self.strides if axis is None else self.strides[axis]
    def numel(self): return self.shape[0]*self.shape[1]
    def element_size(self): return 2 if self.dtype==FP16 else 1
    def data_ptr(self): return getattr(self, 'pointer', 4096+self.offset*self.element_size())
    def is_neg(self): return self.negative
    def is_conj(self): return self.conjugate
    def zero_(self):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]): self.storage[self.offset+i*self.strides[0]+j*self.strides[1]] = 0
        return self
    def values(self):
        return np.array([[self.storage[self.offset+i*self.strides[0]+j*self.strides[1]] for j in range(self.shape[1])]
                         for i in range(self.shape[0])], dtype=np.float32).reshape(self.shape)

class Pointer:
    def __init__(self, runner, tensor, offsets=None):
        self.runner, self.tensor = runner, tensor
        self.offsets = np.asarray(tensor.offset if offsets is None else offsets, dtype=np.int64)
        self.dtype = types.SimpleNamespace(element_ty=tensor.dtype)
    def __add__(self, delta): return Pointer(self.runner, self.tensor, self.offsets+delta)
    __radd__ = __add__
    def access(self, mask, store=False):
        offsets, active = np.broadcast_arrays(self.offsets, np.asarray(mask, dtype=bool))
        # Check even masked formed final pointers for the repaired source.
        if self.runner.repaired:
            assert np.all((0 <= offsets) & (offsets < len(self.tensor.storage)))
        ids = offsets[active]
        assert np.all((0 <= ids) & (ids < len(self.tensor.storage)))
        self.runner.accesses += len(ids)
        if store:
            keys = [(id(self.tensor.storage), int(i)) for i in ids]
            assert len(set(keys)) == len(keys) and not set(keys) & self.runner.written
            self.runner.written.update(keys)
        return offsets, active

class Language:
    constexpr, int64, float32, float16, float8e4nv = int, np.int64, np.float32, FP16, FP8
    def __init__(self, runner): self.runner = runner
    def program_id(self, axis=0): return Lanes(self.runner.pid, dtype=np.int32)
    def cdiv(self, x, y): return (x+y-1)//y
    def arange(self, a, b): return Lanes(np.arange(a, b, dtype=np.int32))
    def zeros(self, shape, dtype): return Lanes(np.zeros(shape, dtype=dtype))
    def where(self, pred, yes, no): return Lanes(np.where(pred, yes, no))
    def multiple_of(self, value, _): return value
    def max_contiguous(self, value, width):
        if len(value) == width and (int(value[0]) % width != 0 or np.any(np.diff(value) != 1)):
            raise RuntimeError('false_full_block_contiguity_hint')
        return value
    def load(self, pointer, mask=True, other=0.):
        offsets, active = pointer.access(mask)
        output = np.full(offsets.shape, other, dtype=np.float32)
        output[active] = pointer.tensor.storage[offsets[active]]
        return Lanes(output)
    def dot(self, a, b, accumulator):
        self.runner.dots += 1
        return Lanes(np.asarray(a, np.float32) @ np.asarray(b, np.float32) + accumulator)
    def store(self, pointer, values, mask=True):
        offsets, active = pointer.access(mask, store=True)
        pointer.tensor.storage[offsets[active]] = np.broadcast_to(values, offsets.shape)[active]

class Runner:
    def __init__(self, path, sms):
        self.repaired, self.sms = path == REPAIR, sms
        self.pid, self.dots, self.accesses = 0, 0, 0
        self.written, self.launches, self.queries, self.allocations = set(), [], [], []
        self.current_device, self.launch_devices, self.fail_launch = 'cuda:0', [], False
        @contextmanager
        def device_context(device):
            previous = self.current_device
            self.current_device = device
            try: yield
            finally: self.current_device = previous
        def properties(device):
            self.queries.append(str(device)); return types.SimpleNamespace(multi_processor_count=sms)
        def empty(shape, device, dtype):
            output = Tensor(np.full(np.prod(shape), 37., np.float32), tuple(shape), (max(shape[1],1),1), dtype, device=device)
            self.allocations.append(output); return output
        host = types.SimpleNamespace(Tensor=Tensor, strided='strided', float16=FP16, float8_e4m3fn=FP8, empty=empty,
                                     cuda=types.SimpleNamespace(get_device_properties=properties,device=device_context))
        nodes = copy.deepcopy([n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)
                               and n.name in ['_matmul_launch_metadata','matmul_kernel_persistent','matmul_persistent']
                               or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_unchecked_matmul_persistent' for t in n.targets)])
        for node in nodes:
            if isinstance(node,ast.FunctionDef): node.decorator_list = []
        ns = {'torch':host,'triton':types.SimpleNamespace(cdiv=lambda x,y:(x+y-1)//y),'tl':Language(self)}
        exec(compile(ast.Module(nodes, []), str(path), 'exec'), ns)
        fn = ns['matmul_kernel_persistent']
        runner = self
        class Kernel:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    if runner.repaired:
                        assert all(value.device == runner.current_device for value in args if isinstance(value,Tensor))
                    runner.launch_devices.append(runner.current_device)
                    if runner.fail_launch: raise RuntimeError('mock submission failure')
                    options = {k:kwargs.pop(k) for k in ['num_stages','num_warps']}
                    count = grid(kwargs)[0]
                    runner.launches.append({'grid':[count],'constants':kwargs.copy(),'options':options})
                    bound = inspect.signature(fn).bind(*args, **kwargs)
                    for key,value in list(bound.arguments.items()):
                        if isinstance(value, Tensor): bound.arguments[key] = Pointer(runner,value)
                        elif isinstance(value,int) and key not in kwargs: bound.arguments[key] = Lanes(value,dtype=np.int64)
                    for pid in range(count):
                        runner.pid = pid; fn(*bound.args, **bound.kwargs)
                return launch
        ns['matmul_kernel_persistent'] = Kernel()
        self.call = ns['matmul_persistent']

def tensor(rows, cols, dtype, row_stride=None, col_stride=1, offset=3):
    row_stride = row_stride if row_stride is not None else max(cols*col_stride+2,1)
    capacity = offset + ((rows-1)*row_stride+(cols-1)*col_stride+1 if rows and cols else 0) + 5
    raw = np.full(capacity, 23., dtype=np.float32)
    t = Tensor(raw,(rows,cols),(row_stride,col_stride),dtype,offset)
    for i in range(rows):
        for j in range(cols): raw[offset+i*row_stride+j*col_stride] = ((i+2*j)%5-2)/8
    raw[:] = quantize(raw,dtype)
    return t

def run(name,m,n,k,dtype,sms=3,broadcast=False,alias=False):
    a = tensor(m,k,dtype,col_stride=2)
    b = tensor(k,n,dtype,row_stride=0 if broadcast else None)
    if alias:
        assert m == n == k
        b = Tensor(a.storage,(k,n),a.strides,dtype,a.offset)
    before_a, before_b = a.storage.copy(), b.storage.copy()
    expected = quantize(a.values().astype(np.float64) @ b.values().astype(np.float64),dtype)
    runner = Runner(REPAIR,sms); result = runner.call(a,b)
    assert result is runner.allocations[0] and result.storage is not a.storage and result.storage is not b.storage
    assert result.shape == (m,n) and result.dtype == dtype and result.device == a.device
    assert result.strides == (max(n,1),1)
    assert np.array_equal(result.values(),expected,equal_nan=True),name
    assert np.array_equal(a.storage,before_a) and np.array_equal(b.storage,before_b)
    assert runner.queries == [a.device]
    assert runner.current_device == 'cuda:0'
    assert runner.launch_devices == (['cuda:1'] if m*n*k else [])
    expected_dots = ((m+127)//128)*((n+255)//256)*((k+(64 if dtype==FP16 else 128)-1)//(64 if dtype==FP16 else 128)) if m*n*k else 0
    assert runner.dots == expected_dots
    if m*n*k: assert len(runner.written) == m*n
    else: assert not runner.launches
    print(name,'PASS',flush=True)
    return {'name':name,'M':m,'N':n,'K':k,'dtype':dtype,'SMS':sms,'dot_calls':runner.dots,
            'launches':runner.launches,'formed_and_active_accesses_checked':True,'fresh_exact_return':True,
            'full_input_storage_frame':True,'offsets_and_strides':[a.offset,list(a.strides),b.offset,list(b.strides)]}

def main():
    cases=[]
    for dtype in [FP16,FP8]:
        for m,n,k,sms in [(1,1,1,132),(129,257,129,132),(257,513,65,2),(1025,257,3,3),(128,256,64,1),
                          (2,3,0,132),(0,3,4,132),(2,0,4,132),(0,0,0,132)]:
            cases.append(run(f'{dtype}_{m}_{n}_{k}_{sms}',m,n,k,dtype,sms))
        cases.append(run(dtype+'_broadcast_B',3,4,5,dtype,broadcast=True))
        cases.append(run(dtype+'_aliased_AB',3,3,3,dtype,alias=True))
    original=[]
    for m,n,k in [(129,257,129),(2,3,0)]:
        runner=Runner(ORIGINAL,132)
        try:
            c=runner.call(tensor(m,k,FP16),tensor(k,n,FP16))
            assert k==0 and np.all(c.values()==37)
            original.append({'shape':[m,n,k],'finding':'K0 retains sentinel allocator cells','native_value_claimed':False})
        except RuntimeError as error:
            assert str(error)=='false_full_block_contiguity_hint'
            original.append({'shape':[m,n,k],'finding':str(error),'native_value_claimed':False})
    unsupported=Tensor(np.zeros(0,np.float32),(0,0),(1,1),'fp32')
    try: Runner(REPAIR,132).call(unsupported,unsupported)
    except TypeError: rejected=True
    else: rejected=False
    assert rejected
    report={'status':'PASS','candidate_sha256':sha(REPAIR),'original_sha256':sha(ORIGINAL),
        'cases':cases,'case_count':len(cases),'original_findings':original,'unsupported_FP32_rejected_even_empty':True,
        'kind':'bounded actual production AST CPU facade with NumPy diagnostic dot and Torch output quantization',
        'GPU_execution':False,'native_numerical_conformance':False,'compiler_hint_miscompilation_claimed':False}
    rejects=0
    for attr,value in [('ndim',1),('layout','sparse'),('is_cuda',False),('negative',True),('conjugate',True),('strides',(-1,1)),('pointer',4097),('device','cuda:7')]:
        a,b=tensor(3,5,FP16),tensor(5,4,FP16)
        setattr(a,attr,value)
        runner=Runner(REPAIR,3)
        try: runner.call(a,b)
        except (ValueError,TypeError): rejects+=1
        else: raise AssertionError(attr)
        assert not runner.allocations and not runner.launches and not runner.queries
    # Ordinary empty-view API data_ptr is zero even beyond the backing allocation.
    for dtype in [torch.float16,torch.float8_e4m3fn]:
        view=torch.as_strided(torch.empty((1,),dtype=dtype),(2,0),(7,2),storage_offset=2**62)
        assert view.numel()==0 and view.data_ptr()==0
    runner=Runner(REPAIR,3)
    runner.fail_launch=True
    try: runner.call(tensor(3,5,FP16),tensor(5,4,FP16))
    except RuntimeError as error: assert str(error)=='mock submission failure'
    else: raise AssertionError('expected mock submission failure')
    assert runner.current_device=='cuda:0' and runner.launch_devices==['cuda:1'] and not runner.written
    print(f'PASS: {len(cases)} actual source CPU executions, both dtype profiles, non-current device, offsets/broadcast/aliases/empty/K0, {rejects+1} guard rejections. No GPU execution.')
    print('PASS: actual submissions select cuda:1 and restore ambient cuda:0 on success and submission exception; empty/K0 skip launch.')

if __name__=='__main__':main()
