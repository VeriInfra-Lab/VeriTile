"""Native source paths, dynamic i32/u64 addresses, original wrapper, and ABI."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import ast
import ctypes
import hashlib
import json
import math
import re
import subprocess
import types

W = Path(__file__).resolve().parent
R = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
O = W / 'evidence/semantics'
O.mkdir(exist_ok=False)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
load = lambda p: json.loads(p.read_text())
native_ref = load(W / 'evidence/native-latest.json')
native = W / native_ref['receipt']
receipt = load(native)
assert sha(native) == native_ref['sha256']
for name, digest in receipt['artifacts'].items():
    assert sha(native.parent / name) == digest
checks = []
for record in receipt['records']:
    assert record['exit_code'] == 0
    row, name = record['result'], record['dtype']
    if row['status'] != 'compiled':
        assert ('not supported in this architecture' in row['exception'] if 'fnuz' in name
                else row['failure_stage'] == 'signature' and 'KeyError' in row['exception'])
        continue
    text = (native.parent / (name + '.raw.ttir')).read_text()
    assert row['signature']['n_elements'] == 'i32' and row['constexprs'] == {'(3,)': 64}
    assert text.count('tt.load') == 1 and text.count('tt.store') == 1 and text.count('math.cos') == 1
    assert text.index('tt.load') < text.index('math.cos') < text.index('tt.store')
    assert 'arith.cmpi slt' in text and 'tt.assert' not in text
    definitions = {m[1]: m[2] for m in re.finditer(r'^\s*(%[\w]+) = (.*)$', text, re.M)}
    stored = re.search(r'tt.store\s+%[\w]+,\s*(%[\w]+),', text)[1]
    chain = []
    current = stored
    while True:
        line = definitions[current]
        operation = line.split()[0]
        chain.append(operation)
        if operation == 'tt.load':
            break
        operands = re.findall(r'%[\w]+', line)
        assert operands, (name, line)
        current = operands[0]
    expected = (
        ['arith.fptosi', 'math.cos', 'arith.uitofp', 'arith.cmpi', 'tt.load'] if name == 'bool' else
        ['arith.fptosi', 'math.cos', 'arith.sitofp', 'tt.load'] if name.startswith('int') else
        ['arith.fptoui', 'math.cos', 'arith.uitofp', 'tt.load'] if name.startswith('uint') else
        ['arith.truncf', 'math.cos', 'arith.extf', 'tt.load'] if name in ['float16', 'bfloat16'] else
        ['arith.extf', 'math.cos', 'arith.truncf', 'tt.load'] if name == 'float64' else
        ['math.cos', 'tt.load'] if name == 'float32' else
        ['tt.fp_to_fp', 'math.cos', 'tt.fp_to_fp', 'tt.load'])
    assert chain == expected, (name, chain, expected)
    cosine = next(line for line in definitions.values() if line.startswith('math.cos'))
    assert 'tensor<64xf32>' in cosine
    if name.startswith('float8'):
        assert 'rounding = rtne' in definitions[stored]
    checks.append({'dtype': name, 'backward_store_to_load_operations': chain,
                   'cosine_precision': 'fp32', 'raw_ttir_sha256': sha(native.parent / (name + '.raw.ttir'))})
assert len(checks) == 15

c = O / 'arithmetic.c'
c.write_text('''#include <stdint.h>
uint32_t off(uint32_t block,uint32_t pid,uint32_t lane){return pid*block+lane;}
int mask(uint32_t n,uint32_t block,uint32_t pid,uint32_t lane){return (int32_t)off(block,pid,lane)<(int32_t)n;}
uint64_t pointer(uint64_t data,uint32_t width,uint32_t block,uint32_t pid,uint32_t lane){return data+(uint64_t)(int64_t)(int32_t)off(block,pid,lane)*width;}
''')
command = ['cc', '-shared', '-fPIC', '-O2', str(c), '-o', str(O / 'arithmetic.so')]
result = subprocess.run(command, capture_output=True, text=True)
(O / 'build.log').write_text(result.stdout + result.stderr)
assert result.returncode == 0
lib = ctypes.CDLL(str(O / 'arithmetic.so'))
lib.mask.argtypes = [ctypes.c_uint32] * 4
lib.mask.restype = ctypes.c_int
lib.pointer.argtypes = [ctypes.c_uint64] + [ctypes.c_uint32] * 4
lib.pointer.restype = ctypes.c_uint64
extractor = W / 'evidence/inherited/annotated-i32-scope-review/extract_i32.so'
driver = ctypes.PyDLL(str(extractor))
driver.extractI32.argtypes = [ctypes.c_void_p, ctypes.py_object]
driver.extractI32.restype = ctypes.c_bool
import triton
signed = lambda n: ((n + 2**31) % 2**32) - 2**31
extents = [0, 1, 2, 3, 4, 5, 4095, 4096, 4097, 2**31-1, 2**31, 2**31+1,
           2**32-1, 2**32, 2**32+1, 2**40, 2**40+1, 2**52+1, 2**52+2, 2**61, 2**63-1]
count = 0
abi_rows = []
for n in extents:
    block = triton.next_power_of_2(math.ceil(math.sqrt(n)))
    word = ctypes.c_int32()
    assert driver.extractI32(ctypes.byref(word), n) and word.value == signed(n)
    abi_rows.append({'N': n, 'i32': word.value, 'block': block,
                     'grid': triton.cdiv(n, block) if block else None})
    if block == 0:
        continue
    pids = {0, 1, (2**31 // block) - 1, 2**31 // block, 2**32 // block,
            max(0, triton.cdiv(n, block) - 1)}
    lanes = set(range(min(block, 64))) | {block-1, block//2}
    for pid in sorted(p for p in pids if 0 <= p < 2**32):
        for lane in sorted(lanes):
            offset = signed(pid * block + lane)
            assert bool(lib.mask(n % 2**32, block % 2**32, pid, lane)) == (offset < signed(n))
            for width in [1, 2, 4, 8]:
                for data in [0, 32, 2**33+32, 2**64-16]:
                    assert lib.pointer(data, width, block % 2**32, pid, lane) == (data + offset*width) % 2**64
                    count += 1

source = R / 'bench/tritonbench_g/cosine_compute/cosine_compute.py'
tree = ast.parse(source.read_text())
functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
wrapper = functions['cos']
traces = []
for n, allocation_error in [(n, False) for n in extents] + [(0, True), (2**63-1, True)]:
    events = []
    class Tensor:
        def __init__(self, name): self.name = name
        def numel(self): events.append(['numel', self.name, n]); return n
        @property
        def is_cuda(self): raise AssertionError('original wrapper must not query CUDA status')
        @property
        def device(self): raise AssertionError('original wrapper must not enter a device context')
    x, output = Tensor('A'), Tensor('B')
    def allocate(value):
        assert value is x
        events.append(['empty_like', value.name])
        if allocation_error: raise MemoryError('instrumented allocation failure')
        return output
    def sqrt(value): events.append(['sqrt', value]); return math.sqrt(value)
    def ceil(value): events.append(['ceil', value.hex()]); return math.ceil(value)
    def next_power(value): events.append(['next_power_of_2', value]); return triton.next_power_of_2(value)
    def cdiv(a, b): events.append(['cdiv', a, b]); return triton.cdiv(a, b)
    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                block = triton.next_power_of_2(math.ceil(math.sqrt(n)))
                assert args == (x, output, n, block) and kwargs == {}
                assert grid == (triton.cdiv(n, block), 1, 1)
                events.append(['launch', list(grid), n, block])
            return launch
    environment = {'torch': types.SimpleNamespace(empty_like=allocate), 'math': types.SimpleNamespace(sqrt=sqrt, ceil=ceil),
                   'triton': types.SimpleNamespace(next_power_of_2=next_power, cdiv=cdiv), 'cos_func': Kernel()}
    exec(compile(ast.Module(body=[wrapper], type_ignores=[]), str(source), 'exec'), environment)
    try:
        returned = environment['cos'](x)
        assert returned is output
        outcome = 'returned_same_allocation'
    except (MemoryError, ZeroDivisionError) as error:
        outcome = type(error).__name__
    expected = 'MemoryError' if allocation_error else 'ZeroDivisionError' if n == 0 else 'returned_same_allocation'
    assert outcome == expected
    assert [event[0] for event in events] == (['empty_like'] if allocation_error else
        ['empty_like', 'numel', 'sqrt', 'ceil', 'next_power_of_2', 'cdiv'] + ([] if n == 0 else ['launch']))
    traces.append({'N': n, 'allocation_error': allocation_error, 'events': events, 'outcome': outcome})

import torch
metadata = []
for tensor in [torch.empty((0, 2**31)), torch.empty((2, 0, 3)), torch.empty(()),
               torch.arange(64).reshape(8, 8).T, torch.arange(64)[7:19:2], torch.arange(64)[7:7]]:
    output = torch.empty_like(tensor)
    metadata.append({'shape': list(tensor.shape), 'numel': tensor.numel(), 'strides': list(tensor.stride()),
                     'storage_offset': tensor.storage_offset(), 'storage_bytes': tensor.untyped_storage().nbytes(),
                     'data_ptr_zero': tensor.data_ptr() == 0, 'output_shape': list(output.shape),
                     'output_strides': list(output.stride()), 'output_offset': output.storage_offset(),
                     'output_data_ptr_zero': output.data_ptr() == 0})
for name, data in [('source-paths.json', checks), ('abi-i32.json', abi_rows),
                   ('wrapper-traces.json', traces), ('metadata.json', metadata)]:
    (O / name).write_text(json.dumps(data, indent=2) + '\n')
summary = {'status': 'passed', 'source_sha256': sha(source), 'native_receipt': native_ref,
           'scalar_dataflow_profiles': len(checks), 'compiled_C_pointer_comparisons': count,
           'exact_i32_extractor_profiles': len(abi_rows), 'original_wrapper_AST_traces': len(traces),
           'real_CPU_metadata_examples': len(metadata), 'host_extents': extents,
           'GPU_launches': 0, 'large_allocations': 0, 'numeric_scalar_conformance': 'not_validated',
           'artifacts': {p.name: sha(p) for p in O.iterdir() if p.is_file()}}
(O / 'receipt.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k: v for k, v in summary.items() if k != 'artifacts'}))
