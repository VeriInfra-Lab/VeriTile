"""Execute unchanged source AST with a small CPU Triton-language adapter.
This validates address/control and explicit output casts, not native GPU MMA.
"""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import types
import torch

SOURCE = Path(__file__).resolve().parents[1] / 'iv_dependent_matmul.py'
text = SOURCE.read_text()
parsed = ast.parse(text)
original_kernel = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == 'iv_dependent_matmul_kernel')
original_host = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == 'iv_dependent_matmul_wrapper')
original_kernel.decorator_list = []

class Pointer:
    def __init__(self, tensor, offset=0, stats=None):
        self.tensor, self.offset, self.stats = tensor.reshape(-1), torch.as_tensor(offset), stats
    def __add__(self, offset):
        return Pointer(self.tensor, self.offset + offset, self.stats)
    __radd__ = __add__

class Language:
    constexpr = object
    float32, float16 = torch.float32, torch.float16
    pid = 0
    @staticmethod
    def cdiv(n, d): return (n + d - 1) // d
    def program_id(self, axis):
        assert axis == 0
        return self.pid
    @staticmethod
    def arange(start, stop): return torch.arange(start, stop)
    @staticmethod
    def zeros(shape, dtype): return torch.zeros(shape, dtype=dtype)
    @staticmethod
    def load(pointer, mask, other):
        offsets, mask = torch.broadcast_tensors(pointer.offset, mask)
        indices = offsets[mask]
        assert torch.all((indices >= 0) & (indices < pointer.tensor.numel()))
        values = torch.full(offsets.shape, other, dtype=pointer.tensor.dtype)
        values[mask] = pointer.tensor[indices]
        pointer.stats['read_addresses'].update(indices.tolist())
        return values
    @staticmethod
    def dot(a, b): return a @ b
    @staticmethod
    def store(pointer, values, mask):
        offsets, mask = torch.broadcast_tensors(pointer.offset, mask)
        indices = offsets[mask]
        assert torch.all((indices >= 0) & (indices < pointer.tensor.numel()))
        pointer.tensor[indices] = values[mask].to(pointer.tensor.dtype)
        pointer.stats['writes'].extend(indices.tolist())

class Factory:
    device = torch.device
    def __init__(self): self.inputs = []; self.output = None
    def rand(self, shape, device):
        assert device.type == 'cpu'
        # Valid self-generated FP32 samples, all in torch.rand's [0,1) domain.
        t = ((torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32) * 7 + len(self.inputs) * 3) % 16 / 16).reshape(shape)
        self.inputs.append(t)
        return t
    def empty(self, shape, device):
        backing = torch.full((shape[0]*shape[1] + 2,), 7.25, dtype=torch.float32)
        self.guards = backing
        self.output = backing[1:-1].reshape(shape)
        return self.output

tl = Language()
factory = Factory()
namespace = {'torch': factory, 'triton': types.SimpleNamespace(cdiv=tl.cdiv), 'tl': tl}
exec(compile(ast.Module(body=[original_kernel], type_ignores=[]), str(SOURCE), 'exec'), namespace)
source_kernel = namespace['iv_dependent_matmul_kernel']
class Launcher:
    def __getitem__(self, grid):
        def launch(a, b, c, *args, **meta):
            self.grid = grid(meta)
            self.meta = meta.copy()
            self.stats = [{'read_addresses': set(), 'writes': []} for _ in range(3)]
            pointers = [Pointer(t, stats=stats) for t, stats in zip((a,b,c), self.stats)]
            for pid in range(self.grid[0]):
                tl.pid = pid
                source_kernel(*pointers, *args, **{k:v for k,v in meta.items() if k != 'num_stages'})
        return launch
launcher = Launcher()
namespace['iv_dependent_matmul_kernel'] = launcher
exec(compile(ast.Module(body=[original_host], type_ignores=[]), str(SOURCE), 'exec'), namespace)
records=[]
for inner in [0, 1, 32, 33, 65]:
 for schedule in ['pre_load', 'post_load', 'post_pre_mixed', 'post_load_two_iters', 'post_load_three_iters']:
    factory.inputs=[]
    result=namespace['iv_dependent_matmul_wrapper'](35,inner,37,32,32,32,type=schedule,device='cpu')
    a,b=factory.inputs
    ideal=a.double() @ b.double()
    reference=ideal.to(torch.float16).to(torch.float32)
    assert torch.equal(result,reference), (inner,schedule,(result-reference).abs().max())
    assert result is factory.output and result.dtype == torch.float32
    assert factory.guards[0].item()==7.25 and factory.guards[-1].item()==7.25
    writes=launcher.stats[2]['writes']
    assert sorted(writes)==list(range(35*37))
    assert launcher.meta['num_stages']==(4 if schedule=='post_load_three_iters' else 3)
    records.append({'K':inner,'schedule':schedule,'shape':[35,37],'grid':list(launcher.grid),'exact_after_half_then_fp32':True,
                    'every_output_written_once':True,'all_reached_loads_bounded':True,'frame_guard_cells_preserved':True})
print(json.dumps({'task':'iv_dependent_matmul','source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
 'native_gpu_execution':False,'method':'unchanged kernel and wrapper AST executed with CPU pointer/language/launch adapters',
 'precision':'self-generated FP32 operands; source FP32 dot/add; explicit FP16 narrowing and FP32 store',
 'records':records},indent=2))
