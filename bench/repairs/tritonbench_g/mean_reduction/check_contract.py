"""Guard/source checks and a CPU storage replay with mocked CUDA launch."""
import ast
import contextlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'mean_reduction_repaired.py'
ORIGINAL = HERE.parents[2] / 'tritonbench_g/mean_reduction/mean_reduction.py'

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

shared = load(HERE.parent / 'add_example/check_contract.py', 'mean_test_storage')
function, raw_storage = shared.function, shared.raw_storage

class Tensor(shared.TensorFacade):
    @property
    def ndim(self): return self.data.ndim
    def stride(self): return self.data.stride()
    def permute(self, order): return Tensor(self.data.permute(order), str(self.device))
    def contiguous(self): return Tensor(self.data.contiguous(), str(self.device))
    def squeeze(self, dims): return Tensor(self.data.squeeze(dims), str(self.device))

class Pointer:
    def __init__(self, tensor, offsets=0): self.tensor, self.offsets = tensor, offsets
    def __add__(self, offsets): return Pointer(self.tensor, self.offsets + offsets)

class LogicalAnd(ast.NodeTransformer):
    def visit_BoolOp(self, node):
        self.generic_visit(node)
        assert isinstance(node.op, ast.And) and len(node.values) == 2
        return ast.copy_location(ast.BinOp(node.values[0], ast.BitAnd(), node.values[1]), node)

def source_checks():
    for name in ('mean_dim_kernel', 'dim_compress'):
        assert ast.dump(function(SOURCE, name)) == ast.dump(function(ORIGINAL, name))
    original_kernel = function(SOURCE, 'mean_dim_kernel')
    candidate_kernel = function(SOURCE, 'mean_dim_kernel_index_commuted')
    candidate_kernel.name = original_kernel.name
    first = candidate_kernel.body[0].value.left
    assert isinstance(first, ast.BinOp) and isinstance(first.op, ast.Mult)
    first.left, first.right = first.right, first.left
    assert ast.dump(original_kernel) == ast.dump(candidate_kernel)
    old, new = function(ORIGINAL, 'mean_dim'), function(SOURCE, 'mean_dim')
    assert [ast.dump(n) for n in old.body] == [ast.dump(n) for n in new.body[-len(old.body):]]
    checks = [ast.unparse(n.test) for n in new.body[:-len(old.body)] if isinstance(n, ast.If)]
    expected = ['not isinstance(x, torch.Tensor)', 'x.layout != torch.strided',
        'x.dtype != torch.float32', 'dtype is not None and dtype != torch.float32',
        'not x.is_cuda', 'x.is_neg()', 'x.numel() > 0 and x.data_ptr() % 4 != 0',
        'isinstance(dim, int)', 'x.ndim == 0 and checked_dims',
        'len(set(checked_dims)) != len(checked_dims)', 'checked_n == 0',
        'padded_m >= 2 ** 31 or padded_n >= 2 ** 31',
        'checked_m > 0 and (padded_m - 1) * checked_n >= 2 ** 31']
    assert checks == expected
    lean = (HERE / 'Contract.lean').read_text()
    for fragment in ['dims.Nodup', 'columns shape dims = 0', 'metadata.isNeg',
            '((rows shape dims+7)/8)*8 >= 2^31', '((columns shape dims+7)/8)*8 >= 2^31',
            '(((rows shape dims+7)/8)*8-1)*columns shape dims >= 2^31']:
        assert fragment in lean
    assert '.Proofs.' not in (HERE / 'Kernels.lean').read_text()
    assert 'FPEquiv' not in (HERE / 'Correct.lean').read_text()
    assert not any(line.startswith('import ') and line.endswith('.Correct')
                   for line in (HERE / 'FPEquiv.lean').read_text().splitlines())

def kernel_interpreter(name="mean_dim_kernel"):
    pid = [0]
    def load_word(p, mask, other):
        offsets, mask = torch.broadcast_tensors(p.offsets, mask)
        out = torch.full_like(offsets, other, dtype=torch.float32)
        out[mask] = raw_storage(p.tensor)[p.tensor.storage_offset() + offsets[mask]]
        return out
    def store_word(p, value, mask):
        offsets, value, mask = torch.broadcast_tensors(p.offsets, value, mask)
        raw_storage(p.tensor)[p.tensor.storage_offset() + offsets[mask]] = value[mask]
    node = function(SOURCE, name)
    node.decorator_list = []
    for arg in node.args.args: arg.annotation = None
    node = LogicalAnd().visit(node)
    ns = {'tl': SimpleNamespace(program_id=lambda _: pid[0], arange=torch.arange,
        zeros=torch.zeros, float32=torch.float32, load=load_word, store=store_word, sum=torch.sum)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(SOURCE), 'exec'), ns)
    return ns[name], pid

def run_checks():
    module = load(SOURCE, 'mean_contract_review')
    kernel, pid = kernel_interpreter()
    candidate, candidate_pid = kernel_interpreter("mean_dim_kernel_index_commuted")
    calls = []
    def empty(shape, dtype, device):
        calls.append(('allocate', tuple(shape)))
        return Tensor(torch.empty(shape, dtype=dtype), str(device))
    class Launch:
        def __getitem__(self, grid):
            def run(x, out, m, n, **kw):
                dims = grid(kw)
                assert kw == {'BLOCK_M': 8, 'BLOCK_N': 8} and dims == ((m + 7)//8,)
                calls.append(('launch', x, m, n))
                for program in reversed(range(dims[0])):
                    pid[0] = program
                    kernel(Pointer(x.data), Pointer(out.data), m, n, 8, 8)
                candidate_output = torch.empty_like(out.data)
                for program in range(dims[0]):
                    candidate_pid[0] = program
                    candidate(Pointer(x.data), Pointer(candidate_output), m, n, 8, 8)
                assert torch.equal(out.data.reshape(-1).view(torch.int32),
                                   candidate_output.reshape(-1).view(torch.int32))
            return run
    module.torch = SimpleNamespace(Tensor=Tensor, strided=torch.strided, float32=torch.float32,
        empty=empty, cuda=SimpleNamespace(device=lambda _: contextlib.nullcontext()))
    module.mean_dim_kernel = Launch()
    rejected = [([], 0, None, TypeError), (Tensor(torch.eye(2).to_sparse()), 0, None, ValueError),
        (Tensor(torch.ones(2, dtype=torch.float64)), 0, None, TypeError),
        (Tensor(torch.ones(2)), 0, torch.float64, TypeError),
        (Tensor(torch.ones(2), 'cpu', False), 0, None, ValueError),
        (Tensor(torch._neg_view(torch.ones(2))), 0, None, ValueError),
        (Tensor(shared.unaligned_tensor(1)), 0, None, ValueError),
        (Tensor(torch.ones(2)), (0,), None, TypeError),
        (Tensor(torch.ones(2)), [0.0], None, TypeError),
        (Tensor(torch.tensor(1.)), 0, None, ValueError),
        (Tensor(torch.ones(2, 3)), [0, 2], None, ValueError),
        (Tensor(torch.empty(2, 0)), 1, None, ValueError)]
    # Meta inputs exercise host extent/product guards without allocating storage.
    class MetaTensor(Tensor):
        def data_ptr(self): return 4
    rejected += [(MetaTensor(torch.empty(shape, device='meta')), axes, None, ValueError)
        for shape, axes in [((2**31,), 0), ((2**31,), []), ((2**28, 9), 1)]]
    for x, axes, dtype, error in rejected:
        calls.clear()
        try: module.mean_dim(x, axes, dtype=dtype)
        except error: pass
        else: raise AssertionError('unsupported input admitted')
        assert not calls
    good = [(torch.arange(3*5*7, dtype=torch.float32).reshape(3,5,7), [0,2]),
        (torch.arange(3*5*7, dtype=torch.float32).reshape(3,5,7).transpose(0,2), [0,2]),
        (torch.arange(35, dtype=torch.float32).reshape(5,7)[:,::2], -1),
        (torch.arange(70, dtype=torch.float32).reshape(7,10)[1:,1::2], -1),
        (torch.arange(5, dtype=torch.float32).expand(9,5), 0),
        (torch.tensor(-3.), []), (torch.empty(0,9), 1),
        (torch.empty(2,0,9), [2]), (torch.ones(2,3), []),
        (torch.tensor([float('nan'),float('inf'),-0.0]), 0)]
    good += [(torch.arange(m*n, dtype=torch.float32).reshape(m,n)/8, 1)
        for m in (1,7,8,9) for n in (1,7,8,9,17)]
    for x, axes in good:
        for keepdim in (False, True):
            calls.clear()
            before = x.clone()
            result = module.mean_dim(Tensor(x), axes, keepdim=keepdim)
            dims = [axes] if isinstance(axes,int) else axes
            dims = [d % x.ndim for d in dims]
            expected = x.mean(tuple(dims), keepdim=keepdim) if dims else x.clone()
            torch.testing.assert_close(result.data, expected, rtol=1e-6, atol=1e-6, equal_nan=True)
            torch.testing.assert_close(x, before, rtol=0, atol=0, equal_nan=True)
            assert result.dtype == torch.float32 and result.device == torch.device('cuda:0')
            assert len(calls) == 2
            # The original intentional compression must follow descending stride.
            order = [i for i in range(x.ndim) if i not in dims] + sorted(dims, key=lambda d:x.stride(d), reverse=True)
            torch.testing.assert_close(calls[1][1].data, x.permute(order).contiguous(), equal_nan=True)
    negative = torch._neg_view(torch.tensor([1.,2.,3.]))
    assert negative.contiguous().is_neg()
    assert raw_storage(negative).mean().item() != negative.mean().item()
    return len(rejected), 2*len(good)

if __name__ == '__main__':
    source_checks()
    rejected, admitted = run_checks()
    print(f'Passed original and integer-index candidate source binding, {rejected} rejections, {admitted} admissions per variant, exact word equality, and negative-view counterexample.')
    print('CPU storage replay with mocked CUDA; no GPU execution or numerical-transformation experiment claimed.')
