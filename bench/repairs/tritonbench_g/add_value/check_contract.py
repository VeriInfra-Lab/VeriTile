"""Targeted source/host checks using CPU storage and mocked CUDA launches."""
import ast
import contextlib
import importlib.util
from pathlib import Path
import re
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'add_value_repaired.py'
ORIGINAL = HERE.parents[2] / 'tritonbench_g/add_value/add_value.py'
KERNELS = ('puzzle1_kernel', 'puzzle1_commuted_kernel')

# Reuse the existing actual-Tensor test helpers; no production wrapper is reused.
spec = importlib.util.spec_from_file_location('add_example_test_helpers', HERE.parent / 'add_example/check_contract.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
TensorFacade, Pointer = helpers.TensorFacade, helpers.Pointer
raw_storage, unaligned_tensor = helpers.raw_storage, helpers.unaligned_tensor


def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)


CHECKS = [
    ('x.layout != torch.strided', 'x.layout != .strided', 'layout : x.layout = .strided'),
    ('x.dtype != torch.float32', 'x.dtype != .floating .fp32', 'dtype : x.dtype = .floating .fp32'),
    ('not x.is_cuda', '!x.isCuda', 'cuda : x.device.kind = "cuda"'),
    ('not x.is_contiguous()', '!x.isContiguous', 'contiguous : x.isContiguous = Bool.true'),
    ('x.is_neg()', 'x.isNeg', 'noNeg : x.isNeg = Bool.false'),
    ('N >= 2**31', 'x.numel >= 2^31', 'extent : x.numel < 2^31'),
    ('N > 0 and x.data_ptr() % 4 != 0', 'x.numel > 0 && x.dataPtr.toNat % 4 != 0',
     'alignment : 0 < x.numel → x.dataPtr.toNat % 4 = 0'),
]


def check_source():
    # The kernel change is exactly binding the original value parameter to 10
    # and removing that parameter; all remaining instructions stay identical.
    expected_kernel = function(ORIGINAL, 'puzzle1_kernel')
    assert expected_kernel.args.args.pop().arg == 'value'
    addition = next(n for n in expected_kernel.body if isinstance(n, ast.Assign)
                    and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'output')
    assert ast.dump(addition.value) == ast.dump(ast.parse('x + value', mode='eval').body)
    addition.value = ast.parse('x + 10', mode='eval').body
    assert ast.dump(function(SOURCE, 'puzzle1_kernel')) == ast.dump(expected_kernel)
    # The candidate adds one local constant and reverses only the add operands.
    expected_kernel.name = 'puzzle1_commuted_kernel'
    addition_index = expected_kernel.body.index(addition)
    expected_kernel.body[addition_index:addition_index+1] = ast.parse('y = 10\noutput = y + x').body
    assert ast.dump(function(SOURCE, 'puzzle1_commuted_kernel')) == ast.dump(expected_kernel)
    wrapper = function(SOURCE, 'puzzle1_repaired')
    guards = [n.test for n in wrapper.body if isinstance(n, ast.If)]
    expected = ['not isinstance(x, torch.Tensor)', *(r[0] for r in CHECKS), 'N == 0']
    assert list(map(ast.dump, guards)) == [ast.dump(ast.parse(e, mode='eval').body) for e in expected]
    assert sum(isinstance(n, ast.Raise) for n in ast.walk(wrapper)) == 8
    lean = (HERE / 'Contract.lean').read_text()
    contract = lean.split('structure InputContract', 1)[1].split('inductive InputError', 1)[0]
    checks = lean.split('def checkTensorInput', 1)[1].split('def checkInput', 1)[0]
    assert re.findall(r'\bif (.*?) then', checks) == [r[1] for r in CHECKS]
    assert all(r[2] in contract for r in CHECKS)
    assert '| .tensor x => checkTensorInput x' in lean
    assert '| .nonTensor => .error .tensor' in lean
    launches = [n for n in ast.walk(wrapper)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Subscript)]
    expected_launch = ast.parse('puzzle1_kernel[grid](x, output, N, BLOCK_SIZE=1024)', mode='eval').body
    assert [ast.dump(n) for n in launches] == [ast.dump(expected_launch)]
    source = SOURCE.read_text()
    assert 'output = torch.empty_like(x)' in source
    assert 'with torch.cuda.device(x.device)' in source
    assert not re.search(r'\.(contiguous|clone|resolve_neg|view)\(', source)
    pending, visited = [HERE / 'Kernels.lean'], set()
    while pending:
        path = pending.pop()
        if path in visited: continue
        visited.add(path)
        for line in path.read_text().splitlines():
            if line.startswith('import '):
                for module in line.split()[1:]:
                    assert '.Proofs.' not in module
                    if module.startswith('bench.'):
                        pending.append(HERE.parents[3] / (module.replace('.', '/') + '.lean'))
    print('Exact x + 10 and local-y commuted kernel ASTs, eight guards, Lean domain, original-only launch, source-only imports passed.')


def interpreter(name):
    pid = [0]
    def load(pointer, mask):
        values = torch.zeros_like(pointer.offsets, dtype=torch.float32)
        indices = pointer.tensor.storage_offset() + pointer.offsets[mask].to(torch.int64)
        values[mask] = raw_storage(pointer.tensor)[indices]
        return values
    def store(pointer, values, mask):
        indices = pointer.tensor.storage_offset() + pointer.offsets[mask].to(torch.int64)
        raw_storage(pointer.tensor)[indices] = values[mask]
    kernel = function(SOURCE, name)
    kernel.decorator_list = []
    for a in kernel.args.args: a.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[kernel], type_ignores=[]))
    ns = {'tl': SimpleNamespace(program_id=lambda axis: pid[0],
          arange=lambda lo, hi: torch.arange(lo, hi, dtype=torch.int32), load=load, store=store)}
    exec(compile(module, str(SOURCE), 'exec'), ns)
    return ns[name], pid


def signed32(n):
    return ((n + 2**31) % 2**32) - 2**31


def check_counterexamples(kernel, pid):
    backing = torch.tensor([1., 70., 2., 80., 3., 90.])
    x = backing[::2]
    out = torch.empty_like(x)
    pid[0] = 0
    kernel(Pointer(x), Pointer(out), 3, 1024)
    assert out.tolist() == [11., 80., 12.]
    assert not torch.equal(out, x + 10)
    neg = torch._neg_view(torch.tensor([1., 2., 3.]))
    kernel(Pointer(neg), Pointer(out), 3, 1024)
    assert out.tolist() == [11., 12., 13.] and not torch.equal(out, neg + 10)
    n = 2**31
    assert (n + 1023)//1024 == 2**21 and signed32(n) == -2**31
    for program in (0, 1, 2**21-1):
        offsets = program * 1024 + torch.arange(1024, dtype=torch.int32)
        assert not (offsets < signed32(n)).any()
    n = 2**31 - 1
    offsets = ((n+1023)//1024-1)*1024 + torch.arange(1024, dtype=torch.int32)
    assert offsets.min() >= 0 and offsets.max() == 2**31-1
    assert (offsets < n).sum() == 1023
    print('Original stride2, negative-view and int32-boundary counterexamples retained; last admitted tail checked.')


def check_host(kernel, pid):
    spec = importlib.util.spec_from_file_location('add_value_contract_review', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls, allocated = [], []
    def empty_like(x):
        calls.append('allocation')
        out = TensorFacade(torch.empty_like(x.data), str(x.device))
        allocated.append(out)
        return out
    class Launch:
        def __getitem__(self, grid):
            def run(x, out, n, BLOCK_SIZE):
                assert BLOCK_SIZE == 1024
                count = grid({'BLOCK_SIZE': BLOCK_SIZE})[0]
                assert count == (n+1023)//1024
                calls.append(('launch', x, out, n, count))
                if x.data.device.type != 'meta':
                    # Reverse complete program order exercises fresh-output independence.
                    for p in reversed(range(count)):
                        pid[0] = p
                        kernel(Pointer(x.data), Pointer(out.data), signed32(n), BLOCK_SIZE)
            return run
    module.torch = SimpleNamespace(Tensor=TensorFacade, strided=torch.strided,
        float32=torch.float32, empty_like=empty_like,
        cuda=SimpleNamespace(device=lambda _: contextlib.nullcontext()))
    module.puzzle1_kernel = Launch()
    bad = [([], TypeError), (TensorFacade(torch.eye(3).to_sparse()), ValueError),
        (TensorFacade(torch.ones(3, dtype=torch.float64)), TypeError),
        (TensorFacade(torch.ones(3), device='cpu', cuda=False), ValueError),
        (TensorFacade(torch.ones(6)[::2]), ValueError),
        (TensorFacade(torch.ones(2, 3).T), ValueError),
        (TensorFacade(torch._neg_view(torch.ones(3))), ValueError),
        (TensorFacade(torch.empty(2**31, device='meta')), ValueError)]
    bad += [(TensorFacade(unaligned_tensor(k)), ValueError) for k in (1,2,3)]
    for x, error in bad:
        calls.clear()
        try: module.puzzle1_repaired(x)
        except error: pass
        else: raise AssertionError('unsupported input admitted')
        assert not calls, 'rejection must precede allocation'
    backing = torch.arange(32, dtype=torch.float32)
    words = torch.tensor([0, -2147483648, 1, 8388607, 8388608, 2139095039,
                          -8388609, 2139095040, -8388608, 2143289345], dtype=torch.int32).view(torch.float32)
    good = [torch.arange(n, dtype=torch.float32) for n in (1, 1023, 1024, 1025, 2049)]
    good += [torch.tensor(-2.), torch.ones(2,3), torch.empty(2,0), backing[1:6],
        torch.as_strided(backing, (1,4), (99,1), 2),
        torch.as_strided(backing, (0,), (1,), 1000), unaligned_tensor(1, empty=True), words]
    for data in good:
        calls.clear()
        x = TensorFacade(data)
        before = data.untyped_storage().clone()
        out = module.puzzle1_repaired(x)
        assert out is allocated[-1] and out is not x
        assert out.shape == x.shape and out.dtype == x.dtype and out.device == x.device
        assert out.is_contiguous() and not out.is_neg()
        torch.testing.assert_close(out.data, data + 10, rtol=0, atol=0, equal_nan=True)
        assert list(before) == list(data.untyped_storage()), 'input storage changed'
        launches = [c for c in calls if isinstance(c, tuple)]
        assert len(launches) == (1 if data.numel() else 0)
        if launches: assert launches[0][1] is x and launches[0][2] is out
    # Admission/one-launch boundary is metadata-only; no huge allocation or GPU work.
    class Huge(TensorFacade):
        def data_ptr(self): return 4096
    huge = Huge(torch.empty(2**31-1, device='meta'))
    calls.clear()
    out = module.puzzle1_repaired(huge)
    assert out is allocated[-1] and calls[-1][3:] == (2**31-1, 2**21)
    print(f'{len(bad)} rejection cases and {len(good)+1} admission cases passed; huge admission is metadata-only.')


if __name__ == '__main__':
    check_source()
    helpers.check_tensor_semantics()
    for name in KERNELS:
        kernel, pid = interpreter(name)
        check_counterexamples(kernel, pid)
        # Reuse the launch mock for both ASTs; the production wrapper remains
        # bound to puzzle1_kernel, as checked above.
        check_host(kernel, pid)
        print(f'{name}: CPU AST interpretation and checked-host cases passed.')
    print(f'CUDA available: {torch.cuda.is_available()}; no GPU execution or numerical conformance claimed.')
