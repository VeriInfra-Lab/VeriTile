"""Focused source/contract/host checks; CPU arithmetic is not native GPU evidence."""
import ast
import contextlib
import hashlib
import importlib.util
from pathlib import Path
import re
from types import SimpleNamespace
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'l2_norm_triton2_repaired.py'
ORIGINAL = HERE.parents[2] / 'tritonbench_g/l2_norm_triton2/l2_norm_triton2.py'
shared_spec = importlib.util.spec_from_file_location('l21_checks', HERE.parent / 'l2_norm_triton1/check_contract.py')
shared = importlib.util.module_from_spec(shared_spec)
shared_spec.loader.exec_module(shared)
function, TensorFacade, Pointer, raw_storage = shared.function, shared.TensorFacade, shared.Pointer, shared.raw_storage
CHECKS = [
    ('x.layout != torch.strided or dy.layout != torch.strided', 'x.layout != .strided || dy.layout != .strided', 'layout : x.layout=.strided ∧ dy.layout=.strided'),
    ('x.dtype != torch.float32 or dy.dtype != torch.float32', 'x.dtype != .floating .fp32 || dy.dtype != .floating .fp32', 'dtype : x.dtype=.floating .fp32 ∧ dy.dtype=.floating .fp32'),
    ('not x.is_cuda or not dy.is_cuda', '!(x.isCuda && dy.isCuda)', 'cuda : x.device.kind="cuda" ∧ dy.device.kind="cuda"'),
    ('x.device != dy.device', 'x.device != dy.device', 'sameDevice : x.device=dy.device'),
    ('dy.ndim == 0', 'dy.shape=[]', 'rank : dy.shape≠[]'),
    ('N == 0 or N > 16384', 'dy.shape.getLastD 0=0 || dy.shape.getLastD 0>16384', 'width : 0<dy.shape.getLastD 0 ∧ dy.shape.getLastD 0≤16384'),
    ('x.numel() != dy.numel()', 'x.numel != dy.numel', 'sameCount : x.numel=dy.numel'),
    ('not x.is_contiguous() or not dy.is_contiguous()', '!(x.isContiguous && dy.isContiguous)', 'contiguous : x.isContiguous=Bool.true ∧ dy.isContiguous=Bool.true'),
    ('x.is_neg() or dy.is_neg()', 'x.isNeg || dy.isNeg', 'noNeg : x.isNeg=Bool.false ∧ dy.isNeg=Bool.false'),
    ('x.numel() > 0 and (x.data_ptr() % 16 != 0 or dy.data_ptr() % 16 != 0)', 'x.numel>0 && (x.dataPtr.toNat%16 != 0 || dy.dataPtr.toNat%16 != 0)', 'alignment : 0<x.numel → x.dataPtr.toNat%16=0 ∧ dy.dataPtr.toNat%16=0'),
]


def source_binding():
    assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == '386a7d3abb4c3960d6af1ff90cb29c0733c0ac6e9ed51319bd22309568947c12'
    for filename, name in [('Compilation.lean', 'candidateHash'), ('Forward.lean', 'sourceHash')]:
        text = (HERE / 'Proofs' / filename).read_text()
        pin = re.search(r'def '+name+r' : String := "([0-9a-f]+)"', text).group(1)
        assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == pin
    kernel_hashes = {
        '_l2_norm_fwd_1pass_kernel_repaired': '804494376c6f21285922e49baa9a22b01131e24e13d4db006b8a52ebeed16571',
        '_l2_norm_bwd_kernel_repaired': 'c2f73c9502094050e97310abe92454f86aa381417fd2b9698d7d437a60b6c313',
    }
    for name, pin in kernel_hashes.items():
        assert hashlib.sha256(ast.dump(function(SOURCE, name)).encode()).hexdigest() == pin
    assert ast.dump(function(SOURCE, '_l2_norm_fwd_repaired')) == ast.dump(function(shared.SOURCE, '_l2_norm_fwd_repaired'))
    wrapper = function(SOURCE, '_l2_norm_bwd_repaired')
    guards = [node.test for node in wrapper.body if isinstance(node, ast.If)]
    expected = [ast.parse(test, mode='eval').body for test in ['not isinstance(x, torch.Tensor) or not isinstance(dy, torch.Tensor)', *(r[0] for r in CHECKS)]]
    assert list(map(ast.dump, guards)) == list(map(ast.dump, expected))
    lean = (HERE / 'Contract.lean').read_text()
    contract = lean.split('structure InputContract', 1)[1].split('inductive InputError', 1)[0]
    checks = lean.split('def checkTensorInputs', 1)[1].split('def checkInputs', 1)[0]
    assert re.findall(r'\bif (.*?) then', checks) == [r[1] for r in CHECKS]
    for _, _, premise in CHECKS: assert premise in contract
    assert sum(isinstance(n, ast.Raise) for n in ast.walk(wrapper)) == 11
    calls = [n.func.attr for n in ast.walk(wrapper) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert all(n not in calls for n in ('clone', 'contiguous', 'resolve_neg', 'reshape', 'detach'))
    assert calls.count('view') == 3
    assert '.Proofs.' not in (HERE / 'Kernels.lean').read_text()
    assert ast.literal_eval(wrapper.args.defaults[0]) == 1e-5
    assert ast.literal_eval(function(SOURCE, '_l2_norm_fwd_repaired').args.defaults[0]) == 1e-6
    print('Both historical kernel ASTs unchanged; original and compiler hashes bound; 11 backward guards match Lean; forward wrapper equals L2-1.')


def interpreter(path, name):
    kernel, pid = shared.interpreter(path, name)
    kernel.__globals__['tl'].sum = lambda x, axis=0: x.sum(dim=axis)
    return kernel, pid


def backward_checks():
    spec = importlib.util.spec_from_file_location('l22_source', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    kernel, pid = interpreter(SOURCE, '_l2_norm_bwd_kernel_repaired')
    def empty(shape, dtype, device):
        assert dtype == torch.float32
        out = TensorFacade(torch.empty(shape, dtype=dtype))
        calls.append(('allocation', out))
        return out
    class Launch:
        def __getitem__(self, grid):
            def run(x, dy, dx, n, eps, block, **options):
                calls.append((grid, x, dy, dx, n, eps, block, options))
                for row in range(grid[0]):
                    pid[0] = row
                    kernel(Pointer(x.data), Pointer(dy.data), Pointer(dx.data), n, eps, block)
            return run
    module.torch = SimpleNamespace(Tensor=TensorFacade, strided=torch.strided, float32=torch.float32,
        empty=empty, cuda=SimpleNamespace(device=lambda _: contextlib.nullcontext()))
    module._l2_norm_bwd_kernel_repaired = Launch()
    base = torch.arange(160, dtype=torch.float32)
    good_one = TensorFacade(torch.ones(8))
    bad = []
    for side in (0, 1):
        unsupported = [(None, TypeError), (TensorFacade(torch.eye(2).to_sparse()), TypeError),
            (TensorFacade(torch.ones(8, dtype=torch.float64)), TypeError),
            (TensorFacade(torch.ones(8), cuda=False), ValueError),
            (TensorFacade(torch.ones(4, 2).T), ValueError),
            (TensorFacade(torch._neg_view(torch.ones(8))), ValueError)]
        for offset in (1, 2, 3): unsupported.append((TensorFacade(base[offset:offset+8]), ValueError))
        for item, error in unsupported:
            pair = [good_one, good_one]
            pair[side] = item
            bad.append((*pair, error))
    another_device = TensorFacade(torch.ones(8)); another_device.device = torch.device('cuda:1')
    bad += [(good_one, another_device, ValueError),
        (TensorFacade(torch.tensor(1.)), TensorFacade(torch.tensor(1.)), ValueError),
        (TensorFacade(torch.empty(0)), TensorFacade(torch.empty(2, 0)), ValueError),
        (TensorFacade(torch.ones(16385)), TensorFacade(torch.ones(1, 16385)), ValueError),
        (TensorFacade(torch.ones(7)), good_one, ValueError)]
    for x, dy, error in bad:
        calls.clear()
        try: module._l2_norm_bwd_repaired(x, dy)
        except error: pass
        else: raise AssertionError('unsupported backward arguments admitted')
        assert not calls
    byte_storage = torch.arange(80, dtype=torch.uint8).untyped_storage()[1:65]
    empty_unaligned = torch.empty(0, dtype=torch.float32).set_(byte_storage, 0, (0, 8), (8, 1))
    assert empty_unaligned.untyped_storage().data_ptr() % 16 == 1
    good = [(torch.randn(2, n), torch.randn(2, n), 1e-5) for n in (1, 3, 16, 32, 128, 129, 16384)]
    good += [(torch.randn(2, 3, 5), torch.randn(6, 5), 1e-5),
        (torch.randn(2, 6), torch.randn(3, 4), 1e-5),
        (torch.tensor(2.), torch.randn(1), 1e-5),
        (torch.empty(0, 3, 5), torch.empty(0, 8), 1e-5),
        (empty_unaligned, empty_unaligned, 1e-5),
        (base[4:36].view(4, 8), base[8:40].view(2, 16), 1e-5),
        (torch.as_strided(base, (1, 8), (99, 1), 4), torch.randn(8), 1e-5),
        (base[:16], base[:16], 1e-5),
        (torch.tensor([[float('nan'), -0., float('inf'), -float('inf')]]), torch.ones(1, 4), 1e-5)]
    good += [(torch.randn(2, 4), torch.randn(2, 4), eps) for eps in (-0., -1., float('nan'), float('inf'))]
    for x_data, dy_data, eps in good:
        calls.clear()
        before = [raw_storage(t).view(torch.int32).clone() for t in (x_data, dy_data)]
        x, dy = TensorFacade(x_data), TensorFacade(dy_data)
        out = module._l2_norm_bwd_repaired(x, dy, eps)
        n = dy_data.shape[-1]
        xx, dd = x_data.view(-1, n), dy_data.view(-1, n)
        var = (xx * xx).sum(-1, keepdim=True)
        rstd = 1 / torch.sqrt(var + eps)
        expected = (dd*rstd - (dd*xx).sum(-1, keepdim=True)*(1/(var+eps))*rstd*xx).view(x_data.shape)
        torch.testing.assert_close(out.data, expected, rtol=5e-6, atol=2e-6, equal_nan=True)
        assert out.shape == x.shape and out.data.is_contiguous()
        for original, snapshot in zip((x_data, dy_data), before):
            assert torch.equal(raw_storage(original).view(torch.int32), snapshot)
        allocation = calls[0][1]
        assert out.data.untyped_storage()._cdata == allocation.data.untyped_storage()._cdata
        assert all(out.data.untyped_storage()._cdata != t.untyped_storage()._cdata for t in (x_data, dy_data))
        if x_data.numel():
            assert len(calls) == 2
            grid, kx, kd, ky, width, got_eps, block, options = calls[1]
            for k, t in ((kx, x_data), (kd, dy_data)):
                assert k.data.data_ptr() == t.data_ptr()
                assert k.data.untyped_storage()._cdata == t.untyped_storage()._cdata
            assert ky is allocation and grid == (x_data.numel()//n,)
            assert width == n and block == 1 << (n-1).bit_length()
            assert options == {'num_warps': 4, 'num_ctas': 1}
        else: assert len(calls) == 1
    old, old_pid = interpreter(ORIGINAL, '_l2_norm_bwd_kernel')
    x = torch.ones(4, 8)[::2]; dy = torch.ones(2, 8); dx = torch.empty_like(x)
    old_pid[0] = 1
    try: old(Pointer(x), Pointer(dy), Pointer(dx), x.stride(0), 8, 1e-5, 8)
    except (AssertionError, IndexError): pass
    else: raise AssertionError('original shared-row-stride defect was not reproduced')
    print(f'Backward: {len(bad)} rejections and {len(good)} direct-view cases passed; DY grouping, scalar X, aliases, empty unaligned storage, and input frames checked.')
    print('Original shared-row-stride defect reproduced. CPU AST checks are structural, not native FMA/reduction conformance.')


if __name__ == '__main__':
    source_binding()
    shared.SOURCE, shared.ORIGINAL = SOURCE, ORIGINAL
    shared.host_checks()
    backward_checks()
