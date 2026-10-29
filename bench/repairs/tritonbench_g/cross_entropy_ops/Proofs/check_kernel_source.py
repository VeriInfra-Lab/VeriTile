"""Pinned source inventory and kernel-only CPU controls; no GPU/FP conformance claim."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import warnings
sys.dont_write_bytecode = True
D = Path(__file__).resolve().parents[1]
ROOT = D.parents[3]
spec = importlib.util.spec_from_file_location('ceops_kernel_controls', D.parent / 'cross_entropy1/Proofs/check_kernel.py')
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)
torch = support.torch
selected = D / 'cross_entropy_ops_repaired.py'
resident = D / 'cross_entropy_ops_conditional.py'
original = ROOT / 'bench/tritonbench_g/cross_entropy_ops/cross_entropy_ops.py'
pins = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (selected, resident, original)}
assert list(pins.values()) == [
    '5f106a09beaeb4807b9a8f6409e0f015478be518e8a31d9ca889a32f9cd7e768',
    'b690a712bd968686ee78818e81dbbb21a8d15331e132eb1c994d1ed3eb2b4ba5',
    'a06de9e0f3d5fa4d785de59bcf988d2945ec20c98e47a05cdc358e6f504aa11a',
]
nodes = [n for n in ast.parse(resident.read_text()).body if isinstance(n, ast.FunctionDef) and n.name.endswith('_kernel')]
assert [len(n.body) for n in nodes] == [14, 16]
lean = (D / 'Kernels.lean').read_text()
sys.path.insert(0, str(ROOT))
from bench.jit_correspondence import check_manifest
pairs = check_manifest(ROOT, D / 'jit-correspondence.json')
assert [pair['statements'] for pair in pairs] == [14, 14, 16, 16]
assert all(pair['mode'] == 'pasted_body' for pair in pairs)
for expression in ('smoothing * sum_logits / total_classes',
                   'probs += 2.0 * lse_square_scale * lse * probs',
                   'z_loss = lse_square_scale * lse * lse',
                   '(dloss * logit_scale) * probs'):
    assert expression in lean
for field in ('logitScale', 'smoothing', 'squareScale'):
    assert '$((p.' + field + ' : ℝ))' not in lean
old = support.load_kernels(original, 'ceops_original_control')
new = support.load_kernels(resident, 'ceops_selected_control')
results = []
for scale in (0., -2.):
    x = torch.tensor([[0., 1., 2.]])
    y = torch.tensor([1], dtype=torch.int64)
    outputs = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for module in (old, new):
            loss, lse, zloss = (torch.full((1,), 79.) for _ in range(3))
            module.cross_entropy_fwd_kernel[(1, 1)](
                loss, lse, zloss, x, y, .25, scale, .125, -100, 3, 0, 3, 1, 3,
                BLOCK_SIZE=4, SPLIT=False)
            outputs.append((loss, lse, zloss))
    assert torch.isnan(outputs[0][0]).all() and torch.isnan(outputs[0][1]).all()
    u = x.double() * scale
    lse = torch.logsumexp(u, 1)
    zloss = .125 * lse.square()
    loss = lse - .75 * u[:, 1] - .25 * u.mean(1) + zloss
    for got, expected in zip(outputs[1], (loss, lse, zloss)):
        support.close(got, expected)
    results.append({'original_padding_scale': scale, 'original': 'NaN counterexample reproduced', 'repair': 'all three outputs passed'})
for module, repaired in ((old, False), (new, True)):
    x = torch.tensor([[float('nan'), 1.]])
    y = torch.tensor([-100], dtype=torch.int64)
    lse = torch.tensor([float('nan')])
    upstream = torch.ones(1)
    out = torch.full_like(x, 79.)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        module.cross_entropy_bwd_kernel[(1, 1)](
            out, upstream, x, lse, y, 0., 1., 0., -100, 2, 0, 2, 2, 2, 1, BLOCK_SIZE=2)
    assert torch.equal(out.view(torch.int32), torch.zeros_like(out, dtype=torch.int32)) if repaired else torch.isnan(out).all()
results.append({'original_ignored_NaN': 'counterexample reproduced', 'repair': 'exact zero words'})
print(json.dumps({'pins': pins, 'statement_counts': [14, 16], 'controls': results,
                  'gpu_runs': [], 'native_numerical_conformance': 'not claimed',
                  'kernel_source_statements': {n.name: [ast.unparse(x) for x in n.body] for n in nodes}}, indent=2))
