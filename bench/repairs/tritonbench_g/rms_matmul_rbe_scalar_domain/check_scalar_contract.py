"""Exact candidate guard/wrapper controls and immutable JIT-AST comparison."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import math

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = HERE / 'rms_matmul_rbe_repaired.py'
ORIGINAL = HERE.parent / 'rms_matmul_rbe/rms_matmul_rbe_repaired.py'
EVIDENCE = ROOT / '.veritile-cache/kernel-evidence/rms_matmul_rbe_scalar_domain'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def functions(path):
    return {n.name: n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}


def main():
    original, candidate = functions(ORIGINAL), functions(SOURCE)
    assert original.keys() == candidate.keys()
    unchanged = []
    for name in original:
        if name != '_checked_scalars':
            assert ast.dump(original[name]) == ast.dump(candidate[name]), name
            unchanged.append(name)
    # The exact original checker is an unchanged prefix; only the two explicit
    # dispatch guards follow its original finite-positive scalar loop.
    assert ast.dump(ast.Module(body=original['_checked_scalars'].body, type_ignores=[])) == ast.dump(
        ast.Module(body=candidate['_checked_scalars'].body[:-2], type_ignores=[]))
    mod = module('rms_candidate_scalar_contract', SOURCE)
    h = module('rms_retained_scalar_controls', ORIGINAL.with_name('check_source.py'))
    ns, _ = h.environment(SOURCE)
    import torch

    class Capture:
        def __init__(self):
            self.calls = []

        def __getitem__(self, grid):
            def call(**kwargs):
                self.calls.append((grid, kwargs))
            return call

    capture = Capture()
    ns['rms_matmul_rbe_qkv'] = capture
    old_controls = json.loads((EVIDENCE / 'scalar-domains.json').read_text())
    lo = 2.0 ** -126
    hi = (2.0 - 2.0 ** -23) * 2.0 ** 127
    cases = [('default', 1e-6, 10000.0, True)]
    for label, value in [('bool', True), ('int32_max', 2**31-1), ('uint32_min', 2**31),
                         ('uint32_max', 2**32-1), ('int64_min_positive', 2**32),
                         ('int64_max', 2**63-1), ('uint64_min', 2**63), ('uint64_max', 2**64-1),
                         ('subnormal', 1e-40), ('underflow', 5e-324), ('overflow', 1e300)]:
        cases.append(('eps_' + label, value, 10000.0, True))
    cases.extend([('theta_normal_min', 1e-6, lo, True), ('theta_normal_max', 1e-6, hi, True)])
    for label, value in [('bool', True), ('int32', 10000), ('uint32', 2**31),
                         ('int64', 2**32), ('uint64', 2**63), ('integer_2pow64', 2**64),
                         ('below_normal', math.nextafter(lo, 0.0)),
                         ('above_max', math.nextafter(hi, math.inf)),
                         ('subnormal', 1e-40), ('huge', 1e300),
                         ('zero', 0.0), ('negative', -1.0), ('nan', math.nan), ('inf', math.inf)]:
        cases.append(('theta_' + label, 1e-6, value, False))
    for label, value in [('int_outside', 2**64), ('huge_int', 10**1000), ('zero', 0),
                         ('false', False), ('negative', -1.0), ('nan', math.nan), ('inf', math.inf)]:
        cases.append(('eps_' + label, value, 10000.0, False))

    records = []
    known = {r['case']: r for r in old_controls['cases']}
    for label, eps, theta, accept in cases:
        try:
            mod._checked_scalars(0, 1, 4, 1, 2, True, eps, theta)
            admitted = True
        except ValueError:
            admitted = False
        assert admitted == accept, label
        x = h.Tensor(torch.ones((1, 2, 3), dtype=torch.float32))
        w = h.Tensor(torch.ones((4, 3), dtype=torch.float16))
        rms = h.Tensor(torch.ones(3, dtype=torch.float16))
        k = h.Tensor(torch.zeros(8, dtype=torch.float16))
        v = h.Tensor(torch.zeros(8, dtype=torch.float16))
        before = len(capture.calls)
        try:
            result = ns['rms_matmul_rbe_qkv_wrapper'](x, 0, w, w, w, rms, 1, 4, k, v,
                                                    eps=eps, theta=theta)
            assert accept, label
            assert len(capture.calls) == before + 1
            assert capture.calls[-1][1]['EPS'] is eps
            assert capture.calls[-1][1]['THETA'] is theta
            assert result[1].data_ptr() == k.data_ptr() and result[2].data_ptr() == v.data_ptr()
        except ValueError:
            assert not accept, label
            assert len(capture.calls) == before
        if accept:
            assert known[label]['compilation'] == 'passed', label
        records.append(dict(case=label, epsilon=repr(eps), theta=repr(theta), admitted=accept,
                            reached_dispatch=accept))

    report = dict(source=str(SOURCE.relative_to(ROOT)), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  original_sha256=hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
                  unchanged_functions=unchanged, original_checker_preserved_prefix=True,
                  added_checks=2, source_scalar_conversion=False,
                  retained_compiler_controls='Original JIT AST is identical; accepted cases all have actual offline compilation evidence.',
                  GPU_execution=False, cases=records)
    (EVIDENCE / 'candidate-scalar-contract.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(accepted=sum(r['admitted'] for r in records), rejected=sum(not r['admitted'] for r in records),
                         unchanged_functions=len(unchanged), source_scalar_conversion=False)))


if __name__ == '__main__':
    main()
