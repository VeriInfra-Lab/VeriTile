"""Exact source-delta and actual wrapper guard/launch-capture checks.

Tensor metadata, CUDA device context and kernel launch are fixtures. No tensors
are allocated and no kernel arithmetic, compiler or GPU is executed.
"""
import ast
import hashlib
import itertools
import json
import math
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
OLD = HERE.with_name('ksoftmax_triton') / 'ksoftmax_triton_repaired.py'
NEW = HERE / 'ksoftmax_triton_grid_y_checked.py'
ADDITION = ('    if x.shape[1] > 65535:\n'
            '        raise ValueError("query rows exceed the CUDA grid y limit 65535")\n')


class Tensor:
    """Metadata-only fixture implementing the ordinary queried Tensor facts."""
    def __init__(self, shape, pointer):
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.layout = 'strided'
        self.dtype = 'float32'
        self.is_cuda = True
        self.device = 'cuda:0'
        self.pointer = pointer
        self.pointer_queries = 0
        self.forbid_pointer = False

    def numel(self):
        return math.prod(self.shape)

    def is_contiguous(self):
        return True

    def is_neg(self):
        return False

    def data_ptr(self):
        self.pointer_queries += 1
        assert not self.forbid_pointer, 'empty or axis-rejected input queried a pointer'
        return self.pointer


class Capture:
    def __init__(self, name, calls):
        self.name, self.calls = name, calls

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((self.name, grid, args, kwargs))
        return launch


@contextmanager
def device_context(device):
    assert device == 'cuda:0'
    yield


def load():
    old, new = OLD.read_text(), NEW.read_text()
    assert new.count(ADDITION) == 1
    assert new.replace(ADDITION, '') == old, 'unexpected byte-level source delta'
    old_ast, new_ast = ast.parse(old), ast.parse(new)
    extent = next(n for n in new_ast.body if isinstance(n, ast.FunctionDef) and n.name == '_extent')
    extra = ast.parse(ADDITION.strip()).body[0]
    assert ast.dump(extent.body[-1]) == ast.dump(extra)
    extent.body.pop()
    assert ast.dump(new_ast) == ast.dump(old_ast), 'unexpected module/JIT AST delta'
    nodes = [n for n in ast.parse(new).body if isinstance(n, ast.FunctionDef)
             and n.name in ('_tensor', '_ranges', '_extent', 'softmax', 'softmax_backward')]
    assert len(nodes) == 5
    calls = []
    namespace = {'torch': SimpleNamespace(Tensor=Tensor, strided='strided', float32='float32',
        cuda=SimpleNamespace(device=device_context)),
        '_softmax': Capture('forward', calls), '_softmax_backward': Capture('backward', calls)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NEW), 'exec'), namespace)
    return namespace, calls


def main():
    source, calls = load()
    records = []
    shapes = [(1, 65535, 1), (1, 65536, 1), (1, 65537, 1),
              (0, 65536, 1), (1, 65536, 0), (0, 2**31, 2**30+1),
              (2**31, 0, 2**30+1), (1, 0, 1),
              (2**31-1, 1, 1), (2**31, 1, 1),
              (65536, 65535, 1)]
    for entry, shape, log, causal in itertools.product(
            ('forward', 'backward'), shapes, (False, True), (False, True)):
        for mode in (None, 'qk', 'bk') if entry == 'forward' else (None,):
            B, Q, K = shape
            size = math.prod(shape)
            accepted = size == 0 or (B < 2**31 and Q <= 65535 and K <= 2**30 and size < 2**61)
            roles = [Tensor(shape, 4096 + i * (4*size+4096)) for i in range(3)]
            mask = Tensor((Q if mode == 'qk' else B, K), 4096+3*(4*size+4096)) if mode else object()
            active = roles[:2]+([mask] if mode else []) if entry == 'forward' else roles
            for t in active:
                t.forbid_pointer = size == 0 or not accepted
            previous = len(calls)
            try:
                result = (source['softmax'](roles[0], roles[1], mask, log, mode, causal)
                          if entry == 'forward' else source['softmax_backward'](*roles, log, causal))
            except ValueError as error:
                assert not accepted, (entry, shape, error)
                if B < 2**31 and Q < 2**31 and K <= 2**30 and size < 2**61:
                    assert str(error) == 'query rows exceed the CUDA grid y limit 65535'
            else:
                assert accepted and result is None
            launched = accepted and size > 0
            assert len(calls) == previous+int(launched)
            if launched:
                name, grid, args, kwargs = calls[-1]
                assert name == entry and grid == (B, Q)
                assert kwargs == dict(LOG=log, CAUSAL=causal, enable_fp_fusion=True,
                                      **({'MASK_TYPE': mode} if entry == 'forward' else {}))
                assert args[3:] == ((Q*K, K, Q*K, K, K if mode else 0, K)
                                   if entry == 'forward' else (Q*K, K, Q*K, K, Q*K, K, K))
                assert args[:2] == tuple(roles[:2])
                assert args[2] is (mask if mode else roles[1]) if entry == 'forward' else args[2] is roles[2]
            else:
                assert all(t.pointer_queries == 0 for t in active)
            records.append(dict(entry=entry, shape=shape, log=log, causal=causal, mask=mode,
                                accepted=accepted, launch_captured=launched,
                                pointer_queries=sum(t.pointer_queries for t in active)))
    # Metadata and exact-bool rejection still precede the unchanged empty return.
    prefix_cases = []
    for entry in ('forward', 'backward'):
        for defect in ('flag', 'rank', 'device', 'dtype'):
            roles = [Tensor((0, 65536, 1), 0) for _ in range(3)]
            for t in roles:
                t.forbid_pointer = True
            if defect == 'rank':
                for t in roles:
                    t.shape, t.ndim = (0, 65536), 2
            if defect == 'device':
                roles[0].device = 'cuda:1'
            if defect == 'dtype':
                roles[0].dtype = 'float16'
            previous = len(calls)
            try:
                kwargs = {'log': 1} if defect == 'flag' else {}
                if entry == 'forward':
                    source['softmax'](*roles[:2], **kwargs)
                else:
                    source['softmax_backward'](*roles, **kwargs)
            except (TypeError, ValueError):
                pass
            else:
                raise AssertionError((entry, defect))
            assert len(calls) == previous and not any(t.pointer_queries for t in roles)
            prefix_cases.append(dict(entry=entry, defect=defect, rejected_before_empty=True))
    report = {'status': 'passed', 'candidate_sha256': hashlib.sha256(NEW.read_bytes()).hexdigest(),
              'original_sha256': hashlib.sha256(OLD.read_bytes()).hexdigest(),
              'sole_delta': 'Two lines add Q > 65535 rejection inside shared _extent, after old extent checks; both wrappers call it after empty return.',
              'byte_delta_verified': True, 'remaining_module_and_jit_AST_identical': True,
              'boundary_cases': len(records), 'prefix_cases': len(prefix_cases),
              'large_grid_product_admitted': 65536*65535 > 2**31,
              'evidence_scope': 'Actual unchanged wrapper/helper AST with metadata-only Tensor fixtures and launch capture. No allocation, arithmetic, compiler, or GPU execution.',
              'records': records, 'prefix_records': prefix_cases}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
