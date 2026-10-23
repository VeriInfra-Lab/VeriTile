"""Exact AST delta and metadata-only execution of the actual host checks.

No kernel arithmetic, producer script, compiler probe, or CUDA execution runs.
"""
import ast
import contextlib
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = HERE / 'int8_matmul_quantization_i32_safe.py'
BASE = HERE.parent / 'int8_matmul_quantization_scalar_cast/int8_matmul_quantization_scalar_cast.py'
EVIDENCE = ROOT / '.veritile-cache/kernel-evidence/int8_matmul_quantization_i32_safe'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def function(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)

def dump(tree):
    return ast.dump(tree, include_attributes=False)

def verify_source(tree, baseline):
    restored = copy.deepcopy(tree)
    fn = function(restored, '_check_gemm')
    active = next(n for n in fn.body if isinstance(n, ast.If) and
                  dump(n.test) == dump(ast.parse('M and N', mode='eval').body))
    expected = ast.parse('if K > 131071:\n    raise ValueError("active GEMM K exceeds signed INT32 accumulation range")').body[0]
    assert dump(active.body[0]) == dump(expected), 'guard predicate/location/error differs'
    del active.body[0]
    assert dump(restored) == dump(baseline), 'unexpected change outside the single guard'

class Tensor:
    next_storage = 100
    def __init__(self, shape, dtype='int8', column=False):
        if isinstance(shape,int): shape=(shape,)
        self.shape, self.dtype = tuple(shape), dtype
        self.ndim, self.layout = len(shape), 'strided'
        self.is_cuda, self.device = True, 'cuda:0'
        self.column = column
        self.storage = Tensor.next_storage
        Tensor.next_storage += 1
        self.events = None
    def is_neg(self): return False
    def is_contiguous(self): return not self.column
    def t(self):
        out = Tensor(self.shape[::-1], self.dtype, not self.column)
        out.storage = self.storage
        return out
    def numel(self):
        n = 1
        for d in self.shape: n *= d
        return n
    def data_ptr(self): return self.storage * 4096
    def element_size(self): return {'int8': 1, 'float16': 2, 'float32': 4}[self.dtype]
    def storage_offset(self): return 0
    def fill_(self, value):
        if self.events is not None: self.events.append('clear-output')
        return self

def host(tree):
    events, launches = [], []
    @contextlib.contextmanager
    def device(owner):
        events.append('device-enter')
        try: yield
        finally: events.append('device-exit')
    def allocate(shape, device, dtype):
        events.append('allocation')
        return Tensor(shape, dtype)
    class Launch:
        def __init__(self, name): self.name = name
        def __getitem__(self, grid):
            def launch(*args):
                events.append(self.name)
                launches.append((self.name, grid, args))
            return launch
    torch = SimpleNamespace(Tensor=Tensor, strided='strided', int8='int8', float16='float16', float32='float32',
        _C=SimpleNamespace(_overlaps=lambda a,b: a.storage == b.storage),
        empty=allocate, zeros=allocate, cuda=SimpleNamespace(device=device))
    triton = SimpleNamespace(cdiv=lambda a,b: (a+b-1)//b,
                             next_power_of_2=lambda k: 1 << max(0, (k-1).bit_length()))
    env = dict(torch=torch, triton=triton, quantize_int8_perrow_kernel=Launch('producer'), matmul_kernel=Launch('gemm'))
    names = ['_tensor', '_check_row', '_check_gemm', '_check_helper', 'quantize_int8_perrow',
             'matmul_int8', 'matmul_quantize_int8']
    nodes = [n for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0],ast.Name)
             and n.targets[0].id == '_GEMM_TILES']
    nodes += [function(tree,n) for n in names]
    exec(compile(ast.Module(nodes,[]),str(SOURCE),'exec'),env)
    return env,events,launches

def main():
    assert digest(BASE) == '2c6ceae59a42edbb9ec315beae4bedf9a156aace6fae0c19b6f0a4cd06065047'
    tree, baseline = ast.parse(SOURCE.read_text()), ast.parse(BASE.read_text())
    verify_source(tree, baseline)
    literal = next(line.split(' := ',1)[1] for line in (HERE/'Source.lean').read_text().splitlines()
                   if line.startswith('def productionAST : String'))
    assert json.loads(literal) == dump(tree)
    mutants = []
    for label, old, new in [
        ('off-by-one', 'K > 131071', 'K >= 131071'),
        ('unsafe-next-dimension', 'K > 131071', 'K > 131072'),
        ('composed-only', 'K > 131071', 'composed and K > 131071'),
        ('drop-bound', 'K > 131071', 'False'),
        ('change-accumulator', 'dtype=tl.int32', 'dtype=tl.float32'),
        ('drop-half-partial', ').to(tl.float16)', ').to(tl.float32)'),
        ('change-split-config', "'SPLIT_K': 2", "'SPLIT_K': 1"),
    ]:
        text = SOURCE.read_text()
        assert old in text
        mutant = ast.parse(text.replace(old,new,1))
        try: verify_source(mutant,baseline)
        except AssertionError: mutants.append(label)
        else: raise AssertionError(('mutant accepted',label))
    records = []
    for composed in (False,True):
        for column in (False,True):
            for k in (1,31,32,33,255,256,257,131070,131071,131072,262144,1048576):
                env,events,launches = host(tree)
                a=Tensor((1,k),'float32' if composed else 'int8')
                b,bs=Tensor((k,1),column=column),Tensor((1,),'float32')
                out=Tensor((1,1),'float32'); out.events=events
                try:
                    if composed: result=env['matmul_quantize_int8'](a,b,bs,out)
                    else: result=env['matmul_int8'](a,Tensor((1,),'float16'),b,bs,out)
                    accepted=True
                except ValueError as error:
                    assert str(error)=='active GEMM K exceeds signed INT32 accumulation range'
                    accepted=False
                assert accepted==(k<=131071)
                if accepted:
                    assert result is out
                    assert [x[0] for x in launches]==(['producer','gemm'] if composed else ['gemm'])
                    assert events.count('clear-output')==1
                    _,grid,_=launches[-1]
                    assert all(grid(dict(BLOCK_SIZE_M=bm,BLOCK_SIZE_N=bn,SPLIT_K=s))==(1,s)
                               for bm,bn,bk,s,g in env['_GEMM_TILES'])
                else: assert events==[] and launches==[]
                records.append(dict(composed=composed,column_major=column,K=k,accepted=accepted,events=events))
    for m,n,k in [(0,1,2**70),(1,0,2**70),(0,0,0),(1,1,0)]:
        env,events,launches=host(tree)
        result=env['matmul_int8'](Tensor((m,k)),Tensor((m,),'float32'),Tensor((k,n)),Tensor((n,),'float32'))
        assert result.shape==(m,n)
        assert len(launches)==int(bool(m and n))
        records.append(dict(api='standalone-empty-or-K0',M=m,N=n,K=k,events=events))
    for m,n,k in [(0,1,2**70),(1,0,262144),(1,0,1048576)]:
        env,events,launches=host(tree)
        result=env['matmul_quantize_int8'](Tensor((m,k),'float32'),Tensor((k,n)),Tensor((n,),'float32'))
        assert result.shape==(m,n)
        assert [x[0] for x in launches]==(['producer'] if m else [])
        records.append(dict(api='composed-empty',M=m,N=n,K=k,events=events))
    for k in (131072,262144,1048576):
        env,events,launches=host(tree)
        env['quantize_int8_perrow'](Tensor((1,k),'float32'))
        assert [x[0] for x in launches]==['producer']
        records.append(dict(api='independent-producer',K=k,events=events))
    for axis in (-2,-1,0,1):
        env,_,_=host(tree)
        env['_check_helper'](Tensor((1048577,1048577),'float32'),axis)
        records.append(dict(api='helper-domain',axis=axis,accepted=True))
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    report=dict(status='PASS',source_sha256=digest(SOURCE),baseline_sha256=digest(BASE),
                exact_single_guard_delta=True,complete_AST_binding=True,gemm_configurations=34,
                producer_configurations=4,mutants_rejected=mutants,cases=records,
                boundary='Actual host AST on metadata-only tensor doubles; no tensor payload, kernel arithmetic, native compilation or GPU execution.')
    (EVIDENCE/'candidate-check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',host_cases=len(records),mutants_rejected=len(mutants))))

if __name__=='__main__': main()
