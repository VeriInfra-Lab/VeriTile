"""Targeted source, checked-domain and Python-int ABI tests; no GPU execution."""
import ast
import contextlib
import importlib.util
import math
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import torch
from triton.runtime.jit import mangle_type

HERE = Path(__file__).resolve().parent.parent
SOURCE = HERE / 'seeded_dropout_repaired.py'
ORIGINAL = HERE.parents[2] / 'tritonbench_g/seeded_dropout/seeded_dropout.py'

def functions(path):
    return {n.name: n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}

class Tensor:
    def __init__(self, data, device='cuda:0', pointer=None):
        self.data, self.device, self.pointer = data, torch.device(device), pointer
    @property
    def shape(self): return self.data.shape
    @property
    def layout(self): return self.data.layout
    @property
    def dtype(self): return self.data.dtype
    @property
    def is_cuda(self): return self.device.type == 'cuda'
    def is_contiguous(self): return self.data.is_contiguous()
    def is_neg(self): return self.data.is_neg()
    def numel(self): return self.data.numel()
    def data_ptr(self): return self.data.data_ptr() if self.pointer is None else self.pointer

def check_candidate():
    source = functions(SOURCE)
    expected = ast.parse(ast.unparse(source['_seeded_dropout'])).body[0]
    expected.name = '_seeded_dropout_index_commuted'
    product = next(node.value for node in expected.body if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'block_start')
    product.left, product.right = product.right, product.left
    assert ast.dump(expected) == ast.dump(source['_seeded_dropout_index_commuted'])
    public = (HERE / 'FPEquiv.lean').read_text()
    assert public.count('specification ') == 1 and 'HostSpec' not in public
    assert 'abbrev originalIO' in public and ': ProgramIO α α where' in public
    assert 'ProgramIO.Variant (originalIO M k X O n p seed pid)' in public
    headline = public.split('specification index_commuted_equiv', 1)[1]
    for condition in ('M.compareLt (some .fp32) .real = some lt',
                      '(hn : n<2^31)', '(hp : pid<gridCount n)', '(hs : SeedABI k seed)'):
        assert condition in headline
    assert '(originalIO M k X O n p seed pid).Equivalent (commutedIO M k X O n p seed pid)' in headline
    assert 'KernelPublic.equivalent' in headline
    retained = (HERE / 'Proofs/KernelEquivalence.lean').read_text()
    assert 'theorem successful_index_commutation' in retained
    assert 'assumptions : Spec.Assumptions FP.GuardedFragment := []' in retained
    assert '#print_fp_assumptions index_commuted_equiv' in retained


def check_kernels():
    spec = importlib.util.spec_from_file_location('dropout_rng_controls', HERE.parent / 'uniform_sampling/check_contract.py')
    rng = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rng)
    assert rng.philox(0, 0, 0) == [0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8]
    pid = [0]

    @lru_cache(maxsize=64)
    def random_block(seed, start, block):
        return rng.uniform([rng.philox(seed, i, 0)[0] for i in range(start, start + block)])

    class Pointer:
        def __init__(self, backing, base, offsets=0):
            self.backing, self.base, self.offsets = backing, base, offsets
        def __add__(self, offsets):
            return Pointer(self.backing, self.base, self.offsets + offsets)

    def load(pointer, mask):
        value = torch.full(mask.shape, float('nan'), dtype=torch.float32)
        value[mask] = pointer.backing[pointer.base + pointer.offsets[mask]]
        return value

    def store(pointer, value, mask):
        pointer.backing[pointer.base + pointer.offsets[mask]] = value[mask]

    def random(seed, offsets):
        return random_block(seed, int(offsets[0]), offsets.numel())

    kernels = []
    for name in ('_seeded_dropout', '_seeded_dropout_index_commuted'):
        node = functions(SOURCE)[name]
        node.decorator_list = []
        for arg in node.args.args:
            arg.annotation = None
        ns = {'tl': SimpleNamespace(program_id=lambda axis=0: pid[0], arange=torch.arange,
                                    rand=random, load=load, store=store, where=torch.where)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(SOURCE), 'exec'), ns)
        kernels.append(ns[name])
    seeds = [-(1 << 63), -(1 << 31) - 1, -(1 << 31), 0, (1 << 31) - 1,
             1 << 31, (1 << 63) - 1, 1 << 63, (1 << 64) - 1]
    probabilities = [0., -0., .5, math.nextafter(1., 0.)]
    words = [0x00000000, 0x80000000, 0x00000001, 0x7F800000, 0xFF800000,
             0x7FC12345, 0x3F800000, 0xC0000000]
    pattern = torch.tensor([w if w < 2**31 else w - 2**32 for w in words], dtype=torch.int32).view(torch.float32)
    cases = executions = 0
    for n in (0, 1, 7, 1023, 1024, 1025, 2049):
        inputs = torch.full((n + 8,), -12345., dtype=torch.float32)
        inputs[4:4+n] = pattern.repeat((n + 7) // 8)[:n]
        initial = inputs.view(torch.uint8).clone()
        for seed in seeds:
            for p in probabilities:
                probability = torch.tensor(p, dtype=torch.float32)
                outputs = []
                for kernel in kernels:
                    output = torch.full((n + 8,), -98765., dtype=torch.float32)
                    for program in reversed(range((n + 1023) // 1024)):
                        pid[0] = program
                        kernel(Pointer(inputs, 4), Pointer(output, 4), n, probability, seed, 1024)
                        executions += 1
                    assert bool((output[:4] == -98765.).all()) and bool((output[-4:] == -98765.).all())
                    outputs.append(output)
                assert torch.equal(inputs.view(torch.uint8), initial)
                assert torch.equal(outputs[0].view(torch.uint8), outputs[1].view(torch.uint8))
                randoms = torch.cat([random_block(seed, i, 1024) for i in range(0, n, 1024)])[:n] if n else torch.empty(0)
                expected = torch.where(randoms > probability, inputs[4:4+n] / (1 - probability), 0.)
                torch.testing.assert_close(outputs[0][4:4+n], expected, rtol=0, atol=0, equal_nan=True)
                cases += 1
    print(f'{cases} complete paired kernel cases and {executions} program executions passed; exact memory equality and exterior bytes checked.')


def main():
    check_candidate()
    original, revised = functions(ORIGINAL), functions(SOURCE)
    assert ast.dump(original['_seeded_dropout']) == ast.dump(revised['_seeded_dropout'])
    retained_body = revised['seeded_dropout'].body[-1]
    assert isinstance(retained_body, ast.With)
    assert ast.dump(ast.Module(body=original['seeded_dropout'].body,type_ignores=[])) == ast.dump(
        ast.Module(body=retained_body.body,type_ignores=[]))
    assert '.Proofs.' not in (HERE/'Kernels.lean').read_text()
    spec=importlib.util.spec_from_file_location('seeded_revision',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    events=[]
    allocated=[]
    def allocate(x):
        events.append(('allocate',x));out=Tensor(torch.empty_like(x.data),str(x.device));allocated.append(out);return out
    @contextlib.contextmanager
    def device(d):
        events.append(('device',d));yield
    module.torch=SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,
        empty_like=allocate,cuda=SimpleNamespace(device=device))
    class Capture:
        def __getitem__(self,grid):
            def run(x,out,n,p,s,*,BLOCK_SIZE):
                assert BLOCK_SIZE==1024
                events.append(('launch',x,out,n,p,s,grid({'BLOCK_SIZE':BLOCK_SIZE})[0]))
            return run
    module._seeded_dropout=Capture()
    cases=[Tensor(torch.tensor(2.)),Tensor(torch.empty(0),pointer=1),
        Tensor(torch.ones(2,3)),Tensor(torch.ones(1025)),Tensor(torch.arange(10.)[2:8]),
        Tensor(torch.tensor([float('nan'),float('inf'),-float('inf'),-0.]))]
    probabilities=[0.,-0.,.5,math.nextafter(1.,0.)]
    seeds=[-(1<<63),-(1<<31)-1,-(1<<31),0,(1<<31)-1,1<<31,(1<<63)-1,1<<63,(1<<64)-1]
    for s in seeds:
        expected='i32' if -(1<<31)<=s<1<<31 else 'i64' if s<1<<63 else 'u64'
        assert mangle_type(s)==expected,(s,mangle_type(s),expected)
    count=0
    for x in cases:
        for p in probabilities:
            for s in seeds:
                events.clear();result=module.seeded_dropout(x,p,s)
                assert result is allocated[-1]
                assert [e[0] for e in events]==['device','allocate','launch']
                launch=events[-1]
                assert launch[1] is x and launch[2] is result and launch[3]==x.numel()
                assert launch[4] is p and launch[5] is s
                assert launch[6]==(x.numel()+1023)//1024
                count+=1
    assert torch.tensor(probabilities[-1],dtype=torch.float32).item()==1.
    x=Tensor(torch.ones(3))
    bad=[([],0.5,0),(Tensor(torch.eye(3).to_sparse()),.5,0),
        (Tensor(torch.ones(3,dtype=torch.float16)),.5,0),(Tensor(torch.ones(3),'cpu'),.5,0),
        (Tensor(torch.arange(6.)[::2]),.5,0),(Tensor(torch._neg_view(torch.ones(3))),.5,0),
        (Tensor(torch.empty(1<<31,device='meta')),.5,0),(Tensor(torch.ones(3),pointer=3),.5,0),
        (x,0,0),(x,True,0),(x,torch.tensor(.5),0),(x,float('nan'),0),(x,float('inf'),0),
        (x,-float('inf'),0),(x,-.1,0),(x,1.,0),(x,.5,True),(x,.5,1.),
        (x,.5,torch.tensor(1)),(x,.5,-(1<<63)-1),(x,.5,1<<64)]
    for args in bad:
        events.clear()
        try:module.seeded_dropout(*args)
        except (ValueError,TypeError):pass
        else:raise AssertionError(('unsupported input admitted',args))
        assert events==[],events
    print(f'{count} admitted launch/identity cases, {len(bad)} rejected cases, 9 native seed ABI endpoints passed.')
    check_kernels()
    print('Original ASTs and sole index-product candidate edit passed. CPU RNG/storage replay only; no GPU numerical claim.')

if __name__=='__main__':main()
