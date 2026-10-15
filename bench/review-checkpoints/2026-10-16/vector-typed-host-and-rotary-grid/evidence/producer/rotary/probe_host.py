"""Run only the unchanged original wrapper AST with recorded primitive stubs.

This checks host wiring, not GPU execution or native launch admission. Python
int/float conversion is the real interpreter primitive; all shapes/strides and
the kernel launcher are instrumented values, so no Torch/Triton import occurs.
"""
from pathlib import Path
import ast
import hashlib
import json
import sys

WORK = Path(__file__).resolve().parent
ROOT = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
SOURCE = ROOT / 'bench/tritonbench_g/rotary_emb/rotary_emb.py'


class Tensor:
    def __init__(self, name, shape, strides):
        self.name, self.shape, self.strides = name, shape, strides

    def stride(self, index):
        return self.strides[index]


class Launcher:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
        return launch


class Triton:
    @staticmethod
    def cdiv(a, b):
        return (a+b-1)//b


tree = ast.parse(SOURCE.read_text())
wrapper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'rotary_emb_fwd')
wrapper.decorator_list = []
module = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
launcher = Launcher()
namespace = {'triton': Triton, '_rotary_kernel': launcher}
exec(compile(module, str(SOURCE), 'exec'), namespace)
original = namespace['rotary_emb_fwd']

cases = []
for width, factor in [(0,1.0),(1,1.0),(3,1.0),(7,0.75),(64,0.5),(127,1.0),
                      (128,1.0),(129,1.0),(130,1.0),(256,0.5),(256,0.499),
                      (64,0.0),(64,-1.0),(64,1.5),(64,float('nan')),
                      (64,float('inf')),(64,-float('inf'))]:
    cases.append((17,5,10,width,factor,(17,17,17)))
for length in [0,1,15,16,17,31,32,33]:
    for heads in [0,1,4,5,8]:
        cases.append((length,heads,heads+7,64,1.0,(length,length,length)))
for lengths in [(16,17,17),(17,16,17),(17,17,16),(16,16,16),(16,17,16)]:
    cases.append((17,5,10,64,1.0,lengths))
cases.append((17,5,10,64,float('nan'),(16,16,16)))

records = []
lean = ['import HostMetadata', '', 'namespace VeriTile.Bench.TritonBenchG.RotaryEmb.HostGrid',
        'open VeriTile.Triton', '']
for index,(length,heads,kheads,width,factor,lengths) in enumerate(cases):
    kl,cl,sl = lengths
    q=Tensor('Q',(length,heads,width),(1000,23,2))
    k=Tensor('K',(kl,kheads,width+3),(2000,31,3))
    cos=Tensor('Cos',(cl,width),(300,4))
    sin=Tensor('Sin',(sl,width+1),(400,5))
    start=len(launcher.calls)
    try:
        result=original(q,k,cos,sin,factor)
        actual='launched'
        assert result is None
    except (ValueError,OverflowError,AssertionError) as error:
        actual=type(error).__name__
    try:
        dimension=int(width*factor)
        expected='launched' if length==cl==sl and kl==cl==sl else 'AssertionError'
    except (ValueError,OverflowError) as error:
        dimension=None
        expected=type(error).__name__
    assert actual==expected,(index,actual,expected)
    calls=launcher.calls[start:]
    assert len(calls)==int(expected=='launched')
    record={'case':index,'q_shape':q.shape,'k_shape':k.shape,'cos_shape':cos.shape,
            'sin_shape':sin.shape,'factor':str(factor),'outcome':actual,'dimension':dimension}
    if calls:
        grid,args,kwargs=calls[0]
        assert grid==((heads+3)//4,(length+15)//16)
        assert args[:4]==(q,k,cos,sin)
        assert args[4:]==q.strides+k.strides+cos.strides+sin.strides+(length,heads,kheads)
        assert kwargs=={'BLOCK_HEAD':4,'BLOCK_SEQ':16,'BLOCK_DMODEL':dimension,
                        'num_warps':8 if dimension>=128 else 4,'num_stages':1}
        record.update(grid=grid,arguments=args[4:],keywords=kwargs)
        factor_literal=repr(factor)
        lean += [f'def probe{index} : Request :=',
                 f'  ⟨⟨{length},{heads},{width},1000,23,2⟩,⟨{kl},{kheads},{width+3},2000,31,3⟩,',
                 f'    ⟨{cl},{width},300,4⟩,⟨{sl},{width+1},400,5⟩,{factor_literal}⟩',
                 f'example : prepare (fun _ _ => some ({dimension})) probe{index} =',
                 f'    some ⟨probe{index},{dimension}⟩ := rfl',
                 f'example : (Plan.mk probe{index} ({dimension})).grid.dims = [{grid[0]},{grid[1]}] ∧',
                 f'    (Plan.mk probe{index} ({dimension})).numWarps = {kwargs["num_warps"]} ∧',
                 f'    (Plan.mk probe{index} ({dimension})).numStages = 1 := by decide','']
    elif dimension is not None:
        lean += [f'def probe{index} : Request :=',
                 f'  ⟨⟨{length},{heads},{width},1000,23,2⟩,⟨{kl},{kheads},{width+3},2000,31,3⟩,',
                 f'    ⟨{cl},{width},300,4⟩,⟨{sl},{width+1},400,5⟩,1.0⟩',
                 f'example : prepare (fun _ _ => some ({dimension})) probe{index} = none := by decide','']
    records.append(record)
lean += ['end VeriTile.Bench.TritonBenchG.RotaryEmb.HostGrid','']
(WORK/'HostProbe.lean').write_text('\n'.join(lean))
receipt={'source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
         'command':sys.argv,'case_count':len(records),'all_passed':True,
         'method':'unchanged original host FunctionDef AST; instrumentation of shape/stride/cdiv/launch primitives',
         'boundary':'No GPU/native execution. Lean probe conversion is supplied as the measured trusted primitive result.',
         'cases':records}
(WORK/'evidence'/'HostProbe-source-ast.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(f'{len(records)} original-source host probes passed; HostProbe.lean generated.')
