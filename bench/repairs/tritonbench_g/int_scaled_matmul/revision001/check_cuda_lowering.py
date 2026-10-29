"""Offline INT8 saturation/fusion audit. Does not execute a GPU."""
import argparse
import ast
import importlib.util
import os
from pathlib import Path
import re
import sys

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

def run(output):
    os.environ.pop('TRITON_INTERPRET',None)
    os.environ['TRITON_CACHE_DIR']=str(output/'cache')
    sys.dont_write_bytecode=True
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    source=Path(__file__).with_name('int_scaled_matmul_repaired.py')
    output.mkdir(parents=True,exist_ok=True)
    module=load(source,'int_scaled_offline_candidate')
    # Reproduce the historical += tl.dot lowering with the same repaired
    # masks and metadata; the original benchmark file remains unchanged.
    old=ast.parse(source.read_text())
    class OriginalAdd(ast.NodeTransformer):
        def visit_Assign(self,node):
            if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='inline_asm_elementwise':
                values=next(k.value for k in node.value.keywords if k.arg=='args')
                return ast.copy_location(ast.AugAssign(target=node.targets[0],op=ast.Add(),value=values.elts[1]),node)
            return self.generic_visit(node)
    old=ast.fix_missing_locations(OriginalAdd().visit(old))
    baseline=output/'before_explicit_add.py';baseline.write_text(ast.unparse(old)+'\n')
    previous=load(baseline,'int_scaled_offline_previous')
    total=0
    for before in [True,False]:
        for scaled in [False,True]:
            for tile in ([16] if before else [16,64]):
                for even in ([False] if before or not scaled else [False,True]):
                    owner=previous if before else module
                    kernel=getattr(owner,'scaled_matmul_kernel_with_block_pointers' if scaled else 'matmul_kernel_with_block_pointers')
                    constants=dict(BLOCK_M=tile,BLOCK_N=tile,BLOCK_K=32,GROUP_M=8)
                    if scaled:constants.update(EVEN_K=even,ACC_TYPE=triton.language.int32)
                    signature={name:('*i8' if i<2 else '*i32' if i==2 else '*fp32' if scaled and i==3 else
                                     'constexpr' if name in constants else 'i32') for i,name in enumerate(kernel.arg_names)}
                    compiled=triton.compile(ASTSource(kernel,signature=signature,constexprs=constants),
                        target=GPUTarget('cuda',90,32),options={'num_warps':4,'num_stages':2})
                    name=f'{"before" if before else "after"}_scaled{int(scaled)}_tile{tile}_even{int(even)}'
                    for ext in ['ttir','llir','ptx']:(output/f'{name}.{ext}').write_text(compiled.asm[ext])
                    ptx,ttir=compiled.asm['ptx'],compiled.asm['ttir']
                    assert '.s32.s8.s8' in ptx and '.address_size 64' in ptx
                    dots=[line for line in ttir.splitlines() if '= tt.dot ' in line]
                    assert dots
                    if before:
                        assert '.satfinite.s32.s8.s8.s32' in ptx
                        for line in dots:
                            seed=re.search(r'tt.dot %\w+, %\w+, (%\w+)',line).group(1)
                            assert not seed.startswith('%cst')
                            assert any(seed+' = ' in loop for loop in ttir.splitlines() if 'scf.for' in loop)
                    else:
                        assert all(re.search(r'tt.dot %\w+, %\w+, %cst',line) for line in dots),dots
                        assert 'tt.elementwise_inline_asm "add.u32 $0, $1, $2;"' in ttir
                        assert 'add.u32' in ptx
                        # The zero seed is the typed int32 dense constant, not
                        # the loop-carried value observed in the old lowering.
                        for line in dots:
                            seed=re.search(r'tt.dot %\w+, %\w+, (%cst\w*)',line).group(1)
                            assert re.search(re.escape(seed)+r' = arith.constant dense<0> : tensor<\d+x\d+xi32>',ttir)
                    if scaled:
                        assert 'cvt.rn.f32.s32' in ptx and 'mul.f32' in ptx and 'cvt.rzi.s32.f32' in ptx
                    total+=1
                    family='wgmma32' if 'wgmma.mma_async' in ptx else 'mma.sync32'
                    print(name,family,'passed',flush=True)
    # The observed old positive recurrence saturates on the last iteration;
    # the explicit modular carry reaches the signed32 interpretation of 2**31.
    saturated=wrapped=0
    for _ in range(4096):
        saturated=min(2**31-1,saturated+32*128*128)
        wrapped=(wrapped+32*128*128)%2**32
    signed=wrapped if wrapped<2**31 else wrapped-2**32
    assert saturated==2147483647 and signed==-2147483648
    print(f'{total} SM90 offline compilations passed; old saturation witness {saturated}, '
          f'repaired modular result {signed}. No GPU execution.')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=Path('/tmp/veritile-int-scaled-lowering-audit'))
    run(parser.parse_args().output_dir)
