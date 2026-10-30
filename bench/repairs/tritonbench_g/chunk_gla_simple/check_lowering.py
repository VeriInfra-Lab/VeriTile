"""One offline source profile checks dot carry/FMA lowering; no GPU execution."""
import importlib.util
import os
import re
from pathlib import Path
out=Path('/tmp/chunk-gla-simple-lowering');out.mkdir(exist_ok=True)
os.environ['TRITON_CACHE_DIR']=str(out/'cache')
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
source=Path(__file__).with_name('chunk_gla_simple_repaired.py')
spec=importlib.util.spec_from_file_location('simple_lowering',source)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
constants={'T':17,'K':129,'V':17,'BT':16,'BK':64,'BV':32}
signature={name:'*fp32' for name in ['q','k','v','h','g','o']}
signature.update({name:'i32' for name in ['s_k_h','s_k_t','s_v_h','s_v_t','s_h_h','s_h_t']})
signature['scale']='fp32';signature.update({name:'constexpr' for name in constants})
compiled=triton.compile(ASTSource(m.chunk_simple_gla_fwd_kernel_o_repaired.fn,signature=signature,constexprs=constants),
                       target=GPUTarget('cuda',80,32),options={'num_warps':4})
for ext in ['ttir','ttgir','llir','ptx']:(out/f'bt16-k129-v17-w4-sm80.{ext}').write_text(compiled.asm[ext])
ir=compiled.asm['ttir'];ptx=compiled.asm['ptx']
dots=re.findall(r'tt.dot ([^\n]+)',ir)
assert len(dots)==3,dots
assert all('tf32' not in x for x in dots)
assert 'scf.for' in ir and 'iter_args' in ir
assert 'fma.rn.f32' in ptx and 'mma.sync' not in ptx
assert 'ex2.approx' in ptx
assert compiled.metadata.enable_fp_fusion is True
# QH and QK consume the two live loop accumulators; final SV consumes gated history.
loop=re.search(r'iter_args\((%\w+) = %\w+, (%\w+) = %\w+\)',ir)
assert loop,ir
assert re.search(r'tt.dot [^,\n]+, [^,\n]+, '+re.escape(loop[1])+r' ',ir)
assert re.search(r'tt.dot [^,\n]+, [^,\n]+, '+re.escape(loop[2])+r' ',ir)
final_carry=re.findall(r'tt.dot [^,\n]+, [^,\n]+, (%\w+) ',ir)[-1]
assert re.search(re.escape(final_carry)+r' = arith.mulf ',ir),final_carry
print(f'PASS: Triton {triton.__version__}, SM80, BT16 K129 V17 BK64 BV32, four warps; three live-carry IEEE dots, scalar FMA and exp2 lowering')
