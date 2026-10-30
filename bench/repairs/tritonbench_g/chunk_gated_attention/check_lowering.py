"""Narrow offline SM80 carry/FMA check for both gate branches and prefix dot."""
import importlib.util
import os
import re
from pathlib import Path
out=Path('/tmp/chunk-gated-attention-lowering');out.mkdir(exist_ok=True)
os.environ['TRITON_CACHE_DIR']=str(out/'cache')
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
spec=importlib.util.spec_from_file_location('gated_lowering',Path(__file__).with_name('chunk_gated_attention_repaired.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
for gate in [False,True]:
    constants={'T':32,'K':32,'V':32,'BT':16,'BK':16,'BV':16,'NT':2,'GATEK':gate,'USE_INITIAL_STATE':True,'STORE_FINAL_STATE':True,'PROGRAM_BASE':0}
    sig={name:'*fp32' for name in ['k','v','g','h','h0','ht']}
    sig.update({name:'i32' for name in ['s_k_h','s_k_t','s_k_d','s_v_h','s_v_t','s_v_d','s_h_h','s_h_t','s_h_d']})
    sig.update({name:'constexpr' for name in constants})
    compiled=triton.compile(ASTSource(m.chunk_gated_abc_fwd_kernel_h,signature=sig,constexprs=constants),target=GPUTarget('cuda',80,32),options={'num_warps':2,'num_stages':1})
    for ext in ['ttir','ttgir','llir','ptx']:(out/f'inner-gatek{int(gate)}.{ext}').write_text(compiled.asm[ext])
    ir=compiled.asm['ttir'];ptx=compiled.asm['ptx']
    dots=re.findall(r'tt.dot [^,\n]+, [^,\n]+, (%\w+) ',ir)
    assert len(dots)==1,dots
    assert re.search(re.escape(dots[0])+r' = arith.mulf ',ir)
    assert 'scf.for' in ir and 'iter_args' in ir
    assert 'arith.addf' in ir,'missing zero-plus-seed'
    assert 'fma.rn.f32' in ptx and 'mma.sync' not in ptx and 'ex2.approx' in ptx
    assert compiled.metadata.enable_fp_fusion is True
constants={'T':32,'S':64,'BT':16,'BS':16,'BH':2,'PROGRAM_BASE':0}
sig={'s':'*fp32','o':'*fp32','s_s_h':'i32','s_s_t':'i32','s_s_d':'i32',**{name:'constexpr' for name in constants}}
compiled=triton.compile(ASTSource(m.chunk_gated_abc_fwd_kernel_cum.fn,signature=sig,constexprs=constants),target=GPUTarget('cuda',80,32),options={'num_warps':2})
for ext in ['ttir','ttgir','llir','ptx']:(out/f'prefix-bs16.{ext}').write_text(compiled.asm[ext])
assert 'tt.dot' in compiled.asm['ttir'] and 'fma.rn.f32' in compiled.asm['ptx'] and 'mma.sync' not in compiled.asm['ptx']
print(f'PASS: Triton {triton.__version__}, SM80; both gate branches BT/BK/BV16, two chunks, seed/final; live gated dot carry, zero-plus-seed, scalar FMA/exp2; prefix BT/BS16 scalar FMA')
