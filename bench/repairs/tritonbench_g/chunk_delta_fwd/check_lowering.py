"""Narrow offline SM80 checks of one- and multiple-subblock carry lowering."""
import importlib.util
import os
import re
from pathlib import Path
out=Path('/tmp/chunk-delta-fwd-lowering');out.mkdir(exist_ok=True)
os.environ['TRITON_CACHE_DIR']=str(out/'cache')
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
spec=importlib.util.spec_from_file_location('delta_lowering',Path(__file__).with_name('chunk_delta_fwd_repaired.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
for BT in [16,80]:
    BC=min(BT,64)
    constants={'H':2,'T':81,'K':16,'V':17,'BT':BT,'BC':BC,'BK':16,'BV':64,'NT':triton.cdiv(81,BT),'USE_INITIAL_STATE':True,'STORE_FINAL_STATE':True}
    sig={name:'*fp32' for name in ['k','v','d','v_new','h','initial_state','final_state']}
    sig.update({name:'i32' for name in ['s_qk_h','s_qk_t','s_qk_d','s_w_b','s_w_h','s_w_t','s_w_d','s_vo_h','s_vo_t','s_vo_d','s_h_h','s_h_t']})
    sig.update({name:'constexpr' for name in constants})
    compiled=triton.compile(ASTSource(m.chunk_delta_rule_fwd_kernel_h.fn,signature=sig,constexprs=constants),target=GPUTarget('cuda',80,32),options={'num_warps':4,'num_stages':1})
    for ext in ['ttir','ttgir','llir','ptx']:(out/f'bt{BT}.{ext}').write_text(compiled.asm[ext])
    ir=compiled.asm['ttir'];ptx=compiled.asm['ptx']
    assert 'fma.rn.f32' in ptx and 'mma.sync' not in ptx
    assert compiled.metadata.enable_fp_fusion is True
    dots=re.findall(r'(%\w+) = tt.dot [^,\n]+, [^,\n]+, (%\w+) ',ir)
    assert len(dots)==2,dots
    assert re.search(re.escape(dots[0][1])+r' = arith.constant dense<0.000000e\+00>',ir)
    assert re.search(r'arith.subf .*'+re.escape(dots[0][0]),ir)
    if BT==16:
        assert 'arith.addf' not in ir,'one-subblock update must retain the fused history carry'
        assert not re.search(re.escape(dots[1][1])+r' = arith.constant',ir)
    else:
        assert re.search(r'arith.addf ',ir),'multiple subblocks retain separate outer add'
        assert len(re.findall(r'scf.for',ir))==2
print(f'PASS: Triton {triton.__version__}, SM80, 4 warps; NC=1 live state carry and NC=2 separate outer add; zero-seeded correction, scalar FMA, zero padding')
