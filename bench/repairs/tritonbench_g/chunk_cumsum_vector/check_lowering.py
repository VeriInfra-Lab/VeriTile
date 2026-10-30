"""One offline profile checks the accumulating-dot compiler boundary; no GPU run."""
import importlib.util, os, re
from pathlib import Path
out=Path('/tmp/chunk-vector-lowering'); out.mkdir(exist_ok=True)
os.environ['TRITON_CACHE_DIR']=str(out/'cache')
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
source=Path(__file__).with_name('chunk_cumsum_vector_repaired.py')
spec=importlib.util.spec_from_file_location('chunk_vector_lowering',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
c={'H':1,'s_s_b':1056,'s_s_h':1056,'s_s_t':32,'s_s_d':1,'s_z_b':1056,'s_z_h':1056,'s_z_t':32,'s_z_d':1,'T':33,'S':32,'BT':16,'BS':32}
sig={'s':'*fp32','z':'*fp32',**{k:'constexpr' for k in c}}
k=triton.compile(ASTSource(m.chunk_global_cumsum_vector_kernel.fn,signature=sig,constexprs=c), target=GPUTarget('cuda',90,32),options={'num_warps':4})
for ext in ['ttir','ttgir','llir','ptx']:(out/f'bt16-w4-sm90.{ext}').write_text(k.asm[ext])
ttir=k.asm['ttir']
assert k.metadata.enable_fp_fusion is True
dot=re.search(r'tt.dot [^,\n]+, [^,\n]+, (%[\w]+) :',ttir)
assert dot, 'missing accumulating dot'
broadcast=re.search(re.escape(dot[1])+r' = tt.broadcast (%[\w]+) ',ttir)
assert broadcast, 'dot accumulator is not broadcast'
expand=re.search(re.escape(broadcast[1])+r' = tt.expand_dims (%[\w]+) \{axis = 0',ttir)
assert expand, 'missing carry expansion'
assert re.search(r'iter_args\('+re.escape(expand[1])+r' = ',ttir), 'dot accumulator is not incoming loop carry'
assert 'arith.addf' in ttir, 'missing separate carry/reduction addition'
print('Triton' ,triton.__version__, 'SM90 BT16 BS32 warps4 default fp fusion',k.metadata)
for line in k.asm['ttir'].splitlines():
 if 'tt.dot' in line or 'arith.addf' in line or 'scf.for' in line:print(line)
