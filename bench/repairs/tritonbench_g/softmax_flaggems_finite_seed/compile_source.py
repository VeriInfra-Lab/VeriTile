"""Two task-local offline SM80 profiles; no CUDA execution is performed."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
OUT=ROOT/'.veritile-cache/softmax-flaggems-finite-seed-evidence/compiler'
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'tmp').mkdir(exist_ok=True)
os.environ['TRITON_CACHE_DIR']=str(OUT/'cache')
os.environ['TMPDIR']=str(OUT/'tmp')
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.runtime.jit import JITFunction
source=HERE/'softmax_flaggems_repaired.py'
spec=importlib.util.spec_from_file_location('flaggems_finite_seed',source)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
records=[]
for name,n,b,c in [('softmax_kernel_inner',32769,4096,1),('softmax_kernel_non_inner',8193,8192,1)]:
    fn=getattr(m,name)
    while not isinstance(fn,JITFunction):fn=fn.fn
    noninner='non_inner' in name
    sig={a:'*fp32' for a in fn.arg_names[:2]}
    sig.update({a:'i64' for a in fn.arg_names[2:] if a in ('M','N','K')})
    constants={'TILE_N':b,'ONE_TILE_PER_CTA':False}
    if noninner:constants.update(TILE_K=c,NUM_SMS=80)
    compiled=triton.compile(ASTSource(fn,signature=sig,constexprs=constants),
        target=GPUTarget('cuda',80,32),options={'num_warps':16,'enable_fp_fusion':True})
    files={}
    for ext in ('ttir','ttgir','llir','ptx'):
        p=OUT/(name+'.'+ext)
        p.write_text(compiled.asm[ext])
        files[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    ptx=compiled.asm['ptx']
    assert 'FF7FFFFF' in ptx.upper(), 'finite FP32 seed absent from generated PTX'
    assert 'ex2.approx' in ptx and 'st.global' in ptx
    records.append({'kernel':name,'shape':[1,n] if not noninner else [1,n,2],
        'tile_n':b,'tile_k':c,'manual_signature':sig,'constants':constants,
        'finite_seed_word_found':True,'fp32_exp2_found':True,
        'fma':[line.strip() for line in ptx.splitlines() if 'fma.' in line],
        'files':files})
result={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'triton_version':triton.__version__,
    'target':'CUDA SM80','offline_profiles':records,'gpu_execution':False,
    'scope':'Two explicit manual-signature profiles of the actual changed JITs; no universal compiler/resource claim.'}
(OUT/'report.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'offline_profiles':len(records),'finite_seed_confirmed':True,'gpu_execution':False}))
