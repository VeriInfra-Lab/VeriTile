"""Offline SM90 compile of actual repaired arithmetic and pointer source."""
import argparse,hashlib,importlib.util,json,os
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
out=p.parse_args().output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
os.environ['TRITON_CACHE_DIR']=str(out/'cache')
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
source=Path(__file__).with_name('fifth_order_sph_harmonics_repaired.py')
spec=importlib.util.spec_from_file_location('fifth_order_native',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
cases=[]
for phase in ['fwd','bwd']:
 for block,rows,width,col in [(1,1,11,0),(64,65,17,3),(128,129,19,4),(1024,2**20,2**31-1,2**31-12)]:
  abis=['none'] if phase=='fwd' else ['i32','u32','i64','constexpr']
  for abi in abis:
   kernel=getattr(m,'fifth_order_'+phase+'_repaired')
   sig=dict(coord_ptr='*fp32')
   if phase=='fwd':sig['output_ptr']='*fp32'
   else:sig.update(coord_grad_ptr='*fp32',sph_grad_ptr='*fp32')
   constants=dict(block_size=block,coord_numel=3*rows,output_numel=rows*width,col_offset=col,output_stride=width)
   sig.update({k:'constexpr' for k in constants})
   if phase=='bwd':
    sig.update(upstream_row_stride=abi,upstream_col_stride=abi)
    if abi=='constexpr':constants.update(upstream_row_stride=0,upstream_col_stride=0)
   result=triton.compile(ASTSource(kernel,signature=sig,constexprs=constants,attrs={}),target=GPUTarget('cuda',90,32),options={'enable_fp_fusion':False})
   name=f'{phase}-{block}-{abi}'
   for stage in ['ttir','ptx']:(out/(name+'.'+stage)).write_text(result.asm[stage])
   ptx=result.asm['ptx']
   assert '.address_size 64' in ptx and 'fma.rn.f32' not in ptx and '.ftz' not in ptx
   assert 'mul.rn.f32' in ptx and 'add.rn.f32' in ptx
   cases.append(dict(phase=phase,block=block,rows=rows,width=width,column=col,stride_abi=abi))
report=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),triton=triton.__version__,sm=90,cases=cases,gpu_execution=False,numerical_conformance=False,enable_fp_fusion=False)
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(dict(offline_compilations=len(cases),sm=90,gpu_execution=False))
