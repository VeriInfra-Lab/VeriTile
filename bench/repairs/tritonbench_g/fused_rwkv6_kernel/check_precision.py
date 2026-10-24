"""Source/lowering binding for the explicit FP32 contract; no GPU claim."""
from pathlib import Path
import argparse,ast,hashlib,json,re
A=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,default=Path('/tmp/veritile-rwkv-lowering/report.json'));args=parser.parse_args()
manifest=json.loads((A/'precision.json').read_text());source=A/'fused_rwkv6_kernel_repaired.py';original=A.parents[2]/'tritonbench_g/fused_rwkv6_kernel/fused_rwkv6_kernel.py';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(source)==manifest['source_sha256'];assert sha(original)==manifest['original_sha256']
name='fused_recurrent_rwkv6_fwd_kernel';tree=ast.parse(source.read_text());old=ast.parse(original.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name);before=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
assert hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest()==manifest['kernel_ast_sha256']
# Exactly the allocated output-head pitch changes in the source kernel.
assignment=next(n for n in before.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='p_o' for t in n.targets))
class Pitch(ast.NodeTransformer):
 def visit_Name(self,n):return ast.parse('T * V',mode='eval').body if n.id=='s_v_h' else n
Pitch().visit(assignment)
assert ast.dump(before,include_attributes=False)==ast.dump(fn,include_attributes=False)
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef));assert {n.name for n in cls.body if isinstance(n,ast.FunctionDef)}=={'forward'}
assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('contiguous','clone','copy_') for n in ast.walk(tree))
for public,forbidden in [('Correct.lean',{'Native.lean','Storage.lean','Trace.lean','Binding.lean','FPEquiv.lean','HistoricalFPEquiv.lean'}),('FPEquiv.lean',{'Real.lean','RealReference.lean','HistoricalCorrect.lean','Correct.lean'})]:
 seen=set()
 def visit(p):
  if p in seen:return
  seen.add(p);assert p.name not in forbidden,(public,p)
  for line in p.read_text().splitlines():
   prefix='import bench.repairs.tritonbench_g.fused_rwkv6_kernel.'
   if line.startswith(prefix):visit(A/(line[len(prefix):].replace('.','/')+'.lean'))
 visit(A/public)
report=json.loads(args.report.read_text());assert report['status']=='PASS' and report['actual_JIT_binder'] and not report['GPU_execution'];assert len(report['results'])==8
for row in report['results']:
 assert row['source_sha256']==sha(source) and row['status']=='compiled';assert row['sm'] in (80,90)
 for artifact in row['artifacts'].values():assert sha(Path(artifact['path']))==artifact['sha256']
 ptx=Path(row['artifacts']['ptx']['path']).read_text();T=row['shape'][2]
 assert 'mma' not in ptx and 'st.global' in ptx and '.ftz' not in ptx
 assert row['options']=={'num_warps':1,'num_stages':1,'enable_fp_fusion':True}
 if T>1 or (T>0 and row['final']):
  assert re.search(r'mul\.f32\s+([^,]+),\s*[^,]+,\s*0f3FB8AA3B;\s*ex2\.approx\.f32\s+[^,]+,\s*\1;',ptx),row['name']
 if T and row['shape'][3]>1:assert 'fma.rn.f32' in ptx and 'add.f32' in ptx
 if T==0 and row['initial']:
  llir=Path(row['artifacts']['llir']['path']).read_text();assert re.search(r'fadd float [^,]+, 0\.000000e\+00',llir)
print('PASS: hashes, exact one-expression kernel repair, no payload copies/backward, independent Real/native roots, 8 actual-JIT SM80/SM90 profiles, exp2 multiplier, FP32 FMA/add and no FTZ/MMA')
print('Deferred boundaries: ordinary Tensor/JIT/allocation/launch interfaces and legal source lowering/reduction plans; local scalar numerical conformance. No CUDA execution.')
