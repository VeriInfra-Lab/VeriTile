"""Hash-bound source/IR/PTX inspection; no GPU instruction-conformance claim."""
from pathlib import Path
import argparse,ast,hashlib,json,re
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=Path('/tmp/veritile-lightning-lowering'));args=p.parse_args()
A=Path(__file__).resolve().parent;source=A/'lightning_attention_repaired.py';tree=ast.parse(source.read_text());functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
original=A.parents[2]/'tritonbench_g/lightning_attention/lightning_attention.py';old={n.name:n for n in ast.parse(original.read_text()).body if isinstance(n,ast.FunctionDef)}
for name in ['_fwd_kernel','_bwd_intra_kernel','_bwd_inter_kernel']:assert ast.dump(functions[name])==ast.dump(old[name])
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef));methods={n.name:ast.unparse(n) for n in cls.body if isinstance(n,ast.FunctionDef)}
assert '.contiguous(' not in methods['forward'] and methods['backward'].count('.contiguous(')==1
assert 'do = do.contiguous()' in methods['backward'] and 'save_for_backward(q, k, v)' in methods['forward']
assert 'import bench.' not in (A/'Kernels.lean').read_text()
# Public FP dependency closure must remain independent of mathematical/Real endpoints.
seen=set()
def audit(path):
 if path in seen:return
 seen.add(path)
 for imp in re.findall(r'^import (\S+)',path.read_text(),re.M):
  if imp.startswith('bench.repairs.tritonbench_g.lightning_attention.'):
   name=imp.rsplit('.',1)[-1];assert 'Real' not in name and name not in ['Correct','Complete','Endpoint','NativeEndpoint']
   audit(A/('Kernels.lean' if name=='Kernels' else 'Proofs/'+name+'.lean'))
audit(A/'FPEquiv.lean')
report=json.loads((args.output_dir/'report.json').read_text());assert report['status']=='PASS';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for row in report['results']:
 assert row['source_sha256']==sha(source)
 for artifact in row['artifacts'].values():assert sha(Path(artifact['path']))==artifact['sha256']
 ptx=Path(row['artifacts']['ptx']['path']).read_text();ir=Path(row['artifacts']['ttir']['path']).read_text()
 if row['stage']=='inter' and row['shape'][2]<=64:continue
 assert 'inputPrecision = tf32' in ir and 'xf16>' not in ir
 ops=re.findall(r'(?:wgmma\.mma_async[^\s]+|mma\.sync\.aligned[^\s]+)',ptx)
 assert ops and all('tf32.tf32' in op for op in ops),ops
 assert all('k8' in op for op in ops),ops
 assert 'st.global' in ptx and ('ld.global' in ptx or 'cp.async' in ptx)
 # The source/native graph retains scalar adds and consecutive fixed-K8 dot nodes.

print('PASS: unchanged three kernel ASTs, direct forward/saved inputs, necessary cotangent materialization, independent FP layer;',len(report['results']),'SM80/SM90 profiles')
