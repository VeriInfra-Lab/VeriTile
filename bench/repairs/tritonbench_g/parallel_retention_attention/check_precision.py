"""Pinned source, independent proof imports and offline scalar-lowering audit."""
from pathlib import Path
import ast,hashlib,json,re
A=Path(__file__).resolve().parent;root=A.parents[3];P='bench.repairs.tritonbench_g.parallel_retention_attention.'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();manifest=json.loads((A/'precision.json').read_text());source=A/'parallel_retention_attention_repaired.py';original=root/'bench/tritonbench_g/parallel_retention_attention/parallel_retention_attention.py'
# Historical compiler evidence binds the unchanged original definitions.
selected=source.read_text().split('\n\n# Strict accumulator-assignment fusion candidates.')[0].encode()
assert hashlib.sha256(selected).hexdigest()==manifest['source_sha256'] and sha(original)==manifest['original_sha256']
assert manifest['source_sha256'] in (A/'Kernels.lean').read_text()
functions={n.name:n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef)}
for name,want in manifest['kernel_ast_sha256'].items():
 node=functions[name];assert hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()==want
 for n in ast.walk(node):
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):
   if n.func.attr=='dot':assert any(k.arg=='allow_tf32' and isinstance(k.value,ast.Constant) and k.value.value is False for k in n.keywords)
   if n.func.attr=='load':assert any(k.arg=='padding_option' and isinstance(k.value,ast.Constant) and k.value.value=='zero' for k in n.keywords)
# Original copy decorator is retained exactly; changes are scoped to guards,
# min16 tiles, zero padding, integer width and explicit source scalar casts.
old={n.name:n for n in ast.parse(original.read_text()).body if isinstance(n,ast.FunctionDef)}
assert ast.dump(functions['contiguous'],include_attributes=False)==ast.dump(old['contiguous'],include_attributes=False)
seen=set()
def closure(name):
 if name in seen:return
 seen.add(name);path=root/(name.replace('.','/')+'.lean')
 for line in path.read_text().splitlines():
  if line.startswith('import '+P):closure(line.removeprefix('import '))
closure(P+'FPEquiv')
for forbidden in ['Correct','Proofs.Real','Proofs.GraphReal','Proofs.Results','Proofs.HistoricalAggregate','Proofs.Differential','Proofs.Specification','Proofs.SourceCorrect']:
 assert P+forbidden not in seen,(forbidden,seen)
seen.clear();closure(P+'Correct');assert P+'FPEquiv' not in seen and P+'Proofs.HistoricalFPEquiv' not in seen
report=json.loads(Path(manifest['lowering_report']).read_text());assert report['status']=='PASS' and report['actual_JIT_binder'] and report['GPU_execution'] is False
assert len(report['results'])==8
control=report['original_minimum_tile_control'];assert control['status']=='failed_as_expected' and sha(Path(control['path']))==control['sha256']
counts=[]
for row in report['results']:
 assert row['source_sha256']==manifest['source_sha256'] and row['status']=='compiled'
 text={}
 for kind,item in row['artifacts'].items():
  path=Path(item['path']);assert sha(path)==item['sha256'];text[kind]=path.read_text()
 ptx=text['ptx'];ttir=text['ttir'];assert not any(op in ptx for op in ['mma.sync','wgmma','tcgen05'])
 assert 'fma.rn.f32' in ptx and 'ex2.approx.ftz.f32' in ptx and 'ld.global' in ptx and 'st.global' in ptx
 assert 'tf32' not in ttir.lower() and 'xf16' not in ttir and 'xbf16' not in ttir
 dots=[line for line in ttir.splitlines() if ' = tt.dot ' in line];assert len(dots)>=4
 carries=[line for line in dots if not re.search(r'tt.dot [^,]+, [^,]+, %cst',line)];assert carries
 counts.append({'name':row['name'],'fp32_dot_sites':len(dots),'live_carry_sites':len(carries)})
print('PASS: source hashes, original contiguous decorator, independent Real/native imports,',len(counts),'original SM80/SM90 JIT profiles; new candidate compiler admission is not claimed')
print(json.dumps(counts,indent=2))
