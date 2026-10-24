"""Exact normalized AST comparison: selected Python versus the quoted source."""
from pathlib import Path
import ast,re,json,hashlib
ROOT=Path.cwd();TASK=ROOT/'bench/repairs/tritonbench_g/flash_attn_scalar_abi_checked';BASE=TASK/'kernel-evidence'
py=TASK/'flash_attn_scalar_abi_checked.py';lean=TASK.parent/'flash_attn/KernelOnly/SourceSyntax.lean'
original=next(x for x in ast.parse(py.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='_fwd_kernel')
class BindBlocks(ast.NodeTransformer):
 def visit_Name(self,n):
  if n.id in ('BLOCK_M','BLOCK_N'):return ast.copy_location(ast.Constant(128 if n.id=='BLOCK_M' else 64),n)
  return n
original=BindBlocks().visit(original)
text=lean.read_text().split('(triton {',1)[1].split('}).surfaceBody',1)[0]
text=re.sub(r'\$\(\(p\.(\w+) : Nat\)\)',r'\1',text)
text=re.sub(r'\$\((\d+)\)',r'\1',text)
text=re.sub(r'boundary_check=\(\[(\d+)\] : List Nat\)',r'boundary_check=(\1,)',text)
lines=[];indent=1;declared=False
for raw in text.splitlines():
 line=raw.strip()
 if not line or line.startswith('--') or line.startswith('tl.region'):continue
 if line.startswith('sm_scale = '):assert not declared;declared=True;continue
 if line=='}':indent-=1;continue
 if line.endswith('{'):
  lines.append('    '*indent+line[:-1].strip()+':');indent+=1
 else:lines.append('    '*indent+line)
assert declared and indent==1
quoted=ast.parse('def _fwd_kernel():\n'+'\n'.join(lines)+'\n').body[0]
a=ast.dump(ast.Module(body=original.body,type_ignores=[]),include_attributes=False)
b=ast.dump(ast.Module(body=quoted.body,type_ignores=[]),include_attributes=False)
if a!=b:
 (BASE/'source-python.ast').write_text(a);(BASE/'source-lean.ast').write_text(b)
 raise AssertionError('Actual selected source AST differs from shared quotation')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report={'status':'pass','selected_source':{'path':str(py.relative_to(ROOT)),'sha256':sha(py)},'shared_source':{'path':str(lean.relative_to(ROOT)),'sha256':sha(lean)},'source_AST_sha256':hashlib.sha256(a.encode()).hexdigest(),'statement_count':len(original.body),'bindings':{'BLOCK_M':128,'BLOCK_N':64},'normalizations':['Lean Nat parameter antiquotation spelling','boundary-axis List Nat syntax to the identical Python singleton tuple','compile-time block constants from unchanged wrapper','syntactic scalar declaration removed by actual kernel surfaceBody.drop 1; incoming register is retained'],'primitive_numerical_experiment':False,'GPU_execution':False}
(BASE/'source-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
