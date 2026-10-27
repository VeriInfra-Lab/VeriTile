from pathlib import Path
import ast,copy,hashlib,json,subprocess
r=Path.cwd();rel='bench/repairs/tritonbench_g/isfinite_kernel/isfinite_kernel_repaired.py';source=(r/rel).read_text();parent='aea3449603970c3c56d8dc5e8a259737588ce58b';assert source.encode()==subprocess.check_output(['git','show',parent+':'+rel],cwd=r)
tree=ast.parse(source);nodes={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
left=copy.deepcopy(nodes['isfinite_strided_chunk']);right=copy.deepcopy(nodes['isfinite_strided_chunk_index_commuted']);right.name=left.name;prod=right.body[1].value.left;assert isinstance(prod,ast.BinOp) and isinstance(prod.op,ast.Mult);prod.left,prod.right=prod.right,prod.left;assert ast.dump(left,include_attributes=False)==ast.dump(right,include_attributes=False)
original=ast.parse((r/'bench/tritonbench_g/isfinite_kernel/isfinite_kernel.py').read_text());old={n.name:n for n in original.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
for name in ['StridedBuffer','isfinite_func']:assert ast.dump(nodes[name],include_attributes=False)==ast.dump(old[name],include_attributes=False)
print('PASS: unchanged Python/wrapper, original helper/StridedBuffer, and exactly one candidate integer multiplication commute')
