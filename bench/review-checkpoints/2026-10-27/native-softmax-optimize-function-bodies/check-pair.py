import ast
from pathlib import Path
root=Path('/tmp/veritile-pr18-review/bench/repairs/tritonbench_g/softmax_optimize')
module=ast.parse((root/'check_index_candidate.py').read_text())
fn=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='check_ast')
ns={'ast':ast,'HERE':root};exec(compile(ast.Module(body=[fn],type_ignores=[]),'check_index_candidate.py','exec'),ns);ns['check_ast']()
print('PASS: original helper identical; candidate exactly four integer pid_m*N -> N*pid_m commutations; complete function AST compared.')
