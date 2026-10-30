"""Current source pins and exact accepted JIT/host AST comparison; no GPU claim."""
import ast
import hashlib
import json
from pathlib import Path
D=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
checked=D/'cross_entropy1_repaired.py'
conditional=D/'cross_entropy1_conditional.py'
assert sha(conditional)=='edfc7f0c6adb8c9a2050660c04d43034083fdefb58f068192208d91f676278d5'
assert sha(checked) in (D/'Contract.lean').read_text()
assert sha(conditional) in (D/'Proofs/ActualCalls.lean').read_text()
def source(path,strip):
    nodes=[]
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,ast.FunctionDef) and node.name.endswith('_kernel'): nodes.append(node)
        if isinstance(node,ast.ClassDef) and node.name=='CrossEntropyLoss':
            if strip:
                for f in node.body:
                    if isinstance(f,ast.FunctionDef):
                        f.body=[n for n in f.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
                            and isinstance(n.value.func,ast.Name) and n.value.func.id.startswith('_check_'))]
            nodes.append(node)
    return [ast.dump(n,include_attributes=False) for n in nodes]
assert source(checked,True)==source(conditional,False)
print(json.dumps({'checked_sha256':sha(checked),'conditional_sha256':sha(conditional),
                 'unchanged_kernel_and_host_AST':True,'gpu_execution':False}))
