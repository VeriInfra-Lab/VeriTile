"""Full candidate kernel/typed-DSL AST correspondence; no host/numeric run."""
from pathlib import Path
import ast
import copy
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = HERE / 'kernel-evidence'
SOURCE = HERE / 'int8_dequant_matmul_scalar_cast.py'
DSL = HERE / 'Kernels.lean'
source = next(node for node in ast.parse(SOURCE.read_text()).body
              if isinstance(node, ast.FunctionDef) and node.name == '_int8_matmul_rowwise_dequantize')
body = DSL.read_text().split('(triton {', 1)[1].split('}).surfaceBody', 1)[0]
types = body.splitlines()[1].strip()
assert types == 'tl.region a_ptr = tl.int8, b_ptr = tl.int8, c_ptr = tl.float16, bias_ptr = tl.float32, state_x_ptr = tl.float32, state_w_ptr = tl.float32'
body = body.replace(types, '')
assert 'tl.param divfactor : tl.float32 (carrier=tl.float32)' in body
body = body.replace('tl.param divfactor : tl.float32 (carrier=tl.float32)', '')
body = re.sub(r'\$\(([^()]+)\)', lambda match: '(' + match[1] + ')', body)
body = re.sub(r'\bsteps\b', 'tl.cdiv(K, BLOCK_K * SPLIT_K)', body)
names = dict(a_ptr='A', b_ptr='B', c_ptr='C', bias_ptr='bias',
             acc_int='acc', scaled='acc', out='acc')
names.update({name+'64': name for name in ['M', 'N', 'K', 'stride_am', 'stride_ak',
              'stride_bk', 'stride_bn', 'stride_cm', 'stride_cn']})
for before, after in names.items():
    body = re.sub(r'\b'+before+r'\b', after, body)
lines, indent = [], 0
for line in body.splitlines():
    line = line.strip()
    if not line:
        continue
    if line == '}':
        indent -= 1
        continue
    if line == '} else {':
        indent -= 1
        lines.append('    '*indent+'else:')
        indent += 1
    elif line.endswith('{'):
        lines.append('    '*indent+line[:-1].rstrip()+':')
        indent += 1
    else:
        lines.append('    '*indent+line)
assert indent == 0
translated = '\n'.join(lines)+'\n'
(OUT / 'normalized-dsl.py.txt').write_text(translated)

class Normalize(ast.NodeTransformer):
    def visit_List(self, node):
        return ast.copy_location(ast.Tuple([self.visit(x) for x in node.elts], node.ctx), node)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == 'range' and len(node.args) == 3:
            assert ast.literal_eval(node.args[-1]) == 1
            node.args.pop()
        return node

    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        if ast.unparse(node) == 'C.dtype.element_ty':
            return ast.Attribute(ast.Name('tl', ast.Load()), 'float16', ast.Load())
        return node

def normalize(statements):
    module = ast.Module(copy.deepcopy(statements), [])
    return ast.dump(Normalize().visit(module), include_attributes=False)

dsl = ast.parse(translated)
assert len(dsl.body) == len(source.body)
for index, (left, right) in enumerate(zip(dsl.body, source.body)):
    assert normalize([left]) == normalize([right]), (index, ast.unparse(left), ast.unparse(right))
assert normalize(dsl.body) == normalize(source.body)

record = {'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          'dsl_sha256': hashlib.sha256(DSL.read_bytes()).hexdigest(),
          'full_kernel_AST_equal': True, 'top_level_statements': len(source.body),
          'normalized_ast_sha256': hashlib.sha256(normalize(source.body).encode()).hexdigest(),
          'normalizations': ['SSA register names', 'list/tuple tile syntax',
                             'explicit range step 1', 'steps = cdiv(K,BK*S)',
                             'C pointer dtype is the original fixed FP16',
                             'natural descriptor index specialization after exact signed64 casts'],
          'retained_branches': ['EVEN_K', 'has_bias', 'pid_z == 0', 'SPLIT_K == 1'],
          'runtime_divisor': 'explicit FP32 ABI parameter with FP32-to-Real algorithm view, bound by invocation; full AST preserved',
          'no_host_or_primitive_numerical_experiment': True}
(OUT / 'source-validation.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2))
