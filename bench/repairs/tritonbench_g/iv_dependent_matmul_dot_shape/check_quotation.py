"""Compare every specialized source statement with the readable DSL template."""
import ast
import copy
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'iv_dependent_matmul_dot_shape.py'
QUOTATION = HERE / 'Kernels.lean'
MODES = {'preLoad':'pre_load', 'postLoad':'post_load', 'mixed':'post_pre_mixed',
         'two':'post_load_two_iters', 'three':'post_load_three_iters'}
NAMES = {'sam':'stride_am','sak':'stride_ak','sbk':'stride_bk','sbn':'stride_bn',
         'scm':'stride_cm','scn':'stride_cn','BM':'BLOCK_SIZE_M','BN':'BLOCK_SIZE_N','BK':'BLOCK_SIZE_K'}

class Normalize(ast.NodeTransformer):
    def visit_Name(self, node):
        node.id = NAMES.get(node.id, 'k' if node.id == 'k64' else node.id)
        return node
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and ast.unparse(node.func) == 'tl.cast':
            assert ast.unparse(node.args[1]) == 'tl.int64'
            return node.args[0]
        if ast.unparse(node.func) == 'range' and len(node.args) == 2:
            node.args.append(ast.Constant(1))
        if ast.unparse(node.func) == 'tl.zeros' and isinstance(node.args[0], ast.Tuple):
            node.args[0] = ast.List(elts=node.args[0].elts, ctx=ast.Load())
        return node
    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.op, ast.Mult):
            if isinstance(node.left, ast.Constant) and node.left.value == 1: return node.right
            if isinstance(node.right, ast.Constant) and node.right.value == 1: return node.left
        # Source pointer offset parentheses are associative integer expressions.
        if isinstance(node.op, ast.Add) and isinstance(node.right, ast.BinOp) and isinstance(node.right.op, ast.Add):
            return self.visit_BinOp(ast.BinOp(ast.BinOp(node.left, ast.Add(), node.right.left), ast.Add(), node.right.right))
        return node


def specialized(body, mode):
    result = []
    for stmt in copy.deepcopy(body):
        if isinstance(stmt, ast.If):
            assert isinstance(stmt.test, ast.Compare) and isinstance(stmt.test.left, ast.Name) and stmt.test.left.id == 'type'
            branch = stmt.body if mode == stmt.test.comparators[0].value else stmt.orelse
            result.extend(specialized(branch, mode))
        elif isinstance(stmt, ast.For):
            stmt.body = specialized(stmt.body, mode); result.append(stmt)
        elif isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call) and ast.unparse(stmt.value.func) == 'tl.cast' and isinstance(stmt.value.args[0], ast.Name):
            # Exactly the selected widening-only scalar and induction casts.
            assert ast.unparse(stmt.value.args[1]) == 'tl.int64'
            assert stmt.targets[0].id == stmt.value.args[0].id or stmt.targets[0].id == 'k64'
        else: result.append(stmt)
    return result


def python_body(text, mode, native):
    fn = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'iv_dependent_matmul_kernel_i64')
    body = specialized(fn.body, mode)
    for stmt in body:
        if isinstance(stmt, ast.Assign) and stmt.targets[0].id in ('a_ptr','b_ptr'):
            base = stmt.targets[0].id
            class Base(ast.NodeTransformer):
                def visit_Name(self, n):
                    if n.id == base: n.id = 'A' if base == 'a_ptr' else 'B'
                    return n
            stmt.value = Base().visit(stmt.value)
    class Store(ast.NodeTransformer):
        def visit_Name(self,n):
            if n.id == 'c_ptr': n.id = 'C'
            return n
        def visit_Call(self,n):
            n=self.generic_visit(n)
            if ast.unparse(n.func) == 'tl.store':
                n.args[1]=ast.Call(ast.Attribute(n.args[1],'to',ast.Load()),[ast.Attribute(ast.Name('tl',ast.Load()),'float32',ast.Load())],[])
            return n
        def visit_AugAssign(self,n):
            if native:
                assert n.target.id == 'accumulator' and ast.unparse(n.value.func) == 'tl.dot'
                n.value.args.append(ast.Name('accumulator',ast.Load()))
                return ast.Assign([n.target],n.value)
            return n
    return Normalize().visit(Store().visit(ast.Module(body=body,type_ignores=[])))


def quoted_body(text, schedule, native):
    pieces = {'initial':[], 'before':[], 'after':[]}
    enabled = False
    variable = None
    for line in text.splitlines():
        if line.startswith('    if schedule.getId =='):
            enabled = schedule in re.findall(r'== `(\w+)',line)
            variable = None
        if enabled and re.match(r'      (initial|before|after) :=',line):
            variable=re.match(r'      (\w+)',line).group(1)
        if enabled and variable and '← `(tritonStmt|' in line:
            snippet=line.split('← `(tritonStmt|',1)[1].strip().rstrip(',]')
            assert snippet.endswith(')')
            pieces[variable].append(snippet[:-1])
    dot = re.search(
        r"^    let dot : TSyntax `tritonStmt ← if graph.getId == `source then\n"
        r"      `\(tritonStmt\| (?P<source>[^\n]+)\)\n"
        r"      else `\(tritonStmt\| (?P<native>[^\n]+)\)$",
        text, re.MULTILINE)
    assert dot is not None, 'unrecognized source/native dot quotation'
    pieces['dot'] = [dot.group('native' if native else 'source')]
    template=text.split('      triton {',1)[1].split('      })',1)[0]
    output=[]; level=0
    for line in template.splitlines():
        line=line.strip()
        if not line or line.startswith('tl.region '): continue
        if line == '}': level-=1; continue
        marker=re.fullmatch(r'\$(initial|before|after):tritonStmt\*|\$(dot)',line)
        if marker:
            for snippet in pieces[marker.group(1) or marker.group(2)]: output.append('    '*level+snippet)
            continue
        if line.endswith('{'):
            output.append('    '*level+line[:-1].rstrip()+':');level+=1
        else: output.append('    '*level+line)
    text='\n'.join(output)
    text=re.sub(r'\$\(([^()]*)\)',r'\1',text).replace('DTYPE','tl.float32')
    return Normalize().visit(ast.parse(text))


def audit(py, lean):
    records=[]
    for schedule, mode in MODES.items():
        for native in (False,True):
            left=python_body(py,mode,native);right=quoted_body(lean,schedule,native)
            a,b=ast.dump(left,include_attributes=False),ast.dump(right,include_attributes=False)
            if a != b:
                for i,(x,y) in enumerate(zip(left.body,right.body)):
                    if ast.dump(x)!=ast.dump(y): raise AssertionError((schedule,native,i,ast.unparse(ast.fix_missing_locations(x)),ast.unparse(ast.fix_missing_locations(y))))
                raise AssertionError((schedule,native,len(left.body),len(right.body)))
            records.append({'schedule':mode,'graph':'live_MMA_seed' if native else 'source_dot_plus_add','statements':len(left.body),'normalized_AST_sha256':hashlib.sha256(a.encode()).hexdigest()})
    return records


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    py,lean=SOURCE.read_text(),QUOTATION.read_text();records=audit(py,lean)
    controls=[]
    mutations=[('tail_mask',py.replace(' < K - k64',' <= K - k64',1),lean),
               ('half_cast',py.replace('accumulator.to(tl.float16)','accumulator.to(tl.float32)'),lean),
               ('lookahead',py.replace('(k64 + 2)','(k64 + 3)',1),lean),
               ('DSL_store_cast',py,lean.replace('c.to(DTYPE)','c.to(tl.float16)')),
               ('DSL_source_dot',py,lean.replace('accumulator += tl.dot(a, b)',
                                               'accumulator += tl.dot(a, a)')),
               ('DSL_native_live_carry',py,lean.replace('accumulator = tl.dot(a, b, accumulator)',
                                                       'accumulator = tl.dot(a, b)'))]
    for name,a,b in mutations:
        try: audit(a,b)
        except AssertionError: controls.append({'mutation':name,'rejected':True})
        else: raise AssertionError('negative control accepted: '+name)
    result={'source':{'path':str(SOURCE),'sha256':hashlib.sha256(py.encode()).hexdigest()},'quotation':{'path':str(QUOTATION),'sha256':hashlib.sha256(lean.encode()).hexdigest()},'profiles':records,'negative_controls':controls,'normalizations':['Erase only explicit int64 widening casts; rename k64 to k. Signed64 correspondence is separately proved.','Substitute the constexpr schedule; normalize integer pointer addition parentheses and unit factors.','Make the source implicit FP16-to-FP32 destination widening explicit.','Source dot+add remains distinct from the observed native live-seed dot lowering.'],'GPU_runs':0,'numerical_experiments':0}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'profile_matches':len(records),'negative_controls_rejected':len(controls)}))
if __name__=='__main__': main()
