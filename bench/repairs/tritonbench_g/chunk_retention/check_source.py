"""Compare every selected Python body with its readable shared quotation."""
import ast
import copy
import hashlib
import json
import re
import textwrap
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
EVIDENCE=ROOT/'.veritile-cache/task-deliveries/chunk_retention'
SOURCE=HERE/'Whole001/chunk_retention_repaired.py'
NAMES=['chunk_retention_fwd_kernel_h','chunk_retention_fwd_kernel_o',
       'chunk_retention_bwd_kernel_dqkv','chunk_retention_bwd_kernel_dh']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(n):return ast.dump(n,include_attributes=False)
def fp32():return ast.parse('tl.float32',mode='eval').body

class Normalize(ast.NodeTransformer):
    def visit_Constant(self,n):
        # 1.0 is retained as floating syntax where it forces i_h promotion;
        # the left operand of b_b subtraction is implicitly promoted by Python.
        return n
    def visit_For(self,n):
        n=self.generic_visit(n)
        if dump(n.iter)==dump(ast.parse('range(NT - 1, -1, -1)',mode='eval').body):
            n.target=ast.Name(id='j',ctx=ast.Store())
            n.iter=ast.parse('range(0, NT, 1)',mode='eval').body
            n.body.insert(0,ast.parse('i_t = NT - 1 - j').body[0])
        return n

    def visit_List(self,n):return ast.Tuple(elts=[self.visit(x) for x in n.elts],ctx=n.ctx)
    def visit_Name(self,n):return fp32() if n.id in ['OUT_DTYPE','DTYPE'] else n
    def visit_Attribute(self,n):
        if n.attr=='dtype' or n.attr=='element_ty' and isinstance(n.value,ast.Attribute) and n.value.attr=='dtype':return fp32()
        return self.generic_visit(n)
    def visit_Assign(self,n):
        if len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='b_b':
            n.value.args[0].left=ast.Constant(value=1.0)
        if len(n.targets)==1 and isinstance(n.targets[0],ast.Tuple) and isinstance(n.value,ast.Tuple):
            return [ast.Assign(targets=[self.visit(a)],value=self.visit(b)) for a,b in zip(n.targets[0].elts,n.value.elts)]
        return self.generic_visit(n)
    def visit_Call(self,n):
        n=self.generic_visit(n)
        if isinstance(n.func,ast.Attribute) and n.func.attr=='to' and n.args and dump(n.args[0])==dump(ast.parse('tl.int32',mode='eval').body):
            assert isinstance(n.func.value,ast.Call) and dump(n.func.value.func)==dump(ast.parse('tl.arange',mode='eval').body)
            return n.func.value
        if isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='tl':
            if n.func.attr in ['max_contiguous','multiple_of']:return n.args[0]
            if n.func.attr=='make_block_ptr':
                names=['base','shape','strides','offsets','block_shape','order']
                args=dict(zip(names,n.args));args.update({k.arg:k.value for k in n.keywords})
                assert set(args)==set(names)
                n.args=[args[name] for name in names];n.keywords=[]
        return n

def quote(index):
    text=(HERE/'Whole001/Kernels.lean').read_text().split('(triton {')[index+1].split('}).surfaceBody',1)[0]
    text=re.sub(r'\$\(\(([^()]+?)\s*:\s*ℝ\)\)',r'\1',text)
    text=re.sub(r'\$\((\w+)\)',r'\1',text)
    text=text.replace('do_', 'do')
    text=re.sub(r'\(\((\w+)\s*:\s*ℝ\)\)',r'\1',text)
    text=re.sub(r'\(\((\w+)\s*:\s*ℝ\)\)',r'\1',text)
    text=re.sub(r'\(\[(.*?)\]\s*:\s*List Nat\)',r'(\1)',text)
    text=re.sub(r'\$[qkvg]Ty:tritonDType','tl.float32',text)
    lines=[];in_region=False
    for line in textwrap.dedent(text).splitlines():
        if line.strip().startswith('tl.region ') or in_region:
            in_region=line.rstrip().endswith(',');continue
        if line.strip()=='}':continue
        if line.strip()=='} else {':
            lines.append(line.replace('} else {', 'else:'));continue
        if '{ return }' in line:
            indent=line[:len(line)-len(line.lstrip())]
            line=line.replace('{ return }',':')+'\n'+indent+'  return'
        elif line.rstrip().endswith('{'):line=line.rstrip()[:-1].rstrip()+':'
        lines.append(line)
    result=textwrap.dedent('\n'.join(lines))
    return Normalize().visit(ast.parse(result))

def main():
    assert sha(SOURCE)=='25a61c95d8536363844489c3d618b6264cf7142bec0e26ff7917c0e5dcf172bd'
    source=ast.parse(SOURCE.read_text());comparisons={}
    for index,name in enumerate(NAMES):
        node=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name==name)
        expected=Normalize().visit(ast.Module(body=copy.deepcopy(node.body),type_ignores=[]))
        actual=quote(index)
        assert dump(expected)==dump(actual),(name,ast.unparse(ast.fix_missing_locations(expected)),ast.unparse(ast.fix_missing_locations(actual)))
        comparisons[name]={'exact_match':True,'statements':len(actual.body),
            'normalized_ast_sha256':hashlib.sha256(dump(actual).encode()).hexdigest()}
    result={'source':{'path':str(SOURCE.relative_to(ROOT)),'sha256':sha(SOURCE)},
        'quotation':{'path':str((HERE/'Whole001/Kernels.lean').relative_to(ROOT)),'sha256':sha(HERE/'Whole001/Kernels.lean')},
        'kernels':comparisons,'normalizations':['static FP32 dtype spellings','tuple/list syntax',
        'tuple assignment into independent scalar assignments','block pointer keyword/positional syntax',
        'explicit source arange signed32 type','reverse range represented by forward counter and NT-1-j','implicit floating promotion of b_b left literal'],
        'gpu_runs':0,'tensor_allocations':0,'numerical_experiments':0}
    (EVIDENCE/'quotation-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS: all four complete selected Python and shared quotation bodies match')

if __name__=='__main__':main()
