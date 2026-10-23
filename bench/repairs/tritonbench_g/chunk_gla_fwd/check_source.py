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
EVIDENCE=ROOT/'.veritile-cache/task-deliveries/chunk_gla_fwd'
SOURCE=HERE/'chunk_gla_fwd_repaired.py'
NAMES=['chunk_gla_fwd_A_kernel_intra_sub_inter','chunk_gla_fwd_A_kernel_intra_sub_intra',
       'chunk_gla_fwd_A_kernel_intra_sub_intra_split','chunk_gla_fwd_A_kernel_intra_sub_intra_merge',
       'chunk_gla_fwd_kernel_o']
SELECTED=[{'$add':'b_A += tl.dot(b_qg, b_kg)'},
          {'$scaled':'b_A = tl.where(o_i >= j, b_A * scale, 0.0)'},
          {'$scaled':'b_A = tl.where(o_i >= j, b_A * scale, 0.0)'},
          {'$add':'b_A += tl.load(p_A, boundary_check=(0, 1), padding_option="zero")'},
          {'$finalAdd':'b_o += tl.dot(b_A, b_v, allow_tf32=False)'}]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(n):return ast.dump(n,include_attributes=False)
def fp32():return ast.parse('tl.float32',mode='eval').body

class Normalize(ast.NodeTransformer):
    def visit_List(self,n):return ast.Tuple(elts=[self.visit(x) for x in n.elts],ctx=n.ctx)
    def visit_Name(self,n):return fp32() if n.id in ['OUT_DTYPE','DTYPE'] else n
    def visit_Attribute(self,n):
        if n.attr=='dtype' or n.attr=='element_ty' and isinstance(n.value,ast.Attribute) and n.value.attr=='dtype':return fp32()
        return self.generic_visit(n)
    def visit_Assign(self,n):
        if len(n.targets)==1 and isinstance(n.targets[0],ast.Tuple) and isinstance(n.value,ast.Tuple):
            return [ast.Assign(targets=[self.visit(a)],value=self.visit(b)) for a,b in zip(n.targets[0].elts,n.value.elts)]
        return self.generic_visit(n)
    def visit_Call(self,n):
        n=self.generic_visit(n)
        if isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='tl':
            if n.func.attr in ['max_contiguous','multiple_of']:return n.args[0]
            if n.func.attr=='make_block_ptr':
                names=['base','shape','strides','offsets','block_shape','order']
                args=dict(zip(names,n.args));args.update({k.arg:k.value for k in n.keywords})
                assert set(args)==set(names)
                n.args=[args[name] for name in names];n.keywords=[]
        return n

def quote(index):
    text=(HERE/'Kernels.lean').read_text().split('(triton {')[index+1].split('}).surfaceBody',1)[0]
    for before,after in SELECTED[index].items():text=text.replace(before,after)
    text=re.sub(r'\$\(\((\w+)\s*:\s*ℝ\)\)',r'\1',text)
    text=re.sub(r'\$\((\w+)\)',r'\1',text)
    text=re.sub(r'\(\((\w+)\s*:\s*ℝ\)\)',r'\1',text)
    text=re.sub(r'\(\((\w+)\s*:\s*ℝ\)\)',r'\1',text)
    text=re.sub(r'\(\[(.*?)\]\s*:\s*List Nat\)',r'(\1)',text)
    text=re.sub(r'\$[qkvg]Ty:tritonDType','tl.float32',text)
    lines=[];in_region=False
    for line in textwrap.dedent(text).splitlines():
        if line.strip().startswith('tl.region ') or in_region:
            in_region=line.rstrip().endswith(',');continue
        if line.strip()=='}':continue
        if '{ return }' in line:
            indent=line[:len(line)-len(line.lstrip())]
            line=line.replace('{ return }',':')+'\n'+indent+'  return'
        elif line.rstrip().endswith('{'):line=line.rstrip()[:-1].rstrip()+':'
        lines.append(line)
    result=textwrap.dedent('\n'.join(lines))
    return Normalize().visit(ast.parse(result))

def main():
    assert sha(SOURCE)=='3e983e484088f08c6d8690520227efc0a2d89a0fa0c4a7f4897879fa58faefb5'
    source=ast.parse(SOURCE.read_text());comparisons={}
    for index,name in enumerate(NAMES):
        node=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name==name)
        expected=Normalize().visit(ast.Module(body=copy.deepcopy(node.body),type_ignores=[]))
        actual=quote(index)
        assert dump(expected)==dump(actual),(name,ast.unparse(ast.fix_missing_locations(expected)),ast.unparse(ast.fix_missing_locations(actual)))
        comparisons[name]={'exact_match':True,'statements':len(actual.body),
            'normalized_ast_sha256':hashlib.sha256(dump(actual).encode()).hexdigest()}
    result={'source':{'path':str(SOURCE.relative_to(ROOT)),'sha256':sha(SOURCE)},
        'quotation':{'path':str((HERE/'Kernels.lean').relative_to(ROOT)),'sha256':sha(HERE/'Kernels.lean')},
        'kernels':comparisons,'normalizations':['static FP32 dtype spellings','tuple/list syntax',
        'tuple assignment into independent scalar assignments','block pointer keyword/positional syntax',
        'selected original-order macro branches','erase five value-identity alignment hints absent from Python'],
        'gpu_runs':0,'tensor_allocations':0,'numerical_experiments':0}
    (EVIDENCE/'quotation-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS: all five complete selected Python and shared quotation bodies match')

if __name__=='__main__':main()
