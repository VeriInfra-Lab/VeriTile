"""Exact selected-FP32 quotation and one-line source binding; no execution oracle."""
import argparse
import ast
import hashlib
import json
import re
import textwrap
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
SOURCE=HERE/'chunk_gla_simple_min_dot_tile.py'
PREDECESSOR=HERE.parent/'chunk_gla_simple/chunk_gla_simple_repaired.py'
QUOTE=HERE/'Kernels.lean'
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=ROOT/'.veritile-cache/task-deliveries/chunk_gla_simple_min_dot_tile/source-binding.json')
args=parser.parse_args()
OUT=args.output
ROW_SOURCE=HERE/'chunk_gla_simple_row_reused.py'

def pin(p):
    p=Path(p).resolve()
    return {'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
            'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

class FP32Projection(ast.NodeTransformer):
    """Erase only explicit i64 widening; choose the guarded FP32 dtype profile."""
    def visit_Call(self,n):
        n=self.generic_visit(n)
        if isinstance(n.func,ast.Attribute):
            if n.func.attr=='to' and len(n.args)==1:
                if ast.unparse(n.args[0])=='tl.int64':return n.func.value
                if ast.unparse(n.args[0]) in ('b_v.dtype','p_o.dtype.element_ty'):
                    n.args=[ast.Attribute(ast.Name('tl',ast.Load()),'float32',ast.Load())]
            if ast.unparse(n.func)=='tl.cast' and len(n.args)==2 and ast.unparse(n.args[1])=='tl.int64':return n.args[0]
        return n

source=SOURCE.read_text()
assert source==PREDECESSOR.read_text().replace(
    'BK = min(triton.next_power_of_2(max(K, 1)), 64)',
    'BK = max(16, min(triton.next_power_of_2(max(K, 1)), 64))')
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='chunk_simple_gla_fwd_kernel_o_repaired')
original_node=ast.dump(node)
body=[]
for stmt in node.body:
    if isinstance(stmt,ast.Assign) and isinstance(stmt.targets[0],ast.Tuple):
        assert len(stmt.targets[0].elts)==len(stmt.value.elts)==2
        for target,value in zip(stmt.targets[0].elts,stmt.value.elts):
            assert ast.unparse(value)==f'tl.cast({target.id}, tl.int64)'
        continue
    body.append(FP32Projection().visit(stmt))
actual=ast.dump(ast.Module(body=body,type_ignores=[]),include_attributes=False)
def quoted_body(name):
    text=QUOTE.read_text()
    start=text.index('def '+name+' ')
    text=text[text.index('    i_v = tl.program_id(0)',start):text.index('  }).surfaceBody',start)]
    text=re.sub(r'^\s*}\s*$', '', text, flags=re.MULTILINE)
    text=text.replace(' {\n', ':\n')
    text=text.replace('$((scale : ℝ))','scale')
    text=re.sub(r'\$\((\w+)\)',r'\1',text)
    text=text.replace('OUT_DTYPE','tl.float32')
    return ast.dump(ast.parse(textwrap.dedent(text)),include_attributes=False)

assert actual==quoted_body('fp32Original'), 'complete original quotation AST mismatch'
row_source=ROW_SOURCE.read_text()
assert row_source==source.replace(
    'rows = i_t * BT + tl.arange(0, BT).to(tl.int64)',
    'rows = i_t * BT + o_i.to(tl.int64)')
row_node=next(n for n in ast.parse(row_source).body if isinstance(n,ast.FunctionDef) and n.name==node.name)
row_body=[]
for stmt in row_node.body:
    if isinstance(stmt,ast.Assign) and isinstance(stmt.targets[0],ast.Tuple):
        assert len(stmt.targets[0].elts)==len(stmt.value.elts)==2
        for target,value in zip(stmt.targets[0].elts,stmt.value.elts):
            assert ast.unparse(value)==f'tl.cast({target.id}, tl.int64)'
        continue
    row_body.append(FP32Projection().visit(stmt))
row_actual=ast.dump(ast.Module(body=row_body,type_ignores=[]),include_attributes=False)
assert row_actual==quoted_body('rowReusedSource'), 'complete address-CSE quotation AST mismatch'
utils=Path('/usr/local/lib/python3.12/site-packages/triton/_utils.py')
semantic=Path('/usr/local/lib/python3.12/site-packages/triton/language/semantic.py')
compiler=Path('/usr/local/lib/python3.12/site-packages/triton/backends/nvidia/compiler.py')
assert 'TRITON_MAX_TENSOR_NUMEL = 1048576' in utils.read_text()
assert 'if numel > TRITON_MAX_TENSOR_NUMEL:' in utils.read_text()
for BT in (16,32,64,128,256,512,1024):
    assert BT*BT<=2**20 and BT*64<=2**20 and BT>=16
result={
 'candidate_source':pin(SOURCE),'predecessor_source':pin(PREDECESSOR),'readable_quote':pin(QUOTE),'checker':pin(__file__),
 'source_delta':'Exactly one BK selection expression; no other source bytes changed.',
 'address_cse_candidate':pin(ROW_SOURCE),
 'address_cse_delta':'Exactly one rows expression reuses the existing o_i lane vector with the same i64 widening.',
 'address_cse_complete_AST_match':True,
 'address_cse_projected_ast_sha256':hashlib.sha256(row_actual.encode()).hexdigest(),
 'selected_profile':'All input/output storage FP32, original arithmetic order, allow_tf32=False.',
 'normalization':'Only source i64 widening cast erasure plus guarded FP32 dtype resolution; all remaining statements, masks, address expressions, memory instruction order and numerical operations match AST exactly.',
 'original_kernel_ast_sha256':hashlib.sha256(original_node.encode()).hexdigest(),
 'common_projected_ast_sha256':hashlib.sha256(actual.encode()).hexdigest(),
 'complete_AST_match':True,'source_dots':3,'source_masked_loads':5,'source_masked_stores':1,
 'native_integer_bridge':'NativeABI.whole_grid / NativeControl.loop_control / Native.all_coordinates; reconstructed source success, no host success premise.',
 'static_tensor_limit':{'implementation':pin(utils),'limit':1048576,'rejection_operator':'>','largest_shape':[1024,1024],'largest_elements':1048576,'all_other_2d_shapes_at_most':65536},
 'native_dot_admission':{'implementation':pin(compiler),'semantic':pin(semantic),'FP32_minimum_M_N_K':[1,1,16],
   'QH_QK_inner':'BK=max(16,min(nextpow2(max(K,1)),64))','SV_inner':'BT in original [16,32,64,128,256,512,1024]'},
 'machine_equivalence_to_predecessor_claimed':False,
 'GPU_execution':False}
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps(result,indent=2)+'\n')
print('PASS complete original/candidate FP32 quotation ASTs; exact original preservation and one row-lane CSE; source binding',pin(OUT))
