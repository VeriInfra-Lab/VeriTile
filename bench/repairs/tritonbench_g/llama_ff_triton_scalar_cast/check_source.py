"""Static source/quotation/artifact audit. No Tensor allocation or execution."""
import ast
import copy
import hashlib
import json
import re
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EVIDENCE = ROOT / '.veritile-cache/task-deliveries/llama_ff_triton_scalar_cast'
ORIGINAL = HERE.parent / 'llama_ff_triton/llama_ff_triton_repaired.py'
SOURCE = HERE / 'llama_ff_triton_scalar_cast.py'
SCALARS = ['M', 'N', 'K', 'stride_am', 'stride_ak', 'stride_w1k', 'stride_w1n',
           'stride_w3k', 'stride_w3n', 'stride_outm', 'stride_outn', 'stride_rms_w']
EXTERNAL = dict(zip(SCALARS, ['M','N','K','sam','sak','sw1k','sw1n','sw3k','sw3n','soutm','soutn','srms']))
REGISTERS = {k: v + '64' for k, v in EXTERNAL.items()}
RENAMES = {**REGISTERS, 'a_ptr':'A','w1_ptr':'W1','w3_ptr':'W3','out_ptr':'OUT','rms_w_ptr':'RMS',
           'BLOCK_SIZE_M':'BM','BLOCK_SIZE_N':'BN','BLOCK_SIZE_K':'BK'}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dump(node):
    return ast.dump(node, include_attributes=False)

class Normalize(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'tl' and node.func.attr == 'cdiv':
            assert len(node.args) == 2 and not node.keywords
            a, b = node.args
            return ast.BinOp(left=ast.BinOp(left=ast.BinOp(left=a, op=ast.Add(), right=copy.deepcopy(b)), op=ast.Sub(), right=ast.Constant(value=1)), op=ast.FloorDiv(), right=b)
        return node

    def visit_List(self, node):
        return ast.Tuple(elts=[self.visit(x) for x in node.elts], ctx=node.ctx)

class SourceBranch(Normalize):
    def __init__(self, byte):
        self.byte = byte

    def visit_Name(self, node):
        return ast.Name(id=RENAMES.get(node.id,node.id),ctx=node.ctx)

    def visit_If(self, node):
        assert isinstance(node.test,ast.Name) and node.test.id=='USE_FP8'
        assert not node.orelse
        return [self.visit(n) for n in node.body] if self.byte else []

    def visit_Assign(self,node):
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in SCALARS:
            name=node.targets[0].id
            expected=ast.parse(f'{name} = tl.cast({name}, tl.int64)').body[0]
            assert dump(node)==dump(expected)
            return ast.parse(f'{REGISTERS[name]} = tl.cast({EXTERNAL[name]}, tl.int64)').body[0]
        return self.generic_visit(node)

    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and dump(node.func)==dump(ast.parse('tl.store',mode='eval').body):
            # The shared typed store makes the source's implicit FP16 cast explicit.
            assert dump(node.args[1])==dump(ast.Name(id='accumulator',ctx=ast.Load()))
            node.args[1]=ast.parse('accumulator.to(tl.float16)',mode='eval').body
        return Normalize.visit_Call(self, node)

def quote_branch(name):
    text=(HERE/'Kernels.lean').read_text()
    section=text.split(f'def {name} ',1)[1].split('(triton {',1)[1].split('}).surfaceBody',1)[0]
    lines=[]
    for line in textwrap.dedent(section).splitlines():
        if line.strip().startswith('tl.region ') or line.strip()=='}':
            continue
        if line.rstrip().endswith('{'):
            line=line.rstrip()[:-1].rstrip()+':'
        line=re.sub(r'\$\((\w+)\)',r'\1',line)
        lines.append(line)
    return Normalize().visit(ast.parse(textwrap.dedent('\n'.join(lines))))

def check():
    original=ORIGINAL.read_text()
    expected=original
    for name in SCALARS:
        old=f'{name} = {name}.to(tl.int64)'
        assert expected.count(old)==1
        expected=expected.replace(old,f'{name} = tl.cast({name}, tl.int64)')
    assert SOURCE.read_text()==expected, 'Change exceeds the twelve scalar casts'
    assert sha(SOURCE)=='c603fead96e4e516f81824f1312d3fa80c2e999c2ad50883e9cb7b50d15d71eb'
    historical=json.loads((EVIDENCE.parent/'llama_ff_triton/historical.json').read_text())
    assert len(historical['files'])==25
    for path,digest in historical['files'].items():
        assert sha(ROOT/path)==digest, ('Historical file changed',path)
    node=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='ff_llama')
    body=node.body[1:]
    comparisons={}
    for name,byte in [('half',False),('byte',True)]:
        expected=SourceBranch(byte).visit(ast.Module(body=copy.deepcopy(body),type_ignores=[]))
        actual=quote_branch(name)
        assert dump(expected)==dump(actual), (name,ast.unparse(expected),ast.unparse(actual))
        comparisons[name]={'source_ast_sha256':hashlib.sha256(dump(expected).encode()).hexdigest(),
                           'quotation_ast_sha256':hashlib.sha256(dump(actual).encode()).hexdigest(),
                           'top_level_statements':len(actual.body),'exact_match':True}
    audit=json.loads((EVIDENCE/'source-audit.json').read_text())
    assert audit['actual_jit_compiles']==10 and audit['empty_launches_skipped']==2
    assert audit['specialization_unchanged']
    assert audit['tensor_allocations']==audit['gpu_runs']==audit['numerical_experiments']==0
    artifacts=[]
    for case in audit['cases']:
        if case['launch_count']==0:
            continue
        compiled=case['selected']
        assert compiled['status']=='compiled'
        assert compiled['specialization']['stride_ak']==['constexpr',1]
        for artifact in compiled['artifacts'].values():
            assert sha(Path(artifact['path']))==artifact['sha256']
        ptx=Path(compiled['artifacts']['ptx']['path']).read_text()
        mma=re.findall(r'\bmma\.sync\.aligned\.m16n8k16\.row\.col\.f32\.f16\.f16\.f32\b',ptx)
        assert len(mma)==16, len(mma)
        assert ptx.count('fma.rn.f32')==8
        direct_e5=ptx.count('cvt.rn.f16x2.e5m2x2')
        assert direct_e5==(40 if case['weight_storage']=='int8_e5m2' else 0)
        artifacts.append({'shape':case['shape_B_S_N_K'],'storage':case['weight_storage'],
                          'mma_m16n8k16':len(mma),'terminal_fma':8,
                          'direct_e5m2_to_half_instructions':direct_e5})
    result={'source':{'path':str(SOURCE),'sha256':sha(SOURCE)},'quotation':{'path':str(HERE/'Kernels.lean'),'sha256':sha(HERE/'Kernels.lean')},
            'historical_files_unchanged':25,'scalar_cast_changes':12,'domain_changed':False,
            'branch_correspondence':comparisons,'actual_binder_compiles':10,'empty_launches':2,
            'selected_native_artifacts':artifacts,
            'normalizations':['alpha-renaming of scalar SSA registers and pointer/block parameters',
              'static USE_FP8 branch selection','tuple/list shape syntax','signed cdiv expansion (a+b-1)//b','implicit output half store cast'],
            'tensor_allocations':0,'gpu_runs':0,'numerical_experiments':0}
    (EVIDENCE/'quotation-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    check()
