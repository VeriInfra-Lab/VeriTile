"""Extract only original JIT functions and generate frontend TTIR without a GPU."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton._C.libtriton import ir
import triton.language.semantic as semantic

ROOT = Path(__file__).resolve().parent
REPO = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
OUT = ROOT/'evidence/frontend'
OUT.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
tasks = ['token_softmax_bloom','token_softmax_llama']
types = {'fp16':'fp16','bf16':'bf16','fp32':'fp32'}
target = GPUTarget('cuda',90,32)
backend = CUDABackend(target)
options = backend.parse_options({'num_warps':4})
context = ir.context()
ir.load_dialects(context)
backend.load_dialects(context)
sources, profiles, asts = {}, [], []
for task in tasks:
    source = REPO/'bench/tritonbench_g'/task/(task+'.py')
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n,ast.FunctionDef) and n.name=='_fwd_kernel_token_softmax')
    original_ast = ast.dump(node,include_attributes=False)
    asts.append(original_ast)
    lines = source.read_text().splitlines(keepends=True)
    first = min([node.lineno]+[d.lineno for d in node.decorator_list])
    text = 'import triton\nimport triton.language as tl\n'+''.join(lines[first-1:node.end_lineno])
    extracted = OUT/(task+'_kernel.py')
    extracted.write_text(text)
    extracted_node = next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef))
    assert original_ast == ast.dump(extracted_node,include_attributes=False)
    sources[task] = {'path':source.relative_to(REPO).as_posix(),'sha256':sha(source),
        'kernel_lines':[first,node.end_lineno],'ast_sha256':hashlib.sha256(original_ast.encode()).hexdigest(),
        'extracted_path':extracted.relative_to(ROOT).as_posix(),'extracted_sha256':sha(extracted),
        'ast_identical':True,'original_module_imported':False}
    spec = importlib.util.spec_from_file_location(task+'_mixed_output_probe',extracted)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for input_type in types:
        for output_type in types:
            signature = {'Logics':'*'+types[input_type],'B_Start_Loc':'*i32','B_Seqlen':'*i32',
                'Prob_Out':'*'+types[output_type],'stride_logic_h':'i32','stride_logic_bs':'i32',
                'stride_prob_h':'i32','stride_prob_bs':'i32'}
            row = {'task':task,'input_dtype':input_type,'output_dtype':output_type,'signature':signature}
            try:
                src = ASTSource(module._fwd_kernel_token_softmax,signature,constexprs={'BLOCK_SIZE':1024})
                generated = src.make_ir(target,options,backend.get_codegen_implementation(options),backend.get_module_map(),context)
                path = OUT/(task+'__'+input_type+'__'+output_type+'.raw.ttir')
                path.write_text(str(generated))
                row.update(status='frontend_generated',ttir_path=path.relative_to(ROOT).as_posix(),ttir_sha256=sha(path))
            except Exception as e:
                row.update(status='frontend_error',error_type=type(e).__name__,error=str(e))
            profiles.append(row)
            print(task,input_type,output_type,row['status'],flush=True)
assert asts[0] == asts[1]
result = {'schema':1,'sources':sources,'kernel_ast_identical_between_tasks':True,
    'profiles':profiles,'frontend_successes':sum(r['status']=='frontend_generated' for r in profiles),
    'environment':{'python':sys.version,'triton':triton.__version__,'semantic_path':semantic.__file__,
                   'semantic_sha256':sha(Path(semantic.__file__))},
    'target':{'backend':'cuda','architecture':'sm90','warp_size':32},'options':vars(options),
    'constexprs':{'BLOCK_SIZE':1024},'original_module_imported':False,'native_backend_compiled':False,
    'gpu_execution':False,'scope':'Only the selected 3x3 floating-storage routes at this frontend configuration are observed. No compiler correctness, native numerical conformance, host grid composition, or whole-program acceptance follows.'}
(OUT/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
assert result['frontend_successes']==18
print('receipt',sha(OUT/'receipt.json'))
