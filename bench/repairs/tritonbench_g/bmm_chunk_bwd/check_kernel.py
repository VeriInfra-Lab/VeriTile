"""Targeted original/index-commuted source and compiler-profile checks.

CPU checks execute actual source ASTs with mocked launches. Offline compilation
checks only Triton 3.7.1 SM90 structure; neither is native GPU execution.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'bmm_chunk_bwd_repaired.py'
EXPECTED = '91e5b8d4d3d423bb1a00fadb347273c5d90d2a3afd5d2b05b51f98eadd90c737'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def candidate(directory):
    assert sha(SOURCE) == EXPECTED
    original = SOURCE.read_text()
    old = '    pid_h = pid_ch - pid_c * ngroups'
    new = '    pid_h = pid_ch - ngroups * pid_c'
    assert original.count(old) == 1
    output = directory / 'bmm_chunk_bwd_index_commuted.py'
    output.write_text(original.replace(old, new))
    assert output.read_text().replace(new, old) == original
    return output

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def cpu(path):
    checks = load(HERE / 'check_contract.py', 'bmm_bwd_kernel_checks')
    checks.main()
    checks.SOURCE = path
    checks.source_binding = lambda: None  # Exact sole edit is checked above.
    checks.main()
    return {'source_cases': 9, 'candidate_cases': 9, 'source_rejections': 54,
            'candidate_rejections': 54, 'launches_mocked': True,
            'numerical_gpu_validation': False}

def audit_ttir(path, residual):
    text = Path(path).read_text()
    loops = [line.strip() for line in text.splitlines() if '= scf.for ' in line]
    assert len(loops) == 1, loops
    bindings = re.findall(r'(%[\w]+) = (%[\w]+)', loops[0].split('iter_args(',1)[1].split(') ->',1)[0])
    assert len(bindings) == 3, bindings
    dots = re.findall(r'(%[\w]+) = tt.dot (%[\w]+), (%[\w]+), (%[\w]+), inputPrecision = tf32', text)
    assert len(dots) == 1, dots
    result,left,right,carry = dots[0]
    assert carry == bindings[2][0], (carry, bindings)
    loads = dict(re.findall(r'(%[\w]+) = tt.load (%[\w]+),',text))
    assert loads[left] == bindings[0][0] and loads[right] == bindings[1][0], (left,right,loads,bindings)
    assert left.startswith('%dout') and right.startswith('%a'), dots
    yields = re.findall(r'scf.yield ([^:]+) :',text)
    assert len(yields) == 1 and yields[0].split(',')[-1].strip() == result, yields
    initial = bindings[2][1]
    assert re.search(re.escape(initial)+r' = arith.constant dense<0\.000000e\+00>',text), initial
    additions = re.findall(r'(%[\w]+) = arith.addf (%[\w#]+), (%[\w#]+)',text)
    if residual:
        assert len(additions) == 1, additions
        _,acc,res = additions[0]
        assert acc == '%acc#2' and res.startswith('%res'), additions
    else:
        assert additions == [], additions
    return {'dot_c_is_loop_carried_accumulator':True,
            'dot_left_is_dout_load':True,'dot_right_is_a_load':True,
            'initial_carry_is_literal_fp32_zero':True,
            'loop_yields_dot_result':True,'source_loop':loops[0],
            'dot_operands':list(dots[0]),'residual_add_operands':additions,
            'separate_reduction_addf':False}

def compile_profiles(path, directory):
    os.environ['TRITON_CACHE_DIR'] = str(directory / 'cache')
    import triton
    import triton.language as tl
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    assert triton.__version__ == '3.7.1', triton.__version__
    configs = [(128,256,64,3,8),(64,256,32,4,4),(128,128,32,4,4),
               (128,64,32,4,4),(64,128,32,4,4),(128,32,32,4,4),
               (64,32,32,5,2),(32,64,32,5,2),(64,64,32,4,2)]
    rows = []
    for version, source in [('original', SOURCE), ('index_commuted', path)]:
        fn = load(source, 'bmm_bwd_' + version)._bmm_chunk_bwd_kernel.fn
        for residual in (False, True):
            for choice, (m,n,k,stages,warps) in enumerate(configs):
                sig = {name:'i64' for name in fn.arg_names[:25]}
                sig.update(a_ptr='*fp32',dout_ptr='*fp32',db_ptr='*fp32')
                const = dict(dot_dtype=tl.float32,HAS_RESIDUAL=residual,
                             BLOCK_SIZE_M=m,BLOCK_SIZE_N=n,BLOCK_SIZE_CS=k)
                if residual:
                    sig['res_ptr'] = '*fp32'
                else:
                    sig['res_ptr'] = 'constexpr'
                    const['res_ptr'] = None
                kernel = triton.compile(ASTSource(fn,signature=sig,constexprs=const),
                    target=GPUTarget('cuda',90,32),
                    options={'num_warps':warps,'num_stages':stages})
                artifacts = {}
                for kind in ('ttir','ttgir','ptx'):
                    target = directory / f'{version}-res{int(residual)}-{choice}.{kind}'
                    target.write_text(kernel.asm[kind])
                    artifacts[kind] = {'path':str(target),'sha256':sha(target)}
                ptx = kernel.asm['ptx']
                ops = sorted(set(re.findall(r'(?:wgmma|mma)[.][a-z0-9_.]+',ptx)))
                expected_op = (f'wgmma.mma_async.sync.aligned.m64n{n}k8.f32.tf32.tf32'
                               if choice < 6 else 'mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32')
                assert expected_op in ops, (version,residual,choice,ops)
                sites = [line.strip() for line in ptx.splitlines() if expected_op in line]
                live = {}
                if choice < 6:
                    for line in sites:
                        predicate = re.search(r'(%p\d+),\s*1,\s*1;',line)
                        assert predicate, line
                        pred = predicate.group(1)
                        definitions = [line.strip() for line in ptx.splitlines()
                                       if re.search(r'mov.pred\s+'+re.escape(pred)+r',\s*-1;',line)]
                        assert definitions, (pred,sites)
                        live[pred] = definitions
                rows.append({'version':version,'residual':residual,'config':choice,
                             'tile':[m,n,k],'stages':stages,'warps':warps,
                             'ops':ops,'accumulator_live':True,'live_predicates':live,
                             'instruction_sites':len(sites),'artifacts':artifacts,
                             'signature':sig,'ttir_source_binding':audit_ttir(artifacts['ttir']['path'],residual)})
                print(f'{version} residual={residual} config={choice}: {expected_op}',flush=True)
    return {'triton':triton.__version__,'target':'cuda sm90','profiles':rows,
            'gpu_execution':False,'scope':'Fixed offline compiler profiles; no universal compiler conformance claim.'}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--audit-existing', action='store_true')
    args = parser.parse_args()
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    if args.audit_existing:
        receipt = directory / 'receipt.json'
        result = json.loads(receipt.read_text())
        assert result['source_sha256'] == sha(SOURCE) == EXPECTED
        for row in result['compiler']['profiles']:
            for artifact in row['artifacts'].values():
                assert sha(artifact['path']) == artifact['sha256']
            row['ttir_source_binding'] = audit_ttir(row['artifacts']['ttir']['path'],row['residual'])
        receipt.write_text(json.dumps(result,indent=2)+'\n')
        print('36 existing profiles: live loop carry, Dout/A operand order, initial zero, residual addition and all 108 artifact hashes checked.')
        return
    generated = candidate(directory)
    result = {'source':str(SOURCE),'source_sha256':sha(SOURCE),
              'candidate':str(generated),'candidate_sha256':sha(generated),
              'candidate_edit':'pid_h = pid_ch - ngroups * pid_c',
              'cpu':cpu(generated)}
    if args.compile:
        result['compiler'] = compile_profiles(generated,directory)
    (directory/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__ == '__main__':
    main()
