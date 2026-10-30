"""Offline source/candidate branch compilation; no GPU execution or accuracy claim."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
CANONICAL_SHA = 'f6faaf6e84e960102afca4c63573af27c412e2c36a1a3a9d3a95ecab264665b7'
CONFIGS = [(128,256,64,3,8),(64,256,32,4,4),(128,128,32,4,4),
           (128,64,32,4,4),(64,128,32,4,4),(128,32,32,4,4),
           (64,32,32,5,2),(32,64,32,5,2),(64,64,32,4,2)]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def check_candidate():
    source = ROOT / 'bmm_chunk_fwd_repaired.py'
    candidate = ROOT / 'bmm_chunk_fwd_candidate.py'
    assert sha(source) == CANONICAL_SHA
    left = 'pid_h = pid_ch - pid_c * ngroups'
    right = 'pid_h = pid_ch - ngroups * pid_c'
    assert source.read_text().count(left) == 1
    assert candidate.read_text() == source.read_text().replace(left, right)
    return source, candidate

def audit_staging(ttir, ptx):
    dots = re.findall(r'(%\w+) = tt\.dot (%\w+), (%\w+), (%\w+), inputPrecision = tf32', ttir)
    assert len(dots) == 1 and 'arith.addf' not in ttir
    result, left, right, carry = dots[0]
    loop = next(line.strip() for line in ttir.splitlines() if 'scf.for' in line and 'iter_args' in line)
    yielded = next(line.strip() for line in ttir.splitlines() if 'scf.yield' in line and result in line)
    assert re.search(re.escape(carry) + r'\s*=', loop)
    wgmma = [line for line in ptx.splitlines() if 'wgmma.mma_async.' in line]
    predicates = sorted({p for line in wgmma for p in re.findall(r'%p\d+',line)})
    assert all(re.search(r'mov.pred\s+'+re.escape(p)+r',\s*-1;',ptx) for p in predicates)
    return dict(dot_result=result, left=left, right=right, loop_carry=carry,
                loop=loop, yielded=yielded, separately_rounded_outer_add=False,
                carry_kind='wgmma_live_predicate' if wgmma else 'mma_explicit_c',
                live_predicates=predicates,
                instruction_sites=len(wgmma) if wgmma else len(re.findall(r'mma.sync.aligned',ptx)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'Evidence'/'lowering')
    parser.add_argument('--arch', type=int, default=90)
    args = parser.parse_args()
    source, candidate = check_candidate()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ['TRITON_CACHE_DIR'] = str(args.output/'cache')
    import triton
    import triton.language as tl
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    rows = []
    for variant, path in [('source',source),('candidate',candidate)]:
        module = load(path, 'bmm_fwd_'+variant)
        function = module._bmm_chunk_fwd_kernel_repaired.fn
        for i,(m,n,k,stages,warps) in enumerate(CONFIGS):
            for causal in (False,True):
                for seq in (False,True):
                    constants = dict(IS_CAUSAL=causal, dot_dtype=tl.float32, HAS_SEQ_IDX=seq,
                                     BLOCK_SIZE_M=m, BLOCK_SIZE_N=n, BLOCK_SIZE_K=k)
                    signature = {name: ('*i64' if name=='seq_idx_ptr' else '*fp32' if name.endswith('_ptr') else 'i32')
                                 for name in function.arg_names if name not in constants}
                    options = dict(num_warps=warps, num_stages=stages, enable_fp_fusion=True)
                    compiled = triton.compile(ASTSource(function, signature, constexprs=constants),
                                              target=GPUTarget('cuda',args.arch,32), options=options)
                    label = f'{variant}-config{i}-causal{int(causal)}-seq{int(seq)}'
                    artifacts = {}
                    for ext in ('ttir','ttgir','ptx'):
                        artifact = args.output/(label+'.'+ext)
                        artifact.write_text(compiled.asm[ext])
                        artifacts[ext] = dict(path=str(artifact), sha256=sha(artifact))
                    instructions = sorted(set(re.findall(r'(?:wgmma\.mma_async|mma\.sync)[\w.]+',compiled.asm['ptx'])))
                    assert instructions and all('tf32.tf32' in op for op in instructions)
                    rows.append(dict(variant=variant,config=i,causal=causal,has_seq=seq,
                                     tile=[m,n,k],options=options,signature=signature,
                                     instructions=instructions,shared_bytes=compiled.metadata.shared,artifacts=artifacts,
                                     staging=audit_staging(compiled.asm['ttir'],compiled.asm['ptx'])))
                    print(label, instructions, flush=True)
        function = module._clear_causal_upper
        constants = dict(BLOCK_M=32,BLOCK_N=64)
        signature = dict(Out='*fp32',chunk_size='i32',chunk_heads='i32')
        compiled = triton.compile(ASTSource(function,signature,constexprs=constants),
                                  target=GPUTarget('cuda',args.arch,32),options=dict(num_warps=4))
        artifacts = {}
        for ext in ('ttir','ttgir','ptx'):
            artifact = args.output/(variant+'-cleanup.'+ext)
            artifact.write_text(compiled.asm[ext])
            artifacts[ext]=dict(path=str(artifact),sha256=sha(artifact))
        rows.append(dict(variant=variant,cleanup=True,signature=signature,constants=constants,artifacts=artifacts))
    receipt = dict(source_sha256=sha(source),candidate_sha256=sha(candidate),triton=triton.__version__,
                   target=f'cuda sm{args.arch}',gpu_execution=False,numerical_conformance=False,
                   scope='Recorded i32 runtime signature; all nine configs and all causal/sequence flags. Not exhaustive specialization coverage.',
                   cases=rows)
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__ == '__main__':
    main()
