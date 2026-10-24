#!/usr/bin/env python3
"""Compile the unchanged reverse-cumsum sources offline and retain rejections.

No original GPU tests or host launches are executed. This is an admission and
lowering observation, not a correctness or source-to-Lean refinement proof.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
FLOATS = ('fp16', 'bf16', 'fp32', 'fp64', 'fp8e4nv', 'fp8e5')
SOURCES = {
    'scalar': ('reversed_cumsum_scalar', 'e145461bdb74f2d5ae5de59275181e6e22d39f0b0b133cd191d214ec5ab4f40a', [(16,2),(32,4),(32,2),(64,8),(64,4)]),
    'vector': ('reversed_cumsum', '28e2c4e324b0098cac1aec32d11a0883dbebc491564f20674f6f08f1939fe314', [(b,w) for b in (16,32,64) for w in (2,4,8)]),
}

def digest(raw): return hashlib.sha256(raw).hexdigest()
def ast_digest(node): return digest(ast.dump(node, include_attributes=False).encode())

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--artifacts', required=True, type=Path)
    args=parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('TRITON_CACHE_DIR', str(args.artifacts/'cache'))
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.runtime.jit import JITFunction
    receipt={'triton_version':triton.__version__, 'target':{'backend':'cuda','arch':90,'warp_size':32},
      'scope':'Unchanged decorated original JIT and host-wrapper ASTs, compiled offline. All original autotune configurations, six homogeneous storage types and empty/single/exact/partial chunk lengths; all 36 independent input/output type pairs for BT16/warps2/T17. Vector BS32/S33 includes feature tail and runtime strides use i32 ABI. No GPU execution, universal shape/ABI admission, IEEE correctness, wrapper tensor-layout correctness or independent formal proof replay.',
      'sources':{}, 'results':[], 'complete_task':False}
    for kind,(directory,pin,configs) in SOURCES.items():
        source=ROOT/'bench/tritonbench_g'/directory/(directory+'.py')
        raw=source.read_bytes()
        if digest(raw)!=pin:raise RuntimeError(f'Source pin changed: {source}')
        functions={n.name:n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef)}
        stem='chunk_global_reversed_cumsum_'+kind
        names=[stem+'_kernel',stem]
        extracted=ast.Module(body=[ast.Import(names=[ast.alias(name='torch')]),ast.Import(names=[ast.alias(name='triton')]),ast.Import(names=[ast.alias(name='triton.language',asname='tl')]),ast.ImportFrom(module='typing',names=[ast.alias(name='Optional')],level=0),*[functions[n] for n in names]],type_ignores=[])
        path=args.artifacts/(kind+'_original.py');path.write_text(ast.unparse(extracted)+'\n')
        emitted={n.name:n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
        for name in names:
            if ast_digest(emitted[name])!=ast_digest(functions[name]):raise RuntimeError('Extraction changed '+name)
        spec=importlib.util.spec_from_file_location('veritile_reverse_native_'+kind,path)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        decorated=getattr(module,names[0])
        actual=[(c.kwargs['BT'],c.num_warps) for c in decorated.configs]
        if actual!=configs:raise RuntimeError(f'Config mismatch: {actual}')
        fn=decorated
        while not isinstance(fn,JITFunction):fn=fn.fn
        receipt['sources'][kind]={'path':str(source.relative_to(ROOT)),'sha256':pin,'function_ast_sha256':{n:ast_digest(functions[n]) for n in names},'configs':configs}
        cases=set()
        for dtype in FLOATS:
            for bt,warps in configs:
                for t in (0,1,bt,bt+1):cases.add((dtype,dtype,bt,warps,t))
        for src,dst in itertools.product(FLOATS,repeat=2):cases.add((src,dst,16,2,17))
        for src,dst,bt,warps,t in sorted(cases):
            name=f'{kind}-{src}-{dst}-bt{bt}-w{warps}-t{t}'
            signature={'s':'*'+src, 'o' if kind=='scalar' else 'z':'*'+dst}
            constexprs={'T':t,'BT':bt}
            if kind=='vector':
                signature.update(s_s_h='i32',s_s_t='i32',s_s_d='i32');constexprs.update(S=33,BS=32)
            row={'name':name,'kind':kind,'input':src,'output':dst,'signature':signature,'constexprs':constexprs,'num_warps':warps}
            try:
                compiled=triton.compile(ASTSource(fn,signature,constexprs=constexprs),target=GPUTarget('cuda',90,32),options={'num_warps':warps})
                row['status']='compiled';row['artifacts']={}
                for ir in ('ttir','ttgir','llir','ptx'):
                    data=compiled.asm[ir].encode();out=args.artifacts/(name+'.'+ir);out.write_bytes(data)
                    row['artifacts'][ir]={'file':out.name,'sha256':digest(data)}
                row['precision_lines']=[line.strip() for line in compiled.asm['ttir'].splitlines() if any(s in line for s in ('tt.load','tt.store','tt.scan','tt.reduce','tt.dot','arith.truncf','arith.extf','scf.for','ub.poison'))]
            except Exception as exc:
                row.update(status='rejected',exception_type=type(exc).__name__,message=str(exc))
                (args.artifacts/(name+'.error.txt')).write_text(str(exc)+'\n')
            receipt['results'].append(row)
            receipt['counts']=dict(Counter(r['status'] for r in receipt['results']))
            args.output.write_text(json.dumps(receipt,indent=2)+'\n')
            print(name,row['status'],flush=True)
    print(json.dumps(receipt['counts']),flush=True)
    return int(any(r['status']=='rejected' for r in receipt['results']))

if __name__=='__main__':raise SystemExit(main())
