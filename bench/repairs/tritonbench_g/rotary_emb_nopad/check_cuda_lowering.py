"""Inspect local default-fusion graphs offline; this never executes on a GPU."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re


def graph(ptx, cached):
    registers = {}
    loads = []
    stores = []
    names = ['a','b','ka','kb','c','s'] if cached else ['c','s','a','b','ka','kb']
    for line in ptx.splitlines():
        load = re.search(r'ld\.global\.b32\s+\{\s*(%r\d+)\s*\}', line)
        if load:
            register = load.group(1)
            assert len(loads) < len(names), 'inspect a changed vectorization separately'
            registers[register] = names[len(loads)]
            loads.append(register)
        operation = re.search(r'(mul\.f32|neg\.f32|fma\.rn\.f32)\s+([^;]+);', line)
        if operation:
            arguments = [a.strip() for a in operation.group(2).split(',')]
            name = operation.group(1).split('.')[0]
            registers[arguments[0]] = (name, *(registers[a] for a in arguments[1:]))
        store = re.search(r'st\.global\.b32\s+[^;]+,\s*\{\s*(%r\d+)\s*\}', line)
        if store:
            stores.append(registers[store.group(1)])
    assert len(loads) == 6 and len(stores) == 4
    def pair(a,b):
        if cached:
            return [('fma',a,'c',('neg',('mul',b,'s'))), ('fma',b,'c',('mul',a,'s'))]
        return [('fma','c',a,('neg',('mul','s',b))), ('fma','s',a,('mul','c',b))]
    expected = pair('ka','kb')+pair('a','b') if cached else pair('a','b')+pair('ka','kb')
    assert stores == expected, (stores, expected)
    return stores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ['TRITON_CACHE_DIR'] = str(out/'cache')
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    path = Path(__file__).with_name('rotary_emb_nopad_repaired.py')
    spec = importlib.util.spec_from_file_location('nopad_lowering_review',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = []
    for sm in (80,90):
        for integer in ('i32','i64'):
            for cached in (False,True):
                name = 'fused_rotary_embedding_kernel_v2' if cached else 'rotary_embedding_kernel'
                kernel = getattr(module,name)
                signature = {argument: integer for argument in kernel.arg_names}
                constants = {'Q_HEAD_NUM':4,'HEAD_DIM':12,'BLOCK_PAIRS':8}
                constants.update({'K_HEAD_NUM':2} if cached else {'KV_GROUP_NUM':2,'BLOCK_TOKENS':4})
                for pointer in ('q','k','cos','sin','kv_cache'):
                    if pointer in signature:signature[pointer]='*fp32'
                for pointer in ('BLOCK_TABLES','context_lengths'):
                    if pointer in signature:signature[pointer]='*i64'
                for constant in constants:signature[constant]='constexpr'
                result = triton.compile(ASTSource(kernel,signature=signature,constexprs=constants),
                    target=GPUTarget('cuda',sm,32),options={'num_warps':4})
                label=f'sm{sm}_{integer}_{name}'
                for stage in ('ttir','llir','ptx'):(out/f'{label}.{stage}').write_text(result.asm[stage])
                selected=graph(result.asm['ptx'],cached)
                cases.append({'name':label,'stores':selected})
                print(label+': local scalar graph and original store order passed')
    (out/'report.json').write_text(json.dumps({'triton_version':triton.__version__,
        'head_dim':12,'q_heads':4,'k_heads':2,'num_warps':4,'gpu_execution':False,
        'scope':'Representative offline specializations, not universal compiler or GPU conformance.',
        'cases':cases},indent=2)+'\n')


if __name__=='__main__':main()
