"""Inspect both selected SwiGLU sources' ordered CUDA graphs without a GPU."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_graph(ptx, backward, tail):
    definitions = {}
    for line in ptx.splitlines():
        match = re.match(r'\s*([a-z][a-z0-9_.]+)\s+(%[a-z]+[0-9]+),\s*([^;]+);', line)
        if match:
            op, target, operands = match.groups()
            definitions[target] = op, tuple(x.strip() for x in operands.split(','))
    loads = re.findall(r'ld\.global\.b32\s+\{\s*(%\w+)\s*\}', ptx)
    stores = re.findall(r'st\.global\.b32\s+\[[^\]]+\],\s*\{\s*(%\w+)\s*\}', ptx)
    assert loads and stores and '.ftz.' not in ptx
    accesses = re.findall(r'^\s*(@%p\d+)?\s*((?:ld|st)\.global\S*)\s+', ptx, re.M)
    assert all(op.endswith('.b32') and (pred or not tail) for pred, op in accesses), accesses

    def args(register, opcode):
        op, operands = definitions[register]
        assert op == opcode, (register, op, opcode)
        return operands

    def literal(register, bits):
        if register == bits:
            return
        op, operands = definitions[register]
        assert op in ('mov.b32', 'mov.f32') and operands == (bits,), (register, op, operands, bits)

    def result(opcode, operands):
        matches = [reg for reg, (op, ins) in definitions.items() if op == opcode and ins == operands]
        assert len(matches) == 1, (opcode, operands, matches)
        return matches[0]

    predicted, used_loads, lanes = [], set(), 0
    for sigmoid, (op, ins) in definitions.items():
        if op != 'div.full.f32':
            continue
        lanes += 1
        one, denominator = ins
        literal(one, '0f3F800000')
        exponent, one_added = args(denominator, 'add.f32')
        literal(one_added, '0f3F800000')
        (scaled,) = args(exponent, 'ex2.approx.f32')
        negated, log2e = args(scaled, 'mul.f32')
        literal(log2e, '0f3FB8AA3B')
        zero, a = args(negated, 'sub.f32')
        literal(zero, '0f00000000')
        assert a in loads
        silu = result('mul.f32', (sigmoid, a))
        if backward:
            one_minus = next(reg for reg, (op, ins) in definitions.items()
                             if op == 'sub.f32' and ins[1] == sigmoid)
            literal(args(one_minus, 'sub.f32')[0], '0f3F800000')
            fused = result('fma.rn.f32', (silu, one_minus, sigmoid))
            dc_product = next((reg, ins[1]) for reg, (op, ins) in definitions.items()
                              if op == 'mul.f32' and ins[0] == fused)
            dc = dc_product[1]
            da, b = next((reg, ins[1]) for reg, (op, ins) in definitions.items()
                         if op == 'mul.f32' and ins[0] == dc_product[0])
            db = result('mul.f32', (silu, dc))
            assert len({a, b, dc}) == 3 and b in loads and dc in loads
            assert stores.index(da) < stores.index(db)
            predicted.extend([da, db]); used_loads.update([a, b, dc])
        else:
            output, b = next((reg, ins[1]) for reg, (op, ins) in definitions.items()
                             if op == 'mul.f32' and ins[0] == silu)
            assert b in loads and b != a
            predicted.append(output); used_loads.update([a, b])
    assert lanes and set(predicted) == set(stores) and used_loads == set(loads)
    fmas = sum(op == 'fma.rn.f32' for op, _ in definitions.values())
    assert fmas == (lanes if backward else 0)
    return {'lanes': lanes, 'ordered_outputs_and_inputs_checked': True,
            'inner_fma': backward, 'masked_or_proved_all_active_fp32_memory': True, 'explicit_ftz': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    output = parser.parse_args().output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    os.environ['TRITON_CACHE_DIR'] = str(output/'cache')
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    here = Path(__file__).resolve().parent
    sources = {'swiglu_fwd': here.parent/'swiglu_fwd/swiglu_fwd_checked.py',
               'swiglu_triton': here/'swiglu_triton_repaired.py'}
    modules = {name: load(name+'_native_check', path) for name, path in sources.items()}
    jobs = []
    for block in (32, 64, 128, 256, 512, 1024):
        sig = {k: '*fp32' for k in ('X', 'Y', 'OUT')}
        sig.update({k: 'i32' for k in ('stride_x_row', 'stride_y_row', 'stride_out_row', 'ncols')})
        sig['BLOCK_N'] = 'constexpr'
        jobs.append(('swiglu_fwd', False, block, None, 4,
                     modules['swiglu_fwd']._swiglu_fwd_kernel.fn, sig, {'BLOCK_N': block}))
    for cols, block, warps in ((1,1,4),(33,64,4),(129,256,4),(2047,2048,8),
                                (8191,8192,16),(32767,32768,32),(65536,65536,32)):
        for backward in (False, True):
            ports = ('dc_ptr','a_ptr','b_ptr') if backward else ('a_ptr','b_ptr','c_ptr')
            sig = {k: '*fp32' for k in ports}
            sig.update(stride='i32', n_cols='constexpr', BLOCK_SIZE='constexpr')
            fn = getattr(modules['swiglu_triton'], '_swiglu_backward_kernel' if backward else '_swiglu_forward_kernel')
            jobs.append(('swiglu_triton', backward, block, cols, warps, fn, sig,
                         {'n_cols': cols, 'BLOCK_SIZE': block}))
    cases = []
    for task, backward, block, cols, warps, fn, sig, constants in jobs:
        compiled = triton.compile(ASTSource(fn, signature=sig, constexprs=constants),
            target=GPUTarget('cuda',90,32), options={'num_warps':warps,'enable_fp_fusion':True})
        stem = f'{task}.{"backward" if backward else "forward"}.b{block}'
        ptx = compiled.asm['ptx']; ptx_path = output/(stem+'.ptx'); ptx_path.write_text(ptx)
        (output/(stem+'.llir')).write_text(compiled.asm['llir'])
        case = {'task':task,'backward':backward,'block':block,'cols':cols,'warps':warps,
                'ptx':str(ptx_path),'ptx_sha256':hashlib.sha256(ptx.encode()).hexdigest(),
                **check_graph(ptx, backward, cols is None or cols < block)}
        cases.append(case); print(stem+': ordered graph and memory passed', flush=True)
    (output/'report.json').write_text(json.dumps({'triton':triton.__version__,'sm':90,
        'fp_fusion':True,'gpu_execution':False,'numerical_assumptions':[],
        'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources.values()},
        'cases':cases},indent=2)+'\n')


if __name__ == '__main__': main()
