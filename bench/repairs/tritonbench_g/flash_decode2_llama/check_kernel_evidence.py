"""Validate source pins and the two observed scalar compiler patterns."""
import ast
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = HERE / 'kernel-evidence'
SOURCE = HERE / 'flash_decode2_llama_repaired.py'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
expected = '2d8a58c053404bcc7b8b58117aaea53cb4eed4742ddcadc90c8eaa6075e34c85'
assert sha(SOURCE) == expected
kernel = next(n for n in ast.parse(SOURCE.read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name.startswith('_fwd_kernel'))
normalized = hashlib.sha256(ast.dump(kernel, include_attributes=False).encode()).hexdigest()
assert normalized == '291edeb85857e096f069b1dbe4ac55ef65a9afb388825d3c20f40300d281da02'
profiles = json.loads((OUT/'profiles.json').read_text())
assert len(profiles['results']) == 20 and profiles['actual_binder'] and profiles['actual_wrapper_arguments']
assert profiles['gpu_execution'] is False
checks = []
for row in profiles['results']:
    assert row['status'] == 'compiled' and row['source_sha256'] == expected
    for artifact in row['artifacts']:
        assert sha(Path(artifact['path'])) == artifact['sha256']
    ptx = Path(next(a['path'] for a in row['artifacts'] if a['path'].endswith('.ptx'))).read_text()
    fmas = [re.split(r'\s*,\s*', x.strip()) for x in re.findall(r'fma\.rn\.f32\s+([^;]+);', ptx)]
    assert len(fmas) == 2
    expmul = re.findall(r'mul\.f32\s+(%r\d+),\s*(%r\d+),\s*0f3FB8AA3B;', ptx)
    exps = re.findall(r'ex2\.approx\.f32\s+(%r\d+),\s*(%r\d+);', ptx)
    assert len(expmul) == len(exps) == 2
    assert [r[0] for r in expmul] == [r[1] for r in exps]
    old, fresh = [r[0] for r in exps]
    maximal = row['constants']['BLOCK_SEQ'] == str(2**31-1)
    if maximal:
        assert 'Inner Loop Header' not in ptx
        assert all(f[1] == old and f[2] == '0f00000000' for f in fmas)
    else:
        assert 'Inner Loop Header' in ptx
        assert all(f[0] == f[1] and f[2] == old for f in fmas)
    assert fmas[1][3] == fresh
    product = re.search(r'mul\.f32\s+'+re.escape(fmas[0][3])+r',\s*'+re.escape(fresh)+r',\s*(%r\d+);', ptx)
    assert product
    assert re.search(r'div\.full\.f32\s+%r\d+,\s*'+re.escape(fmas[0][0])+r',\s*'+re.escape(fmas[1][0])+r';', ptx)
    assert len(re.findall(r'max\.f32', ptx)) == 1
    checks.append(dict(name=row['name'], pattern='single_iteration_zero_second' if maximal else 'loop_carry_first',
                       fma_operands=fmas, exp2_operands=exps, log2e_multiplications=expmul))
result = dict(task='flash_decode2_llama', source_sha256=expected, normalized_jit_ast_sha256=normalized,
              source_unchanged=True, independent_target='unshifted weighted stage-1-summary formula',
              native_profile='FP32 payload/output, signed INT32 lengths and INT64 promoted indexing',
              admitted_dimensions=[16,32,64,128], source_defect_found=False,
              grid_product_compatibility='deferred by task scope',
              source_quote='FlashDecode2.llama in flash_decode2_phi/Kernels.lean',
              integer_quote_boundary='Natural loop/index quotient with separately proved signed64 and pointer bounds',
              scalar_model='six total scalar families: sub/mul/div.full/max/exp2.approx/fma.rn; exp32 composition explicit',
              compiler_boundary='Two fixed patterns selected by BLOCK_SEQ; samples support the model, not a proof of Triton lowering for every shape/backend',
              native_numerical_experiment=False, gpu_execution=False, compiler_checks=checks)
(OUT/'source-and-compiler-audit.json').write_text(json.dumps(result, indent=2)+'\n')
print('PASS: canonical source/AST unchanged; 20 instruction-pattern checks; no numerical experiment claim.')
