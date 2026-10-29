import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
OUT = parser.parse_args().output.resolve()
OUT.mkdir(parents=True, exist_ok=False)
TASK = ROOT / 'bench/tritonbench_g/attention_kernel_aligned'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


original = TASK / 'attention_kernel_aligned.py'
assert digest(original) == 'ca0ecc71aeb5d33c1e216ec1891a726281fae909e8ee765945a3bb12414453b9'
old = (TASK/'Proofs/FPSource.lean').read_text()
new = (TASK/'Proofs/FullFp16Source.lean').read_text()
old_body = old[old.index('def typedSurface'):old.index('\n\nstructure Parameters')]
new_body = new[new.index('def typedSurface'):new.index('\ndef source')].rstrip()
assert old_body.rstrip() == new_body
negative = (TASK/'Proofs/BfloatRejected.lean').read_text()
negative_body = negative[negative.index('#check fun'):negative.index('\n#auditModuleAxioms')].rstrip()
expected = old_body.replace('def typedSurface\n', '#check fun\n', 1)
expected = expected.replace('    :\n    ComputeKernel := triton {', '    => (triton {', 1)
expected = expected.replace('tl.float16', 'tl.bfloat16').rstrip() + ' : ComputeKernel)'
assert negative_body == expected

tree = ast.parse(original.read_text())
kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_fwd_kernel_aligned')
wrapper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_attention_rel_h_rel_w_kernel_aligned_device')
calls = [n for n in ast.walk(kernel) if isinstance(n, ast.Call)]
loads = [n for n in calls if ast.unparse(n.func) == 'tl.load']
stores = [n for n in calls if ast.unparse(n.func) == 'tl.store']
assert len(loads) == 5 and len(stores) == 1
assert all(not n.keywords for n in loads+stores)
dot = next(n for n in calls if ast.unparse(n.func) == 'tl.dot' and n.keywords)
assert [(k.arg, ast.unparse(k.value)) for k in dot.keywords] == [('out_dtype', 'OUT_DTYPE')]
assign = {}
for n in ast.walk(kernel):
    if isinstance(n, ast.Assign):
        assign.setdefault(ast.unparse(n.targets[0]), ast.unparse(n.value))
assert assign['q_offset'] == 'off_hz * stride_qh'
assert assign['kv_offset'] == 'off_hz * stride_kh'
assert 'base=V + kv_offset' in assign['V_block_ptr']
assert 'base=Out + q_offset' in assign['O_block_ptr']
assert not any(isinstance(n, ast.Name) and n.ctx.__class__ is ast.Load and n.id in {'stride_vh','stride_oh'} for n in ast.walk(kernel))

# Execute the exact original wrapper AST with real CPU tensor metadata and a
# launch recorder. This runs no kernel and does not invoke a device compiler.
import torch


class LaunchRecorder:
    def __getitem__(self, grid):
        self.grid = grid
        return self.launch

    def launch(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs


recorder = LaunchRecorder()
namespace = {'torch': torch, 'triton': SimpleNamespace(cdiv=lambda n, b: (n+b-1)//b),
             'tl': SimpleNamespace(float16='fp16', bfloat16='bf16'),
             '_fwd_kernel_aligned': recorder}
exec(compile(ast.Module(body=[wrapper], type_ignores=[]), str(original), 'exec'), namespace)
run_wrapper = namespace[wrapper.name]
probes = []
for dtype in [torch.float16, torch.bfloat16]:
    q = torch.empty((1, 2, 65, 64), dtype=dtype)
    k = torch.empty_like(q)
    v = torch.empty((1, 2, 65, 65), dtype=dtype)[..., :64]
    o = torch.empty((1, 2, 65, 66), dtype=dtype)[..., :64]
    bias = torch.empty((1, 2, 65, 128), dtype=dtype)
    run_wrapper(q, k, v, bias, 1.0, o, 64, 64, 4, 2)
    assert recorder.grid == (2, 2, 1)
    assert recorder.kwargs['OUT_DTYPE'] == ('fp16' if dtype == torch.float16 else 'bf16')
    assert recorder.kwargs['BIAS_LAST_SIZE'] == 64
    assert recorder.args[6:18] == (*q.stride()[1:], *k.stride()[1:], *v.stride()[1:], *o.stride()[1:])
    assert k.stride(1) != v.stride(1) and q.stride(1) != o.stride(1)
    probes.append({'dtype': str(dtype), 'shape': list(q.shape), 'grid': list(recorder.grid),
                   'wrapper_completed_to_recorded_launch': True, 'kernel_executed': False,
                   'query_strides': list(q.stride()), 'key_strides': list(k.stride()),
                   'value_strides': list(v.stride()), 'output_strides': list(o.stride()),
                   'source_v_head_1_base': k.stride(1), 'physical_v_head_1_base': v.stride(1),
                   'source_out_head_1_base': q.stride(1), 'physical_out_head_1_base': o.stride(1)})

frozen = ROOT/'bench/review-checkpoints/2026-10-14/aligned-native'
saved = json.loads((frozen/'receipt.json').read_text())
assert saved['source_sha256'] == digest(original)
assert [(r['dtype'],r['compiled']) for r in saved['results']] == [('fp16',True),('bf16',False)]
checked = {}
for name, expected_hash in saved['artifact_sha256'].items():
    # CPython cache is a historical runtime byproduct; source/compiler artifacts
    # are the reproducibility inputs, not a newly generated Python cache file.
    if name.endswith('.pyc'):
        continue
    assert digest(frozen/name) == expected_hash, name
    checked[name] = expected_hash

result = {'status':'passed', 'original_python': str(original.relative_to(ROOT)),
          'original_python_sha256':digest(original),
          'fp16_full_body_equal_preserved_typed_source':True,
          'bf16_negative_full_body_only_dtype_substitution':True,
          'original_unchecked_load_count':len(loads), 'original_unchecked_store_count':len(stores),
          'host_probe_kind':'exact_original_wrapper_AST_with_real_CPU_metadata_and_launch_recorder',
          'host_probes':probes, 'kernel_executed':False,
          'saved_native_receipt_sha256':digest(frozen/'receipt.json'),
          'saved_native_checked_artifact_sha256':checked,
          'source_bugs':['bf16_dot_output_compile_rejection','unchecked_partial_tiles',
                         'value_and_output_head_base_use_other_tensor_strides'],
          'numerical_defer':False}
(OUT/'aligned-full-source-check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
