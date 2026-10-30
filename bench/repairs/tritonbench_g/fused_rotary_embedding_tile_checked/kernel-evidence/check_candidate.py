"""Exact added-guard and shared-quotation checks; CPU host controls, no GPU."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import re
import sys

sys.dont_write_bytecode = True
TASK = Path(__file__).resolve().parents[1]
ROOT = next(p for p in TASK.parents if (p / 'lean-toolchain').is_file())
OLD = TASK.parent / 'fused_rotary_embedding'
SOURCE = TASK / 'fused_rotary_embedding_tile_checked.py'
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

old = ast.parse((OLD / 'fused_rotary_embedding_repaired.py').read_text())
new = ast.parse(SOURCE.read_text())
wrapper = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == 'decoding_fused_rotary_embedding')
guard = ast.parse('if full_block > 1048576:\n    raise ValueError("static rotary tile exceeds Triton maximum 1048576")').body[0]
positions = [i for i, n in enumerate(wrapper.body) if ast.dump(n) == ast.dump(guard)]
assert len(positions) == 1
at = positions[0]
assert ast.dump(wrapper.body[at - 1]) == ast.dump(ast.parse('full_block = triton.next_power_of_2(width)').body[0])
assert any(isinstance(n, ast.If) and ast.unparse(n.test) == 'tokens == 0' for n in wrapper.body[:at])
del wrapper.body[at]
assert ast.dump(old) == ast.dump(new)
kernel = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == 'decoding_fused_rotary_embedding_kernel')
# The function-body frontend retains Python syntax directly. Reuse the
# checked source/ABI/call comparison rather than stripping historical Lean
# directives, antiquotes and brace control flow from the old quotation.
sys.path.insert(0, str(ROOT))
from bench.jit_correspondence import check_manifest
source_pairs = check_manifest(ROOT, TASK / 'jit-correspondence.json')
assert [(p['mode'], p['statements']) for p in source_pairs] == [
    ('pasted_body', 21), ('pasted_body', 21)]
kernel_ast = ast.dump(ast.Module(body=kernel.body, type_ignores=[]))

# Reuse the saved harness definitions without rerunning its old control suite.
harness = ast.parse((OLD / 'check_contract.py').read_text())
stop = next(i for i, n in enumerate(harness.body) if isinstance(n, ast.For))
support = {'__file__': str(OLD / 'check_contract.py'), '__name__': 'candidate_support'}
exec(compile(ast.Module(body=harness.body[:stop], type_ignores=[]), str(OLD / 'check_contract.py'), 'exec'), support)
spec = importlib.util.spec_from_file_location('rotary_tile_candidate', SOURCE)
candidate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = candidate
spec.loader.exec_module(candidate)
candidate.torch = support['m'].torch
candidate.decoding_fused_rotary_embedding_kernel = support['spy']
support['m'] = candidate
records = []
for new_layout in (False, True):
    for tokens, width, accepted in [(1, 1048576, True), (1, 1048578, False),
                                     (0, 1048578, True), (0, 2097152, True), (0, 67108864, True)]:
        args = support['inputs'](T=tokens, HQ=2, HK=1, D=width, B=1, X=2 if new_layout else None)
        if accepted:
            support['admit'](args)
        else:
            support['reject'](args, 'static rotary tile')
        calls = support['spy'].calls
        records.append({'tokens': tokens, 'width': width, 'new_layout': new_layout,
                        'accepted': accepted, 'launched': bool(calls),
                        'metadata_copies': len(support['copies'])})
        if calls:
            kwargs = calls[0][2]
            assert kwargs['BLOCK_DIM'] == 1048576 and kwargs['BLOCK_HALF'] == 524288
        if not tokens:
            assert not support['copies'] and not calls
assert sha(OLD / 'fused_rotary_embedding_repaired.py') == '0e60ae0f98d1d48040ae2d038c39587117f004b41f76a683a8fdc3e1a11f6a32'
report = {'all_passed': True, 'candidate_sha256': sha(SOURCE), 'quote_sha256': sha(TASK / 'Kernels.lean'),
          'source_AST_sha256': hashlib.sha256(kernel_ast.encode()).hexdigest(),
          'sole_guard_AST_change': True, 'complete_quote_AST_identical': True,
          'boundary_cases': records, 'new_gpu_execution': False, 'new_offline_compile': False}
(TASK / 'kernel-evidence/candidate-checks.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
