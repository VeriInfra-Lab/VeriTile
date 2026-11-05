import ast
import copy
import hashlib
from pathlib import Path
import unittest
from scripts.bench.jit_correspondence import check

from tests.python.layout_fixtures import check_pinned_pairs, pinned_pairs, variant_provenance, variant_source

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / 'bench/tritonbench_g/context_attn_llama'


class ContextAttnLlamaBodyMigrationTests(unittest.TestCase):
    def setUp(self):
        self.lean = (DIRECTORY / 'Kernels.lean').read_text()
        self.records = pinned_pairs(DIRECTORY)

    def verify(self, text, index=0):
        r = self.records[index]
        return check((ROOT / r['python']).read_text(), text, r['python_function'],
                     r['lean_definition'], r['signature_definition'], r['expected_signature'],
                     expected_call_bindings=r['expected_call_bindings'])

    def test_all_three_complete_bodies(self):
        rows = check_pinned_pairs(ROOT, DIRECTORY)
        self.assertEqual([r['statements'] for r in rows], [25, 25, 25])

    def test_cast_loop_mask_and_operation_mutations_are_rejected(self):
        for before, after in [
            ('tl.program_id(0).to(tl.int64)', 'tl.program_id(0)'),
            ('kv_loc = tl.load', 'kv_loc = 0 # tl.load'),
            ('p = p.to(v.dtype)', 'p = p'),
            ('range(0, block_mask * block_end_loc, BLOCK_N)', 'range(0, block_end_loc, BLOCK_N)'),
            ('start_n = tl.multiple_of(start_n, BLOCK_N)', 'start_n = start_n'),
            ('mask=start_n + offs_n[None, :] < block_end_loc', 'mask=start_n + offs_n[None, :] <= block_end_loc'),
            ('qk * sm_scale', 'sm_scale * qk'),
            ('tl.dot(p, v, acc)', 'tl.dot(p, v) + acc'),
        ]:
            with self.subTest(before=before):
                self.assertIn(before, self.lean)
                with self.assertRaises(ValueError):
                    self.verify(self.lean.replace(before, after))

    def test_payload_and_metadata_storage_mutations_are_rejected(self):
        for before, after in [('("B_Seqlen",.i32)', '("B_Seqlen",.i64)'),
                              ('| .bf16 => .bf16', '| .bf16 => .fp16')]:
            with self.subTest(before=before):
                self.assertIn(before, self.lean)
                with self.assertRaises(ValueError):
                    self.verify(self.lean.replace(before, after))

    def test_candidates_restore_the_entire_original_module(self):
        original_text = variant_source(DIRECTORY / 'context_attn_llama.py')
        candidate_text = variant_source(DIRECTORY / 'context_attn_llama_candidates.py')
        record = variant_provenance(DIRECTORY)
        self.assertEqual(hashlib.sha256(original_text.encode()).hexdigest(), record['original_sha256'])
        self.assertEqual(hashlib.sha256(candidate_text.encode()).hexdigest(), record['candidate_sha256'])
        original = ast.parse(original_text)
        candidates = ast.parse(candidate_text)
        original_kernel = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == '_fwd_kernel')
        for suffix in ['scale_first', 'block_end_factored']:
            with self.subTest(suffix=suffix):
                names = {'_fwd_kernel_' + suffix: '_fwd_kernel',
                         'context_attention_fwd_' + suffix: 'context_attention_fwd'}

                class RestoreNames(ast.NodeTransformer):
                    def visit_Name(self, node):
                        node.id = names.get(node.id, node.id)
                        return node

                    def visit_FunctionDef(self, node):
                        node.name = names.get(node.name, node.name)
                        return self.generic_visit(node)

                selected = ast.Module(body=[copy.deepcopy(n) for n in candidates.body
                    if not isinstance(n, ast.FunctionDef) or n.name in names
                    or n.name in {'_checked_tensor', '_formed_bound'}], type_ignores=[])
                selected = RestoreNames().visit(selected)
                kernel = next(n for n in selected.body if isinstance(n, ast.FunctionDef) and n.name == '_fwd_kernel')
                if suffix == 'scale_first':
                    loop = next(n for n in kernel.body if isinstance(n, ast.For))
                    statement = next(n for n in loop.body if isinstance(n, ast.Assign)
                        and ast.unparse(n.targets[0]) == 'qk' and isinstance(n.value, ast.Call)
                        and ast.unparse(n.value.func) == 'tl.where')
                    self.assertEqual(ast.unparse(statement.value.args[1]), 'sm_scale * qk')
                    statement.value.args[1] = ast.parse('qk * sm_scale', mode='eval').body
                else:
                    statement = next(n for n in kernel.body if isinstance(n, ast.Assign)
                        and ast.unparse(n.targets[0]) == 'block_end_loc')
                    self.assertEqual(ast.unparse(statement.value),
                        'tl.minimum(block_start_loc + BLOCK_M, cur_batch_seq_len) + prompt_cache_len')
                    statement.value = copy.deepcopy(next(n.value for n in original_kernel.body
                        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'block_end_loc'))
                self.assertEqual(ast.dump(selected, include_attributes=False),
                                 ast.dump(original, include_attributes=False))
