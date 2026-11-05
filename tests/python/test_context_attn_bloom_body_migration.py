import ast
import copy
import hashlib
from pathlib import Path
import unittest
from scripts.bench.jit_correspondence import check

from tests.python.layout_fixtures import check_pinned_pairs, pinned_pairs, variant_provenance, variant_source

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / 'bench/tritonbench_g/context_attn_bloom'


class ContextAttnBloomBodyMigrationTests(unittest.TestCase):
    def setUp(self):
        self.lean = (DIRECTORY / 'Kernels.lean').read_text()
        self.records = pinned_pairs(DIRECTORY)

    def verify(self, text):
        r = self.records[0]
        return check((ROOT / r['python']).read_text(), text, r['python_function'],
                     r['lean_definition'], r['signature_definition'], r['expected_signature'],
                     expected_call_bindings=r['expected_call_bindings'])

    def test_all_three_complete_bodies(self):
        self.assertEqual([r['statements'] for r in check_pinned_pairs(ROOT, DIRECTORY)], [24, 24, 24])

    def test_cast_mask_sentinel_and_final_return_are_checked(self):
        for before, after in [
            ('tl.program_id(0).to(tl.int64)', 'tl.program_id(0)'),
            ('p = p.to(v.dtype)', 'p = p'),
            ('tl.store(out_ptrs, acc,', 'tl.store(out_ptrs, acc.to(tl.float32),'),
            ("float('-100000000.0')", "float('-inf')"),
            ('block_mask * block_end_loc', 'block_end_loc'),
            ('        qk *= sm_scale', '        qk = sm_scale * qk'),
            ('    return\n', ''),
        ]:
            with self.subTest(before=before):
                self.assertIn(before, self.lean)
                with self.assertRaises(ValueError):
                    self.verify(self.lean.replace(before, after))

    def test_call_dtype_and_stride_binding_are_checked(self):
        for before, after in [('("B_Seqlen",.i32)', '("B_Seqlen",.i64)'),
                              ('stride_qbs := stride_qbs', 'stride_qbs := 0')]:
            with self.subTest(before=before):
                self.assertIn(before, self.lean)
                with self.assertRaises(ValueError):
                    self.verify(self.lean.replace(before, after))

    def test_existing_candidates_restore_the_entire_original_module(self):
        original_text = variant_source(DIRECTORY / 'context_attn_bloom.py')
        candidate_text = variant_source(DIRECTORY / 'context_attn_bloom_candidates.py')
        record = variant_provenance(DIRECTORY)
        self.assertEqual(hashlib.sha256(original_text.encode()).hexdigest(), record['original_sha256'])
        self.assertEqual(hashlib.sha256(candidate_text.encode()).hexdigest(), record['candidate_sha256'])
        original, candidates = ast.parse(original_text), ast.parse(candidate_text)
        original_kernel = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == '_fwd_kernel')
        for suffix in ['index_shared', 'scale_first']:
            names = {'_fwd_kernel_' + suffix: '_fwd_kernel',
                     'context_attention_fwd_' + suffix: 'context_attention_fwd'}

            class RestoreNames(ast.NodeTransformer):
                def visit_Name(self, node):
                    node.id = names.get(node.id, node.id)
                    return node

                def visit_FunctionDef(self, node):
                    node.name = names.get(node.name, node.name)
                    return self.generic_visit(node)

            selected = RestoreNames().visit(ast.Module(body=[copy.deepcopy(n) for n in candidates.body
                if not isinstance(n, ast.FunctionDef) or n.name in names or n.name in {'_checked_tensor', '_formed_bound'}], type_ignores=[]))
            kernel = next(n for n in selected.body if isinstance(n, ast.FunctionDef) and n.name == '_fwd_kernel')
            if suffix == 'index_shared':
                statement = next(n for n in kernel.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'offs_m')
                self.assertEqual(ast.unparse(statement.value), 'block_start_loc + tl.arange(0, BLOCK_M).to(tl.int64)')
                statement.value = copy.deepcopy(next(n.value for n in original_kernel.body
                    if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'offs_m'))
            else:
                loop = next(n for n in kernel.body if isinstance(n, ast.For))
                position = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign)
                    and ast.unparse(n.targets[0]) == 'qk' and ast.unparse(n.value) == 'sm_scale * qk')
                loop.body[position] = copy.deepcopy(next(n for n in next(n for n in original_kernel.body
                    if isinstance(n, ast.For)).body if isinstance(n, ast.AugAssign)
                    and isinstance(n.op, ast.Mult) and ast.unparse(n.target) == 'qk'))
            self.assertEqual(ast.dump(selected, include_attributes=False), ast.dump(original, include_attributes=False))
