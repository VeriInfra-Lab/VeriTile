import ast
import hashlib
from pathlib import Path
import unittest
from scripts.bench.jit_correspondence import check

from tests.python.layout_fixtures import check_pinned_pairs, pinned_pairs, variant_provenance, variant_source

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / 'bench/tritonbench_g/bmm_chunk_fwd'


class BmmChunkBodyMigrationTests(unittest.TestCase):
    def setUp(self):
        self.lean = (DIRECTORY / 'Kernels.lean').read_text()
        self.records = pinned_pairs(DIRECTORY)

    def verify(self, text, index=0):
        r = self.records[index]
        return check((ROOT / r['python']).read_text(), text, r['python_function'],
                     r['lean_definition'], r['signature_definition'], r['expected_signature'],
                     expected_call_bindings=r['expected_call_bindings'],
                     expected_integer_args=r.get('expected_integer_args', {}),
                     expected_arange_args=r.get('expected_arange_args', {}))

    def test_six_complete_source_pairs(self):
        results = check_pinned_pairs(ROOT, DIRECTORY)
        self.assertEqual([r['statements'] for r in results], [45,45,45,10,10,10])

    def test_cast_return_mask_and_sequence_mutations_are_rejected(self):
        mutations = [
            ('tl.program_id(axis=1).to(tl.int64)', 'tl.program_id(axis=1)'),
            ('K = tl.cast(K, tl.int64)', 'K = K'),
            ('            return', '            pass'),
            ('other=0.0).to(dot_dtype)', 'other=0.0)'),
            ('other=-2)', 'other=-1)'),
            ('out = acc.to(out_ptr.dtype.element_ty)', 'out = acc'),
            ('offs_m[:, None] < chunk_size)', 'offs_m[:, None] <= chunk_size)'),
        ]
        for before, after in mutations:
            with self.subTest(before=before):
                self.assertIn(before, self.lean)
                with self.assertRaises(ValueError):
                    self.verify(self.lean.replace(before, after))
        with self.assertRaises(ValueError):
            self.verify(self.lean.replace('(column > row)', '(column >= row)'), 3)

    def test_external_dtype_and_call_bindings_are_checked(self):
        for before, after in [('("seq_idx_ptr",.i64)', '("seq_idx_ptr",.i32)'),
                              ('K := K', 'K := 0'),
                              ('match a_dtype with | .fp32 => .fp32', 'match a_dtype with | .fp32 => .fp16')]:
            with self.subTest(before=before):
                with self.assertRaises(ValueError):
                    self.verify(self.lean.replace(before, after))

    def test_only_documented_candidate_transformations(self):
        provenance = variant_provenance(DIRECTORY)
        original_text = variant_source(DIRECTORY / 'bmm_chunk_fwd.py')
        self.assertEqual(hashlib.sha256(original_text.encode()).hexdigest(), provenance['original_sha256'])
        original = ast.parse(original_text)
        for variant in provenance['candidates']:
            text = variant_source(DIRECTORY / variant['python'])
            self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), variant['sha256'])
            module = ast.parse(text)
            if variant['kind'] == 'integer_product_commutation':
                kernel = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == '_bmm_chunk_fwd_kernel_repaired')
                stmt = next(n for n in kernel.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'pid_h')
                self.assertEqual(ast.unparse(stmt.value), 'pid_ch - ngroups * pid_c')
                stmt.value.right.left, stmt.value.right.right = stmt.value.right.right, stmt.value.right.left
            else:
                for node in ast.walk(module):
                    if isinstance(node, (ast.Name, ast.FunctionDef)):
                        attr = 'id' if isinstance(node, ast.Name) else 'name'
                        setattr(node, attr, getattr(node, attr).removesuffix('_dot_first'))
                kernel = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == '_bmm_chunk_fwd_kernel_repaired')
                loop = next(n for n in kernel.body if isinstance(n, ast.For))
                i, stmt = next((i,s) for i,s in enumerate(loop.body) if isinstance(s, ast.Assign) and ast.unparse(s.targets[0]) == 'acc')
                self.assertEqual(ast.unparse(stmt.value), 'tl.dot(a, b) + acc')
                loop.body[i] = ast.AugAssign(target=stmt.targets[0], op=ast.Add(), value=stmt.value.left)
            self.assertEqual(ast.dump(original, include_attributes=False), ast.dump(module, include_attributes=False))
