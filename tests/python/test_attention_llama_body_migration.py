import ast
import hashlib
from pathlib import Path
import re
import unittest
from scripts.bench.jit_correspondence import check, quotation

from tests.python.layout_fixtures import check_pinned_pairs, pinned_pairs, variant_provenance, variant_source

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / 'bench/tritonbench_g/attention_llama'


class AttentionLlamaBodyMigrationTests(unittest.TestCase):
    def setUp(self):
        self.lean = (DIRECTORY / 'Kernels.lean').read_text()
        self.records = pinned_pairs(DIRECTORY)

    def verify(self, text, index=0):
        r = self.records[index]
        result = check((ROOT / r['python']).read_text(), text, r['python_function'],
                       r['lean_definition'], r['signature_definition'], r['expected_signature'],
                       expected_call_bindings=r['expected_call_bindings'])
        signature, _ = quotation(text, r['lean_definition'], raw_python=True)
        flag = 'false' if r['invocation_variant'] == 'half_storage' else 'true'
        if len(re.findall(r'\blet USE_FP8 := Bool\.' + flag + r'\b', signature)) != 1:
            raise ValueError('storage specialization differs from the reviewed caller')
        return result

    def test_four_complete_source_pairs(self):
        result = check_pinned_pairs(ROOT, DIRECTORY)
        self.assertEqual([r['statements'] for r in result], [26,26,26,26])
        for i in range(4):
            self.verify(self.lean, i)

    def test_source_cast_branch_bound_and_store_mutations_are_rejected(self):
        mutations = [
            ('k = k.to(tl.float8e5, bitcast=True)', 'k = k.to(tl.float8e5)'),
            ('p = p.to(Q.dtype.element_ty)', 'p = p'),
            ('range(0, block_n_end, BLOCK_N)', 'range(0, N_CTX, BLOCK_N)'),
            ('block_n_offs[None, :] < N_CTX', 'block_n_offs[None, :] <= N_CTX'),
            ('tl.store(out_ptrs, acc, offs_m[:, None] < H)',
             'tl.store(out_ptrs, acc, offs_m[:, None] <= H)'),
        ]
        for before, after in mutations:
            with self.subTest(before=before):
                self.assertIn(before, self.lean)
                with self.assertRaises(ValueError):
                    self.verify(self.lean.replace(before, after))

    def test_dtype_call_bindings_and_closed_storage_flags_are_checked(self):
        for before, after, index in [
            ('("Q",.fp16)', '("Q",.fp32)', 0),
            ('stride_qz := stride_qz', 'stride_qz := 0', 0),
            ('let USE_FP8 := Bool.false', 'let USE_FP8 := Bool.true', 0),
            ('let USE_FP8 := Bool.true', 'let USE_FP8 := Bool.false', 2),
        ]:
            with self.subTest(before=before):
                with self.assertRaises(ValueError):
                    self.verify(self.lean.replace(before, after), index)

    def test_only_documented_carry_reordering_changes_the_python_module(self):
        p = variant_provenance(DIRECTORY)
        original = variant_source(DIRECTORY / p['original'])
        candidate = variant_source(DIRECTORY / p['candidate'])
        self.assertEqual(hashlib.sha256(original.encode()).hexdigest(), p['original_sha256'])
        self.assertEqual(hashlib.sha256(candidate.encode()).hexdigest(), p['candidate_sha256'])
        module = ast.parse(candidate)
        for node in ast.walk(module):
            if isinstance(node, (ast.Name, ast.FunctionDef)):
                attr = 'id' if isinstance(node, ast.Name) else 'name'
                setattr(node, attr, getattr(node, attr).removesuffix('_carry_first'))
        kernel = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == '_fwd_kernel')
        loop = next(n for n in kernel.body if isinstance(n, ast.For))
        stmt = next(n for n in loop.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'l_curr')
        self.assertEqual(ast.unparse(stmt.value), 'l_prev + tl.sum(p, 1)')
        stmt.value.left, stmt.value.right = stmt.value.right, stmt.value.left
        self.assertEqual(ast.dump(ast.parse(original), include_attributes=False),
                         ast.dump(module, include_attributes=False))
