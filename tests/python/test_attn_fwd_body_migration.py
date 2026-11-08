import ast
import hashlib
from pathlib import Path
import unittest

from scripts.bench.jit_correspondence import check, check_statement_function

from tests.python.layout_fixtures import check_pinned_pairs, pinned_pairs, variant_provenance, variant_source

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / 'bench/tritonbench_g/attn_fwd_triton'


class AttnFwdBodyMigration(unittest.TestCase):
    def setUp(self):
        self.rows = pinned_pairs(CASE)
        self.lean = (CASE / 'Kernels.lean').read_text()

    def test_complete_helper_and_both_callers(self):
        result = check_pinned_pairs(ROOT, CASE)
        self.assertEqual([r['statements'] for r in result], [3, 31, 3, 31])
        self.assertTrue(all(r['status'] == 'PASS' and r['mode'] == 'pasted_body' for r in result))

    def test_helper_stage_cast_mask_and_dot_mutations_are_rejected(self):
        row = self.rows[0]
        for before, after in [
            ('elif STAGE == 2:', 'elif STAGE == 3:'),
            ('tl.dot(q, k).to(tl.float32)', 'tl.dot(q, k)'),
            ('out_dtype=tl.float16', 'out_dtype=tl.float32'),
            ('offs_n[None, :] < (N_CTX - start_n)', 'offs_n[None, :] <= (N_CTX - start_n)'),
            ('return acc, l_i, m_i', 'return acc, m_i, l_i'),
        ]:
            with self.subTest(before=before):
                self.assertIn(before, self.lean)
                with self.assertRaises(ValueError):
                    check_statement_function((ROOT / row['python']).read_text(),
                                             self.lean.replace(before, after),
                                             row['python_function'], row['lean_definition'])

    def test_caller_cast_and_second_helper_call_cannot_disappear(self):
        row = self.rows[1]
        for before, after in [
            ('off_z.to(tl.int64)', 'off_z'),
            ('acc, l_i, _ = _attn_fwd_inner_repaired', 'acc, l_i, m_i = _attn_fwd_inner_repaired'),
            ('acc.to(Out.type.element_ty)', 'acc'),
        ]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                check((ROOT / row['python']).read_text(), self.lean.replace(before, after),
                      row['python_function'], row['lean_definition'], row['signature_definition'],
                      row['expected_signature'], row['expected_call_bindings'],
                      expected_jit_functions=row['expected_jit_functions'])

    def test_candidate_changes_only_the_integer_remainder_expression(self):
        meta = variant_provenance(CASE)
        sources = {k: variant_source(CASE / meta[k]) for k in ['original', 'candidate']}
        for key, source in sources.items():
            self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), meta[key + '_sha256'])
        trees = {k: ast.parse(s) for k, s in sources.items()}
        caller = next(n for n in trees['candidate'].body if isinstance(n, ast.FunctionDef)
                      and n.name == '_attn_fwd_repaired')
        changed = next(n for n in caller.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'off_h' for t in n.targets))
        self.assertEqual(ast.dump(changed), ast.dump(ast.parse('off_h = off_hz - (off_hz // H) * H').body[0]))
        caller.body[caller.body.index(changed)] = ast.parse('off_h = off_hz % H').body[0]
        self.assertEqual(ast.dump(trees['candidate']), ast.dump(trees['original']))
