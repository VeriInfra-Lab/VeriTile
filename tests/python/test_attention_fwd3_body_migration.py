import ast
import hashlib
from pathlib import Path
import unittest

from scripts.bench.jit_correspondence import check, check_statement_function

from tests.python.layout_fixtures import check_pinned_pairs, pinned_pairs, variant_provenance, variant_source

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / 'bench/tritonbench_g/attention_fwd_triton3'
FUNCTIONS = ['_attn_fwd_inner', '_attn_fwd']


class UndoCSE(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id.endswith('_key_mask_cse'):
            node.id = node.id.removesuffix('_key_mask_cse')
        if node.id == 'key_mask':
            return ast.parse('(tl.arange(0, BLOCK_N) + start_n) < N_CTX', mode='eval').body
        return node

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'key_mask':
            return None
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        node.name = node.name.removesuffix('_key_mask_cse')
        return self.generic_visit(node)


class AttentionFwd3BodyMigration(unittest.TestCase):
    def setUp(self):
        self.records = pinned_pairs(CASE)
        self.lean = (CASE / 'Kernels.lean').read_text()

    def test_four_complete_source_pairs(self):
        result = check_pinned_pairs(ROOT, CASE)
        self.assertEqual(len(result), 4)
        self.assertEqual([r['statements'] for r in result], [4, 24, 4, 24])
        self.assertTrue(all(r['status'] == 'PASS' and r['mode'] == 'pasted_body' for r in result))

    def test_omitted_casts_branches_and_clamps_are_rejected(self):
        r = self.records[0]
        py = (ROOT / r['python']).read_text()
        mutations = [('start_m.to(tl.int64)', 'start_m'),
                     ('lo.to(tl.int32)', 'lo'),
                     ('tl.cast(N_CTX, tl.int64)', 'N_CTX'),
                     ('if lo < 0:', 'if lo <= 0:'),
                     ('p.to(v.dtype)', 'p'),
                     ('alpha = tl.where(alpha_mask, 1., alpha)', 'alpha = alpha')]
        for before, after in mutations:
            with self.subTest(before=before):
                self.assertIn(before, self.lean)
                with self.assertRaises(ValueError):
                    check_statement_function(py, self.lean.replace(before, after), r['python_function'], r['lean_definition'])

    def test_call_abi_and_helper_binding_are_checked(self):
        r = self.records[1]
        py = (ROOT / r['python']).read_text()
        for before, after in [('("Q",.fp32)', '("Q",.fp16)'),
                              ('_attn_fwd_inner := innerSource]', '_attn_fwd_inner := innerCSESource]')]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                check(py, self.lean.replace(before, after), r['python_function'], r['lean_definition'],
                      r['signature_definition'], r['expected_signature'], r['expected_call_bindings'],
                      expected_jit_functions=r['expected_jit_functions'])

    def test_candidate_only_cses_the_key_mask(self):
        meta = variant_provenance(CASE)
        sources = {k: variant_source(CASE / meta[k]) for k in ['original', 'candidate']}
        for k, source in sources.items():
            self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), meta[k + '_sha256'])
        def kernels(source):
            return [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                    and n.name.removesuffix('_key_mask_cse') in FUNCTIONS]
        tree = ast.parse(sources['candidate'])
        definitions = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'key_mask' for t in n.targets)]
        self.assertEqual(len(definitions), 1)
        uses = [n for n in ast.walk(tree) if isinstance(n, ast.Name)
                and n.id == 'key_mask' and isinstance(n.ctx, ast.Load)]
        self.assertEqual(len(uses), 2)
        original = kernels(sources['original'])
        candidate = [UndoCSE().visit(n) for n in kernels(sources['candidate'])]
        self.assertEqual([ast.dump(n) for n in original], [ast.dump(n) for n in candidate])
