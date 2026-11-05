import ast
import hashlib
import json
from pathlib import Path
import unittest

from scripts.bench.jit_correspondence import check_control_source

from tests.python.layout_fixtures import check_pinned_pairs, pinned_pairs, variant_provenance, variant_source

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / 'bench/tritonbench_g/spinning_lock_reduction'


class UndoArea(ast.NodeTransformer):
    def rename(self, name):
        return {'spinning_lock_kernel_area': 'spinning_lock_kernel',
                'spinning_lock_area': 'spinning_lock'}.get(name, name)

    def visit_Name(self, node):
        node.id = self.rename(node.id)
        return node

    def visit_FunctionDef(self, node):
        node.name = self.rename(node.name)
        return self.generic_visit(node)

    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.op, ast.Mult) and isinstance(node.left, ast.Name)
                and node.left.id in ['pid', 'next_pid']
                and isinstance(node.right, ast.BinOp) and isinstance(node.right.op, ast.Mult)
                and ast.unparse(node.right.left) == 'BLOCK_SIZE_M'
                and ast.unparse(node.right.right) == 'BLOCK_SIZE_N'):
            return ast.BinOp(ast.BinOp(node.left, ast.Mult(), node.right.left), ast.Mult(), node.right.right)
        return node


class SpinningLockBodyMigration(unittest.TestCase):
    def setUp(self):
        self.records = pinned_pairs(CASE)
        self.lean = (CASE / 'Kernels.lean').read_text()

    def check(self, lean, record=None):
        record = record or self.records[0]
        return check_control_source((ROOT / record['python']).read_text(), lean,
                                    record['python_function'], record['lean_definition'],
                                    record['signature_definition'], record['expected_signature'],
                                    record['expected_constexpr_bools'])

    def test_complete_original_and_candidate(self):
        results = check_pinned_pairs(ROOT, CASE)
        self.assertEqual(results, json.loads(json.dumps(results)))
        self.assertEqual([r['statements'] for r in results], [9, 9])
        self.assertTrue(all(r['mode'] == 'pasted_body' and r['status'] == 'PASS' for r in results))

    def test_reject_source_effect_control_and_cast_mutations(self):
        mutations = [
            ('pid = tl.program_id(0).to(tl.int64)', 'pid = tl.program_id(0)'),
            ('pid.to(tl.uint64) % tl.cast(k, tl.uint64)', 'pid % tl.cast(k, tl.uint64)'),
            ('lock_ready = -1 if LOCK_NEG else 1', 'lock_ready = 1'),
            ('range(1, 10)', 'range(1, 9)'),
            ('tl.atomic_cas(locks + next_pid, lock_ready, lock_ready)',
             'tl.atomic_cas(locks + next_pid, 0, lock_ready)'),
            ('tl.debug_barrier()', 'pass'),
            ('if P_NEG:', 'if False:'),
            ('0x80000000', '0x00000000'),
            ('if C_NEG else acc, mask=mask', 'if C_NEG else acc'),
        ]
        for before, after in mutations:
            with self.subTest(before=before):
                self.assertIn(before, self.lean)
                with self.assertRaises(ValueError):
                    self.check(self.lean.replace(before, after, 1))

    def test_reject_caller_signature_and_flag_mutations(self):
        for before, after in [('("locks",.i32)', '("locks",.fp32)'),
                              ('P_NEG := false', 'P_NEG := true'),
                              ('LOCK_NEG := false', 'LOCK_NEG := false,LOCK_NEG := false'),
                              ('with_control_call callSignature', 'with_control_call otherSignature')]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.check(self.lean.replace(before, after, 1))

    def test_candidate_has_only_documented_area_reassociation(self):
        manifest = variant_provenance(CASE)
        original, candidate = [variant_source(CASE / manifest[k]) for k in ['original', 'candidate']]
        for key, source in [('original', original), ('candidate', candidate)]:
            self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), manifest[key + '_sha256'])
        self.assertEqual(ast.dump(ast.parse(original)), ast.dump(UndoArea().visit(ast.parse(candidate))))

    def test_repair_wrapper_rejects_negative_views_before_flag_queries(self):
        for record in self.records:
            tree = ast.parse((ROOT / record['python']).read_text())
            gate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_check_inputs')
            rejecting = [n for n in ast.walk(gate) if isinstance(n, ast.If)
                         and any(isinstance(x, ast.Call) and ast.unparse(x.func) == 'tensor.is_neg'
                                 for x in ast.walk(n.test))
                         and any(isinstance(x, ast.Raise) for x in n.body)]
            self.assertEqual(len(rejecting), 1)
            wrapper = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                           and n.name in ['spinning_lock', 'spinning_lock_area'])
            self.assertIsInstance(wrapper.body[0], ast.Expr)
            self.assertEqual(ast.unparse(wrapper.body[0].value.func), '_check_inputs')
            flags = [n for n in wrapper.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id in ['p_neg', 'c_neg', 'lock_neg'] for t in n.targets)]
            self.assertEqual([ast.unparse(n.value) for n in flags], ['P.is_neg()', 'C.is_neg()', 'locks.is_neg()'])
