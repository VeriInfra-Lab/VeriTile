import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.bench.jit_correspondence import check_control_source, check_manifest

from tests.python.layout_fixtures import check_pinned_pairs, pinned_pairs, variant_provenance, variant_source

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / 'bench/tritonbench_g/streamk_matmul'
FUNCTIONS = ['swizzle_tile', 'linear_tile', 'mac_loop', 'first_wave', 'full_tiles', 'reduce_streamk']


class UndoArea(ast.NodeTransformer):
    def rename(self, name):
        return {n + '_area': n for n in FUNCTIONS + ['matmul']}.get(name, name)

    def visit_Name(self, node):
        node.id = self.rename(node.id)
        return node

    def visit_FunctionDef(self, node):
        node.name = self.rename(node.name)
        return self.generic_visit(node)

    def visit_ClassDef(self, node):
        node.name = self.rename(node.name)
        return self.generic_visit(node)

    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.op, ast.Mult) and isinstance(node.left, ast.Name)
                and node.left.id in ['program', 'pid'] and isinstance(node.right, ast.BinOp)
                and isinstance(node.right.op, ast.Mult) and ast.unparse(node.right.left) == 'BLOCK_M'
                and ast.unparse(node.right.right) == 'BLOCK_N'):
            return ast.BinOp(ast.BinOp(node.left, ast.Mult(), node.right.left), ast.Mult(), node.right.right)
        return node


class StreamKBodyMigration(unittest.TestCase):
    def setUp(self):
        self.records = pinned_pairs(CASE)
        self.lean = (CASE / 'Kernels.lean').read_text()

    def check(self, lean, function='mac_loop'):
        r = next(r for r in self.records if r['python_function'] == function)
        return check_control_source((ROOT / r['python']).read_text(), lean, r['python_function'],
                                    r['lean_definition'], r['signature_definition'], r['expected_signature'],
                                    r['expected_constexpr_bools'], r['expected_constexpr_dtypes'], r['expected_jit_calls'])

    def test_all_six_functions_and_all_six_candidate_functions(self):
        result = check_pinned_pairs(ROOT, CASE)
        self.assertEqual(len(result), 12)
        self.assertEqual([r['statements'] for r in result], [8, 3, 12, 4, 16, 7] * 2)
        self.assertEqual(result, json.loads(json.dumps(result)))
        self.assertTrue(all(r['status'] == 'PASS' and r['mode'] == 'pasted_body' for r in result))

    def test_source_mutations_fail_closed(self):
        mutations = [
            ('mac_loop', 'rk = tl.arange(0, BLOCK_K).to(tl.int64)', 'rk = tl.arange(0, BLOCK_K)'),
            ('mac_loop', 'dtype=ACC_TYPE', 'dtype=tl.float32'),
            ('mac_loop', 'range(start_iter, end_iter)', 'range(0, end_iter)'),
            ('mac_loop', 'acc += tl.dot(a, b)', 'acc += a * b'),
            ('mac_loop', 'tl.store(partials + slots, acc)', 'tl.store(partials + slots, acc, mask=True)'),
            ('first_wave', 'while start_iter < last_iter:', 'while start_iter <= last_iter:'),
            ('full_tiles', 'acc = acc.to(tl.float16)', 'acc = acc'),
            ('reduce_streamk', 'tl.atomic_add(C + offsets, contribution, mask=mask)',
             'tl.store(C + offsets, contribution, mask=mask)'),
        ]
        for function, before, after in mutations:
            with self.subTest(function=function, before=before):
                self.assertIn(before, self.lean)
                with self.assertRaises(ValueError):
                    self.check(self.lean.replace(before, after), function)

    def test_dtype_helper_and_caller_metadata_mutations_fail_closed(self):
        for before, after in [('ACC_TYPE := .fp32', 'ACC_TYPE := .fp16'),
                              ('swizzle_tile := "swizzle"', 'swizzle_tile := "linear"'),
                              ('with_control_function "mac_loop"', 'with_control_function "other"'),
                              ('("partials",.fp32)', '("partials",.fp16)')]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.check(self.lean.replace(before, after))
        missing = [r for r in self.records if r['python_function'] != 'swizzle_tile']
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / 'manifest.json'
            manifest.write_text(json.dumps(missing))
            with self.assertRaises(ValueError):
                check_manifest(ROOT, manifest)

    def test_complete_module_candidate_provenance(self):
        meta = variant_provenance(CASE)
        original, candidate = [variant_source(CASE / meta[key]) for key in ['original', 'candidate']]
        for key, source in [('original', original), ('candidate', candidate)]:
            self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), meta[key + '_sha256'])
        self.assertEqual(ast.dump(ast.parse(original)), ast.dump(UndoArea().visit(ast.parse(candidate))))

    def test_wrapper_supplies_fp32_storage_and_accumulator(self):
        for filename in ['streamk_matmul.py', 'streamk_matmul_area.py']:
            tree = ast.parse(variant_source(CASE / filename))
            call = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_call')
            gates = [n for n in ast.walk(call) if isinstance(n, ast.If)
                     and ast.unparse(n.test) == 'a.dtype != torch.float32 or b.dtype != torch.float32']
            self.assertEqual(len(gates), 1)
            self.assertIsInstance(gates[0].body[0], ast.Raise)
            dtype = next(n for n in call.body if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == 'ACC_TYPE' for t in n.targets))
            self.assertIsInstance(dtype.value, ast.IfExp)
            self.assertEqual(ast.unparse(dtype.value.body), 'tl.float32')
            self.assertIn('torch.float32', [ast.unparse(n) for n in dtype.value.test.comparators[0].elts])
