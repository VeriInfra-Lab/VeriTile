import ast
import hashlib
import json
from pathlib import Path
import unittest
from scripts.bench.jit_correspondence import check, check_manifest, check_statement_function
ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / 'bench/tritonbench_g/kv_cache_filling'

class KvCacheBodyMigrationTests(unittest.TestCase):
    def test_all_three_policies_and_the_complete_helper(self):
        rows = check_manifest(ROOT, DIRECTORY / 'jit-correspondence.json')
        self.assertEqual([r['statements'] for r in rows], [5, 5] + [8] * 6)
        self.assertTrue(all(r['mode'] == 'pasted_body' for r in rows))
        records = json.loads((DIRECTORY / 'jit-correspondence.json').read_text())[2:]
        self.assertEqual([r['expected_constexpr_bindings']['POLICY'] for r in records],
                         ['0', '0', '8', '8', '4', '4'])
        self.assertEqual([r['expected_signature']['pointers'][2][1] for r in records],
                         ['fp32', 'fp32', 'u8', 'u8', 'u8', 'u8'])

    def test_all_policy_source_statements_are_required(self):
        records = json.loads((DIRECTORY / 'jit-correspondence.json').read_text())[2:]
        for record in records:
            python = (ROOT / record['python']).read_text()
            lean = (ROOT / record['lean']).read_text()
            start = lean.index('def ' + record['lean_definition'] + ' ')
            for before, after in [('tl.program_id(0).to(tl.int64)', 'tl.program_id(0)'),
                                  ('elif POLICY == 8:', 'elif POLICY == 4:'),
                                  ('qk2, _, _ = _quant_row', 'qk2, sk, zk = _quant_row'),
                                  ('qk = qk1 + qk2 * 16', 'qk = qk1 + qk2'),
                                  ('other=-1', 'other=0'),
                                  ('ik < CK, ik, 0', 'ik < DK, ik, 0')]:
                if before not in lean[start:]:
                    continue
                changed = lean[:start] + lean[start:].replace(before, after, 1)
                with self.subTest(definition=record['lean_definition'], before=before), self.assertRaises(ValueError):
                    check(python, changed, record['python_function'], record['lean_definition'],
                          record['signature_definition'], record['expected_signature'],
                          expected_call_bindings=record['expected_call_bindings'],
                          expected_constexpr_bindings=record['expected_constexpr_bindings'],
                          expected_jit_functions=record['expected_jit_functions'])

    def test_helper_quantization_cast_and_return_order_are_required(self):
        record = json.loads((DIRECTORY / 'jit-correspondence.json').read_text())[0]
        python = (ROOT / record['python']).read_text()
        lean = (ROOT / record['lean']).read_text()
        for before, after in [('return code, scale, zero', 'return code, zero, scale'),
                              ('.to(tl.uint8)', '.to(tl.int8)'),
                              ('hi == lo, 1.0, raw_scale', 'hi == lo, 0.0, raw_scale')]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                check_statement_function(python, lean.replace(before, after, 1),
                                         record['python_function'], record['lean_definition'])

    def test_candidate_only_commutes_three_existing_integer_products(self):
        original_path = DIRECTORY / 'kv_cache_filling.py'
        candidate_path = DIRECTORY / 'kv_cache_filling_index_commuted.py'
        original = ast.parse(original_path.read_text())
        candidate = ast.parse(candidate_path.read_text())
        for node in ast.walk(candidate):
            if isinstance(node, ast.FunctionDef) and node.name == '_fill_rows_index_commuted':
                node.name = '_fill_rows'
            if isinstance(node, ast.Name) and node.id == '_fill_rows_index_commuted':
                node.id = '_fill_rows'
        fn = next(n for n in candidate.body if isinstance(n, ast.FunctionDef) and n.name == '_fill_rows')
        products = [n for n in ast.walk(fn) if isinstance(n, ast.BinOp) and ast.unparse(n) == 'DK * src']
        self.assertEqual(len(products), 3)
        for product in products:
            product.left, product.right = product.right, product.left
        self.assertEqual(ast.dump(original), ast.dump(candidate))
        provenance = json.loads((DIRECTORY / 'candidate-provenance.json').read_text())
        for label, path in [('original', original_path), ('candidate', candidate_path)]:
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), provenance[label + '_sha256'])
