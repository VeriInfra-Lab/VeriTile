import ast
import hashlib
import math
import re
import struct
from pathlib import Path
import unittest
from scripts.bench.jit_correspondence import check

from tests.python.layout_fixtures import check_pinned_pairs, pinned_pairs, variant_provenance, variant_source

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / 'bench/tritonbench_g/triton_linear_activation'


class LinearActivationBodyMigrationTests(unittest.TestCase):
    def records(self):
        return pinned_pairs(CASE)

    def test_all_helpers_original_and_existing_candidate(self):
        rows = check_pinned_pairs(ROOT, CASE)
        self.assertEqual([r['statements'] for r in rows], [2] * 8 + [27, 27])
        for r in rows[-2:]:
            self.assertEqual(set(r['runtime_parameters']), {
                'M', 'N', 'K', 'CACHE_KEY_M', 'CACHE_KEY_N', 'CACHE_KEY_K',
                'output_m_stride', 'output_n_stride', 'act_inputs_m_stride',
                'act_inputs_n_stride', 'a_m_stride', 'a_k_stride', 'b_n_stride', 'b_k_stride'})

    def test_caller_cast_mask_store_and_helper_mutations_are_rejected(self):
        for r in self.records()[-2:]:
            python = (ROOT / r['python']).read_text()
            lean = (ROOT / r['lean']).read_text()
            begin = lean.index('def ' + r['lean_definition'] + ' ')
            tail = lean.find('\ndef ', begin + 1)
            tail = len(lean) if tail == -1 else tail
            body = lean[begin:tail]
            for before, after in [('.to(tl.float32)', ''), ('range(K, 0, -BLOCK_K)', 'range(K, 0, BLOCK_K)'),
                                  ('mask=save_mask', 'mask=c_ptr_mask'), ('mask=c_ptr_mask', 'mask=True'),
                                  ('tl.store(C, acc,', 'tl.store(C, acc.to(tl.float32),'),
                                  ('acc = fast_gelu(acc)', 'acc = gelu(acc)'),
                                  ('B += BLOCK_K * b_k_stride', 'B += BLOCK_K')]:
                self.assertIn(before, body)
                changed = lean[:begin] + body.replace(before, after, 1) + lean[tail:]
                with self.subTest(body=r['lean_definition'], mutation=before), self.assertRaises(ValueError):
                    check(python, changed, r['python_function'], r['lean_definition'],
                          r['signature_definition'], r['expected_signature'],
                          expected_call_bindings=r['expected_call_bindings'],
                          expected_jit_functions=r['expected_jit_functions'])

    def test_native_sqrt_words_match_the_actual_host_constants(self):
        literals = (CASE / 'Proofs/Mathematical_FPLiterals.lean').read_text()
        for name, value in [('sqrt2', math.sqrt(2.0)), ('sqrt2pi', math.sqrt(2.0 / math.pi))]:
            declared = re.search(r'def ' + name + r'Word[^\n]*0x([0-9a-fA-F]+)', literals)
            self.assertIsNotNone(declared)
            actual = struct.unpack('<I', struct.pack('<f', value))[0]
            self.assertEqual(actual, int(declared[1], 16))

    def test_existing_candidate_only_changes_the_conditional_width_index(self):
        original = ast.parse(variant_source(CASE / 'triton_linear_activation.py'))
        candidate = ast.parse(variant_source(CASE / 'triton_linear_activation_shifted.py'))

        class Restore(ast.NodeTransformer):
            changed = 0

            def visit_FunctionDef(self, node):
                if node.name == 'kernel_fma_shifted':
                    node.name = 'kernel_fma'
                elif node.name == 'linear_layer_shifted':
                    node.name = 'linear_layer'
                return self.generic_visit(node)

            def visit_Name(self, node):
                if node.id == 'kernel_fma_shifted':
                    node.id = 'kernel_fma'
                elif node.id == 'linear_layer_shifted':
                    node.id = 'linear_layer'
                return node

            def visit_If(self, node):
                if ast.unparse(node.test) == 'GROUP_M == 8':
                    self.changed += 1
                    assert len(node.body) == len(node.orelse) == 1
                    assert ast.unparse(node.body[0]) == 'width = grid_n << 3'
                    assert ast.unparse(node.orelse[0]) == 'width = GROUP_M * grid_n'
                    return node.orelse[0]
                return self.generic_visit(node)

        restore = Restore()
        candidate = restore.visit(candidate)
        self.assertEqual(restore.changed, 1)
        self.assertEqual(ast.dump(candidate), ast.dump(original))
        provenance = variant_provenance(CASE)
        for field in ['original', 'candidate']:
            self.assertEqual(hashlib.sha256(variant_source(CASE / provenance[field]).encode()).hexdigest(),
                             provenance[field + '_sha256'])
