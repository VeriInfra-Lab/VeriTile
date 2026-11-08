import ast
import hashlib
from pathlib import Path
import unittest
from scripts.bench.jit_correspondence import check

from tests.python.layout_fixtures import check_pinned_pairs, pinned_pairs, variant_provenance, variant_source

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / 'bench/tritonbench_g/llama_ff_triton'

class LlamaFFBodyMigrationTests(unittest.TestCase):
    def test_both_storage_routes_keep_the_complete_original_and_candidate(self):
        rows = check_pinned_pairs(ROOT, DIRECTORY)
        self.assertEqual([r['statements'] for r in rows], [37] * 4)
        self.assertTrue(all(r['mode'] == 'pasted_body' for r in rows))
        records = pinned_pairs(DIRECTORY)
        self.assertEqual([r['expected_signature']['pointers'][1][1] for r in records],
                         ['fp16', 'fp16', 'i8', 'i8'])
        self.assertTrue(all(len(r['expected_call_bindings']) == 12 for r in records))

    def test_casts_bitcasts_masks_intrinsics_and_branches_are_required(self):
        records = pinned_pairs(DIRECTORY)
        for record in records:
            python = (ROOT / record['python']).read_text()
            lean = (ROOT / record['lean']).read_text()
            start = lean.index('def ' + record['lean_definition'] + ' ')
            for before, after in [('M = tl.cast(M, tl.int64)', 'M = M'),
                                  ('bitcast=True', 'bitcast=False'),
                                  ('b = b.to(tl.float32)', 'b = b'),
                                  ('if USE_FP8:', 'if False:'),
                                  ('K - step * BLOCK_SIZE_K', 'K'),
                                  ('tl.extra.cuda.libdevice.pow', 'tl.pow'),
                                  ('tl.math.rsqrt', 'tl.sqrt'),
                                  ('tl.store(out_ptrs, accumulator, mask=out_mask)',
                                   'tl.store(out_ptrs, accumulator)')]:
                changed = lean[:start] + lean[start:].replace(before, after, 1)
                self.assertNotEqual(changed, lean)
                with self.subTest(definition=record['lean_definition'], before=before), self.assertRaises(ValueError):
                    check(python, changed, record['python_function'], record['lean_definition'],
                          record['signature_definition'], record['expected_signature'],
                          expected_call_bindings=record['expected_call_bindings'])

    def test_candidate_only_commutes_the_existing_integer_output_index(self):
        original_path = DIRECTORY / 'llama_ff_triton.py'
        candidate_path = DIRECTORY / 'llama_ff_index_commuted.py'
        original = ast.parse(variant_source(original_path))
        candidate = ast.parse(variant_source(candidate_path))
        for node in ast.walk(candidate):
            if isinstance(node, ast.FunctionDef) and node.name == 'ff_llama_index_commuted':
                node.name = 'ff_llama'
            if isinstance(node, ast.Name) and node.id == 'ff_llama_index_commuted':
                node.id = 'ff_llama'
        fn = next(n for n in candidate.body if isinstance(n, ast.FunctionDef) and n.name == 'ff_llama')
        assignment = next(n for n in fn.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'offs_outm')
        self.assertEqual(ast.unparse(assignment.value.left), 'BLOCK_SIZE_M * pid_m')
        assignment.value.left.left, assignment.value.left.right = assignment.value.left.right, assignment.value.left.left
        original_fn = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == 'ff_llama')
        self.assertEqual(ast.dump(original_fn), ast.dump(fn))
        provenance = variant_provenance(DIRECTORY)
        for label, path in [('original', original_path), ('candidate', candidate_path)]:
            self.assertEqual(hashlib.sha256(variant_source(path).encode()).hexdigest(), provenance[label + '_sha256'])

if __name__ == '__main__':
    unittest.main()
