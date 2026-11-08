import ast
import hashlib
from pathlib import Path
import unittest
from scripts.bench.jit_correspondence import check
from tests.python.layout_fixtures import check_pinned_pairs, pinned_pairs, variant_provenance, variant_source

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'bench/tritonbench_g'

class RotaryBodyMigrationTests(unittest.TestCase):
    def test_complete_bodies_and_both_offset_invocations(self):
        for name in ['rotary_transform', 'rotary_transform_ops']:
            rows = check_pinned_pairs(ROOT, BASE / name)
            self.assertEqual([r['statements'] for r in rows], [11] * 4)
            records = pinned_pairs(BASE / name)
            for r in records:
                profile = r['expected_signature']
                self.assertEqual(profile['dtype_parameter_types'], {'wideCu': 'Bool', 'wideOffset': 'Bool'})
                self.assertEqual('SEQLEN_OFFSETS' in dict(profile['pointers']),
                                 r['invocation_variant'].startswith('tensor'))
                self.assertEqual('SEQLEN_OFFSETS' in dict(profile['scalars']),
                                 r['invocation_variant'].startswith('scalar'))

    def test_cast_masks_metadata_and_early_return_are_required(self):
        for name in ['rotary_transform', 'rotary_transform_ops']:
            for r in pinned_pairs(BASE / name):
                python = (ROOT / r['python']).read_text()
                lean = (ROOT / r['lean']).read_text()
                start = lean.index('def ' + r['lean_definition'] + ' ')
                for before, after in [('.to(tl.float32)', ''), ('if not IS_VARLEN:', 'if IS_VARLEN:'),
                                      ('CU_SEQLENS + pid_batch + 1', 'CU_SEQLENS + pid_batch'),
                                      ('if pid_m * BLOCK_M >= seqlen:', 'if pid_m * BLOCK_M > seqlen:'),
                                      ('other=1.0', 'other=0.0'), ('if CONJUGATE:', 'if not CONJUGATE:'),
                                      ('rk_half[None, :] < rotary_dim_half', 'rk_half[None, :] <= rotary_dim_half')]:
                    changed = lean[:start] + lean[start:].replace(before, after, 1)
                    self.assertNotEqual(changed, lean)
                    with self.subTest(task=name, definition=r['lean_definition'], mutation=before), self.assertRaises(ValueError):
                        check(python, changed, r['python_function'], r['lean_definition'],
                              r['signature_definition'], r['expected_signature'],
                              expected_call_bindings=r['expected_call_bindings'])

    def test_cse_candidate_only_reuses_existing_index(self):
        directory = BASE / 'rotary_transform'
        original = ast.parse(variant_source(directory / 'rotary_transform.py'))
        candidate = ast.parse(variant_source(directory / 'rotary_transform_index_cse.py'))
        for node in ast.walk(candidate):
            if isinstance(node, ast.FunctionDef) and node.name == 'rotary_kernel_index_cse':
                node.name = 'rotary_kernel'
            if isinstance(node, ast.Name) and node.id == 'rotary_kernel_index_cse':
                node.id = 'rotary_kernel'
        fn = next(n for n in candidate.body if isinstance(n, ast.FunctionDef) and n.name == 'rotary_kernel')
        assignments = [n for n in ast.walk(fn) if isinstance(n, ast.Assign) and
                       any(isinstance(t, ast.Name) and t.id == 'rk_repeat' for t in n.targets)]
        self.assertEqual(len(assignments), 1)
        self.assertEqual(ast.unparse(assignments[0].value), 'rk // 2')
        assignments[0].value = ast.parse('tl.arange(0, BLOCK_K) // 2', mode='eval').body
        self.assertEqual(ast.dump(original), ast.dump(candidate))
        provenance = variant_provenance(directory)
        for field in ['original', 'candidate']:
            self.assertEqual(hashlib.sha256(variant_source(directory / provenance[field]).encode()).hexdigest(),
                             provenance[field + '_sha256'])

    def test_existing_xor_candidate_changes_only_partner_index(self):
        module = ast.parse((BASE / 'rotary_transform_ops/rotary_transform_ops.py').read_text())
        original, candidate = [next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == name)
                               for name in ['rotary_kernel', 'rotary_kernel_xor']]
        candidate.name = original.name
        assignments = [n for n in ast.walk(candidate) if isinstance(n, ast.Assign) and
                       any(isinstance(t, ast.Name) and t.id == 'rk_swap' for t in n.targets)]
        self.assertEqual(len(assignments), 1)
        self.assertEqual(ast.unparse(assignments[0].value), 'rk ^ 1')
        assignments[0].value = ast.parse('rk + ((rk + 1) % 2) * 2 - 1', mode='eval').body
        self.assertEqual(ast.dump(original), ast.dump(candidate))
