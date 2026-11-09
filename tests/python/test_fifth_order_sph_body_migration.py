import ast
import copy
import hashlib
from pathlib import Path
import unittest
from scripts.bench.jit_correspondence import check

from tests.python.layout_fixtures import check_pinned_pairs, pinned_pairs, variant_provenance, variant_source

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / 'bench/tritonbench_g/fifth_order_sph_harmonics'

class FifthOrderSphBodyMigrationTests(unittest.TestCase):
    def test_both_complete_directions_and_candidates(self):
        rows = check_pinned_pairs(ROOT, DIR)
        self.assertEqual([r['statements'] for r in rows], [73, 122, 73, 122])
        self.assertTrue(all(r['mode'] == 'pasted_body' for r in rows))
        self.assertEqual(rows[0]['runtime_parameters'], {})
        self.assertEqual(rows[1]['runtime_parameters'],
                         {'upstream_row_stride': 'integer', 'upstream_col_stride': 'integer'})

    def test_casts_constants_masks_strides_and_accumulation_cannot_disappear(self):
        records = pinned_pairs(DIR)
        for record, replacements in [
            (records[0], [('.to(tl.int64)', ''), ('CONST006 = 6.21867148191637', ''),
                          ('-CONST021 * VAR17', 'CONST021 * VAR17'),
                          ('mask=coord_row_offset < coord_numel', 'mask=True')]),
            (records[1], [('tl.cast(upstream_row_stride, tl.int64)', 'upstream_row_stride'),
                          ('tl.cast(upstream_col_stride, tl.int64)', 'upstream_col_stride'),
                          ('EXACT000 = 11.63406904311645', 'EXACT000 = 11.6340690431164'),
                          ('g_x +=', 'g_x =')])]:
            python = (ROOT / record['python']).read_text()
            lean = (ROOT / record['lean']).read_text()
            start = lean.index('def ' + record['lean_definition'] + ' (')
            for before, after in replacements:
                changed = lean[:start] + lean[start:].replace(before, after, 1)
                self.assertNotEqual(changed, lean, before)
                with self.subTest(function=record['python_function'], before=before), self.assertRaises(ValueError):
                    check(python, changed, record['python_function'], record['lean_definition'],
                          record['signature_definition'], record['expected_signature'],
                          expected_call_bindings=record['expected_call_bindings'])

    def test_candidate_only_reuses_identical_left_associated_powers(self):
        original_path = DIR / 'fifth_order_sph_harmonics.py'
        candidate_path = DIR / 'fifth_order_sph_harmonics_powers_reuse.py'
        original = ast.parse(variant_source(original_path))
        candidate = ast.parse(variant_source(candidate_path))
        for node in ast.walk(candidate):
            if isinstance(node, ast.FunctionDef) and node.name.endswith('_powers_reuse'):
                node.name = node.name.removesuffix('_powers_reuse')
            if isinstance(node, ast.Name) and node.id.endswith('_powers_reuse'):
                node.id = node.id.removesuffix('_powers_reuse')
        old_functions = {f.name: f for f in original.body if isinstance(f, ast.FunctionDef)}
        for fn in candidate.body:
            if not isinstance(fn, ast.FunctionDef) or fn.name not in {
                    'fifth_order_fwd_repaired', 'fifth_order_bwd_repaired'}:
                continue
            old = old_functions[fn.name]
            def powers(body):
                return [(i, n.targets[0].id, n) for i, n in enumerate(body)
                        if isinstance(n, ast.Assign) and len(n.targets) == 1
                        and isinstance(n.targets[0], ast.Name) and n.targets[0].id.startswith('VAR')]
            previous, current = powers(old.body), powers(fn.body)
            self.assertEqual([i for i, _, _ in previous], [i for i, _, _ in current])
            self.assertEqual(set(n for _, n, _ in previous), set(n for _, n, _ in current))
            known = {}
            class Expand(ast.NodeTransformer):
                def visit_Name(self, node):
                    return copy.deepcopy(known.get(node.id, node))
            for _, name, stmt in current:
                known[name] = Expand().visit(copy.deepcopy(stmt.value))
            for _, name, stmt in previous:
                self.assertEqual(ast.dump(stmt.value), ast.dump(known[name]), name)
            start = previous[0][0]
            self.assertEqual([i for i, _, _ in previous], list(range(start, start + len(previous))))
            fn.body[start:start+len(previous)] = copy.deepcopy(old.body[start:start+len(previous)])
        self.assertEqual(ast.dump(original), ast.dump(candidate))
        provenance = variant_provenance(DIR)
        for label, path in [('original', original_path), ('candidate', candidate_path)]:
            self.assertEqual(hashlib.sha256(variant_source(path).encode()).hexdigest(), provenance[label + '_sha256'])
