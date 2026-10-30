"""Regressions for source literals whose Real value survives dtype erasure."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from scripts.bench.check_imported_kernel_review import exact_finite_masks

ROOT = Path(__file__).resolve().parents[2]
MASK = '-338953138925153547590470800371487866880'


class ImportedKernelLiteralTests(unittest.TestCase):
    def test_rounded_decimal_is_not_the_same_real_mask(self):
        program = f'x = tl.cast({MASK}.0, tl.bfloat16)'
        with self.assertRaisesRegex(AssertionError, 'finite mask differs'):
            exact_finite_masks(program, ['-338953138925153550000000000000000000000'])
        self.assertEqual(exact_finite_masks(program, [MASK, f'({MASK}/1)']), [MASK])

    def test_both_mathematical_entries_must_match(self):
        with self.assertRaisesRegex(AssertionError, 'finite mask differs'):
            exact_finite_masks(f'x = {MASK}.0', [MASK, '-1'])

    def test_ordinary_constants_are_outside_this_targeted_guard(self):
        self.assertEqual(exact_finite_masks('x = 2.0 * y + 0.000001', ['y + y']), [])

    def test_all_imported_original_masks_match_their_real_entries(self):
        from scripts.bench.check_imported_kernel_review import bodies
        reviewed = 0
        for collection in ('bug_detection', 'pretrain_kernels'):
            for task in sorted((ROOT / 'bench' / collection / 'tasks').iterdir()):
                with self.subTest(collection=collection, task=task.name):
                    original = bodies((task / 'Kernels.lean').read_text())['original']
                    masks = exact_finite_masks(original, [(task / 'Contract.lean').read_text()])
                    reviewed += bool(masks)
        self.assertGreaterEqual(reviewed, 108)

    def test_next_multiple_helper_satisfies_round_up_bounds(self):
        # Execute the actual pure JIT helper body, without importing Torch,
        # Triton, running the module's host tests, or launching a GPU kernel.
        source = ROOT / 'bench/tritonbench_g/softmax_flaggems/softmax_flaggems.py'
        function = next(n for n in ast.parse(source.read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == 'next_multiple_of')
        function.decorator_list = []
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        namespace = {'tl': SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b)}
        exec(compile(module, str(source), 'exec'), namespace)
        for a in (0, 1, 15, 16, 17, 127, 128, 129, 8193):
            for b in (1, 2, 16, 128):
                with self.subTest(a=a, b=b):
                    result = namespace['next_multiple_of'](a, b)
                    self.assertEqual(result % b, 0)
                    self.assertGreaterEqual(result, a)
                    self.assertLess(result - b, a)


if __name__ == '__main__':
    unittest.main()
