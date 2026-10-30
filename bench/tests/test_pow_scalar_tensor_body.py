"""Full scalar-power body and external binary import provenance checks."""
import ast
import copy
import json
from pathlib import Path
import unittest
from bench.jit_correspondence import check, check_binary_imports

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'bench/repairs/tritonbench_g/pow_scalar_tensor'

class PowScalarTensorBodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = (BASE / 'pow_scalar_tensor_repaired.py').read_text()
        cls.candidate = (BASE / 'candidate_kernels.py').read_text()
        cls.lean = (BASE / 'Kernels.lean').read_text()
        cls.rows = json.loads((BASE / 'jit-correspondence.json').read_text())
        cls.binding = {'_pow': 'triton.language.extra.cuda.libdevice.pow'}

    def run_row(self, row, python=None, lean=None):
        return check(python if python is not None else (ROOT / row['python']).read_text(),
                     lean if lean is not None else self.lean,
                     **{k: v for k, v in row.items() if k not in ('python', 'lean')})

    def test_complete_correspondence(self):
        self.assertEqual(len(self.rows), 3)
        for row in self.rows:
            self.assertEqual(self.run_row(row)['status'], 'PASS')

    def test_candidate_only_commutes_both_integer_products(self):
        a = next(n for n in ast.parse(self.original).body if isinstance(n, ast.FunctionDef)
                 and n.name == 'pow_func_scalar_tensor_kernel_rank_1')
        b = next(n for n in ast.parse(self.candidate).body if isinstance(n, ast.FunctionDef)
                 and n.name == 'pow_func_scalar_tensor_kernel_rank_1_index_commuted')
        b = copy.deepcopy(b); b.name = a.name
        count = 0
        for n in ast.walk(b):
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'offset0' for t in n.targets):
                self.assertEqual(ast.unparse(n.value), 'tile_size0 * tile_id0')
                n.value.left, n.value.right = n.value.right, n.value.left
                count += 1
        self.assertEqual(count, 2)
        self.assertEqual(ast.dump(a), ast.dump(b))

    def test_input_cast_cannot_disappear(self):
        lean = self.lean.replace('.to(in0_ptr.type.element_ty)', '')
        with self.assertRaises(ValueError): self.run_row(self.rows[1], lean=lean)

    def test_output_cast_cannot_disappear(self):
        lean = self.lean.replace('out0.to(out0_bptr.type.element_ty)', 'out0')
        with self.assertRaises(ValueError): self.run_row(self.rows[1], lean=lean)

    def test_helper_cast_cannot_disappear(self):
        lean = self.lean.replace('_pow(x.to(tl.float32), exponent)', '_pow(x, exponent)')
        with self.assertRaises(ValueError): self.run_row(self.rows[0], lean=lean)

    def test_selected_import_and_fallbacks_are_recorded(self):
        choices = check_binary_imports(self.original, self.binding)
        self.assertEqual(choices['_pow'], [self.binding['_pow'], 'triton.language.math.pow', 'triton.language.libdevice.pow'])

    def test_wrong_primary_is_rejected(self):
        text = self.original.replace('triton.language.extra.cuda.libdevice', 'triton.language.math', 1)
        with self.assertRaises(ValueError): check_binary_imports(text, self.binding)

    def test_computed_fallback_is_rejected(self):
        text = self.original.replace('from triton.language.math import pow as _pow', '_pow = make_pow()')
        with self.assertRaises(ValueError): check_binary_imports(text, self.binding)

    def test_import_rebinding_is_rejected(self):
        with self.assertRaises(ValueError): check_binary_imports(self.original + '\n_pow = lambda x, y: x\n', self.binding)

    def test_import_shadowing_is_rejected(self):
        text = self.original + '\ndef shadow(_pow):\n    return _pow\n'
        with self.assertRaises(ValueError): check_binary_imports(text, self.binding)

    def test_non_importerror_fallback_is_rejected(self):
        text = self.original.replace('except ImportError:', 'except Exception:', 1)
        with self.assertRaises(ValueError): check_binary_imports(text, self.binding)

    def test_implicit_import_and_pattern_bindings_are_rejected(self):
        for suffix in ['\nfrom unknown import *\n',
                       '\ntry:\n    value()\nexcept ValueError as _pow:\n    pass\n',
                       '\nmatch value:\n    case _pow:\n        pass\n']:
            with self.subTest(source=suffix), self.assertRaises(ValueError):
                check_binary_imports(self.original + suffix, self.binding)

if __name__ == '__main__':
    unittest.main()
