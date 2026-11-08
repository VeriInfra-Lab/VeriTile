import ast
import copy
import hashlib
from pathlib import Path
import unittest
from scripts.bench.jit_correspondence import check

from tests.python.layout_fixtures import check_pinned_pairs, pinned_pairs, variant_provenance, variant_source

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / 'bench/tritonbench_g/int_scaled_matmul'


class IntScaledMatmulBodyMigrationTests(unittest.TestCase):
    def setUp(self):
        self.lean = (DIRECTORY / 'Kernels.lean').read_text()
        self.records = pinned_pairs(DIRECTORY)

    def verify(self, lean, index):
        r = self.records[index]
        return check((ROOT / r['python']).read_text(), lean, r['python_function'],
                     r['lean_definition'], r['signature_definition'], r['expected_signature'],
                     expected_call_bindings=r['expected_call_bindings'],
                     expected_constexpr_bindings=r.get('expected_constexpr_bindings'))

    def test_all_four_complete_function_bodies(self):
        rows = check_pinned_pairs(ROOT, DIRECTORY)
        self.assertEqual([r['statements'] for r in rows], [17, 16, 25, 23])

    def test_source_mutation_masks_assembly_and_loop_are_rejected(self):
        for before, after, index in [
            ('GROUP_M = min(num_pid_m - first_pid_m, GROUP_M)', 'GROUP_M = GROUP_M', 0),
            ('padding_option="zero"', 'padding_option="nan"', 0),
            ('add.u32 $0, $1, $2;', 'add.s32 $0, $1, $2;', 0),
            ('constraints="=r,r,r"', 'constraints="=r,r,l"', 0),
            ('is_pure=True', 'is_pure=False', 0),
            ('for k in range(K, 0, -BLOCK_K):', 'for k in range(0, K, BLOCK_K):', 2),
            ('mask=rk[None, :] < k', 'mask=rk[None, :] <= k', 2),
            ('mask.shape', '[BLOCK_M, BLOCK_N]', 2),
            ('acc * tmp0, mask)', '(acc * tmp0).to(tl.int32), mask)', 2),
        ]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.verify(self.lean.replace(before, after), index)

    def test_call_storage_and_dtype_specialization_are_checked(self):
        for before, after, index in [('("a_ptr",.i8)', '("a_ptr",.i32)', 0),
                                     ('ACC_TYPE : SourceDType := .i32', 'ACC_TYPE : SourceDType := .i64', 2)]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.verify(self.lean.replace(before, after), index)

    def test_candidate_only_removes_the_two_existing_redundancies(self):
        original_text = variant_source(DIRECTORY / 'int_scaled_matmul.py')
        candidate_text = variant_source(DIRECTORY / 'int_scaled_matmul_candidates.py')
        provenance = variant_provenance(DIRECTORY)
        self.assertEqual(hashlib.sha256(original_text.encode()).hexdigest(), provenance['original_sha256'])
        self.assertEqual(hashlib.sha256(candidate_text.encode()).hexdigest(), provenance['candidate_sha256'])
        original = ast.parse(original_text)
        candidate = ast.parse(candidate_text)
        names = {'matmul_kernel_with_block_pointers_copy_free': 'matmul_kernel_with_block_pointers',
                 'scaled_matmul_kernel_with_block_pointers_shared_indices': 'scaled_matmul_kernel_with_block_pointers'}

        class RestoreNames(ast.NodeTransformer):
            def visit_Name(self, node):
                node.id = names.get(node.id, node.id)
                return node

            def visit_FunctionDef(self, node):
                node.name = names.get(node.name, node.name)
                return self.generic_visit(node)

        restored = RestoreNames().visit(copy.deepcopy(candidate))
        originals = {n.name: n for n in original.body if isinstance(n, ast.FunctionDef)}
        functions = {n.name: n for n in restored.body if isinstance(n, ast.FunctionDef)}
        unscaled = functions['matmul_kernel_with_block_pointers']
        old = originals[unscaled.name]
        index = next(i for i, n in enumerate(old.body)
                     if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'c')
        unscaled.body.insert(index, copy.deepcopy(old.body[index]))
        store = next(n.value for n in unscaled.body if isinstance(n, ast.Expr)
                     and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'tl.store')
        self.assertEqual(ast.unparse(store.args[1]), 'accumulator')
        store.args[1] = ast.Name(id='c', ctx=ast.Load())
        scaled = functions['scaled_matmul_kernel_with_block_pointers']
        old = originals[scaled.name]
        def idx_m(function):
            return next(i for i, n in enumerate(function.body)
                        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'idx_m')
        position, old_position = idx_m(scaled), idx_m(old)
        scaled.body[position:position] = copy.deepcopy(old.body[old_position-2:old_position])
        self.assertEqual(ast.dump(restored, include_attributes=False), ast.dump(original, include_attributes=False))
