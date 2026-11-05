import ast
import copy
from pathlib import Path
import unittest

from scripts.bench.jit_correspondence import check

from tests.python.layout_fixtures import check_pinned_pairs, pinned_pairs

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / 'bench/tritonbench_g/mixed_sparse_attention'


class MixedSparseBodyMigration(unittest.TestCase):
    def setUp(self):
        self.rows = pinned_pairs(CASE)
        self.lean = (CASE / 'Kernels.lean').read_text()

    def test_both_complete_functions_and_storage_specializations(self):
        rows = check_pinned_pairs(ROOT, CASE)
        self.assertEqual([r['statements'] for r in rows], [30, 30, 30, 30])
        self.assertTrue(all(r['status'] == 'PASS' and r['mode'] == 'pasted_body' for r in rows))

    def checked(self, row, lean):
        return check((ROOT / row['python']).read_text(), lean, row['python_function'],
                     row['lean_definition'], row['signature_definition'], row['expected_signature'],
                     row['expected_call_bindings'], expected_constexpr_bindings=row['expected_constexpr_bindings'],
                     expected_natural_reads=row.get('expected_natural_reads'))

    def mutation(self, before, after, index=0):
        self.assertIn(before, self.lean)
        with self.assertRaises(ValueError):
            self.checked(self.rows[index], self.lean.replace(before, after))

    def test_early_return_dtype_masks_counts_and_both_loops_are_preserved(self):
        for before, after in [
            ('if start_m * BLOCK_M >= seqlen:', 'if start_m * BLOCK_M > seqlen:'),
            ('q = (q * qk_scale).to(dtype)', 'q = q * qk_scale'),
            ('for block_index in range(max_num_blks):', 'for block_index in range(BLOCK_N):'),
            ('for start_n in range(0, max_num_cols, BLOCK_N):', 'for start_n in range(max_num_cols):'),
            ('tl.store(o_ptrs, acc.to(dtype), mask=m_mask)', 'tl.store(o_ptrs, acc, mask=m_mask)'),
        ]:
            with self.subTest(before=before):
                self.mutation(before, after)

    def test_projection_and_storage_metadata_are_checked(self):
        self.mutation('("seqlens",.i32)', '("seqlens",.u32)')
        self.mutation('with_triton_natural_reads [seqlens,block_count,block_offset,column_count,column_index]',
                      'with_triton_natural_reads [seqlens,block_count,column_count,column_index]')
        self.mutation('let dtype : FloatDType := .fp16', 'let dtype : FloatDType := .bf16')
        row = copy.deepcopy(self.rows[0])
        del row['expected_natural_reads']
        with self.assertRaisesRegex(ValueError, 'require reviewed caller metadata'):
            self.checked(row, self.lean)

    def test_existing_candidate_only_fuses_the_column_zero_mask_assignment(self):
        tree = ast.parse((CASE / 'mixed_sparse_attention.py').read_text())
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        source = functions['_triton_mixed_sparse_attn_fwd_kernel']
        fused = copy.deepcopy(functions['_triton_mixed_sparse_attn_fused_mask_kernel'])
        fused.name = source.name
        # The candidate replaces exactly this one zero-plus-where pair.
        loop = next(node for node in fused.body if isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name) and node.target.id == 'start_n')
        assignment = next(node for node in loop.body if isinstance(node, ast.Assign)
                          and ast.unparse(node.targets[0]) == 'qk')
        call = assignment.value
        self.assertEqual(ast.unparse(call.func), 'tl.where')
        self.assertEqual(ast.unparse(call.args[1]), 'tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)')
        zero = ast.Assign(targets=copy.deepcopy(assignment.targets), value=call.args[1])
        call.args[1] = ast.Name(id='qk', ctx=ast.Load())
        pos = loop.body.index(assignment)
        loop.body.insert(pos, zero)
        self.assertEqual(ast.dump(source), ast.dump(fused))
