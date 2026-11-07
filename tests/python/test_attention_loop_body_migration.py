import ast
import copy
import hashlib
from pathlib import Path
import unittest
from scripts.bench.jit_correspondence import check, check_statement_function

from tests.python.layout_fixtures import check_pinned_pairs, pinned_pairs, variant_provenance, variant_source

ROOT = Path(__file__).resolve().parents[2]
CASES = [('attention_forward_triton', 'head_mask', 'fp16', [3, 27, 3, 28]),
         ('attention_fwd_triton2', 'index_shift', 'bf16', [3, 27, 3, 27])]

class AttentionLoopBodyMigrationTests(unittest.TestCase):
    def test_all_complete_helpers_callers_and_distinct_output_types(self):
        for case, _, dtype, counts in CASES:
            directory = ROOT / 'bench/tritonbench_g' / case
            rows = check_pinned_pairs(ROOT, directory)
            self.assertEqual([r['statements'] for r in rows], counts)
            self.assertTrue(all(r['mode'] == 'pasted_body' for r in rows))
            records = pinned_pairs(directory)
            self.assertEqual(records[1]['expected_signature']['pointers'][-1], ['Out', dtype])
            self.assertEqual(len(records[1]['expected_call_bindings']), 19)

    def test_helper_loop_return_masks_and_casts_are_required(self):
        for case, _, _, _ in CASES:
            directory = ROOT / 'bench/tritonbench_g' / case
            record = pinned_pairs(directory)[0]
            python = (ROOT / record['python']).read_text()
            lean = (ROOT / record['lean']).read_text()
            for before, after in [('range(lo, hi, BLOCK_N)', 'range(lo, hi, 1)'),
                                  ('return acc, l_i', 'return acc'),
                                  ('out_dtype=tl.float16', 'out_dtype=tl.float32'),
                                  ('tl.arange(0, 128) < 96', 'tl.arange(0, 128) < 128'),
                                  ('p.to(tl.float16)', 'p')]:
                changed = lean.replace(before, after, 1)
                self.assertNotEqual(changed, lean)
                with self.subTest(case=case, before=before), self.assertRaises(ValueError):
                    check_statement_function(python, changed, record['python_function'], record['lean_definition'])

    def test_caller_casts_and_unused_helper_arguments_are_required(self):
        for case, _, _, _ in CASES:
            directory = ROOT / 'bench/tritonbench_g' / case
            record = pinned_pairs(directory)[1]
            python = (ROOT / record['python']).read_text()
            lean = (ROOT / record['lean']).read_text()
            start = lean.index('def originalKernel ')
            for before, after in [('off_z.to(tl.int64)', 'off_z'),
                                  ('acc.to(Out.type.element_ty)', 'acc'),
                                  ('4 - STAGE, offs_m, offs_n', '0, offs_m, offs_n')]:
                changed = lean[:start] + lean[start:].replace(before, after, 1)
                self.assertNotEqual(changed, lean)
                with self.subTest(case=case, before=before), self.assertRaises(ValueError):
                    check(python, changed, record['python_function'], record['lean_definition'],
                          record['signature_definition'], record['expected_signature'],
                          expected_call_bindings=record['expected_call_bindings'],
                          expected_jit_functions=record['expected_jit_functions'])

    def test_candidates_only_contain_the_existing_reviewed_transformations(self):
        for case, suffix, _, _ in CASES:
            directory = ROOT / 'bench/tritonbench_g' / case
            original_path = directory / (case + '.py')
            candidate_path = directory / (case + '_' + suffix + '.py')
            original = ast.parse(variant_source(original_path))
            candidate = ast.parse(variant_source(candidate_path))
            for node in ast.walk(candidate):
                if isinstance(node, ast.FunctionDef) and node.name.endswith('_' + suffix):
                    node.name = node.name.removesuffix('_' + suffix)
                if isinstance(node, ast.Name) and node.id.endswith('_' + suffix):
                    node.id = node.id.removesuffix('_' + suffix)
            fns = {n.name: n for n in candidate.body if isinstance(n, ast.FunctionDef)}
            caller = fns['_attn_fwd']
            if suffix == 'head_mask':
                helper = fns['_attn_fwd_inner']
                self.assertEqual(helper.args.args[-1].arg, 'head_mask')
                helper.args.args.pop()
                call = next(n for n in ast.walk(caller) if isinstance(n, ast.Call)
                            and isinstance(n.func, ast.Name) and n.func.id == '_attn_fwd_inner')
                self.assertEqual(ast.unparse(call.args[-1]), 'head_mask')
                call.args.pop()
                index, binding = next((i,n) for i,n in enumerate(caller.body) if isinstance(n, ast.Assign)
                                      and ast.unparse(n.targets[0]) == 'head_mask')
                self.assertEqual(ast.unparse(binding.value), 'offs_k < 96')
                caller.body.pop(index)
                class InlineMask(ast.NodeTransformer):
                    def visit_Name(self, node):
                        return ast.parse('tl.arange(0, 128) < 96', mode='eval').body if node.id == 'head_mask' else node
                candidate = InlineMask().visit(candidate)
            else:
                assignment = next(n for n in caller.body if isinstance(n, ast.Assign)
                                  and ast.unparse(n.targets[0]) == 'offs_m')
                self.assertEqual(ast.unparse(assignment.value.left), 'start_m << 7')
                assignment.value.left = ast.parse('start_m * BLOCK_M', mode='eval').body
            self.assertEqual(ast.dump(original), ast.dump(candidate))
            provenance = variant_provenance(directory)
            for label, path in [('original', original_path), ('candidate', candidate_path)]:
                self.assertEqual(hashlib.sha256(variant_source(path).encode()).hexdigest(), provenance[label + '_sha256'])
