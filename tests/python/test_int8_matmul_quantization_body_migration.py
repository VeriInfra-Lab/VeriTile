import ast
import hashlib
import json
from pathlib import Path
import unittest

from scripts.bench.jit_correspondence import check, check_manifest

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / 'bench/tritonbench_g/int8_matmul_quantization'


class Int8MatmulQuantizationBodyMigrationTests(unittest.TestCase):
    def setUp(self):
        self.lean = (DIRECTORY / 'Sources/Body.lean').read_text()
        self.records = json.loads((DIRECTORY / 'jit-correspondence.json').read_text())

    def verify(self, text, index=0):
        r = self.records[index]
        return check((ROOT / r['python']).read_text(), text, r['python_function'],
                     r['lean_definition'], r['signature_definition'], r['expected_signature'],
                     expected_call_bindings=r['expected_call_bindings'])

    def test_ten_complete_source_pairs(self):
        result = check_manifest(ROOT, DIRECTORY / 'jit-correspondence.json')
        self.assertEqual([r['statements'] for r in result], [19, 19] + [39] * 8)

    def test_cast_loop_mask_and_store_mutations_are_rejected(self):
        for before, after, index in [
            ('M = tl.cast(M, tl.int64)', 'M = M', 0),
            ('pid_m = tl.program_id(axis=0).to(tl.int64)', 'pid_m = tl.program_id(axis=0)', 0),
            ('range(0, tl.cdiv(K, BLOCK_SIZE_K))', 'range(0, 1)', 0),
            ('inta = (fpa / a_scale[:, None]).to(tl.int8)', 'inta = fpa.to(tl.int8)', 0),
            ('tl.store(as_ptr + as_offs, a_scale)', 'tl.store(as_ptr + as_offs, a_scale.to(tl.float16))', 0),
            ('accumulator.to(tl.float32)', 'accumulator', 2),
            ('range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K))', 'range(0, 1)', 2),
            ('K - k * BLOCK_SIZE_K * SPLIT_K', 'K - k * BLOCK_SIZE_K', 2),
            ('tl.atomic_add(c_ptrs, c, mask=c_mask)', 'tl.store(c_ptrs, c, mask=c_mask)', 2),
        ]:
            with self.subTest(before=before):
                self.assertIn(before, self.lean)
                with self.assertRaises(ValueError):
                    self.verify(self.lean.replace(before, after), index)

    def test_storage_and_call_bindings_are_checked(self):
        for before, after, index in [
            ('("a_ptr",.i8)', '("a_ptr",.i32)', 0),
            ('("as_ptr",.fp16)', '("as_ptr",.fp32)', 2),
            ('("c_ptr",.fp32)', '("c_ptr",.fp16)', 4),
            ('K := K', 'K := 1', 0),
            ('stride_bk := stride_bk', 'stride_bk := 0', 2),
        ]:
            with self.subTest(before=before):
                with self.assertRaises(ValueError):
                    self.verify(self.lean.replace(before, after), index)

    def test_candidate_changes_only_the_three_documented_index_operations(self):
        p = json.loads((DIRECTORY / 'candidate-provenance.json').read_text())
        original = (DIRECTORY / p['original']).read_text()
        candidate = (DIRECTORY / p['candidate']).read_text()
        self.assertEqual(hashlib.sha256(original.encode()).hexdigest(), p['original_sha256'])
        self.assertEqual(hashlib.sha256(candidate.encode()).hexdigest(), p['candidate_sha256'])
        module = ast.parse(candidate)
        names = {'quantize_int8_perrow_kernel_unwrapped': 'quantize_int8_perrow_kernel',
                 'matmul_kernel_unmasked': 'matmul_kernel'}
        for n in ast.walk(module):
            if isinstance(n, ast.Name):
                n.id = names.get(n.id, n.id)
            elif isinstance(n, ast.FunctionDef):
                n.name = names.get(n.name, n.name)
        producer = next(n for n in module.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'quantize_int8_perrow_kernel')
        offsets = next(n for n in producer.body if isinstance(n, ast.Assign)
                       and ast.unparse(n.targets[0]) == 'offs_am')
        self.assertNotIsInstance(offsets.value.op, ast.Mod)
        offsets.value = ast.BinOp(offsets.value, ast.Mod(), ast.Name('M', ast.Load()))
        gemm = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'matmul_kernel')
        for name, pointer, index, limit in [('a_scale', 'as_ptrs', 'offs_am', 'M'),
                                            ('b_scale', 'bs_ptrs', 'offs_bn', 'N')]:
            assignment = next(n for n in gemm.body if isinstance(n, ast.Assign)
                              and ast.unparse(n.targets[0]) == name)
            self.assertEqual(ast.unparse(assignment.value), f'tl.load({pointer})')
            assignment.value = ast.parse(f'tl.load({pointer}, mask={index} < {limit}, other=0.0)',
                                         mode='eval').body
        self.assertEqual(ast.dump(ast.parse(original), include_attributes=False),
                         ast.dump(module, include_attributes=False))
