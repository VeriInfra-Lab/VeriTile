"""Reject drift in the selected FP4 helper, caller, casts and call metadata."""
import json
from pathlib import Path
import unittest
from bench.jit_correspondence import check, check_manifest

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'bench/repairs/tritonbench_g/fp4_to_bf16'


class PackedFp4Body(unittest.TestCase):
    def setUp(self):
        self.rows = json.loads((BASE / 'jit-correspondence.json').read_text())

    def rejects(self, index, before, after):
        row = self.rows[index]
        source = (ROOT / row['lean']).read_text()
        self.assertIn(before, source)
        options = {k: v for k, v in row.items()
                   if k not in ['python', 'lean', 'python_function', 'lean_definition']}
        with self.assertRaises(ValueError):
            check((ROOT / row['python']).read_text(), source.replace(before, after, 1),
                  row['python_function'], row['lean_definition'], **options)

    def test_complete_helper_and_caller(self):
        rows = check_manifest(ROOT, BASE / 'jit-correspondence.json')
        self.assertEqual([r['statements'] for r in rows], [22, 13])
        self.assertEqual(rows[0]['argument_mode'], 'inferred_helper_values')

    def test_nibble_order(self):
        self.rejects(0, 'tl.interleave(x_low_bits, x_high_bits)',
                     'tl.interleave(x_high_bits, x_low_bits)')

    def test_exponent_cast(self):
        self.rejects(0, 'exp_biased_f32.to(tl.int32)', 'exp_biased_f32')

    def test_exponent_operation_order(self):
        self.rejects(0, 'exp_biased_f4 - f4_e2m1_exp_bias + f32_exp_bias',
                     'exp_biased_f4 + f32_exp_bias - f4_e2m1_exp_bias')

    def test_signed_zero(self):
        self.rejects(0, 'result = result | sign_f32', 'result = result')

    def test_bitcast(self):
        self.rejects(0, 'result.to(tl.float32, bitcast=True)', 'result.to(tl.float32)')

    def test_bf16_conversion(self):
        self.rejects(0, 'output = output.to(tl.bfloat16)', 'output = output')

    def test_runtime_helper_parameters(self):
        self.rejects(0, 'triton_function _fp4_packed_to_bf16(x_packed, sign_mask_f4,',
                     'triton_function _fp4_packed_to_bf16(x_packed, sign_mask_f4: tl.constexpr,')

    def test_program_id_cast(self):
        self.rejects(1, 'tl.program_id(axis=0).to(tl.int64)', 'tl.program_id(axis=0)')

    def test_count_cast(self):
        self.rejects(1, 'n_elements_in = tl.cast(n_elements_in, tl.int64)',
                     'n_elements_in = n_elements_in')

    def test_lane_cast(self):
        self.rejects(1, 'tl.arange(0, BLOCK_SIZE_IN).to(tl.int64)',
                     'tl.arange(0, BLOCK_SIZE_IN)')

    def test_input_mask(self):
        self.rejects(1, 'tl.load(x_ptr + offsets_in, mask=mask_in)',
                     'tl.load(x_ptr + offsets_in)')

    def test_output_mask(self):
        self.rejects(1, 'tl.store(output_ptr + offsets_out, output, mask=mask_out)',
                     'tl.store(output_ptr + offsets_out, output)')

    def test_input_dtype(self):
        self.rejects(1, '("x_ptr",.u8)', '("x_ptr",.i8)')

    def test_call_binding(self):
        self.rejects(1, 'call_args [n_elements_in := n]',
                     'call_args [n_elements_in := BLOCK_SIZE_IN]')

    def test_format_constant(self):
        self.rejects(1, 'let f32_exp_bias : Nat := 127', 'let f32_exp_bias : Nat := 126')


if __name__ == '__main__':
    unittest.main()
