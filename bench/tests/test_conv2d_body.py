"""Protect literal Conv2d bodies, masks, casts, modes and caller bindings."""
import copy
import json
from pathlib import Path
import runpy
import unittest
from bench.jit_correspondence import check, validate_manifest_records
from bench.repairs.tritonbench_g.triton_conv2d_fwd.check_body_source import check_aliases, check_sources

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'bench/repairs/tritonbench_g/triton_conv2d_fwd'

class Conv2dBodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = json.loads((BASE / 'jit-correspondence.json').read_text())

    def reject(self, before, after, row=0, aliases=False):
        record = self.rows[row]
        source = (ROOT / record['lean']).read_text()
        self.assertTrue(before in source, before)
        changed = source.replace(before, after, 1)
        with self.assertRaises(ValueError):
            if aliases:
                check_aliases(changed, record['lean_definition'],
                              record['expected_specialized_booleans']['fp16'], row >= 2)
            else:
                check((ROOT / record['python']).read_text(), changed,
                      **{k:v for k,v in record.items() if k not in ('python','lean','profile_variant')})

    def test_all_four_bodies_and_aliases(self):
        result = check_sources(ROOT)
        self.assertEqual([x['statements'] for x in result['bodies']], [19,19,20,20])
        self.assertTrue(all(x['mode']=='pasted_body' and x['status']=='PASS' for x in result['bodies']))

    def test_original_half_cast(self):
        self.reject('input_block = input_block.to(tl.float16)', 'input_block = input_block')

    def test_original_weight_cast(self):
        self.reject('weight_block = weight_block.to(tl.float16)', 'weight_block = weight_block')

    def test_runtime_tf32_option(self):
        self.reject('allow_tf32=tf32', 'allow_tf32=False')

    def test_channel_stride(self):
        self.reject('range(0, in_group_dim, BLOCK_SIZE_IN_FEAT)', 'range(0, in_group_dim, 1)')

    def test_height_trip_count(self):
        self.reject('range(kernel_height)', 'range(kernel_width)')

    def test_width_trip_count(self):
        self.reject('range(kernel_width)', 'range(kernel_height)')

    def test_signed_padding(self):
        self.reject('h - padding_height', 'h + padding_height')

    def test_input_mask(self):
        self.reject('mask=input_mask', 'mask=weight_mask')

    def test_input_masked_zero(self):
        self.reject('mask=input_mask, other=0.0', 'mask=input_mask, other=1.0')

    def test_candidate_shift(self):
        self.reject('batch_height_width_pid << 7', 'batch_height_width_pid << 6', 2)

    def test_candidate_assertion(self):
        self.reject('tl.static_assert(BLOCK_SIZE_BATCH_HEIGHT_WIDTH == 128)', 'tl.static_assert(BLOCK_SIZE_BATCH_HEIGHT_WIDTH == 64)', 2)

    def test_storage_profile(self):
        self.reject('("input_pointer",.fp32)', '("input_pointer",.fp16)')

    def test_runtime_binding(self):
        self.reject('batch_dim := p.batch_dim', 'batch_dim := p.in_feat_dim')

    def test_boolean_specialization(self):
        self.reject('let fp16 : Bool := Bool.true', 'let fp16 : Bool := Bool.false')

    def test_candidate_tile_binding(self):
        self.reject('let BLOCK_SIZE_BATCH_HEIGHT_WIDTH : Nat := 128', 'let BLOCK_SIZE_BATCH_HEIGHT_WIDTH : Nat := 64', 2)

    def test_padding_alias(self):
        self.reject('let padding_height : Int := (p.padding_height : Int)', 'let padding_height : Int := (p.padding_width : Int)', aliases=True)

    def test_runtime_tf32_alias(self):
        self.reject('let tf32 : Bool := p.tf32', 'let tf32 : Bool := Bool.false', aliases=True)

    def test_pointer_alias(self):
        self.reject('let input_pointer : RegionName := p.Input', 'let input_pointer : RegionName := p.Weight', aliases=True)

    def test_constexpr_tile_alias(self):
        self.reject('let BLOCK_SIZE_IN_FEAT : Nat := p.BLOCK_SIZE_IN_FEAT', 'let BLOCK_SIZE_IN_FEAT : Nat := p.BLOCK_SIZE_OUT_FEAT', aliases=True)

    def test_duplicate_boolean_profile_rejected(self):
        rows = copy.deepcopy(self.rows[:2])
        rows[1]['expected_specialized_booleans'] = {'fp16': True}
        with self.assertRaises(ValueError): validate_manifest_records(rows)

    def test_full_existing_candidate_source_delta(self):
        runpy.run_path(str(BASE/'check_pair_source.py'))

if __name__ == '__main__': unittest.main()
