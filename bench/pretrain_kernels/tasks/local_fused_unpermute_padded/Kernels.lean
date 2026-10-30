import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_fused_unpermute_padded
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Kernels

def original : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  triton {
    r0_z = tl.arange($(524288))
    r0_value = tl.load($(input0_ptr) + ((((r0_z / $(2048)) % $(256)) * $(2048)) + (r0_z % $(2048))), dtype=tl.float64)
    r0_value = r0_value.to(tl.bfloat16)
    r0_value = tl.broadcast_to(r0_value, [$(524288)])
    tl.store($(vto_copy_1_ptr) + r0_z, r0_value)
    z = tl.arange($(524288))
    value = tl.load($(vto_copy_1_ptr) + ((((z / $(2048)) % $(256)) * $(2048)) + (z % $(2048))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(524288)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let parameter0_ptr : RegionName := "parameter0"
  let out_ptr : RegionName := "out"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let candidate_vtopk_value_ptr : RegionName := "candidate_vtopk_value"
  let candidate_vtopk_index_ptr : RegionName := "candidate_vtopk_index"
  let candidate_vscatter_ptr : RegionName := "candidate_vscatter"
  let candidate_vsoftmax_ptr : RegionName := "candidate_vsoftmax"
  let candidate_vscatter_1_ptr : RegionName := "candidate_vscatter_1"
  let candidate_vto_copy_ptr : RegionName := "candidate_vto_copy"
  let candidate_vto_copy_1_ptr : RegionName := "candidate_vto_copy_1"
  let candidate_vto_copy_2_ptr : RegionName := "candidate_vto_copy_2"
  let candidate_vto_copy_3_ptr : RegionName := "candidate_vto_copy_3"
  let candidate_vto_copy_4_ptr : RegionName := "candidate_vto_copy_4"
  let candidate_vsum_2_ptr : RegionName := "candidate_vsum_2"
  let candidate_vadd_ptr : RegionName := "candidate_vadd"
  let candidate_vsub_ptr : RegionName := "candidate_vsub"
  let candidate_vfloor_divide_ptr : RegionName := "candidate_vfloor_divide"
  let candidate_vmul_ptr : RegionName := "candidate_vmul"
  let candidate_vcumsum_ptr : RegionName := "candidate_vcumsum"
  let candidate_vsub_1_ptr : RegionName := "candidate_vsub_1"
  let candidate_vto_copy_5_ptr : RegionName := "candidate_vto_copy_5"
  let candidate_vcumsum_1_ptr : RegionName := "candidate_vcumsum_1"
  let candidate_vsub_2_ptr : RegionName := "candidate_vsub_2"
  let candidate_vadd_1_ptr : RegionName := "candidate_vadd_1"
  let candidate_vwhere_ptr : RegionName := "candidate_vwhere"
  let candidate_vwhere_1_ptr : RegionName := "candidate_vwhere_1"
  let candidate_vwhere_2_ptr : RegionName := "candidate_vwhere_2"
  let candidate_vscatter_add_ptr : RegionName := "candidate_vscatter_add"
  let candidate_vge_ptr : RegionName := "candidate_vge"
  let candidate_vclamp_min_ptr : RegionName := "candidate_vclamp_min"
  let candidate_vindex_ptr : RegionName := "candidate_vindex"
  let candidate_vto_copy_6_ptr : RegionName := "candidate_vto_copy_6"
  let candidate_vwhere_4_ptr : RegionName := "candidate_vwhere_4"
  let candidate_vto_copy_7_ptr : RegionName := "candidate_vto_copy_7"
  let candidate_vmul_1_ptr : RegionName := "candidate_vmul_1"
  let candidate_vsum_3_ptr : RegionName := "candidate_vsum_3"
  let candidate_vto_copy_8_ptr : RegionName := "candidate_vto_copy_8"
  triton {
    r0_z = tl.arange($(1536))
    r0_lane = tl.arange($(128))
    r0_row_values = tl.load($(parameter0_ptr) + ((((r0_z[:, None] / $(6)) % $(256)) * $(128)) + (r0_lane[None, :] % $(128))), dtype=tl.float32)
    r0_rank = tl.full([$(1536), $(128)], $(0), dtype=tl.int64)
    for r0_other in range($(128)) {
      r0_other_value = tl.load($(parameter0_ptr) + ((((r0_z[:, None] / $(6)) % $(256)) * $(128)) + (r0_other % $(128))), dtype=tl.float32)
      r0_other_nan = r0_other_value != r0_other_value
      r0_row_nan = r0_row_values != r0_row_values
      r0_strict = (r0_other_nan & (~r0_row_nan)) | (((~r0_other_nan) & (~r0_row_nan)) & (r0_other_value > r0_row_values))
      r0_equal = (r0_other_nan & r0_row_nan) | (r0_other_value == r0_row_values)
      r0_rank = r0_rank + tl.where(r0_strict | (r0_equal & (r0_other < r0_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r0_selected = r0_rank == tl.cast((r0_z % $(6))[:, None], tl.int64)
    r0_value = tl.full([$(1536)], $(0), dtype=tl.int64)
    for r0_cutoff in range($(128)) {
      r0_value = r0_value + tl.cast(tl.sum(r0_selected & (r0_lane[None, :] > r0_cutoff), axis=1), tl.int64)
    }
    r0_value = tl.load($(parameter0_ptr) + ((((r0_z / $(6)) % $(256)) * $(128)) + (tl.cast(r0_value, tl.uint32) % $(128))), dtype=tl.float32)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(1536)])
    tl.store($(candidate_vtopk_value_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(1536))
    r1_lane = tl.arange($(128))
    r1_row_values = tl.load($(parameter0_ptr) + ((((r1_z[:, None] / $(6)) % $(256)) * $(128)) + (r1_lane[None, :] % $(128))), dtype=tl.float32)
    r1_rank = tl.full([$(1536), $(128)], $(0), dtype=tl.int64)
    for r1_other in range($(128)) {
      r1_other_value = tl.load($(parameter0_ptr) + ((((r1_z[:, None] / $(6)) % $(256)) * $(128)) + (r1_other % $(128))), dtype=tl.float32)
      r1_other_nan = r1_other_value != r1_other_value
      r1_row_nan = r1_row_values != r1_row_values
      r1_strict = (r1_other_nan & (~r1_row_nan)) | (((~r1_other_nan) & (~r1_row_nan)) & (r1_other_value > r1_row_values))
      r1_equal = (r1_other_nan & r1_row_nan) | (r1_other_value == r1_row_values)
      r1_rank = r1_rank + tl.where(r1_strict | (r1_equal & (r1_other < r1_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r1_selected = r1_rank == tl.cast((r1_z % $(6))[:, None], tl.int64)
    r1_value = tl.full([$(1536)], $(0), dtype=tl.int64)
    for r1_cutoff in range($(128)) {
      r1_value = r1_value + tl.cast(tl.sum(r1_selected & (r1_lane[None, :] > r1_cutoff), axis=1), tl.int64)
    }
    r1_value = tl.broadcast_to(r1_value, [$(1536)])
    tl.store($(candidate_vtopk_index_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(32768))
    r2_value = tl.broadcast_to(false, [$(32768)])
    for r2_lane in range($(6)) {
      r2_value = tl.where(tl.load($(candidate_vtopk_index_ptr) + ((((r2_z / $(128)) % $(256)) * $(6)) + (r2_lane % $(6))), dtype=tl.int64) == tl.cast((r2_z % $(128)), tl.int64), true, r2_value)
    }
    r2_value = tl.broadcast_to(r2_value, [$(32768)])
    tl.store($(candidate_vscatter_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(1536))
    r3_lane = tl.arange($(6))
    r3_row_values = tl.load($(candidate_vtopk_value_ptr) + ((((r3_z[:, None] / $(6)) % $(256)) * $(6)) + (r3_lane[None, :] % $(6))), dtype=tl.float32)
    r3_maximum = tl.max(r3_row_values, axis=1, keep_dims=true)
    r3_weights = tl.exp(r3_row_values - r3_maximum)
    r3_total = tl.sum(r3_weights, axis=1)
    r3_maximum = tl.max(r3_row_values, axis=1)
    r3_value = tl.exp(tl.load($(candidate_vtopk_value_ptr) + ((((r3_z / $(6)) % $(256)) * $(6)) + (r3_z % $(6))), dtype=tl.float32) - r3_maximum) / r3_total
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(1536)])
    tl.store($(candidate_vsoftmax_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(32768))
    r4_value = tl.broadcast_to(0.0, [$(32768)])
    for r4_lane in range($(6)) {
      r4_value = tl.where(tl.load($(candidate_vtopk_index_ptr) + ((((r4_z / $(128)) % $(256)) * $(6)) + (r4_lane % $(6))), dtype=tl.int64) == tl.cast((r4_z % $(128)), tl.int64), tl.load($(candidate_vsoftmax_ptr) + ((((r4_z / $(128)) % $(256)) * $(6)) + (r4_lane % $(6))), dtype=tl.float32), r4_value)
    }
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(32768)])
    tl.store($(candidate_vscatter_1_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(32768))
    r5_value = tl.load($(candidate_vscatter_1_ptr) + ((((r5_z / $(128)) % $(256)) * $(128)) + (r5_z % $(128))), dtype=tl.float32)
    r5_value = r5_value.to(tl.bfloat16)
    r5_value = tl.broadcast_to(r5_value, [$(32768)])
    tl.store($(candidate_vto_copy_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(524288))
    r6_value = tl.load($(input0_ptr) + ((((r6_z / $(2048)) % $(256)) * $(2048)) + (r6_z % $(2048))), dtype=tl.float64)
    r6_value = r6_value.to(tl.bfloat16)
    r6_value = tl.broadcast_to(r6_value, [$(524288)])
    tl.store($(candidate_vto_copy_1_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(32768))
    r7_value = tl.where(tl.load($(candidate_vscatter_ptr) + ((((r7_z / $(128)) % $(256)) * $(128)) + (r7_z % $(128))), dtype=tl.int1), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    r7_value = tl.broadcast_to(r7_value, [$(32768)])
    tl.store($(candidate_vto_copy_2_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(32768))
    r8_value = (tl.load($(candidate_vto_copy_2_ptr) + ((((r8_z / $(128)) % $(256)) * $(128)) + (r8_z % $(128))), dtype=tl.int64) != tl.full([], $(0), dtype=tl.int64))
    r8_value = tl.broadcast_to(r8_value, [$(32768)])
    tl.store($(candidate_vto_copy_3_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(32768))
    r9_value = tl.where(tl.load($(candidate_vto_copy_3_ptr) + ((((r9_z / $(128)) % $(256)) * $(128)) + (r9_z % $(128))), dtype=tl.int1), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    r9_value = tl.broadcast_to(r9_value, [$(32768)])
    tl.store($(candidate_vto_copy_4_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(128))
    r10_value = tl.full([$(128)], $(0), dtype=tl.int64)
    for r10_lane in range($(256)) {
      r10_value = r10_value + tl.load($(candidate_vto_copy_4_ptr) + ((r10_lane * $(128)) + r10_z), dtype=tl.int64)
    }
    r10_value = tl.broadcast_to(r10_value, [$(128)])
    tl.store($(candidate_vsum_2_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(128))
    r11_value = (tl.load($(candidate_vsum_2_ptr) + r11_z, dtype=tl.int64) + tl.full([], $(16), dtype=tl.int64))
    r11_value = tl.broadcast_to(r11_value, [$(128)])
    tl.store($(candidate_vadd_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(128))
    r12_value = (tl.load($(candidate_vadd_ptr) + r12_z, dtype=tl.int64) - tl.full([], $(1), dtype=tl.int64))
    r12_value = tl.broadcast_to(r12_value, [$(128)])
    tl.store($(candidate_vsub_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(128))
    r13_value = (tl.load($(candidate_vsub_ptr) + r13_z, dtype=tl.int64) / tl.full([], $(16), dtype=tl.int64))
    r13_value = tl.broadcast_to(r13_value, [$(128)])
    tl.store($(candidate_vfloor_divide_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(128))
    r14_value = (tl.load($(candidate_vfloor_divide_ptr) + r14_z, dtype=tl.int64) * tl.full([], $(16), dtype=tl.int64))
    r14_value = tl.broadcast_to(r14_value, [$(128)])
    tl.store($(candidate_vmul_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(128))
    r15_value = tl.broadcast_to(tl.full([], $(0), dtype=tl.int64), [$(128)])
    for r15_lane in range($(128)) {
      r15_value = tl.where(r15_lane <= r15_z, (r15_value + tl.load($(candidate_vmul_ptr) + (r15_lane % $(128)), dtype=tl.int64)), r15_value)
    }
    r15_value = tl.broadcast_to(r15_value, [$(128)])
    tl.store($(candidate_vcumsum_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(128))
    r16_value = (tl.load($(candidate_vcumsum_ptr) + r16_z, dtype=tl.int64) - tl.load($(candidate_vmul_ptr) + r16_z, dtype=tl.int64))
    r16_value = tl.broadcast_to(r16_value, [$(128)])
    tl.store($(candidate_vsub_1_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(32768))
    r17_value = tl.where(tl.load($(candidate_vto_copy_3_ptr) + ((((r17_z / $(128)) % $(256)) * $(128)) + (r17_z % $(128))), dtype=tl.int1), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    r17_value = tl.broadcast_to(r17_value, [$(32768)])
    tl.store($(candidate_vto_copy_5_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(32768))
    r18_value = tl.broadcast_to(tl.full([], $(0), dtype=tl.int64), [$(32768)])
    for r18_lane in range($(256)) {
      r18_value = tl.where(r18_lane <= ((r18_z / $(128)) % $(256)), (r18_value + tl.load($(candidate_vto_copy_5_ptr) + (((r18_lane % $(256)) * $(128)) + (r18_z % $(128))), dtype=tl.int64)), r18_value)
    }
    r18_value = tl.broadcast_to(r18_value, [$(32768)])
    tl.store($(candidate_vcumsum_1_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(32768))
    r19_value = (tl.load($(candidate_vcumsum_1_ptr) + ((((r19_z / $(128)) % $(256)) * $(128)) + (r19_z % $(128))), dtype=tl.int64) - tl.full([], $(1), dtype=tl.int64))
    r19_value = tl.broadcast_to(r19_value, [$(32768)])
    tl.store($(candidate_vsub_2_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(32768))
    r20_value = (tl.load($(candidate_vsub_2_ptr) + ((((r20_z / $(128)) % $(256)) * $(128)) + (r20_z % $(128))), dtype=tl.int64) + tl.load($(candidate_vsub_1_ptr) + (r20_z % $(128)), dtype=tl.int64))
    r20_value = tl.broadcast_to(r20_value, [$(32768)])
    tl.store($(candidate_vadd_1_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(32768))
    r21_value = tl.where(tl.load($(candidate_vto_copy_3_ptr) + ((((r21_z / $(128)) % $(256)) * $(128)) + (r21_z % $(128))), dtype=tl.int1), tl.load($(candidate_vadd_1_ptr) + ((((r21_z / $(128)) % $(256)) * $(128)) + (r21_z % $(128))), dtype=tl.int64), (-tl.full([], $(1), dtype=tl.int64)))
    r21_value = tl.broadcast_to(r21_value, [$(32768)])
    tl.store($(candidate_vwhere_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(32768))
    r22_value = tl.where(tl.load($(candidate_vto_copy_3_ptr) + ((((r22_z / $(128)) % $(256)) * $(128)) + (r22_z % $(128))), dtype=tl.int1), tl.load($(candidate_vwhere_ptr) + ((((r22_z / $(128)) % $(256)) * $(128)) + (r22_z % $(128))), dtype=tl.int64), tl.full([], $(2176), dtype=tl.int64))
    r22_value = tl.broadcast_to(r22_value, [$(32768)])
    tl.store($(candidate_vwhere_1_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(67108864))
    r23_value = tl.where(tl.load($(candidate_vto_copy_3_ptr) + ((((r23_z / $(262144)) % $(256)) * $(128)) + ((r23_z / $(2048)) % $(128))), dtype=tl.int1), tl.load($(candidate_vto_copy_1_ptr) + ((((r23_z / $(262144)) % $(256)) * $(2048)) + (r23_z % $(2048))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r23_value = r23_value.to(tl.bfloat16)
    r23_value = tl.broadcast_to(r23_value, [$(67108864)])
    tl.store($(candidate_vwhere_2_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(4458496))
    r24_value = tl.broadcast_to(tl.cast(0.0, tl.bfloat16), [$(4458496)])
    for r24_lane in range($(32768)) {
      r24_value = tl.where(tl.load($(candidate_vwhere_1_ptr) + ((((r24_lane / $(128)) % $(256)) * $(128)) + (r24_lane % $(128))), dtype=tl.int64) == tl.cast(((r24_z / $(2048)) % $(2177)), tl.int64), (r24_value + tl.load($(candidate_vwhere_2_ptr) + ((((((r24_lane * $(2048)) + (r24_z % $(2048))) / $(262144)) % $(256)) * $(262144)) + ((r24_lane % $(128)) * $(2048)) + (r24_z % $(2048))), dtype=tl.bfloat16)), r24_value)
    }
    r24_value = r24_value.to(tl.bfloat16)
    r24_value = tl.broadcast_to(r24_value, [$(4458496)])
    tl.store($(candidate_vscatter_add_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(32768))
    r25_value = (tl.load($(candidate_vwhere_ptr) + ((((r25_z / $(128)) % $(256)) * $(128)) + (r25_z % $(128))), dtype=tl.int64) >= tl.full([], $(0), dtype=tl.int64))
    r25_value = tl.broadcast_to(r25_value, [$(32768)])
    tl.store($(candidate_vge_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(32768))
    r26_value = tl.maximum(tl.load($(candidate_vwhere_ptr) + ((((r26_z / $(128)) % $(256)) * $(128)) + (r26_z % $(128))), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    r26_value = tl.broadcast_to(r26_value, [$(32768)])
    tl.store($(candidate_vclamp_min_ptr) + r26_z, r26_value)
    r27_z = tl.arange($(67108864))
    r27_value = tl.load($(candidate_vscatter_add_ptr) + (((tl.load($(candidate_vclamp_min_ptr) + ((((((r27_z / $(2048)) % $(32768)) / $(128)) % $(256)) * $(128)) + (((r27_z / $(2048)) % $(32768)) % $(128))), dtype=tl.int64) % $(2177)) * $(2048)) + (r27_z % $(2048))), dtype=tl.bfloat16)
    r27_value = r27_value.to(tl.bfloat16)
    r27_value = tl.broadcast_to(r27_value, [$(67108864)])
    tl.store($(candidate_vindex_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(67108864))
    r28_value = tl.load($(candidate_vindex_ptr) + ((((r28_z / $(262144)) % $(256)) * $(262144)) + (((r28_z / $(2048)) % $(128)) * $(2048)) + (r28_z % $(2048))), dtype=tl.bfloat16)
    r28_value = r28_value.to(tl.float32)
    r28_value = tl.broadcast_to(r28_value, [$(67108864)])
    tl.store($(candidate_vto_copy_6_ptr) + r28_z, r28_value)
    r29_z = tl.arange($(67108864))
    r29_value = tl.where(tl.load($(candidate_vge_ptr) + ((((r29_z / $(262144)) % $(256)) * $(128)) + ((r29_z / $(2048)) % $(128))), dtype=tl.int1), tl.load($(candidate_vto_copy_6_ptr) + ((((r29_z / $(262144)) % $(256)) * $(262144)) + (((r29_z / $(2048)) % $(128)) * $(2048)) + (r29_z % $(2048))), dtype=tl.float32), 0.0)
    r29_value = r29_value.to(tl.float32)
    r29_value = tl.broadcast_to(r29_value, [$(67108864)])
    tl.store($(candidate_vwhere_4_ptr) + r29_z, r29_value)
    r30_z = tl.arange($(32768))
    r30_value = tl.load($(candidate_vto_copy_ptr) + ((((r30_z / $(128)) % $(256)) * $(128)) + (r30_z % $(128))), dtype=tl.bfloat16)
    r30_value = r30_value.to(tl.float32)
    r30_value = tl.broadcast_to(r30_value, [$(32768)])
    tl.store($(candidate_vto_copy_7_ptr) + r30_z, r30_value)
    r31_z = tl.arange($(67108864))
    r31_value = (tl.load($(candidate_vwhere_4_ptr) + ((((r31_z / $(262144)) % $(256)) * $(262144)) + (((r31_z / $(2048)) % $(128)) * $(2048)) + (r31_z % $(2048))), dtype=tl.float32) * tl.load($(candidate_vto_copy_7_ptr) + ((((r31_z / $(262144)) % $(256)) * $(128)) + ((r31_z / $(2048)) % $(128))), dtype=tl.float32))
    r31_value = r31_value.to(tl.float32)
    r31_value = tl.broadcast_to(r31_value, [$(67108864)])
    tl.store($(candidate_vmul_1_ptr) + r31_z, r31_value)
    r32_z = tl.arange($(524288))
    r32_lane = tl.arange($(128))
    r32_value = tl.sum(tl.load($(candidate_vmul_1_ptr) + ((((r32_z[:, None] / $(2048)) % $(256)) * $(262144)) + (r32_lane[None, :] * $(2048)) + (r32_z[:, None] % $(2048))), dtype=tl.float32), axis=1)
    r32_value = r32_value.to(tl.float32)
    r32_value = tl.broadcast_to(r32_value, [$(524288)])
    tl.store($(candidate_vsum_3_ptr) + r32_z, r32_value)
    r33_z = tl.arange($(524288))
    r33_value = tl.load($(candidate_vsum_3_ptr) + ((((r33_z / $(2048)) % $(256)) * $(2048)) + (r33_z % $(2048))), dtype=tl.float32)
    r33_value = r33_value.to(tl.bfloat16)
    r33_value = tl.broadcast_to(r33_value, [$(524288)])
    tl.store($(candidate_vto_copy_8_ptr) + r33_z, r33_value)
    z = tl.arange($(524288))
    value = tl.load($(candidate_vto_copy_8_ptr) + ((((z / $(2048)) % $(256)) * $(2048)) + (z % $(2048))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(524288)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_fused_unpermute_padded
