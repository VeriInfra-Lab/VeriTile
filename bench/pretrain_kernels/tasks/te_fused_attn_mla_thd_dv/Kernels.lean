import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.te_fused_attn_mla_thd_dv
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Kernels

def original : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let input1_ptr : RegionName := "input1"
  let input2_ptr : RegionName := "input2"
  let out_ptr : RegionName := "out"
  let vfill_ptr : RegionName := "vfill"
  let vslice_scatter_ptr : RegionName := "vslice_scatter"
  let vfill_1_ptr : RegionName := "vfill_1"
  let vslice_scatter_1_ptr : RegionName := "vslice_scatter_1"
  let vfill_2_ptr : RegionName := "vfill_2"
  let vslice_scatter_2_ptr : RegionName := "vslice_scatter_2"
  let vfill_3_ptr : RegionName := "vfill_3"
  let vslice_scatter_3_ptr : RegionName := "vslice_scatter_3"
  let veq_ptr : RegionName := "veq"
  let vge_ptr : RegionName := "vge"
  let vbitwise_and_ptr : RegionName := "vbitwise_and"
  let vbitwise_not_ptr : RegionName := "vbitwise_not"
  let vmasked_fill_ptr : RegionName := "vmasked_fill"
  let vbmm_ptr : RegionName := "vbmm"
  let vmul_ptr : RegionName := "vmul"
  let vadd_ptr : RegionName := "vadd"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vsoftmax_ptr : RegionName := "vsoftmax"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let vbmm_2_ptr : RegionName := "vbmm_2"
  triton {
    r0_z = tl.arange($(37))
    r0_value = tl.full([], $(0), dtype=tl.int64)
    r0_value = tl.broadcast_to(r0_value, [$(37)])
    tl.store($(vfill_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(128))
    r1_value = tl.where(((r1_z >= $(0)) & (r1_z < $(37)) & ((r1_z - $(0)) % $(1) == $(0))), tl.load($(vfill_ptr) + ((r1_z - $(0)) % $(37)), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    r1_value = tl.broadcast_to(r1_value, [$(128)])
    tl.store($(vslice_scatter_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(27))
    r2_value = tl.full([], $(37), dtype=tl.int64)
    r2_value = tl.broadcast_to(r2_value, [$(27)])
    tl.store($(vfill_1_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(128))
    r3_value = tl.where(((r3_z >= $(37)) & (r3_z < $(64)) & ((r3_z - $(37)) % $(1) == $(0))), tl.load($(vfill_1_ptr) + ((r3_z - $(37)) % $(27)), dtype=tl.int64), tl.load($(vslice_scatter_ptr) + r3_z, dtype=tl.int64))
    r3_value = tl.broadcast_to(r3_value, [$(128)])
    tl.store($(vslice_scatter_1_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(27))
    r4_value = tl.full([], $(64), dtype=tl.int64)
    r4_value = tl.broadcast_to(r4_value, [$(27)])
    tl.store($(vfill_2_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(128))
    r5_value = tl.where(((r5_z >= $(64)) & (r5_z < $(91)) & ((r5_z - $(64)) % $(1) == $(0))), tl.load($(vfill_2_ptr) + ((r5_z - $(64)) % $(27)), dtype=tl.int64), tl.load($(vslice_scatter_1_ptr) + r5_z, dtype=tl.int64))
    r5_value = tl.broadcast_to(r5_value, [$(128)])
    tl.store($(vslice_scatter_2_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(37))
    r6_value = tl.full([], $(91), dtype=tl.int64)
    r6_value = tl.broadcast_to(r6_value, [$(37)])
    tl.store($(vfill_3_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(128))
    r7_value = tl.where(((r7_z >= $(91)) & (r7_z < $(128)) & ((r7_z - $(91)) % $(1) == $(0))), tl.load($(vfill_3_ptr) + ((r7_z - $(91)) % $(37)), dtype=tl.int64), tl.load($(vslice_scatter_2_ptr) + r7_z, dtype=tl.int64))
    r7_value = tl.broadcast_to(r7_value, [$(128)])
    tl.store($(vslice_scatter_3_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(16384))
    r8_value = (tl.load($(vslice_scatter_3_ptr) + ((r8_z / $(128)) % $(128)), dtype=tl.int64) == tl.load($(vslice_scatter_3_ptr) + (r8_z % $(128)), dtype=tl.int64))
    r8_value = tl.broadcast_to(r8_value, [$(16384)])
    tl.store($(veq_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(16384))
    r9_value = (tl.cast(((r9_z / $(128)) % $(128)), tl.int64) >= tl.cast((r9_z % $(128)), tl.int64))
    r9_value = tl.broadcast_to(r9_value, [$(16384)])
    tl.store($(vge_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(16384))
    r10_value = (tl.load($(veq_ptr) + ((((r10_z / $(128)) % $(128)) * $(128)) + (r10_z % $(128))), dtype=tl.int1) & tl.load($(vge_ptr) + ((((r10_z / $(128)) % $(128)) * $(128)) + (r10_z % $(128))), dtype=tl.int1))
    r10_value = tl.broadcast_to(r10_value, [$(16384)])
    tl.store($(vbitwise_and_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(16384))
    r11_value = (~tl.load($(vbitwise_and_ptr) + ((((r11_z / $(128)) % $(128)) * $(128)) + (r11_z % $(128))), dtype=tl.int1))
    r11_value = tl.broadcast_to(r11_value, [$(16384)])
    tl.store($(vbitwise_not_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(16384))
    r12_value = tl.where(tl.load($(vbitwise_not_ptr) + ((((r12_z / $(128)) % $(128)) * $(128)) + (r12_z % $(128))), dtype=tl.int1), tl.cast(-338953138925153547590470800371487866880.0, tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r12_value = r12_value.to(tl.bfloat16)
    r12_value = tl.broadcast_to(r12_value, [$(16384)])
    tl.store($(vmasked_fill_ptr) + r12_z, r12_value)
    r13_row = tl.arange($(128))
    r13_col = tl.arange($(128))
    r13_lane = tl.arange($(192))
    for r13_group in range($(4)) {
      r13_left = tl.load($(input0_ptr) + (((((((r13_group % $(4)) * $(24576)) + (r13_row[:, None] * $(192)) + r13_lane[None, :]) / $(192)) % $(128)) * $(768)) + ((((((r13_group % $(4)) * $(24576)) + (r13_row[:, None] * $(192)) + r13_lane[None, :]) / $(24576)) % $(4)) * $(192)) + ((((r13_group % $(4)) * $(24576)) + (r13_row[:, None] * $(192)) + r13_lane[None, :]) % $(192))), dtype=tl.bfloat16)
      r13_right = tl.load($(input1_ptr) + ((((((r13_group % $(4)) * $(24576)) + (r13_lane[:, None] * $(128)) + r13_col[None, :]) % $(128)) * $(768)) + ((((((r13_group % $(4)) * $(24576)) + (r13_lane[:, None] * $(128)) + r13_col[None, :]) / $(24576)) % $(4)) * $(192)) + (((((r13_group % $(4)) * $(24576)) + (r13_lane[:, None] * $(128)) + r13_col[None, :]) / $(128)) % $(192))), dtype=tl.bfloat16)
      r13_value = tl.dot(r13_left, r13_right)
      r13_value = r13_value.to(tl.bfloat16)
      tl.store($(vbmm_ptr) + ((r13_group * $(16384)) + (r13_row[:, None] * $(128)) + r13_col[None, :]), r13_value)
    }
    r14_z = tl.arange($(65536))
    r14_value = (tl.load($(vbmm_ptr) + ((((r14_z / $(16384)) % $(4)) * $(16384)) + (((r14_z / $(128)) % $(128)) * $(128)) + (r14_z % $(128))), dtype=tl.bfloat16) * tl.cast(0.07216878364870323, tl.bfloat16))
    r14_value = r14_value.to(tl.bfloat16)
    r14_value = tl.broadcast_to(r14_value, [$(65536)])
    tl.store($(vmul_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(65536))
    r15_value = (tl.load($(vmul_ptr) + ((((r15_z / $(16384)) % $(4)) * $(16384)) + (((r15_z / $(128)) % $(128)) * $(128)) + (r15_z % $(128))), dtype=tl.bfloat16) + tl.load($(vmasked_fill_ptr) + ((((r15_z / $(128)) % $(128)) * $(128)) + (r15_z % $(128))), dtype=tl.bfloat16))
    r15_value = r15_value.to(tl.bfloat16)
    r15_value = tl.broadcast_to(r15_value, [$(65536)])
    tl.store($(vadd_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(65536))
    r16_value = tl.load($(vadd_ptr) + ((((r16_z / $(16384)) % $(4)) * $(16384)) + (((r16_z / $(128)) % $(128)) * $(128)) + (r16_z % $(128))), dtype=tl.bfloat16)
    r16_value = r16_value.to(tl.float32)
    r16_value = tl.broadcast_to(r16_value, [$(65536)])
    tl.store($(vto_copy_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(65536))
    r17_lane = tl.arange($(128))
    r17_row_values = tl.load($(vto_copy_ptr) + ((((r17_z[:, None] / $(16384)) % $(4)) * $(16384)) + (((r17_z[:, None] / $(128)) % $(128)) * $(128)) + (r17_lane[None, :] % $(128))), dtype=tl.float32)
    r17_maximum = tl.max(r17_row_values, axis=1, keep_dims=true)
    r17_weights = tl.exp(r17_row_values - r17_maximum)
    r17_total = tl.sum(r17_weights, axis=1)
    r17_maximum = tl.max(r17_row_values, axis=1)
    r17_value = tl.exp(tl.load($(vto_copy_ptr) + ((((r17_z / $(16384)) % $(4)) * $(16384)) + (((r17_z / $(128)) % $(128)) * $(128)) + (r17_z % $(128))), dtype=tl.float32) - r17_maximum) / r17_total
    r17_value = r17_value.to(tl.float32)
    r17_value = tl.broadcast_to(r17_value, [$(65536)])
    tl.store($(vsoftmax_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(65536))
    r18_value = tl.load($(vsoftmax_ptr) + ((((r18_z / $(16384)) % $(4)) * $(16384)) + (((r18_z / $(128)) % $(128)) * $(128)) + (r18_z % $(128))), dtype=tl.float32)
    r18_value = r18_value.to(tl.bfloat16)
    r18_value = tl.broadcast_to(r18_value, [$(65536)])
    tl.store($(vto_copy_1_ptr) + r18_z, r18_value)
    r19_row = tl.arange($(128))
    r19_col = tl.arange($(128))
    r19_lane = tl.arange($(128))
    for r19_group in range($(4)) {
      r19_left = tl.load($(vto_copy_1_ptr) + (((((((r19_group % $(4)) * $(16384)) + (r19_lane[None, :] * $(128)) + r19_row[:, None]) / $(16384)) % $(4)) * $(16384)) + ((((((r19_group % $(4)) * $(16384)) + (r19_lane[None, :] * $(128)) + r19_row[:, None]) / $(128)) % $(128)) * $(128)) + ((((r19_group % $(4)) * $(16384)) + (r19_lane[None, :] * $(128)) + r19_row[:, None]) % $(128))), dtype=tl.bfloat16)
      r19_right = tl.load($(input2_ptr) + (((((((r19_group % $(4)) * $(16384)) + (r19_lane[:, None] * $(128)) + r19_col[None, :]) / $(128)) % $(128)) * $(512)) + ((((((r19_group % $(4)) * $(16384)) + (r19_lane[:, None] * $(128)) + r19_col[None, :]) / $(16384)) % $(4)) * $(128)) + ((((r19_group % $(4)) * $(16384)) + (r19_lane[:, None] * $(128)) + r19_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r19_value = tl.dot(r19_left, r19_right)
      r19_value = r19_value.to(tl.bfloat16)
      tl.store($(vbmm_2_ptr) + ((r19_group * $(16384)) + (r19_row[:, None] * $(128)) + r19_col[None, :]), r19_value)
    }
    z = tl.arange($(65536))
    value = tl.load($(vbmm_2_ptr) + ((((z / $(128)) % $(4)) * $(16384)) + (((z / $(512)) % $(128)) * $(128)) + (z % $(128))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(65536)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let input1_ptr : RegionName := "input1"
  let input2_ptr : RegionName := "input2"
  let out_ptr : RegionName := "out"
  let vfill_ptr : RegionName := "vfill"
  let vslice_scatter_ptr : RegionName := "vslice_scatter"
  let vfill_1_ptr : RegionName := "vfill_1"
  let vslice_scatter_1_ptr : RegionName := "vslice_scatter_1"
  let vfill_2_ptr : RegionName := "vfill_2"
  let vslice_scatter_2_ptr : RegionName := "vslice_scatter_2"
  let vfill_3_ptr : RegionName := "vfill_3"
  let vslice_scatter_3_ptr : RegionName := "vslice_scatter_3"
  let veq_ptr : RegionName := "veq"
  let vge_ptr : RegionName := "vge"
  let vbitwise_and_ptr : RegionName := "vbitwise_and"
  let vbitwise_not_ptr : RegionName := "vbitwise_not"
  let vmasked_fill_ptr : RegionName := "vmasked_fill"
  let vbmm_ptr : RegionName := "vbmm"
  let vmul_ptr : RegionName := "vmul"
  let vadd_ptr : RegionName := "vadd"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vsoftmax_ptr : RegionName := "vsoftmax"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let vbmm_2_ptr : RegionName := "vbmm_2"
  let candidate_vfill_ptr : RegionName := "candidate_vfill"
  let candidate_vslice_scatter_ptr : RegionName := "candidate_vslice_scatter"
  let candidate_vfill_1_ptr : RegionName := "candidate_vfill_1"
  let candidate_vslice_scatter_1_ptr : RegionName := "candidate_vslice_scatter_1"
  let candidate_vfill_2_ptr : RegionName := "candidate_vfill_2"
  let candidate_vslice_scatter_2_ptr : RegionName := "candidate_vslice_scatter_2"
  let candidate_vfill_3_ptr : RegionName := "candidate_vfill_3"
  let candidate_vslice_scatter_3_ptr : RegionName := "candidate_vslice_scatter_3"
  let candidate_veq_ptr : RegionName := "candidate_veq"
  let candidate_vge_ptr : RegionName := "candidate_vge"
  let candidate_vbitwise_and_ptr : RegionName := "candidate_vbitwise_and"
  let candidate_vbitwise_not_ptr : RegionName := "candidate_vbitwise_not"
  let candidate_vmasked_fill_ptr : RegionName := "candidate_vmasked_fill"
  let candidate_vbmm_ptr : RegionName := "candidate_vbmm"
  let candidate_vmul_ptr : RegionName := "candidate_vmul"
  let candidate_vadd_ptr : RegionName := "candidate_vadd"
  let candidate_vto_copy_ptr : RegionName := "candidate_vto_copy"
  let candidate_vsoftmax_ptr : RegionName := "candidate_vsoftmax"
  let candidate_vto_copy_1_ptr : RegionName := "candidate_vto_copy_1"
  let candidate_vbmm_2_ptr : RegionName := "candidate_vbmm_2"
  triton {
    r0_z = tl.arange($(37))
    r0_value = tl.full([], $(0), dtype=tl.int64)
    r0_value = tl.broadcast_to(r0_value, [$(37)])
    tl.store($(candidate_vfill_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(128))
    r1_value = tl.where(((r1_z >= $(0)) & (r1_z < $(37)) & ((r1_z - $(0)) % $(1) == $(0))), tl.load($(candidate_vfill_ptr) + ((r1_z - $(0)) % $(37)), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    r1_value = tl.broadcast_to(r1_value, [$(128)])
    tl.store($(candidate_vslice_scatter_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(27))
    r2_value = tl.full([], $(37), dtype=tl.int64)
    r2_value = tl.broadcast_to(r2_value, [$(27)])
    tl.store($(candidate_vfill_1_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(128))
    r3_value = tl.where(((r3_z >= $(37)) & (r3_z < $(64)) & ((r3_z - $(37)) % $(1) == $(0))), tl.load($(candidate_vfill_1_ptr) + ((r3_z - $(37)) % $(27)), dtype=tl.int64), tl.load($(candidate_vslice_scatter_ptr) + r3_z, dtype=tl.int64))
    r3_value = tl.broadcast_to(r3_value, [$(128)])
    tl.store($(candidate_vslice_scatter_1_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(27))
    r4_value = tl.full([], $(64), dtype=tl.int64)
    r4_value = tl.broadcast_to(r4_value, [$(27)])
    tl.store($(candidate_vfill_2_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(128))
    r5_value = tl.where(((r5_z >= $(64)) & (r5_z < $(91)) & ((r5_z - $(64)) % $(1) == $(0))), tl.load($(candidate_vfill_2_ptr) + ((r5_z - $(64)) % $(27)), dtype=tl.int64), tl.load($(candidate_vslice_scatter_1_ptr) + r5_z, dtype=tl.int64))
    r5_value = tl.broadcast_to(r5_value, [$(128)])
    tl.store($(candidate_vslice_scatter_2_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(37))
    r6_value = tl.full([], $(91), dtype=tl.int64)
    r6_value = tl.broadcast_to(r6_value, [$(37)])
    tl.store($(candidate_vfill_3_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(128))
    r7_value = tl.where(((r7_z >= $(91)) & (r7_z < $(128)) & ((r7_z - $(91)) % $(1) == $(0))), tl.load($(candidate_vfill_3_ptr) + ((r7_z - $(91)) % $(37)), dtype=tl.int64), tl.load($(candidate_vslice_scatter_2_ptr) + r7_z, dtype=tl.int64))
    r7_value = tl.broadcast_to(r7_value, [$(128)])
    tl.store($(candidate_vslice_scatter_3_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(16384))
    r8_value = (tl.load($(candidate_vslice_scatter_3_ptr) + ((r8_z / $(128)) % $(128)), dtype=tl.int64) == tl.load($(candidate_vslice_scatter_3_ptr) + (r8_z % $(128)), dtype=tl.int64))
    r8_value = tl.broadcast_to(r8_value, [$(16384)])
    tl.store($(candidate_veq_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(16384))
    r9_value = (tl.cast(((r9_z / $(128)) % $(128)), tl.int64) >= tl.cast((r9_z % $(128)), tl.int64))
    r9_value = tl.broadcast_to(r9_value, [$(16384)])
    tl.store($(candidate_vge_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(16384))
    r10_value = (tl.load($(candidate_veq_ptr) + ((((r10_z / $(128)) % $(128)) * $(128)) + (r10_z % $(128))), dtype=tl.int1) & tl.load($(candidate_vge_ptr) + ((((r10_z / $(128)) % $(128)) * $(128)) + (r10_z % $(128))), dtype=tl.int1))
    r10_value = tl.broadcast_to(r10_value, [$(16384)])
    tl.store($(candidate_vbitwise_and_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(16384))
    r11_value = (~tl.load($(candidate_vbitwise_and_ptr) + ((((r11_z / $(128)) % $(128)) * $(128)) + (r11_z % $(128))), dtype=tl.int1))
    r11_value = tl.broadcast_to(r11_value, [$(16384)])
    tl.store($(candidate_vbitwise_not_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(16384))
    r12_value = tl.where(tl.load($(candidate_vbitwise_not_ptr) + ((((r12_z / $(128)) % $(128)) * $(128)) + (r12_z % $(128))), dtype=tl.int1), tl.cast(-338953138925153547590470800371487866880.0, tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r12_value = r12_value.to(tl.bfloat16)
    r12_value = tl.broadcast_to(r12_value, [$(16384)])
    tl.store($(candidate_vmasked_fill_ptr) + r12_z, r12_value)
    r13_row = tl.arange($(128))
    r13_col = tl.arange($(128))
    r13_lane = tl.arange($(192))
    for r13_group in range($(4)) {
      r13_left = tl.load($(input0_ptr) + (((((((r13_group % $(4)) * $(24576)) + (r13_row[:, None] * $(192)) + r13_lane[None, :]) / $(192)) % $(128)) * $(768)) + ((((((r13_group % $(4)) * $(24576)) + (r13_row[:, None] * $(192)) + r13_lane[None, :]) / $(24576)) % $(4)) * $(192)) + ((((r13_group % $(4)) * $(24576)) + (r13_row[:, None] * $(192)) + r13_lane[None, :]) % $(192))), dtype=tl.bfloat16)
      r13_right = tl.load($(input1_ptr) + ((((((r13_group % $(4)) * $(24576)) + (r13_lane[:, None] * $(128)) + r13_col[None, :]) % $(128)) * $(768)) + ((((((r13_group % $(4)) * $(24576)) + (r13_lane[:, None] * $(128)) + r13_col[None, :]) / $(24576)) % $(4)) * $(192)) + (((((r13_group % $(4)) * $(24576)) + (r13_lane[:, None] * $(128)) + r13_col[None, :]) / $(128)) % $(192))), dtype=tl.bfloat16)
      r13_value = tl.dot(r13_left, r13_right)
      r13_value = r13_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_ptr) + ((r13_group * $(16384)) + (r13_row[:, None] * $(128)) + r13_col[None, :]), r13_value)
    }
    r14_z = tl.arange($(65536))
    r14_value = (tl.load($(candidate_vbmm_ptr) + ((((r14_z / $(16384)) % $(4)) * $(16384)) + (((r14_z / $(128)) % $(128)) * $(128)) + (r14_z % $(128))), dtype=tl.bfloat16) * tl.cast(0.07216878364870323, tl.bfloat16))
    r14_value = r14_value.to(tl.bfloat16)
    r14_value = tl.broadcast_to(r14_value, [$(65536)])
    tl.store($(candidate_vmul_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(65536))
    r15_value = (tl.load($(candidate_vmul_ptr) + ((((r15_z / $(16384)) % $(4)) * $(16384)) + (((r15_z / $(128)) % $(128)) * $(128)) + (r15_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vmasked_fill_ptr) + ((((r15_z / $(128)) % $(128)) * $(128)) + (r15_z % $(128))), dtype=tl.bfloat16))
    r15_value = r15_value.to(tl.bfloat16)
    r15_value = tl.broadcast_to(r15_value, [$(65536)])
    tl.store($(candidate_vadd_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(65536))
    r16_value = tl.load($(candidate_vadd_ptr) + ((((r16_z / $(16384)) % $(4)) * $(16384)) + (((r16_z / $(128)) % $(128)) * $(128)) + (r16_z % $(128))), dtype=tl.bfloat16)
    r16_value = r16_value.to(tl.float32)
    r16_value = tl.broadcast_to(r16_value, [$(65536)])
    tl.store($(candidate_vto_copy_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(65536))
    r17_lane = tl.arange($(128))
    r17_row_values = tl.load($(candidate_vto_copy_ptr) + ((((r17_z[:, None] / $(16384)) % $(4)) * $(16384)) + (((r17_z[:, None] / $(128)) % $(128)) * $(128)) + (r17_lane[None, :] % $(128))), dtype=tl.float32)
    r17_maximum = tl.max(r17_row_values, axis=1, keep_dims=true)
    r17_weights = tl.exp(r17_row_values - r17_maximum)
    r17_total = tl.sum(r17_weights, axis=1)
    r17_maximum = tl.max(r17_row_values, axis=1)
    r17_value = tl.exp(tl.load($(candidate_vto_copy_ptr) + ((((r17_z / $(16384)) % $(4)) * $(16384)) + (((r17_z / $(128)) % $(128)) * $(128)) + (r17_z % $(128))), dtype=tl.float32) - r17_maximum) / r17_total
    r17_value = r17_value.to(tl.float32)
    r17_value = tl.broadcast_to(r17_value, [$(65536)])
    tl.store($(candidate_vsoftmax_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(65536))
    r18_value = tl.load($(candidate_vsoftmax_ptr) + ((((r18_z / $(16384)) % $(4)) * $(16384)) + (((r18_z / $(128)) % $(128)) * $(128)) + (r18_z % $(128))), dtype=tl.float32)
    r18_value = r18_value.to(tl.bfloat16)
    r18_value = tl.broadcast_to(r18_value, [$(65536)])
    tl.store($(candidate_vto_copy_1_ptr) + r18_z, r18_value)
    r19_row = tl.arange($(128))
    r19_col = tl.arange($(128))
    r19_lane = tl.arange($(128))
    for r19_group in range($(4)) {
      r19_left = tl.load($(candidate_vto_copy_1_ptr) + (((((((r19_group % $(4)) * $(16384)) + (r19_lane[None, :] * $(128)) + r19_row[:, None]) / $(16384)) % $(4)) * $(16384)) + ((((((r19_group % $(4)) * $(16384)) + (r19_lane[None, :] * $(128)) + r19_row[:, None]) / $(128)) % $(128)) * $(128)) + ((((r19_group % $(4)) * $(16384)) + (r19_lane[None, :] * $(128)) + r19_row[:, None]) % $(128))), dtype=tl.bfloat16)
      r19_right = tl.load($(input2_ptr) + (((((((r19_group % $(4)) * $(16384)) + (r19_lane[:, None] * $(128)) + r19_col[None, :]) / $(128)) % $(128)) * $(512)) + ((((((r19_group % $(4)) * $(16384)) + (r19_lane[:, None] * $(128)) + r19_col[None, :]) / $(16384)) % $(4)) * $(128)) + ((((r19_group % $(4)) * $(16384)) + (r19_lane[:, None] * $(128)) + r19_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r19_value = tl.dot(r19_left, r19_right)
      r19_value = r19_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_2_ptr) + ((r19_group * $(16384)) + (r19_row[:, None] * $(128)) + r19_col[None, :]), r19_value)
    }
    z = tl.arange($(65536))
    value = tl.load($(candidate_vbmm_2_ptr) + ((((z / $(128)) % $(4)) * $(16384)) + (((z / $(512)) % $(128)) * $(128)) + (z % $(128))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(65536)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.te_fused_attn_mla_thd_dv
