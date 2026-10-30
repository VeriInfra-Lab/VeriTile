import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_fa4_mla_decode
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
  let input3_ptr : RegionName := "input3"
  let out_ptr : RegionName := "out"
  let vcat_ptr : RegionName := "vcat"
  let vcat_1_ptr : RegionName := "vcat_1"
  let vbmm_ptr : RegionName := "vbmm"
  let vmul_ptr : RegionName := "vmul"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vsoftmax_ptr : RegionName := "vsoftmax"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let vbmm_1_ptr : RegionName := "vbmm_1"
  triton {
    r0_z = tl.arange($(18432))
    r0_value = tl.where((r0_z % $(576)) < $(64), tl.load($(input0_ptr) + ((((r0_z / $(4608)) % $(4)) * $(512)) + (((r0_z / $(576)) % $(8)) * $(64)) + (((r0_z % $(576)) - $(0)) % $(64))), dtype=tl.bfloat16), tl.load($(input2_ptr) + ((((r0_z / $(4608)) % $(4)) * $(4096)) + (((r0_z / $(576)) % $(8)) * $(512)) + (((r0_z % $(576)) - $(64)) % $(512))), dtype=tl.bfloat16))
    r0_value = r0_value.to(tl.bfloat16)
    r0_value = tl.broadcast_to(r0_value, [$(18432)])
    tl.store($(vcat_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(589824))
    r1_value = tl.where((r1_z % $(576)) < $(64), tl.load($(input1_ptr) + ((((r1_z / $(147456)) % $(4)) * $(16384)) + (((r1_z / $(576)) % $(256)) * $(64)) + (((r1_z % $(576)) - $(0)) % $(64))), dtype=tl.bfloat16), tl.load($(input3_ptr) + ((((r1_z / $(147456)) % $(4)) * $(131072)) + (((r1_z / $(576)) % $(256)) * $(512)) + (((r1_z % $(576)) - $(64)) % $(512))), dtype=tl.bfloat16))
    r1_value = r1_value.to(tl.bfloat16)
    r1_value = tl.broadcast_to(r1_value, [$(589824)])
    tl.store($(vcat_1_ptr) + r1_z, r1_value)
    r2_row = tl.arange($(1))
    r2_col = tl.arange($(256))
    r2_lane = tl.arange($(576))
    for r2_group in range($(32)) {
      r2_left = tl.load($(vcat_ptr) + (((((((r2_group % $(32)) * $(576)) + (r2_row[:, None] * $(576)) + r2_lane[None, :]) / $(4608)) % $(4)) * $(4608)) + ((((((r2_group % $(32)) * $(576)) + (r2_row[:, None] * $(576)) + r2_lane[None, :]) / $(576)) % $(8)) * $(576)) + ((((r2_group % $(32)) * $(576)) + (r2_row[:, None] * $(576)) + r2_lane[None, :]) % $(576))), dtype=tl.bfloat16)
      r2_right = tl.load($(vcat_1_ptr) + (((((((r2_group % $(32)) * $(147456)) + (r2_lane[:, None] * $(256)) + r2_col[None, :]) / $(1179648)) % $(4)) * $(147456)) + (((((r2_group % $(32)) * $(147456)) + (r2_lane[:, None] * $(256)) + r2_col[None, :]) % $(256)) * $(576)) + (((((r2_group % $(32)) * $(147456)) + (r2_lane[:, None] * $(256)) + r2_col[None, :]) / $(256)) % $(576))), dtype=tl.bfloat16)
      r2_value = tl.dot(r2_left, r2_right)
      r2_value = r2_value.to(tl.bfloat16)
      tl.store($(vbmm_ptr) + ((r2_group * $(256)) + (r2_row[:, None] * $(256)) + r2_col[None, :]), r2_value)
    }
    r3_z = tl.arange($(8192))
    r3_value = (tl.load($(vbmm_ptr) + ((((r3_z / $(2048)) % $(4)) * $(2048)) + (((r3_z / $(256)) % $(8)) * $(256)) + (r3_z % $(256))), dtype=tl.bfloat16) * tl.cast(0.041666666666666664, tl.bfloat16))
    r3_value = r3_value.to(tl.bfloat16)
    r3_value = tl.broadcast_to(r3_value, [$(8192)])
    tl.store($(vmul_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(8192))
    r4_value = tl.load($(vmul_ptr) + ((((r4_z / $(2048)) % $(4)) * $(2048)) + (((r4_z / $(256)) % $(8)) * $(256)) + (r4_z % $(256))), dtype=tl.bfloat16)
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(8192)])
    tl.store($(vto_copy_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(8192))
    r5_lane = tl.arange($(256))
    r5_row_values = tl.load($(vto_copy_ptr) + ((((r5_z[:, None] / $(2048)) % $(4)) * $(2048)) + (((r5_z[:, None] / $(256)) % $(8)) * $(256)) + (r5_lane[None, :] % $(256))), dtype=tl.float32)
    r5_maximum = tl.max(r5_row_values, axis=1, keep_dims=true)
    r5_weights = tl.exp(r5_row_values - r5_maximum)
    r5_total = tl.sum(r5_weights, axis=1)
    r5_maximum = tl.max(r5_row_values, axis=1)
    r5_value = tl.exp(tl.load($(vto_copy_ptr) + ((((r5_z / $(2048)) % $(4)) * $(2048)) + (((r5_z / $(256)) % $(8)) * $(256)) + (r5_z % $(256))), dtype=tl.float32) - r5_maximum) / r5_total
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(8192)])
    tl.store($(vsoftmax_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(8192))
    r6_value = tl.load($(vsoftmax_ptr) + ((((r6_z / $(2048)) % $(4)) * $(2048)) + (((r6_z / $(256)) % $(8)) * $(256)) + (r6_z % $(256))), dtype=tl.float32)
    r6_value = r6_value.to(tl.bfloat16)
    r6_value = tl.broadcast_to(r6_value, [$(8192)])
    tl.store($(vto_copy_1_ptr) + r6_z, r6_value)
    r7_row = tl.arange($(1))
    r7_col = tl.arange($(512))
    r7_lane = tl.arange($(256))
    for r7_group in range($(32)) {
      r7_left = tl.load($(vto_copy_1_ptr) + (((((((r7_group % $(32)) * $(256)) + (r7_row[:, None] * $(256)) + r7_lane[None, :]) / $(2048)) % $(4)) * $(2048)) + ((((((r7_group % $(32)) * $(256)) + (r7_row[:, None] * $(256)) + r7_lane[None, :]) / $(256)) % $(8)) * $(256)) + ((((r7_group % $(32)) * $(256)) + (r7_row[:, None] * $(256)) + r7_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r7_right = tl.load($(input3_ptr) + (((((((r7_group % $(32)) * $(131072)) + (r7_lane[:, None] * $(512)) + r7_col[None, :]) / $(1048576)) % $(4)) * $(131072)) + ((((((r7_group % $(32)) * $(131072)) + (r7_lane[:, None] * $(512)) + r7_col[None, :]) / $(512)) % $(256)) * $(512)) + ((((r7_group % $(32)) * $(131072)) + (r7_lane[:, None] * $(512)) + r7_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r7_value = tl.dot(r7_left, r7_right)
      r7_value = r7_value.to(tl.bfloat16)
      tl.store($(vbmm_1_ptr) + ((r7_group * $(512)) + (r7_row[:, None] * $(512)) + r7_col[None, :]), r7_value)
    }
    z = tl.arange($(16384))
    value = tl.load($(vbmm_1_ptr) + ((((z / $(4096)) % $(4)) * $(4096)) + (((z / $(512)) % $(8)) * $(512)) + (z % $(512))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(16384)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let input1_ptr : RegionName := "input1"
  let input2_ptr : RegionName := "input2"
  let input3_ptr : RegionName := "input3"
  let out_ptr : RegionName := "out"
  let vcat_ptr : RegionName := "vcat"
  let vbmm_ptr : RegionName := "vbmm"
  let vmul_ptr : RegionName := "vmul"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vsoftmax_ptr : RegionName := "vsoftmax"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let vbmm_1_ptr : RegionName := "vbmm_1"
  let candidate_vto_copy_ptr : RegionName := "candidate_vto_copy"
  let candidate_vto_copy_1_ptr : RegionName := "candidate_vto_copy_1"
  let candidate_vbmm_ptr : RegionName := "candidate_vbmm"
  let candidate_vto_copy_2_ptr : RegionName := "candidate_vto_copy_2"
  let candidate_vto_copy_3_ptr : RegionName := "candidate_vto_copy_3"
  let candidate_vbmm_1_ptr : RegionName := "candidate_vbmm_1"
  let candidate_vadd_ptr : RegionName := "candidate_vadd"
  let candidate_vmul_ptr : RegionName := "candidate_vmul"
  let candidate_vadd_1_ptr : RegionName := "candidate_vadd_1"
  let candidate_vsub_ptr : RegionName := "candidate_vsub"
  let candidate_vgt_ptr : RegionName := "candidate_vgt"
  let candidate_vmasked_fill_ptr : RegionName := "candidate_vmasked_fill"
  let candidate_vsoftmax_ptr : RegionName := "candidate_vsoftmax"
  let candidate_vto_copy_4_ptr : RegionName := "candidate_vto_copy_4"
  let candidate_vmm_ptr : RegionName := "candidate_vmm"
  let candidate_vto_copy_5_ptr : RegionName := "candidate_vto_copy_5"
  let candidate_vto_copy_6_ptr : RegionName := "candidate_vto_copy_6"
  let candidate_vbmm_2_ptr : RegionName := "candidate_vbmm_2"
  let candidate_vto_copy_7_ptr : RegionName := "candidate_vto_copy_7"
  let candidate_vto_copy_8_ptr : RegionName := "candidate_vto_copy_8"
  let candidate_vbmm_3_ptr : RegionName := "candidate_vbmm_3"
  let candidate_vadd_2_ptr : RegionName := "candidate_vadd_2"
  let candidate_vmul_1_ptr : RegionName := "candidate_vmul_1"
  let candidate_vadd_3_ptr : RegionName := "candidate_vadd_3"
  let candidate_vsub_1_ptr : RegionName := "candidate_vsub_1"
  let candidate_vgt_1_ptr : RegionName := "candidate_vgt_1"
  let candidate_vmasked_fill_1_ptr : RegionName := "candidate_vmasked_fill_1"
  let candidate_vsoftmax_1_ptr : RegionName := "candidate_vsoftmax_1"
  let candidate_vto_copy_9_ptr : RegionName := "candidate_vto_copy_9"
  let candidate_vmm_1_ptr : RegionName := "candidate_vmm_1"
  let candidate_vto_copy_10_ptr : RegionName := "candidate_vto_copy_10"
  let candidate_vto_copy_11_ptr : RegionName := "candidate_vto_copy_11"
  let candidate_vbmm_4_ptr : RegionName := "candidate_vbmm_4"
  let candidate_vto_copy_12_ptr : RegionName := "candidate_vto_copy_12"
  let candidate_vto_copy_13_ptr : RegionName := "candidate_vto_copy_13"
  let candidate_vbmm_5_ptr : RegionName := "candidate_vbmm_5"
  let candidate_vadd_4_ptr : RegionName := "candidate_vadd_4"
  let candidate_vmul_2_ptr : RegionName := "candidate_vmul_2"
  let candidate_vadd_5_ptr : RegionName := "candidate_vadd_5"
  let candidate_vsub_2_ptr : RegionName := "candidate_vsub_2"
  let candidate_vgt_2_ptr : RegionName := "candidate_vgt_2"
  let candidate_vmasked_fill_2_ptr : RegionName := "candidate_vmasked_fill_2"
  let candidate_vsoftmax_2_ptr : RegionName := "candidate_vsoftmax_2"
  let candidate_vto_copy_14_ptr : RegionName := "candidate_vto_copy_14"
  let candidate_vmm_2_ptr : RegionName := "candidate_vmm_2"
  let candidate_vto_copy_15_ptr : RegionName := "candidate_vto_copy_15"
  let candidate_vto_copy_16_ptr : RegionName := "candidate_vto_copy_16"
  let candidate_vbmm_6_ptr : RegionName := "candidate_vbmm_6"
  let candidate_vto_copy_17_ptr : RegionName := "candidate_vto_copy_17"
  let candidate_vto_copy_18_ptr : RegionName := "candidate_vto_copy_18"
  let candidate_vbmm_7_ptr : RegionName := "candidate_vbmm_7"
  let candidate_vadd_6_ptr : RegionName := "candidate_vadd_6"
  let candidate_vmul_3_ptr : RegionName := "candidate_vmul_3"
  let candidate_vadd_7_ptr : RegionName := "candidate_vadd_7"
  let candidate_vsub_3_ptr : RegionName := "candidate_vsub_3"
  let candidate_vgt_3_ptr : RegionName := "candidate_vgt_3"
  let candidate_vmasked_fill_3_ptr : RegionName := "candidate_vmasked_fill_3"
  let candidate_vsoftmax_3_ptr : RegionName := "candidate_vsoftmax_3"
  let candidate_vto_copy_19_ptr : RegionName := "candidate_vto_copy_19"
  let candidate_vmm_3_ptr : RegionName := "candidate_vmm_3"
  let candidate_vcat_ptr : RegionName := "candidate_vcat"
  triton {
    r0_z = tl.arange($(512))
    r0_value = tl.load($(input0_ptr) + ((((r0_z / $(64)) % $(8)) * $(64)) + (r0_z % $(64))), dtype=tl.bfloat16)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(512)])
    tl.store($(candidate_vto_copy_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(16384))
    r1_value = tl.load($(input1_ptr) + ((((r1_z / $(64)) % $(256)) * $(64)) + (r1_z % $(64))), dtype=tl.bfloat16)
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(16384)])
    tl.store($(candidate_vto_copy_1_ptr) + r1_z, r1_value)
    r2_row = tl.arange($(1))
    r2_col = tl.arange($(256))
    r2_lane = tl.arange($(64))
    for r2_group in range($(8)) {
      r2_left = tl.load($(candidate_vto_copy_ptr) + (((((((r2_group % $(8)) * $(64)) + (r2_row[:, None] * $(64)) + r2_lane[None, :]) / $(64)) % $(8)) * $(64)) + ((((r2_group % $(8)) * $(64)) + (r2_row[:, None] * $(64)) + r2_lane[None, :]) % $(64))), dtype=tl.float32)
      r2_right = tl.load($(candidate_vto_copy_1_ptr) + ((((((r2_group % $(8)) * $(16384)) + (r2_lane[:, None] * $(256)) + r2_col[None, :]) % $(256)) * $(64)) + (((((r2_group % $(8)) * $(16384)) + (r2_lane[:, None] * $(256)) + r2_col[None, :]) / $(256)) % $(64))), dtype=tl.float32)
      r2_value = tl.dot(r2_left, r2_right)
      r2_value = r2_value.to(tl.float32)
      tl.store($(candidate_vbmm_ptr) + ((r2_group * $(256)) + (r2_row[:, None] * $(256)) + r2_col[None, :]), r2_value)
    }
    r3_z = tl.arange($(4096))
    r3_value = tl.load($(input2_ptr) + ((((r3_z / $(512)) % $(8)) * $(512)) + (r3_z % $(512))), dtype=tl.bfloat16)
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(4096)])
    tl.store($(candidate_vto_copy_2_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(131072))
    r4_value = tl.load($(input3_ptr) + ((((r4_z / $(512)) % $(256)) * $(512)) + (r4_z % $(512))), dtype=tl.bfloat16)
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(131072)])
    tl.store($(candidate_vto_copy_3_ptr) + r4_z, r4_value)
    r5_row = tl.arange($(1))
    r5_col = tl.arange($(256))
    r5_lane = tl.arange($(512))
    for r5_group in range($(8)) {
      r5_left = tl.load($(candidate_vto_copy_2_ptr) + (((((((r5_group % $(8)) * $(512)) + (r5_row[:, None] * $(512)) + r5_lane[None, :]) / $(512)) % $(8)) * $(512)) + ((((r5_group % $(8)) * $(512)) + (r5_row[:, None] * $(512)) + r5_lane[None, :]) % $(512))), dtype=tl.float32)
      r5_right = tl.load($(candidate_vto_copy_3_ptr) + ((((((r5_group % $(8)) * $(131072)) + (r5_lane[:, None] * $(256)) + r5_col[None, :]) % $(256)) * $(512)) + (((((r5_group % $(8)) * $(131072)) + (r5_lane[:, None] * $(256)) + r5_col[None, :]) / $(256)) % $(512))), dtype=tl.float32)
      r5_value = tl.dot(r5_left, r5_right)
      r5_value = r5_value.to(tl.float32)
      tl.store($(candidate_vbmm_1_ptr) + ((r5_group * $(256)) + (r5_row[:, None] * $(256)) + r5_col[None, :]), r5_value)
    }
    r6_z = tl.arange($(2048))
    r6_value = (tl.load($(candidate_vbmm_ptr) + ((((r6_z / $(256)) % $(8)) * $(256)) + (r6_z % $(256))), dtype=tl.float32) + tl.load($(candidate_vbmm_1_ptr) + ((((r6_z / $(256)) % $(8)) * $(256)) + (r6_z % $(256))), dtype=tl.float32))
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(2048)])
    tl.store($(candidate_vadd_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(2048))
    r7_value = (tl.load($(candidate_vadd_ptr) + ((((r7_z / $(256)) % $(8)) * $(256)) + (r7_z % $(256))), dtype=tl.float32) * 0.041666666666666664)
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(2048)])
    tl.store($(candidate_vmul_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(1))
    r8_value = (tl.cast($(0), tl.int64) + tl.full([], $(256), dtype=tl.int64))
    r8_value = tl.broadcast_to(r8_value, [$(1)])
    tl.store($(candidate_vadd_1_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(1))
    r9_value = (tl.load($(candidate_vadd_1_ptr) + $(0), dtype=tl.int64) - tl.full([], $(1), dtype=tl.int64))
    r9_value = tl.broadcast_to(r9_value, [$(1)])
    tl.store($(candidate_vsub_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(256))
    r10_value = (tl.cast(r10_z, tl.int64) > tl.load($(candidate_vsub_ptr) + $(0), dtype=tl.int64))
    r10_value = tl.broadcast_to(r10_value, [$(256)])
    tl.store($(candidate_vgt_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(2048))
    r11_value = tl.where(tl.load($(candidate_vgt_ptr) + (r11_z % $(256)), dtype=tl.int1), -inf, tl.load($(candidate_vmul_ptr) + ((((r11_z / $(256)) % $(8)) * $(256)) + (r11_z % $(256))), dtype=tl.float32))
    r11_value = r11_value.to(tl.float32)
    r11_value = tl.broadcast_to(r11_value, [$(2048)])
    tl.store($(candidate_vmasked_fill_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(2048))
    r12_lane = tl.arange($(256))
    r12_row_values = tl.load($(candidate_vmasked_fill_ptr) + ((((r12_z[:, None] / $(256)) % $(8)) * $(256)) + (r12_lane[None, :] % $(256))), dtype=tl.float32)
    r12_maximum = tl.max(r12_row_values, axis=1, keep_dims=true)
    r12_weights = tl.exp(r12_row_values - r12_maximum)
    r12_total = tl.sum(r12_weights, axis=1)
    r12_maximum = tl.max(r12_row_values, axis=1)
    r12_value = tl.exp(tl.load($(candidate_vmasked_fill_ptr) + ((((r12_z / $(256)) % $(8)) * $(256)) + (r12_z % $(256))), dtype=tl.float32) - r12_maximum) / r12_total
    r12_value = r12_value.to(tl.float32)
    r12_value = tl.broadcast_to(r12_value, [$(2048)])
    tl.store($(candidate_vsoftmax_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(2048))
    r13_value = tl.load($(candidate_vsoftmax_ptr) + ((((r13_z / $(256)) % $(8)) * $(256)) + (r13_z % $(256))), dtype=tl.float32)
    r13_value = r13_value.to(tl.bfloat16)
    r13_value = tl.broadcast_to(r13_value, [$(2048)])
    tl.store($(candidate_vto_copy_4_ptr) + r13_z, r13_value)
    r14_row = tl.arange($(8))
    r14_col = tl.arange($(512))
    r14_lane = tl.arange($(256))
    r14_left = tl.load($(candidate_vto_copy_4_ptr) + ((((((r14_row[:, None] * $(256)) + r14_lane[None, :]) / $(256)) % $(8)) * $(256)) + (((r14_row[:, None] * $(256)) + r14_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r14_right = tl.load($(input3_ptr) + ((((((r14_lane[:, None] * $(512)) + r14_col[None, :]) / $(512)) % $(256)) * $(512)) + (((r14_lane[:, None] * $(512)) + r14_col[None, :]) % $(512))), dtype=tl.bfloat16)
    r14_value = tl.dot(r14_left, r14_right)
    r14_value = r14_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_ptr) + ((r14_row[:, None] * $(512)) + r14_col[None, :]), r14_value)
    r15_z = tl.arange($(512))
    r15_value = tl.load($(input0_ptr) + ((((r15_z / $(64)) % $(8)) * $(64)) + (r15_z % $(64)) + $(512)), dtype=tl.bfloat16)
    r15_value = r15_value.to(tl.float32)
    r15_value = tl.broadcast_to(r15_value, [$(512)])
    tl.store($(candidate_vto_copy_5_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(16384))
    r16_value = tl.load($(input1_ptr) + ((((r16_z / $(64)) % $(256)) * $(64)) + (r16_z % $(64)) + $(16384)), dtype=tl.bfloat16)
    r16_value = r16_value.to(tl.float32)
    r16_value = tl.broadcast_to(r16_value, [$(16384)])
    tl.store($(candidate_vto_copy_6_ptr) + r16_z, r16_value)
    r17_row = tl.arange($(1))
    r17_col = tl.arange($(256))
    r17_lane = tl.arange($(64))
    for r17_group in range($(8)) {
      r17_left = tl.load($(candidate_vto_copy_5_ptr) + (((((((r17_group % $(8)) * $(64)) + (r17_row[:, None] * $(64)) + r17_lane[None, :]) / $(64)) % $(8)) * $(64)) + ((((r17_group % $(8)) * $(64)) + (r17_row[:, None] * $(64)) + r17_lane[None, :]) % $(64))), dtype=tl.float32)
      r17_right = tl.load($(candidate_vto_copy_6_ptr) + ((((((r17_group % $(8)) * $(16384)) + (r17_lane[:, None] * $(256)) + r17_col[None, :]) % $(256)) * $(64)) + (((((r17_group % $(8)) * $(16384)) + (r17_lane[:, None] * $(256)) + r17_col[None, :]) / $(256)) % $(64))), dtype=tl.float32)
      r17_value = tl.dot(r17_left, r17_right)
      r17_value = r17_value.to(tl.float32)
      tl.store($(candidate_vbmm_2_ptr) + ((r17_group * $(256)) + (r17_row[:, None] * $(256)) + r17_col[None, :]), r17_value)
    }
    r18_z = tl.arange($(4096))
    r18_value = tl.load($(input2_ptr) + ((((r18_z / $(512)) % $(8)) * $(512)) + (r18_z % $(512)) + $(4096)), dtype=tl.bfloat16)
    r18_value = r18_value.to(tl.float32)
    r18_value = tl.broadcast_to(r18_value, [$(4096)])
    tl.store($(candidate_vto_copy_7_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(131072))
    r19_value = tl.load($(input3_ptr) + ((((r19_z / $(512)) % $(256)) * $(512)) + (r19_z % $(512)) + $(131072)), dtype=tl.bfloat16)
    r19_value = r19_value.to(tl.float32)
    r19_value = tl.broadcast_to(r19_value, [$(131072)])
    tl.store($(candidate_vto_copy_8_ptr) + r19_z, r19_value)
    r20_row = tl.arange($(1))
    r20_col = tl.arange($(256))
    r20_lane = tl.arange($(512))
    for r20_group in range($(8)) {
      r20_left = tl.load($(candidate_vto_copy_7_ptr) + (((((((r20_group % $(8)) * $(512)) + (r20_row[:, None] * $(512)) + r20_lane[None, :]) / $(512)) % $(8)) * $(512)) + ((((r20_group % $(8)) * $(512)) + (r20_row[:, None] * $(512)) + r20_lane[None, :]) % $(512))), dtype=tl.float32)
      r20_right = tl.load($(candidate_vto_copy_8_ptr) + ((((((r20_group % $(8)) * $(131072)) + (r20_lane[:, None] * $(256)) + r20_col[None, :]) % $(256)) * $(512)) + (((((r20_group % $(8)) * $(131072)) + (r20_lane[:, None] * $(256)) + r20_col[None, :]) / $(256)) % $(512))), dtype=tl.float32)
      r20_value = tl.dot(r20_left, r20_right)
      r20_value = r20_value.to(tl.float32)
      tl.store($(candidate_vbmm_3_ptr) + ((r20_group * $(256)) + (r20_row[:, None] * $(256)) + r20_col[None, :]), r20_value)
    }
    r21_z = tl.arange($(2048))
    r21_value = (tl.load($(candidate_vbmm_2_ptr) + ((((r21_z / $(256)) % $(8)) * $(256)) + (r21_z % $(256))), dtype=tl.float32) + tl.load($(candidate_vbmm_3_ptr) + ((((r21_z / $(256)) % $(8)) * $(256)) + (r21_z % $(256))), dtype=tl.float32))
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(2048)])
    tl.store($(candidate_vadd_2_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(2048))
    r22_value = (tl.load($(candidate_vadd_2_ptr) + ((((r22_z / $(256)) % $(8)) * $(256)) + (r22_z % $(256))), dtype=tl.float32) * 0.041666666666666664)
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(2048)])
    tl.store($(candidate_vmul_1_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(1))
    r23_value = (tl.cast($(0), tl.int64) + tl.full([], $(256), dtype=tl.int64))
    r23_value = tl.broadcast_to(r23_value, [$(1)])
    tl.store($(candidate_vadd_3_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(1))
    r24_value = (tl.load($(candidate_vadd_3_ptr) + $(0), dtype=tl.int64) - tl.full([], $(1), dtype=tl.int64))
    r24_value = tl.broadcast_to(r24_value, [$(1)])
    tl.store($(candidate_vsub_1_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(256))
    r25_value = (tl.cast(r25_z, tl.int64) > tl.load($(candidate_vsub_1_ptr) + $(0), dtype=tl.int64))
    r25_value = tl.broadcast_to(r25_value, [$(256)])
    tl.store($(candidate_vgt_1_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(2048))
    r26_value = tl.where(tl.load($(candidate_vgt_1_ptr) + (r26_z % $(256)), dtype=tl.int1), -inf, tl.load($(candidate_vmul_1_ptr) + ((((r26_z / $(256)) % $(8)) * $(256)) + (r26_z % $(256))), dtype=tl.float32))
    r26_value = r26_value.to(tl.float32)
    r26_value = tl.broadcast_to(r26_value, [$(2048)])
    tl.store($(candidate_vmasked_fill_1_ptr) + r26_z, r26_value)
    r27_z = tl.arange($(2048))
    r27_lane = tl.arange($(256))
    r27_row_values = tl.load($(candidate_vmasked_fill_1_ptr) + ((((r27_z[:, None] / $(256)) % $(8)) * $(256)) + (r27_lane[None, :] % $(256))), dtype=tl.float32)
    r27_maximum = tl.max(r27_row_values, axis=1, keep_dims=true)
    r27_weights = tl.exp(r27_row_values - r27_maximum)
    r27_total = tl.sum(r27_weights, axis=1)
    r27_maximum = tl.max(r27_row_values, axis=1)
    r27_value = tl.exp(tl.load($(candidate_vmasked_fill_1_ptr) + ((((r27_z / $(256)) % $(8)) * $(256)) + (r27_z % $(256))), dtype=tl.float32) - r27_maximum) / r27_total
    r27_value = r27_value.to(tl.float32)
    r27_value = tl.broadcast_to(r27_value, [$(2048)])
    tl.store($(candidate_vsoftmax_1_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(2048))
    r28_value = tl.load($(candidate_vsoftmax_1_ptr) + ((((r28_z / $(256)) % $(8)) * $(256)) + (r28_z % $(256))), dtype=tl.float32)
    r28_value = r28_value.to(tl.bfloat16)
    r28_value = tl.broadcast_to(r28_value, [$(2048)])
    tl.store($(candidate_vto_copy_9_ptr) + r28_z, r28_value)
    r29_row = tl.arange($(8))
    r29_col = tl.arange($(512))
    r29_lane = tl.arange($(256))
    r29_left = tl.load($(candidate_vto_copy_9_ptr) + ((((((r29_row[:, None] * $(256)) + r29_lane[None, :]) / $(256)) % $(8)) * $(256)) + (((r29_row[:, None] * $(256)) + r29_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r29_right = tl.load($(input3_ptr) + ((((((r29_lane[:, None] * $(512)) + r29_col[None, :]) / $(512)) % $(256)) * $(512)) + (((r29_lane[:, None] * $(512)) + r29_col[None, :]) % $(512)) + $(131072)), dtype=tl.bfloat16)
    r29_value = tl.dot(r29_left, r29_right)
    r29_value = r29_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_1_ptr) + ((r29_row[:, None] * $(512)) + r29_col[None, :]), r29_value)
    r30_z = tl.arange($(512))
    r30_value = tl.load($(input0_ptr) + ((((r30_z / $(64)) % $(8)) * $(64)) + (r30_z % $(64)) + $(1024)), dtype=tl.bfloat16)
    r30_value = r30_value.to(tl.float32)
    r30_value = tl.broadcast_to(r30_value, [$(512)])
    tl.store($(candidate_vto_copy_10_ptr) + r30_z, r30_value)
    r31_z = tl.arange($(16384))
    r31_value = tl.load($(input1_ptr) + ((((r31_z / $(64)) % $(256)) * $(64)) + (r31_z % $(64)) + $(32768)), dtype=tl.bfloat16)
    r31_value = r31_value.to(tl.float32)
    r31_value = tl.broadcast_to(r31_value, [$(16384)])
    tl.store($(candidate_vto_copy_11_ptr) + r31_z, r31_value)
    r32_row = tl.arange($(1))
    r32_col = tl.arange($(256))
    r32_lane = tl.arange($(64))
    for r32_group in range($(8)) {
      r32_left = tl.load($(candidate_vto_copy_10_ptr) + (((((((r32_group % $(8)) * $(64)) + (r32_row[:, None] * $(64)) + r32_lane[None, :]) / $(64)) % $(8)) * $(64)) + ((((r32_group % $(8)) * $(64)) + (r32_row[:, None] * $(64)) + r32_lane[None, :]) % $(64))), dtype=tl.float32)
      r32_right = tl.load($(candidate_vto_copy_11_ptr) + ((((((r32_group % $(8)) * $(16384)) + (r32_lane[:, None] * $(256)) + r32_col[None, :]) % $(256)) * $(64)) + (((((r32_group % $(8)) * $(16384)) + (r32_lane[:, None] * $(256)) + r32_col[None, :]) / $(256)) % $(64))), dtype=tl.float32)
      r32_value = tl.dot(r32_left, r32_right)
      r32_value = r32_value.to(tl.float32)
      tl.store($(candidate_vbmm_4_ptr) + ((r32_group * $(256)) + (r32_row[:, None] * $(256)) + r32_col[None, :]), r32_value)
    }
    r33_z = tl.arange($(4096))
    r33_value = tl.load($(input2_ptr) + ((((r33_z / $(512)) % $(8)) * $(512)) + (r33_z % $(512)) + $(8192)), dtype=tl.bfloat16)
    r33_value = r33_value.to(tl.float32)
    r33_value = tl.broadcast_to(r33_value, [$(4096)])
    tl.store($(candidate_vto_copy_12_ptr) + r33_z, r33_value)
    r34_z = tl.arange($(131072))
    r34_value = tl.load($(input3_ptr) + ((((r34_z / $(512)) % $(256)) * $(512)) + (r34_z % $(512)) + $(262144)), dtype=tl.bfloat16)
    r34_value = r34_value.to(tl.float32)
    r34_value = tl.broadcast_to(r34_value, [$(131072)])
    tl.store($(candidate_vto_copy_13_ptr) + r34_z, r34_value)
    r35_row = tl.arange($(1))
    r35_col = tl.arange($(256))
    r35_lane = tl.arange($(512))
    for r35_group in range($(8)) {
      r35_left = tl.load($(candidate_vto_copy_12_ptr) + (((((((r35_group % $(8)) * $(512)) + (r35_row[:, None] * $(512)) + r35_lane[None, :]) / $(512)) % $(8)) * $(512)) + ((((r35_group % $(8)) * $(512)) + (r35_row[:, None] * $(512)) + r35_lane[None, :]) % $(512))), dtype=tl.float32)
      r35_right = tl.load($(candidate_vto_copy_13_ptr) + ((((((r35_group % $(8)) * $(131072)) + (r35_lane[:, None] * $(256)) + r35_col[None, :]) % $(256)) * $(512)) + (((((r35_group % $(8)) * $(131072)) + (r35_lane[:, None] * $(256)) + r35_col[None, :]) / $(256)) % $(512))), dtype=tl.float32)
      r35_value = tl.dot(r35_left, r35_right)
      r35_value = r35_value.to(tl.float32)
      tl.store($(candidate_vbmm_5_ptr) + ((r35_group * $(256)) + (r35_row[:, None] * $(256)) + r35_col[None, :]), r35_value)
    }
    r36_z = tl.arange($(2048))
    r36_value = (tl.load($(candidate_vbmm_4_ptr) + ((((r36_z / $(256)) % $(8)) * $(256)) + (r36_z % $(256))), dtype=tl.float32) + tl.load($(candidate_vbmm_5_ptr) + ((((r36_z / $(256)) % $(8)) * $(256)) + (r36_z % $(256))), dtype=tl.float32))
    r36_value = r36_value.to(tl.float32)
    r36_value = tl.broadcast_to(r36_value, [$(2048)])
    tl.store($(candidate_vadd_4_ptr) + r36_z, r36_value)
    r37_z = tl.arange($(2048))
    r37_value = (tl.load($(candidate_vadd_4_ptr) + ((((r37_z / $(256)) % $(8)) * $(256)) + (r37_z % $(256))), dtype=tl.float32) * 0.041666666666666664)
    r37_value = r37_value.to(tl.float32)
    r37_value = tl.broadcast_to(r37_value, [$(2048)])
    tl.store($(candidate_vmul_2_ptr) + r37_z, r37_value)
    r38_z = tl.arange($(1))
    r38_value = (tl.cast($(0), tl.int64) + tl.full([], $(256), dtype=tl.int64))
    r38_value = tl.broadcast_to(r38_value, [$(1)])
    tl.store($(candidate_vadd_5_ptr) + r38_z, r38_value)
    r39_z = tl.arange($(1))
    r39_value = (tl.load($(candidate_vadd_5_ptr) + $(0), dtype=tl.int64) - tl.full([], $(1), dtype=tl.int64))
    r39_value = tl.broadcast_to(r39_value, [$(1)])
    tl.store($(candidate_vsub_2_ptr) + r39_z, r39_value)
    r40_z = tl.arange($(256))
    r40_value = (tl.cast(r40_z, tl.int64) > tl.load($(candidate_vsub_2_ptr) + $(0), dtype=tl.int64))
    r40_value = tl.broadcast_to(r40_value, [$(256)])
    tl.store($(candidate_vgt_2_ptr) + r40_z, r40_value)
    r41_z = tl.arange($(2048))
    r41_value = tl.where(tl.load($(candidate_vgt_2_ptr) + (r41_z % $(256)), dtype=tl.int1), -inf, tl.load($(candidate_vmul_2_ptr) + ((((r41_z / $(256)) % $(8)) * $(256)) + (r41_z % $(256))), dtype=tl.float32))
    r41_value = r41_value.to(tl.float32)
    r41_value = tl.broadcast_to(r41_value, [$(2048)])
    tl.store($(candidate_vmasked_fill_2_ptr) + r41_z, r41_value)
    r42_z = tl.arange($(2048))
    r42_lane = tl.arange($(256))
    r42_row_values = tl.load($(candidate_vmasked_fill_2_ptr) + ((((r42_z[:, None] / $(256)) % $(8)) * $(256)) + (r42_lane[None, :] % $(256))), dtype=tl.float32)
    r42_maximum = tl.max(r42_row_values, axis=1, keep_dims=true)
    r42_weights = tl.exp(r42_row_values - r42_maximum)
    r42_total = tl.sum(r42_weights, axis=1)
    r42_maximum = tl.max(r42_row_values, axis=1)
    r42_value = tl.exp(tl.load($(candidate_vmasked_fill_2_ptr) + ((((r42_z / $(256)) % $(8)) * $(256)) + (r42_z % $(256))), dtype=tl.float32) - r42_maximum) / r42_total
    r42_value = r42_value.to(tl.float32)
    r42_value = tl.broadcast_to(r42_value, [$(2048)])
    tl.store($(candidate_vsoftmax_2_ptr) + r42_z, r42_value)
    r43_z = tl.arange($(2048))
    r43_value = tl.load($(candidate_vsoftmax_2_ptr) + ((((r43_z / $(256)) % $(8)) * $(256)) + (r43_z % $(256))), dtype=tl.float32)
    r43_value = r43_value.to(tl.bfloat16)
    r43_value = tl.broadcast_to(r43_value, [$(2048)])
    tl.store($(candidate_vto_copy_14_ptr) + r43_z, r43_value)
    r44_row = tl.arange($(8))
    r44_col = tl.arange($(512))
    r44_lane = tl.arange($(256))
    r44_left = tl.load($(candidate_vto_copy_14_ptr) + ((((((r44_row[:, None] * $(256)) + r44_lane[None, :]) / $(256)) % $(8)) * $(256)) + (((r44_row[:, None] * $(256)) + r44_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r44_right = tl.load($(input3_ptr) + ((((((r44_lane[:, None] * $(512)) + r44_col[None, :]) / $(512)) % $(256)) * $(512)) + (((r44_lane[:, None] * $(512)) + r44_col[None, :]) % $(512)) + $(262144)), dtype=tl.bfloat16)
    r44_value = tl.dot(r44_left, r44_right)
    r44_value = r44_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_2_ptr) + ((r44_row[:, None] * $(512)) + r44_col[None, :]), r44_value)
    r45_z = tl.arange($(512))
    r45_value = tl.load($(input0_ptr) + ((((r45_z / $(64)) % $(8)) * $(64)) + (r45_z % $(64)) + $(1536)), dtype=tl.bfloat16)
    r45_value = r45_value.to(tl.float32)
    r45_value = tl.broadcast_to(r45_value, [$(512)])
    tl.store($(candidate_vto_copy_15_ptr) + r45_z, r45_value)
    r46_z = tl.arange($(16384))
    r46_value = tl.load($(input1_ptr) + ((((r46_z / $(64)) % $(256)) * $(64)) + (r46_z % $(64)) + $(49152)), dtype=tl.bfloat16)
    r46_value = r46_value.to(tl.float32)
    r46_value = tl.broadcast_to(r46_value, [$(16384)])
    tl.store($(candidate_vto_copy_16_ptr) + r46_z, r46_value)
    r47_row = tl.arange($(1))
    r47_col = tl.arange($(256))
    r47_lane = tl.arange($(64))
    for r47_group in range($(8)) {
      r47_left = tl.load($(candidate_vto_copy_15_ptr) + (((((((r47_group % $(8)) * $(64)) + (r47_row[:, None] * $(64)) + r47_lane[None, :]) / $(64)) % $(8)) * $(64)) + ((((r47_group % $(8)) * $(64)) + (r47_row[:, None] * $(64)) + r47_lane[None, :]) % $(64))), dtype=tl.float32)
      r47_right = tl.load($(candidate_vto_copy_16_ptr) + ((((((r47_group % $(8)) * $(16384)) + (r47_lane[:, None] * $(256)) + r47_col[None, :]) % $(256)) * $(64)) + (((((r47_group % $(8)) * $(16384)) + (r47_lane[:, None] * $(256)) + r47_col[None, :]) / $(256)) % $(64))), dtype=tl.float32)
      r47_value = tl.dot(r47_left, r47_right)
      r47_value = r47_value.to(tl.float32)
      tl.store($(candidate_vbmm_6_ptr) + ((r47_group * $(256)) + (r47_row[:, None] * $(256)) + r47_col[None, :]), r47_value)
    }
    r48_z = tl.arange($(4096))
    r48_value = tl.load($(input2_ptr) + ((((r48_z / $(512)) % $(8)) * $(512)) + (r48_z % $(512)) + $(12288)), dtype=tl.bfloat16)
    r48_value = r48_value.to(tl.float32)
    r48_value = tl.broadcast_to(r48_value, [$(4096)])
    tl.store($(candidate_vto_copy_17_ptr) + r48_z, r48_value)
    r49_z = tl.arange($(131072))
    r49_value = tl.load($(input3_ptr) + ((((r49_z / $(512)) % $(256)) * $(512)) + (r49_z % $(512)) + $(393216)), dtype=tl.bfloat16)
    r49_value = r49_value.to(tl.float32)
    r49_value = tl.broadcast_to(r49_value, [$(131072)])
    tl.store($(candidate_vto_copy_18_ptr) + r49_z, r49_value)
    r50_row = tl.arange($(1))
    r50_col = tl.arange($(256))
    r50_lane = tl.arange($(512))
    for r50_group in range($(8)) {
      r50_left = tl.load($(candidate_vto_copy_17_ptr) + (((((((r50_group % $(8)) * $(512)) + (r50_row[:, None] * $(512)) + r50_lane[None, :]) / $(512)) % $(8)) * $(512)) + ((((r50_group % $(8)) * $(512)) + (r50_row[:, None] * $(512)) + r50_lane[None, :]) % $(512))), dtype=tl.float32)
      r50_right = tl.load($(candidate_vto_copy_18_ptr) + ((((((r50_group % $(8)) * $(131072)) + (r50_lane[:, None] * $(256)) + r50_col[None, :]) % $(256)) * $(512)) + (((((r50_group % $(8)) * $(131072)) + (r50_lane[:, None] * $(256)) + r50_col[None, :]) / $(256)) % $(512))), dtype=tl.float32)
      r50_value = tl.dot(r50_left, r50_right)
      r50_value = r50_value.to(tl.float32)
      tl.store($(candidate_vbmm_7_ptr) + ((r50_group * $(256)) + (r50_row[:, None] * $(256)) + r50_col[None, :]), r50_value)
    }
    r51_z = tl.arange($(2048))
    r51_value = (tl.load($(candidate_vbmm_6_ptr) + ((((r51_z / $(256)) % $(8)) * $(256)) + (r51_z % $(256))), dtype=tl.float32) + tl.load($(candidate_vbmm_7_ptr) + ((((r51_z / $(256)) % $(8)) * $(256)) + (r51_z % $(256))), dtype=tl.float32))
    r51_value = r51_value.to(tl.float32)
    r51_value = tl.broadcast_to(r51_value, [$(2048)])
    tl.store($(candidate_vadd_6_ptr) + r51_z, r51_value)
    r52_z = tl.arange($(2048))
    r52_value = (tl.load($(candidate_vadd_6_ptr) + ((((r52_z / $(256)) % $(8)) * $(256)) + (r52_z % $(256))), dtype=tl.float32) * 0.041666666666666664)
    r52_value = r52_value.to(tl.float32)
    r52_value = tl.broadcast_to(r52_value, [$(2048)])
    tl.store($(candidate_vmul_3_ptr) + r52_z, r52_value)
    r53_z = tl.arange($(1))
    r53_value = (tl.cast($(0), tl.int64) + tl.full([], $(256), dtype=tl.int64))
    r53_value = tl.broadcast_to(r53_value, [$(1)])
    tl.store($(candidate_vadd_7_ptr) + r53_z, r53_value)
    r54_z = tl.arange($(1))
    r54_value = (tl.load($(candidate_vadd_7_ptr) + $(0), dtype=tl.int64) - tl.full([], $(1), dtype=tl.int64))
    r54_value = tl.broadcast_to(r54_value, [$(1)])
    tl.store($(candidate_vsub_3_ptr) + r54_z, r54_value)
    r55_z = tl.arange($(256))
    r55_value = (tl.cast(r55_z, tl.int64) > tl.load($(candidate_vsub_3_ptr) + $(0), dtype=tl.int64))
    r55_value = tl.broadcast_to(r55_value, [$(256)])
    tl.store($(candidate_vgt_3_ptr) + r55_z, r55_value)
    r56_z = tl.arange($(2048))
    r56_value = tl.where(tl.load($(candidate_vgt_3_ptr) + (r56_z % $(256)), dtype=tl.int1), -inf, tl.load($(candidate_vmul_3_ptr) + ((((r56_z / $(256)) % $(8)) * $(256)) + (r56_z % $(256))), dtype=tl.float32))
    r56_value = r56_value.to(tl.float32)
    r56_value = tl.broadcast_to(r56_value, [$(2048)])
    tl.store($(candidate_vmasked_fill_3_ptr) + r56_z, r56_value)
    r57_z = tl.arange($(2048))
    r57_lane = tl.arange($(256))
    r57_row_values = tl.load($(candidate_vmasked_fill_3_ptr) + ((((r57_z[:, None] / $(256)) % $(8)) * $(256)) + (r57_lane[None, :] % $(256))), dtype=tl.float32)
    r57_maximum = tl.max(r57_row_values, axis=1, keep_dims=true)
    r57_weights = tl.exp(r57_row_values - r57_maximum)
    r57_total = tl.sum(r57_weights, axis=1)
    r57_maximum = tl.max(r57_row_values, axis=1)
    r57_value = tl.exp(tl.load($(candidate_vmasked_fill_3_ptr) + ((((r57_z / $(256)) % $(8)) * $(256)) + (r57_z % $(256))), dtype=tl.float32) - r57_maximum) / r57_total
    r57_value = r57_value.to(tl.float32)
    r57_value = tl.broadcast_to(r57_value, [$(2048)])
    tl.store($(candidate_vsoftmax_3_ptr) + r57_z, r57_value)
    r58_z = tl.arange($(2048))
    r58_value = tl.load($(candidate_vsoftmax_3_ptr) + ((((r58_z / $(256)) % $(8)) * $(256)) + (r58_z % $(256))), dtype=tl.float32)
    r58_value = r58_value.to(tl.bfloat16)
    r58_value = tl.broadcast_to(r58_value, [$(2048)])
    tl.store($(candidate_vto_copy_19_ptr) + r58_z, r58_value)
    r59_row = tl.arange($(8))
    r59_col = tl.arange($(512))
    r59_lane = tl.arange($(256))
    r59_left = tl.load($(candidate_vto_copy_19_ptr) + ((((((r59_row[:, None] * $(256)) + r59_lane[None, :]) / $(256)) % $(8)) * $(256)) + (((r59_row[:, None] * $(256)) + r59_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r59_right = tl.load($(input3_ptr) + ((((((r59_lane[:, None] * $(512)) + r59_col[None, :]) / $(512)) % $(256)) * $(512)) + (((r59_lane[:, None] * $(512)) + r59_col[None, :]) % $(512)) + $(393216)), dtype=tl.bfloat16)
    r59_value = tl.dot(r59_left, r59_right)
    r59_value = r59_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_3_ptr) + ((r59_row[:, None] * $(512)) + r59_col[None, :]), r59_value)
    r60_z = tl.arange($(16384))
    r60_value = tl.where(((r60_z / $(4096)) % $(4)) < $(1), tl.load($(candidate_vmm_ptr) + ((((((r60_z / $(512)) % $(8)) + (((r60_z / $(4096)) % $(4)) - $(0))) % $(8)) * $(512)) + (r60_z % $(512))), dtype=tl.bfloat16), tl.where(((r60_z / $(4096)) % $(4)) < $(2), tl.load($(candidate_vmm_1_ptr) + ((((((r60_z / $(512)) % $(8)) + (((r60_z / $(4096)) % $(4)) - $(1))) % $(8)) * $(512)) + (r60_z % $(512))), dtype=tl.bfloat16), tl.where(((r60_z / $(4096)) % $(4)) < $(3), tl.load($(candidate_vmm_2_ptr) + ((((((r60_z / $(512)) % $(8)) + (((r60_z / $(4096)) % $(4)) - $(2))) % $(8)) * $(512)) + (r60_z % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vmm_3_ptr) + ((((((r60_z / $(512)) % $(8)) + (((r60_z / $(4096)) % $(4)) - $(3))) % $(8)) * $(512)) + (r60_z % $(512))), dtype=tl.bfloat16))))
    r60_value = r60_value.to(tl.bfloat16)
    r60_value = tl.broadcast_to(r60_value, [$(16384)])
    tl.store($(candidate_vcat_ptr) + r60_z, r60_value)
    z = tl.arange($(16384))
    value = tl.load($(candidate_vcat_ptr) + ((((z / $(4096)) % $(4)) * $(4096)) + (((z / $(512)) % $(8)) * $(512)) + (z % $(512))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(16384)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_fa4_mla_decode
