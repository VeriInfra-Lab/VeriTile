import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_fa4_mla_prefill
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
  let vtriu_ptr : RegionName := "vtriu"
  let vmasked_fill_ptr : RegionName := "vmasked_fill"
  let vbmm_ptr : RegionName := "vbmm"
  let vmul_ptr : RegionName := "vmul"
  let vadd_ptr : RegionName := "vadd"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vsoftmax_ptr : RegionName := "vsoftmax"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let vbmm_1_ptr : RegionName := "vbmm_1"
  triton {
    r0_z = tl.arange($(1179648))
    r0_value = tl.where((r0_z % $(576)) < $(64), tl.load($(input0_ptr) + ((((r0_z / $(589824)) % $(2)) * $(65536)) + (((r0_z / $(4608)) % $(128)) * $(512)) + (((r0_z / $(576)) % $(8)) * $(64)) + (((r0_z % $(576)) - $(0)) % $(64))), dtype=tl.bfloat16), tl.load($(input2_ptr) + ((((r0_z / $(589824)) % $(2)) * $(524288)) + (((r0_z / $(4608)) % $(128)) * $(4096)) + (((r0_z / $(576)) % $(8)) * $(512)) + (((r0_z % $(576)) - $(64)) % $(512))), dtype=tl.bfloat16))
    r0_value = r0_value.to(tl.bfloat16)
    r0_value = tl.broadcast_to(r0_value, [$(1179648)])
    tl.store($(vcat_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(147456))
    r1_value = tl.where((r1_z % $(576)) < $(64), tl.load($(input1_ptr) + ((((r1_z / $(73728)) % $(2)) * $(8192)) + (((r1_z / $(576)) % $(128)) * $(64)) + (((r1_z % $(576)) - $(0)) % $(64))), dtype=tl.bfloat16), tl.load($(input3_ptr) + ((((r1_z / $(73728)) % $(2)) * $(65536)) + (((r1_z / $(576)) % $(128)) * $(512)) + (((r1_z % $(576)) - $(64)) % $(512))), dtype=tl.bfloat16))
    r1_value = r1_value.to(tl.bfloat16)
    r1_value = tl.broadcast_to(r1_value, [$(147456)])
    tl.store($(vcat_1_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(16384))
    r2_value = tl.where((r2_z % $(128)) >= ((r2_z / $(128)) % $(128)) + $(1), true, false)
    r2_value = tl.broadcast_to(r2_value, [$(16384)])
    tl.store($(vtriu_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(32768))
    r3_value = tl.where(tl.load($(vtriu_ptr) + ((((r3_z / $(128)) % $(128)) * $(128)) + (r3_z % $(128))), dtype=tl.int1), tl.cast(-338953138925153547590470800371487866880.0, tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r3_value = r3_value.to(tl.bfloat16)
    r3_value = tl.broadcast_to(r3_value, [$(32768)])
    tl.store($(vmasked_fill_ptr) + r3_z, r3_value)
    r4_row = tl.arange($(128))
    r4_col = tl.arange($(128))
    r4_lane = tl.arange($(576))
    for r4_group in range($(16)) {
      r4_left = tl.load($(vcat_ptr) + (((((((r4_group % $(16)) * $(73728)) + (r4_row[:, None] * $(576)) + r4_lane[None, :]) / $(589824)) % $(2)) * $(589824)) + ((((((r4_group % $(16)) * $(73728)) + (r4_row[:, None] * $(576)) + r4_lane[None, :]) / $(576)) % $(128)) * $(4608)) + ((((((r4_group % $(16)) * $(73728)) + (r4_row[:, None] * $(576)) + r4_lane[None, :]) / $(73728)) % $(8)) * $(576)) + ((((r4_group % $(16)) * $(73728)) + (r4_row[:, None] * $(576)) + r4_lane[None, :]) % $(576))), dtype=tl.bfloat16)
      r4_right = tl.load($(vcat_1_ptr) + (((((((r4_group % $(16)) * $(73728)) + (r4_lane[:, None] * $(128)) + r4_col[None, :]) / $(589824)) % $(2)) * $(73728)) + (((((r4_group % $(16)) * $(73728)) + (r4_lane[:, None] * $(128)) + r4_col[None, :]) % $(128)) * $(576)) + (((((r4_group % $(16)) * $(73728)) + (r4_lane[:, None] * $(128)) + r4_col[None, :]) / $(128)) % $(576))), dtype=tl.bfloat16)
      r4_value = tl.dot(r4_left, r4_right)
      r4_value = r4_value.to(tl.bfloat16)
      tl.store($(vbmm_ptr) + ((r4_group * $(16384)) + (r4_row[:, None] * $(128)) + r4_col[None, :]), r4_value)
    }
    r5_z = tl.arange($(262144))
    r5_value = (tl.load($(vbmm_ptr) + ((((r5_z / $(131072)) % $(2)) * $(131072)) + (((r5_z / $(16384)) % $(8)) * $(16384)) + (((r5_z / $(128)) % $(128)) * $(128)) + (r5_z % $(128))), dtype=tl.bfloat16) * tl.cast(0.041666666666666664, tl.bfloat16))
    r5_value = r5_value.to(tl.bfloat16)
    r5_value = tl.broadcast_to(r5_value, [$(262144)])
    tl.store($(vmul_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(262144))
    r6_value = (tl.load($(vmul_ptr) + ((((r6_z / $(131072)) % $(2)) * $(131072)) + (((r6_z / $(16384)) % $(8)) * $(16384)) + (((r6_z / $(128)) % $(128)) * $(128)) + (r6_z % $(128))), dtype=tl.bfloat16) + tl.load($(vmasked_fill_ptr) + ((((r6_z / $(131072)) % $(2)) * $(16384)) + (((r6_z / $(128)) % $(128)) * $(128)) + (r6_z % $(128))), dtype=tl.bfloat16))
    r6_value = r6_value.to(tl.bfloat16)
    r6_value = tl.broadcast_to(r6_value, [$(262144)])
    tl.store($(vadd_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(262144))
    r7_value = tl.load($(vadd_ptr) + ((((r7_z / $(131072)) % $(2)) * $(131072)) + (((r7_z / $(16384)) % $(8)) * $(16384)) + (((r7_z / $(128)) % $(128)) * $(128)) + (r7_z % $(128))), dtype=tl.bfloat16)
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(262144)])
    tl.store($(vto_copy_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(262144))
    r8_lane = tl.arange($(128))
    r8_row_values = tl.load($(vto_copy_ptr) + ((((r8_z[:, None] / $(131072)) % $(2)) * $(131072)) + (((r8_z[:, None] / $(16384)) % $(8)) * $(16384)) + (((r8_z[:, None] / $(128)) % $(128)) * $(128)) + (r8_lane[None, :] % $(128))), dtype=tl.float32)
    r8_maximum = tl.max(r8_row_values, axis=1, keep_dims=true)
    r8_weights = tl.exp(r8_row_values - r8_maximum)
    r8_total = tl.sum(r8_weights, axis=1)
    r8_maximum = tl.max(r8_row_values, axis=1)
    r8_value = tl.exp(tl.load($(vto_copy_ptr) + ((((r8_z / $(131072)) % $(2)) * $(131072)) + (((r8_z / $(16384)) % $(8)) * $(16384)) + (((r8_z / $(128)) % $(128)) * $(128)) + (r8_z % $(128))), dtype=tl.float32) - r8_maximum) / r8_total
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(262144)])
    tl.store($(vsoftmax_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(262144))
    r9_value = tl.load($(vsoftmax_ptr) + ((((r9_z / $(131072)) % $(2)) * $(131072)) + (((r9_z / $(16384)) % $(8)) * $(16384)) + (((r9_z / $(128)) % $(128)) * $(128)) + (r9_z % $(128))), dtype=tl.float32)
    r9_value = r9_value.to(tl.bfloat16)
    r9_value = tl.broadcast_to(r9_value, [$(262144)])
    tl.store($(vto_copy_1_ptr) + r9_z, r9_value)
    r10_row = tl.arange($(128))
    r10_col = tl.arange($(512))
    r10_lane = tl.arange($(128))
    for r10_group in range($(16)) {
      r10_left = tl.load($(vto_copy_1_ptr) + (((((((r10_group % $(16)) * $(16384)) + (r10_row[:, None] * $(128)) + r10_lane[None, :]) / $(131072)) % $(2)) * $(131072)) + ((((((r10_group % $(16)) * $(16384)) + (r10_row[:, None] * $(128)) + r10_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + ((((((r10_group % $(16)) * $(16384)) + (r10_row[:, None] * $(128)) + r10_lane[None, :]) / $(128)) % $(128)) * $(128)) + ((((r10_group % $(16)) * $(16384)) + (r10_row[:, None] * $(128)) + r10_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r10_right = tl.load($(input3_ptr) + (((((((r10_group % $(16)) * $(65536)) + (r10_lane[:, None] * $(512)) + r10_col[None, :]) / $(524288)) % $(2)) * $(65536)) + ((((((r10_group % $(16)) * $(65536)) + (r10_lane[:, None] * $(512)) + r10_col[None, :]) / $(512)) % $(128)) * $(512)) + ((((r10_group % $(16)) * $(65536)) + (r10_lane[:, None] * $(512)) + r10_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r10_value = tl.dot(r10_left, r10_right)
      r10_value = r10_value.to(tl.bfloat16)
      tl.store($(vbmm_1_ptr) + ((r10_group * $(65536)) + (r10_row[:, None] * $(512)) + r10_col[None, :]), r10_value)
    }
    z = tl.arange($(1048576))
    value = tl.load($(vbmm_1_ptr) + ((((z / $(524288)) % $(2)) * $(524288)) + (((z / $(512)) % $(8)) * $(65536)) + (((z / $(4096)) % $(128)) * $(512)) + (z % $(512))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(1048576)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let input1_ptr : RegionName := "input1"
  let input2_ptr : RegionName := "input2"
  let input3_ptr : RegionName := "input3"
  let out_ptr : RegionName := "out"
  let vcat_ptr : RegionName := "vcat"
  let vmasked_fill_ptr : RegionName := "vmasked_fill"
  let vbmm_ptr : RegionName := "vbmm"
  let vmul_ptr : RegionName := "vmul"
  let vadd_ptr : RegionName := "vadd"
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
  let candidate_vcat_ptr : RegionName := "candidate_vcat"
  triton {
    r0_z = tl.arange($(65536))
    r0_value = tl.load($(input0_ptr) + ((((r0_z / $(64)) % $(128)) * $(512)) + (((r0_z / $(8192)) % $(8)) * $(64)) + (r0_z % $(64))), dtype=tl.bfloat16)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(65536)])
    tl.store($(candidate_vto_copy_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(8192))
    r1_value = tl.load($(input1_ptr) + ((((r1_z / $(64)) % $(128)) * $(64)) + (r1_z % $(64))), dtype=tl.bfloat16)
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(8192)])
    tl.store($(candidate_vto_copy_1_ptr) + r1_z, r1_value)
    r2_row = tl.arange($(128))
    r2_col = tl.arange($(128))
    r2_lane = tl.arange($(64))
    for r2_group in range($(8)) {
      r2_left = tl.load($(candidate_vto_copy_ptr) + (((((((r2_group % $(8)) * $(8192)) + (r2_row[:, None] * $(64)) + r2_lane[None, :]) / $(8192)) % $(8)) * $(8192)) + ((((((r2_group % $(8)) * $(8192)) + (r2_row[:, None] * $(64)) + r2_lane[None, :]) / $(64)) % $(128)) * $(64)) + ((((r2_group % $(8)) * $(8192)) + (r2_row[:, None] * $(64)) + r2_lane[None, :]) % $(64))), dtype=tl.float32)
      r2_right = tl.load($(candidate_vto_copy_1_ptr) + ((((((r2_group % $(8)) * $(8192)) + (r2_lane[:, None] * $(128)) + r2_col[None, :]) % $(128)) * $(64)) + (((((r2_group % $(8)) * $(8192)) + (r2_lane[:, None] * $(128)) + r2_col[None, :]) / $(128)) % $(64))), dtype=tl.float32)
      r2_value = tl.dot(r2_left, r2_right)
      r2_value = r2_value.to(tl.float32)
      tl.store($(candidate_vbmm_ptr) + ((r2_group * $(16384)) + (r2_row[:, None] * $(128)) + r2_col[None, :]), r2_value)
    }
    r3_z = tl.arange($(524288))
    r3_value = tl.load($(input2_ptr) + ((((r3_z / $(512)) % $(128)) * $(4096)) + (((r3_z / $(65536)) % $(8)) * $(512)) + (r3_z % $(512))), dtype=tl.bfloat16)
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(524288)])
    tl.store($(candidate_vto_copy_2_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(65536))
    r4_value = tl.load($(input3_ptr) + ((((r4_z / $(512)) % $(128)) * $(512)) + (r4_z % $(512))), dtype=tl.bfloat16)
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(65536)])
    tl.store($(candidate_vto_copy_3_ptr) + r4_z, r4_value)
    r5_row = tl.arange($(128))
    r5_col = tl.arange($(128))
    r5_lane = tl.arange($(512))
    for r5_group in range($(8)) {
      r5_left = tl.load($(candidate_vto_copy_2_ptr) + (((((((r5_group % $(8)) * $(65536)) + (r5_row[:, None] * $(512)) + r5_lane[None, :]) / $(65536)) % $(8)) * $(65536)) + ((((((r5_group % $(8)) * $(65536)) + (r5_row[:, None] * $(512)) + r5_lane[None, :]) / $(512)) % $(128)) * $(512)) + ((((r5_group % $(8)) * $(65536)) + (r5_row[:, None] * $(512)) + r5_lane[None, :]) % $(512))), dtype=tl.float32)
      r5_right = tl.load($(candidate_vto_copy_3_ptr) + ((((((r5_group % $(8)) * $(65536)) + (r5_lane[:, None] * $(128)) + r5_col[None, :]) % $(128)) * $(512)) + (((((r5_group % $(8)) * $(65536)) + (r5_lane[:, None] * $(128)) + r5_col[None, :]) / $(128)) % $(512))), dtype=tl.float32)
      r5_value = tl.dot(r5_left, r5_right)
      r5_value = r5_value.to(tl.float32)
      tl.store($(candidate_vbmm_1_ptr) + ((r5_group * $(16384)) + (r5_row[:, None] * $(128)) + r5_col[None, :]), r5_value)
    }
    r6_z = tl.arange($(131072))
    r6_value = (tl.load($(candidate_vbmm_ptr) + ((((r6_z / $(16384)) % $(8)) * $(16384)) + (((r6_z / $(128)) % $(128)) * $(128)) + (r6_z % $(128))), dtype=tl.float32) + tl.load($(candidate_vbmm_1_ptr) + ((((r6_z / $(16384)) % $(8)) * $(16384)) + (((r6_z / $(128)) % $(128)) * $(128)) + (r6_z % $(128))), dtype=tl.float32))
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(131072)])
    tl.store($(candidate_vadd_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(131072))
    r7_value = (tl.load($(candidate_vadd_ptr) + ((((r7_z / $(16384)) % $(8)) * $(16384)) + (((r7_z / $(128)) % $(128)) * $(128)) + (r7_z % $(128))), dtype=tl.float32) * 0.041666666666666664)
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(131072)])
    tl.store($(candidate_vmul_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(128))
    r8_value = (tl.cast(r8_z, tl.int64) + tl.full([], $(128), dtype=tl.int64))
    r8_value = tl.broadcast_to(r8_value, [$(128)])
    tl.store($(candidate_vadd_1_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(128))
    r9_value = (tl.load($(candidate_vadd_1_ptr) + r9_z, dtype=tl.int64) - tl.full([], $(128), dtype=tl.int64))
    r9_value = tl.broadcast_to(r9_value, [$(128)])
    tl.store($(candidate_vsub_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(16384))
    r10_value = (tl.cast((r10_z % $(128)), tl.int64) > tl.load($(candidate_vsub_ptr) + ((r10_z / $(128)) % $(128)), dtype=tl.int64))
    r10_value = tl.broadcast_to(r10_value, [$(16384)])
    tl.store($(candidate_vgt_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(131072))
    r11_value = tl.where(tl.load($(candidate_vgt_ptr) + ((((r11_z / $(128)) % $(128)) * $(128)) + (r11_z % $(128))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_ptr) + ((((r11_z / $(16384)) % $(8)) * $(16384)) + (((r11_z / $(128)) % $(128)) * $(128)) + (r11_z % $(128))), dtype=tl.float32))
    r11_value = r11_value.to(tl.float32)
    r11_value = tl.broadcast_to(r11_value, [$(131072)])
    tl.store($(candidate_vmasked_fill_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(131072))
    r12_lane = tl.arange($(128))
    r12_row_values = tl.load($(candidate_vmasked_fill_ptr) + ((((r12_z[:, None] / $(16384)) % $(8)) * $(16384)) + (((r12_z[:, None] / $(128)) % $(128)) * $(128)) + (r12_lane[None, :] % $(128))), dtype=tl.float32)
    r12_maximum = tl.max(r12_row_values, axis=1, keep_dims=true)
    r12_weights = tl.exp(r12_row_values - r12_maximum)
    r12_total = tl.sum(r12_weights, axis=1)
    r12_maximum = tl.max(r12_row_values, axis=1)
    r12_value = tl.exp(tl.load($(candidate_vmasked_fill_ptr) + ((((r12_z / $(16384)) % $(8)) * $(16384)) + (((r12_z / $(128)) % $(128)) * $(128)) + (r12_z % $(128))), dtype=tl.float32) - r12_maximum) / r12_total
    r12_value = r12_value.to(tl.float32)
    r12_value = tl.broadcast_to(r12_value, [$(131072)])
    tl.store($(candidate_vsoftmax_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(131072))
    r13_value = tl.load($(candidate_vsoftmax_ptr) + ((((r13_z / $(16384)) % $(8)) * $(16384)) + (((r13_z / $(128)) % $(128)) * $(128)) + (r13_z % $(128))), dtype=tl.float32)
    r13_value = r13_value.to(tl.bfloat16)
    r13_value = tl.broadcast_to(r13_value, [$(131072)])
    tl.store($(candidate_vto_copy_4_ptr) + r13_z, r13_value)
    r14_row = tl.arange($(1024))
    r14_col = tl.arange($(512))
    r14_lane = tl.arange($(128))
    r14_left = tl.load($(candidate_vto_copy_4_ptr) + ((((((r14_row[:, None] * $(128)) + r14_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + (((((r14_row[:, None] * $(128)) + r14_lane[None, :]) / $(128)) % $(128)) * $(128)) + (((r14_row[:, None] * $(128)) + r14_lane[None, :]) % $(128))), dtype=tl.bfloat16)
    r14_right = tl.load($(input3_ptr) + ((((((r14_lane[:, None] * $(512)) + r14_col[None, :]) / $(512)) % $(128)) * $(512)) + (((r14_lane[:, None] * $(512)) + r14_col[None, :]) % $(512))), dtype=tl.bfloat16)
    r14_value = tl.dot(r14_left, r14_right)
    r14_value = r14_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_ptr) + ((r14_row[:, None] * $(512)) + r14_col[None, :]), r14_value)
    r15_z = tl.arange($(65536))
    r15_value = tl.load($(input0_ptr) + ((((r15_z / $(64)) % $(128)) * $(512)) + (((r15_z / $(8192)) % $(8)) * $(64)) + (r15_z % $(64)) + $(65536)), dtype=tl.bfloat16)
    r15_value = r15_value.to(tl.float32)
    r15_value = tl.broadcast_to(r15_value, [$(65536)])
    tl.store($(candidate_vto_copy_5_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(8192))
    r16_value = tl.load($(input1_ptr) + ((((r16_z / $(64)) % $(128)) * $(64)) + (r16_z % $(64)) + $(8192)), dtype=tl.bfloat16)
    r16_value = r16_value.to(tl.float32)
    r16_value = tl.broadcast_to(r16_value, [$(8192)])
    tl.store($(candidate_vto_copy_6_ptr) + r16_z, r16_value)
    r17_row = tl.arange($(128))
    r17_col = tl.arange($(128))
    r17_lane = tl.arange($(64))
    for r17_group in range($(8)) {
      r17_left = tl.load($(candidate_vto_copy_5_ptr) + (((((((r17_group % $(8)) * $(8192)) + (r17_row[:, None] * $(64)) + r17_lane[None, :]) / $(8192)) % $(8)) * $(8192)) + ((((((r17_group % $(8)) * $(8192)) + (r17_row[:, None] * $(64)) + r17_lane[None, :]) / $(64)) % $(128)) * $(64)) + ((((r17_group % $(8)) * $(8192)) + (r17_row[:, None] * $(64)) + r17_lane[None, :]) % $(64))), dtype=tl.float32)
      r17_right = tl.load($(candidate_vto_copy_6_ptr) + ((((((r17_group % $(8)) * $(8192)) + (r17_lane[:, None] * $(128)) + r17_col[None, :]) % $(128)) * $(64)) + (((((r17_group % $(8)) * $(8192)) + (r17_lane[:, None] * $(128)) + r17_col[None, :]) / $(128)) % $(64))), dtype=tl.float32)
      r17_value = tl.dot(r17_left, r17_right)
      r17_value = r17_value.to(tl.float32)
      tl.store($(candidate_vbmm_2_ptr) + ((r17_group * $(16384)) + (r17_row[:, None] * $(128)) + r17_col[None, :]), r17_value)
    }
    r18_z = tl.arange($(524288))
    r18_value = tl.load($(input2_ptr) + ((((r18_z / $(512)) % $(128)) * $(4096)) + (((r18_z / $(65536)) % $(8)) * $(512)) + (r18_z % $(512)) + $(524288)), dtype=tl.bfloat16)
    r18_value = r18_value.to(tl.float32)
    r18_value = tl.broadcast_to(r18_value, [$(524288)])
    tl.store($(candidate_vto_copy_7_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(65536))
    r19_value = tl.load($(input3_ptr) + ((((r19_z / $(512)) % $(128)) * $(512)) + (r19_z % $(512)) + $(65536)), dtype=tl.bfloat16)
    r19_value = r19_value.to(tl.float32)
    r19_value = tl.broadcast_to(r19_value, [$(65536)])
    tl.store($(candidate_vto_copy_8_ptr) + r19_z, r19_value)
    r20_row = tl.arange($(128))
    r20_col = tl.arange($(128))
    r20_lane = tl.arange($(512))
    for r20_group in range($(8)) {
      r20_left = tl.load($(candidate_vto_copy_7_ptr) + (((((((r20_group % $(8)) * $(65536)) + (r20_row[:, None] * $(512)) + r20_lane[None, :]) / $(65536)) % $(8)) * $(65536)) + ((((((r20_group % $(8)) * $(65536)) + (r20_row[:, None] * $(512)) + r20_lane[None, :]) / $(512)) % $(128)) * $(512)) + ((((r20_group % $(8)) * $(65536)) + (r20_row[:, None] * $(512)) + r20_lane[None, :]) % $(512))), dtype=tl.float32)
      r20_right = tl.load($(candidate_vto_copy_8_ptr) + ((((((r20_group % $(8)) * $(65536)) + (r20_lane[:, None] * $(128)) + r20_col[None, :]) % $(128)) * $(512)) + (((((r20_group % $(8)) * $(65536)) + (r20_lane[:, None] * $(128)) + r20_col[None, :]) / $(128)) % $(512))), dtype=tl.float32)
      r20_value = tl.dot(r20_left, r20_right)
      r20_value = r20_value.to(tl.float32)
      tl.store($(candidate_vbmm_3_ptr) + ((r20_group * $(16384)) + (r20_row[:, None] * $(128)) + r20_col[None, :]), r20_value)
    }
    r21_z = tl.arange($(131072))
    r21_value = (tl.load($(candidate_vbmm_2_ptr) + ((((r21_z / $(16384)) % $(8)) * $(16384)) + (((r21_z / $(128)) % $(128)) * $(128)) + (r21_z % $(128))), dtype=tl.float32) + tl.load($(candidate_vbmm_3_ptr) + ((((r21_z / $(16384)) % $(8)) * $(16384)) + (((r21_z / $(128)) % $(128)) * $(128)) + (r21_z % $(128))), dtype=tl.float32))
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(131072)])
    tl.store($(candidate_vadd_2_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(131072))
    r22_value = (tl.load($(candidate_vadd_2_ptr) + ((((r22_z / $(16384)) % $(8)) * $(16384)) + (((r22_z / $(128)) % $(128)) * $(128)) + (r22_z % $(128))), dtype=tl.float32) * 0.041666666666666664)
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(131072)])
    tl.store($(candidate_vmul_1_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(128))
    r23_value = (tl.cast(r23_z, tl.int64) + tl.full([], $(128), dtype=tl.int64))
    r23_value = tl.broadcast_to(r23_value, [$(128)])
    tl.store($(candidate_vadd_3_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(128))
    r24_value = (tl.load($(candidate_vadd_3_ptr) + r24_z, dtype=tl.int64) - tl.full([], $(128), dtype=tl.int64))
    r24_value = tl.broadcast_to(r24_value, [$(128)])
    tl.store($(candidate_vsub_1_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(16384))
    r25_value = (tl.cast((r25_z % $(128)), tl.int64) > tl.load($(candidate_vsub_1_ptr) + ((r25_z / $(128)) % $(128)), dtype=tl.int64))
    r25_value = tl.broadcast_to(r25_value, [$(16384)])
    tl.store($(candidate_vgt_1_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(131072))
    r26_value = tl.where(tl.load($(candidate_vgt_1_ptr) + ((((r26_z / $(128)) % $(128)) * $(128)) + (r26_z % $(128))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_1_ptr) + ((((r26_z / $(16384)) % $(8)) * $(16384)) + (((r26_z / $(128)) % $(128)) * $(128)) + (r26_z % $(128))), dtype=tl.float32))
    r26_value = r26_value.to(tl.float32)
    r26_value = tl.broadcast_to(r26_value, [$(131072)])
    tl.store($(candidate_vmasked_fill_1_ptr) + r26_z, r26_value)
    r27_z = tl.arange($(131072))
    r27_lane = tl.arange($(128))
    r27_row_values = tl.load($(candidate_vmasked_fill_1_ptr) + ((((r27_z[:, None] / $(16384)) % $(8)) * $(16384)) + (((r27_z[:, None] / $(128)) % $(128)) * $(128)) + (r27_lane[None, :] % $(128))), dtype=tl.float32)
    r27_maximum = tl.max(r27_row_values, axis=1, keep_dims=true)
    r27_weights = tl.exp(r27_row_values - r27_maximum)
    r27_total = tl.sum(r27_weights, axis=1)
    r27_maximum = tl.max(r27_row_values, axis=1)
    r27_value = tl.exp(tl.load($(candidate_vmasked_fill_1_ptr) + ((((r27_z / $(16384)) % $(8)) * $(16384)) + (((r27_z / $(128)) % $(128)) * $(128)) + (r27_z % $(128))), dtype=tl.float32) - r27_maximum) / r27_total
    r27_value = r27_value.to(tl.float32)
    r27_value = tl.broadcast_to(r27_value, [$(131072)])
    tl.store($(candidate_vsoftmax_1_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(131072))
    r28_value = tl.load($(candidate_vsoftmax_1_ptr) + ((((r28_z / $(16384)) % $(8)) * $(16384)) + (((r28_z / $(128)) % $(128)) * $(128)) + (r28_z % $(128))), dtype=tl.float32)
    r28_value = r28_value.to(tl.bfloat16)
    r28_value = tl.broadcast_to(r28_value, [$(131072)])
    tl.store($(candidate_vto_copy_9_ptr) + r28_z, r28_value)
    r29_row = tl.arange($(1024))
    r29_col = tl.arange($(512))
    r29_lane = tl.arange($(128))
    r29_left = tl.load($(candidate_vto_copy_9_ptr) + ((((((r29_row[:, None] * $(128)) + r29_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + (((((r29_row[:, None] * $(128)) + r29_lane[None, :]) / $(128)) % $(128)) * $(128)) + (((r29_row[:, None] * $(128)) + r29_lane[None, :]) % $(128))), dtype=tl.bfloat16)
    r29_right = tl.load($(input3_ptr) + ((((((r29_lane[:, None] * $(512)) + r29_col[None, :]) / $(512)) % $(128)) * $(512)) + (((r29_lane[:, None] * $(512)) + r29_col[None, :]) % $(512)) + $(65536)), dtype=tl.bfloat16)
    r29_value = tl.dot(r29_left, r29_right)
    r29_value = r29_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_1_ptr) + ((r29_row[:, None] * $(512)) + r29_col[None, :]), r29_value)
    r30_z = tl.arange($(1048576))
    r30_value = tl.where(((r30_z / $(4096)) % $(256)) < $(128), tl.load($(candidate_vmm_ptr) + (((((((r30_z / $(512)) % $(8)) * $(128)) + (((r30_z / $(4096)) % $(256)) - $(0))) % $(1024)) * $(512)) + (r30_z % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vmm_1_ptr) + (((((((r30_z / $(512)) % $(8)) * $(128)) + (((r30_z / $(4096)) % $(256)) - $(128))) % $(1024)) * $(512)) + (r30_z % $(512))), dtype=tl.bfloat16))
    r30_value = r30_value.to(tl.bfloat16)
    r30_value = tl.broadcast_to(r30_value, [$(1048576)])
    tl.store($(candidate_vcat_ptr) + r30_z, r30_value)
    z = tl.arange($(1048576))
    value = tl.load($(candidate_vcat_ptr) + ((((z / $(524288)) % $(2)) * $(524288)) + (((z / $(4096)) % $(128)) * $(4096)) + (((z / $(512)) % $(8)) * $(512)) + (z % $(512))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(1048576)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_fa4_mla_prefill
