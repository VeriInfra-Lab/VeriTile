import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.te_fused_attn_mla_dk
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
  let vtriu_ptr : RegionName := "vtriu"
  let vmasked_fill_ptr : RegionName := "vmasked_fill"
  let vbmm_ptr : RegionName := "vbmm"
  let vmul_ptr : RegionName := "vmul"
  let vadd_ptr : RegionName := "vadd"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vsoftmax_ptr : RegionName := "vsoftmax"
  let vbmm_2_ptr : RegionName := "vbmm_2"
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  let vsoftmax_backward_data_ptr : RegionName := "vsoftmax_backward_data"
  let vto_copy_3_ptr : RegionName := "vto_copy_3"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vbmm_3_ptr : RegionName := "vbmm_3"
  triton {
    r0_z = tl.arange($(16384))
    r0_value = tl.where((r0_z % $(128)) >= ((r0_z / $(128)) % $(128)) + $(1), true, false)
    r0_value = tl.broadcast_to(r0_value, [$(16384)])
    tl.store($(vtriu_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(32768))
    r1_value = tl.where(tl.load($(vtriu_ptr) + ((((r1_z / $(128)) % $(128)) * $(128)) + (r1_z % $(128))), dtype=tl.int1), tl.cast(-338953138925153547590470800371487866880.0, tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r1_value = r1_value.to(tl.bfloat16)
    r1_value = tl.broadcast_to(r1_value, [$(32768)])
    tl.store($(vmasked_fill_ptr) + r1_z, r1_value)
    r2_row = tl.arange($(128))
    r2_col = tl.arange($(128))
    r2_lane = tl.arange($(192))
    for r2_group in range($(8)) {
      r2_left = tl.load($(input0_ptr) + (((((((r2_group % $(8)) * $(24576)) + (r2_row[:, None] * $(192)) + r2_lane[None, :]) / $(98304)) % $(2)) * $(98304)) + ((((((r2_group % $(8)) * $(24576)) + (r2_row[:, None] * $(192)) + r2_lane[None, :]) / $(24576)) % $(4)) * $(24576)) + ((((((r2_group % $(8)) * $(24576)) + (r2_row[:, None] * $(192)) + r2_lane[None, :]) / $(192)) % $(128)) * $(192)) + ((((r2_group % $(8)) * $(24576)) + (r2_row[:, None] * $(192)) + r2_lane[None, :]) % $(192))), dtype=tl.bfloat16)
      r2_right = tl.load($(input1_ptr) + (((((((r2_group % $(8)) * $(24576)) + (r2_lane[:, None] * $(128)) + r2_col[None, :]) / $(98304)) % $(2)) * $(98304)) + ((((((r2_group % $(8)) * $(24576)) + (r2_lane[:, None] * $(128)) + r2_col[None, :]) / $(24576)) % $(4)) * $(24576)) + (((((r2_group % $(8)) * $(24576)) + (r2_lane[:, None] * $(128)) + r2_col[None, :]) % $(128)) * $(192)) + (((((r2_group % $(8)) * $(24576)) + (r2_lane[:, None] * $(128)) + r2_col[None, :]) / $(128)) % $(192))), dtype=tl.bfloat16)
      r2_value = tl.dot(r2_left, r2_right)
      r2_value = r2_value.to(tl.bfloat16)
      tl.store($(vbmm_ptr) + ((r2_group * $(16384)) + (r2_row[:, None] * $(128)) + r2_col[None, :]), r2_value)
    }
    r3_z = tl.arange($(131072))
    r3_value = (tl.load($(vbmm_ptr) + ((((r3_z / $(65536)) % $(2)) * $(65536)) + (((r3_z / $(16384)) % $(4)) * $(16384)) + (((r3_z / $(128)) % $(128)) * $(128)) + (r3_z % $(128))), dtype=tl.bfloat16) * tl.cast(0.07216878364870323, tl.bfloat16))
    r3_value = r3_value.to(tl.bfloat16)
    r3_value = tl.broadcast_to(r3_value, [$(131072)])
    tl.store($(vmul_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(131072))
    r4_value = (tl.load($(vmul_ptr) + ((((r4_z / $(65536)) % $(2)) * $(65536)) + (((r4_z / $(16384)) % $(4)) * $(16384)) + (((r4_z / $(128)) % $(128)) * $(128)) + (r4_z % $(128))), dtype=tl.bfloat16) + tl.load($(vmasked_fill_ptr) + ((((r4_z / $(65536)) % $(2)) * $(16384)) + (((r4_z / $(128)) % $(128)) * $(128)) + (r4_z % $(128))), dtype=tl.bfloat16))
    r4_value = r4_value.to(tl.bfloat16)
    r4_value = tl.broadcast_to(r4_value, [$(131072)])
    tl.store($(vadd_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(131072))
    r5_value = tl.load($(vadd_ptr) + ((((r5_z / $(65536)) % $(2)) * $(65536)) + (((r5_z / $(16384)) % $(4)) * $(16384)) + (((r5_z / $(128)) % $(128)) * $(128)) + (r5_z % $(128))), dtype=tl.bfloat16)
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(131072)])
    tl.store($(vto_copy_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(131072))
    r6_lane = tl.arange($(128))
    r6_row_values = tl.load($(vto_copy_ptr) + ((((r6_z[:, None] / $(65536)) % $(2)) * $(65536)) + (((r6_z[:, None] / $(16384)) % $(4)) * $(16384)) + (((r6_z[:, None] / $(128)) % $(128)) * $(128)) + (r6_lane[None, :] % $(128))), dtype=tl.float32)
    r6_maximum = tl.max(r6_row_values, axis=1, keep_dims=true)
    r6_weights = tl.exp(r6_row_values - r6_maximum)
    r6_total = tl.sum(r6_weights, axis=1)
    r6_maximum = tl.max(r6_row_values, axis=1)
    r6_value = tl.exp(tl.load($(vto_copy_ptr) + ((((r6_z / $(65536)) % $(2)) * $(65536)) + (((r6_z / $(16384)) % $(4)) * $(16384)) + (((r6_z / $(128)) % $(128)) * $(128)) + (r6_z % $(128))), dtype=tl.float32) - r6_maximum) / r6_total
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(131072)])
    tl.store($(vsoftmax_ptr) + r6_z, r6_value)
    r7_row = tl.arange($(128))
    r7_col = tl.arange($(128))
    r7_lane = tl.arange($(128))
    for r7_group in range($(8)) {
      r7_left = tl.load($(input3_ptr) + (((((((r7_group % $(8)) * $(16384)) + (r7_row[:, None] * $(128)) + r7_lane[None, :]) / $(65536)) % $(2)) * $(65536)) + ((((((r7_group % $(8)) * $(16384)) + (r7_row[:, None] * $(128)) + r7_lane[None, :]) / $(128)) % $(128)) * $(512)) + ((((((r7_group % $(8)) * $(16384)) + (r7_row[:, None] * $(128)) + r7_lane[None, :]) / $(16384)) % $(4)) * $(128)) + ((((r7_group % $(8)) * $(16384)) + (r7_row[:, None] * $(128)) + r7_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r7_right = tl.load($(input2_ptr) + (((((((r7_group % $(8)) * $(16384)) + (r7_col[None, :] * $(128)) + r7_lane[:, None]) / $(65536)) % $(2)) * $(65536)) + ((((((r7_group % $(8)) * $(16384)) + (r7_col[None, :] * $(128)) + r7_lane[:, None]) / $(16384)) % $(4)) * $(16384)) + ((((((r7_group % $(8)) * $(16384)) + (r7_col[None, :] * $(128)) + r7_lane[:, None]) / $(128)) % $(128)) * $(128)) + ((((r7_group % $(8)) * $(16384)) + (r7_col[None, :] * $(128)) + r7_lane[:, None]) % $(128))), dtype=tl.bfloat16)
      r7_value = tl.dot(r7_left, r7_right)
      r7_value = r7_value.to(tl.bfloat16)
      tl.store($(vbmm_2_ptr) + ((r7_group * $(16384)) + (r7_row[:, None] * $(128)) + r7_col[None, :]), r7_value)
    }
    r8_z = tl.arange($(131072))
    r8_value = tl.load($(vbmm_2_ptr) + ((((r8_z / $(65536)) % $(2)) * $(65536)) + (((r8_z / $(16384)) % $(4)) * $(16384)) + (((r8_z / $(128)) % $(128)) * $(128)) + (r8_z % $(128))), dtype=tl.bfloat16)
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(131072)])
    tl.store($(vto_copy_2_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(131072))
    r9_lane = tl.arange($(128))
    r9_row_values = tl.load($(vsoftmax_ptr) + ((((r9_z[:, None] / $(65536)) % $(2)) * $(65536)) + (((r9_z[:, None] / $(16384)) % $(4)) * $(16384)) + (((r9_z[:, None] / $(128)) % $(128)) * $(128)) + (r9_lane[None, :] % $(128))), dtype=tl.float32)
    r9_row_grad = tl.load($(vto_copy_2_ptr) + ((((r9_z[:, None] / $(65536)) % $(2)) * $(65536)) + (((r9_z[:, None] / $(16384)) % $(4)) * $(16384)) + (((r9_z[:, None] / $(128)) % $(128)) * $(128)) + (r9_lane[None, :] % $(128))), dtype=tl.float32)
    r9_weighted = r9_row_grad * r9_row_values
    r9_dot_grad = tl.sum(r9_weighted, axis=1)
    r9_value = (tl.load($(vto_copy_2_ptr) + ((((r9_z / $(65536)) % $(2)) * $(65536)) + (((r9_z / $(16384)) % $(4)) * $(16384)) + (((r9_z / $(128)) % $(128)) * $(128)) + (r9_z % $(128))), dtype=tl.float32) - r9_dot_grad) * tl.load($(vsoftmax_ptr) + ((((r9_z / $(65536)) % $(2)) * $(65536)) + (((r9_z / $(16384)) % $(4)) * $(16384)) + (((r9_z / $(128)) % $(128)) * $(128)) + (r9_z % $(128))), dtype=tl.float32)
    r9_value = r9_value.to(tl.float32)
    r9_value = tl.broadcast_to(r9_value, [$(131072)])
    tl.store($(vsoftmax_backward_data_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(131072))
    r10_value = tl.load($(vsoftmax_backward_data_ptr) + ((((r10_z / $(65536)) % $(2)) * $(65536)) + (((r10_z / $(16384)) % $(4)) * $(16384)) + (((r10_z / $(128)) % $(128)) * $(128)) + (r10_z % $(128))), dtype=tl.float32)
    r10_value = r10_value.to(tl.bfloat16)
    r10_value = tl.broadcast_to(r10_value, [$(131072)])
    tl.store($(vto_copy_3_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(131072))
    r11_value = (tl.load($(vto_copy_3_ptr) + ((((r11_z / $(65536)) % $(2)) * $(65536)) + (((r11_z / $(16384)) % $(4)) * $(16384)) + (((r11_z / $(128)) % $(128)) * $(128)) + (r11_z % $(128))), dtype=tl.bfloat16) * tl.cast(0.07216878364870323, tl.bfloat16))
    r11_value = r11_value.to(tl.bfloat16)
    r11_value = tl.broadcast_to(r11_value, [$(131072)])
    tl.store($(vmul_1_ptr) + r11_z, r11_value)
    r12_row = tl.arange($(192))
    r12_col = tl.arange($(128))
    r12_lane = tl.arange($(128))
    for r12_group in range($(8)) {
      r12_left = tl.load($(input0_ptr) + (((((((r12_group % $(8)) * $(24576)) + (r12_lane[None, :] * $(192)) + r12_row[:, None]) / $(98304)) % $(2)) * $(98304)) + ((((((r12_group % $(8)) * $(24576)) + (r12_lane[None, :] * $(192)) + r12_row[:, None]) / $(24576)) % $(4)) * $(24576)) + ((((((r12_group % $(8)) * $(24576)) + (r12_lane[None, :] * $(192)) + r12_row[:, None]) / $(192)) % $(128)) * $(192)) + ((((r12_group % $(8)) * $(24576)) + (r12_lane[None, :] * $(192)) + r12_row[:, None]) % $(192))), dtype=tl.bfloat16)
      r12_right = tl.load($(vmul_1_ptr) + (((((((r12_group % $(8)) * $(16384)) + (r12_lane[:, None] * $(128)) + r12_col[None, :]) / $(65536)) % $(2)) * $(65536)) + ((((((r12_group % $(8)) * $(16384)) + (r12_lane[:, None] * $(128)) + r12_col[None, :]) / $(16384)) % $(4)) * $(16384)) + ((((((r12_group % $(8)) * $(16384)) + (r12_lane[:, None] * $(128)) + r12_col[None, :]) / $(128)) % $(128)) * $(128)) + ((((r12_group % $(8)) * $(16384)) + (r12_lane[:, None] * $(128)) + r12_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r12_value = tl.dot(r12_left, r12_right)
      r12_value = r12_value.to(tl.bfloat16)
      tl.store($(vbmm_3_ptr) + ((r12_group * $(24576)) + (r12_row[:, None] * $(128)) + r12_col[None, :]), r12_value)
    }
    z = tl.arange($(196608))
    value = tl.load($(vbmm_3_ptr) + ((((z / $(98304)) % $(2)) * $(98304)) + (((z / $(24576)) % $(4)) * $(24576)) + ((z % $(192)) * $(128)) + ((z / $(192)) % $(128))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(196608)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let input1_ptr : RegionName := "input1"
  let input2_ptr : RegionName := "input2"
  let input3_ptr : RegionName := "input3"
  let out_ptr : RegionName := "out"
  let vmasked_fill_ptr : RegionName := "vmasked_fill"
  let vbmm_ptr : RegionName := "vbmm"
  let vmul_ptr : RegionName := "vmul"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vsoftmax_ptr : RegionName := "vsoftmax"
  let vbmm_2_ptr : RegionName := "vbmm_2"
  let vsoftmax_backward_data_ptr : RegionName := "vsoftmax_backward_data"
  let vto_copy_3_ptr : RegionName := "vto_copy_3"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vbmm_3_ptr : RegionName := "vbmm_3"
  let candidate_vto_copy_ptr : RegionName := "candidate_vto_copy"
  let candidate_vto_copy_1_ptr : RegionName := "candidate_vto_copy_1"
  let candidate_vbmm_ptr : RegionName := "candidate_vbmm"
  let candidate_vmul_ptr : RegionName := "candidate_vmul"
  let candidate_vgt_ptr : RegionName := "candidate_vgt"
  let candidate_vmasked_fill_ptr : RegionName := "candidate_vmasked_fill"
  let candidate_vsoftmax_ptr : RegionName := "candidate_vsoftmax"
  let candidate_vbmm_2_ptr : RegionName := "candidate_vbmm_2"
  let candidate_vto_copy_3_ptr : RegionName := "candidate_vto_copy_3"
  let candidate_vsoftmax_backward_data_ptr : RegionName := "candidate_vsoftmax_backward_data"
  let candidate_vmasked_fill_1_ptr : RegionName := "candidate_vmasked_fill_1"
  let candidate_vmul_1_ptr : RegionName := "candidate_vmul_1"
  let candidate_vbmm_3_ptr : RegionName := "candidate_vbmm_3"
  let candidate_vto_copy_4_ptr : RegionName := "candidate_vto_copy_4"
  triton {
    r0_z = tl.arange($(196608))
    r0_value = tl.load($(input0_ptr) + ((((r0_z / $(98304)) % $(2)) * $(98304)) + (((r0_z / $(24576)) % $(4)) * $(24576)) + (((r0_z / $(192)) % $(128)) * $(192)) + (r0_z % $(192))), dtype=tl.bfloat16)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(196608)])
    tl.store($(candidate_vto_copy_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(196608))
    r1_value = tl.load($(input1_ptr) + ((((r1_z / $(98304)) % $(2)) * $(98304)) + (((r1_z / $(24576)) % $(4)) * $(24576)) + (((r1_z / $(192)) % $(128)) * $(192)) + (r1_z % $(192))), dtype=tl.bfloat16)
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(196608)])
    tl.store($(candidate_vto_copy_1_ptr) + r1_z, r1_value)
    r2_row = tl.arange($(128))
    r2_col = tl.arange($(128))
    r2_lane = tl.arange($(192))
    for r2_group in range($(8)) {
      r2_left = tl.load($(candidate_vto_copy_ptr) + (((((((r2_group % $(8)) * $(24576)) + (r2_row[:, None] * $(192)) + r2_lane[None, :]) / $(98304)) % $(2)) * $(98304)) + ((((((r2_group % $(8)) * $(24576)) + (r2_row[:, None] * $(192)) + r2_lane[None, :]) / $(24576)) % $(4)) * $(24576)) + ((((((r2_group % $(8)) * $(24576)) + (r2_row[:, None] * $(192)) + r2_lane[None, :]) / $(192)) % $(128)) * $(192)) + ((((r2_group % $(8)) * $(24576)) + (r2_row[:, None] * $(192)) + r2_lane[None, :]) % $(192))), dtype=tl.float32)
      r2_right = tl.load($(candidate_vto_copy_1_ptr) + (((((((r2_group % $(8)) * $(24576)) + (r2_lane[:, None] * $(128)) + r2_col[None, :]) / $(98304)) % $(2)) * $(98304)) + ((((((r2_group % $(8)) * $(24576)) + (r2_lane[:, None] * $(128)) + r2_col[None, :]) / $(24576)) % $(4)) * $(24576)) + (((((r2_group % $(8)) * $(24576)) + (r2_lane[:, None] * $(128)) + r2_col[None, :]) % $(128)) * $(192)) + (((((r2_group % $(8)) * $(24576)) + (r2_lane[:, None] * $(128)) + r2_col[None, :]) / $(128)) % $(192))), dtype=tl.float32)
      r2_value = tl.dot(r2_left, r2_right)
      r2_value = r2_value.to(tl.float32)
      tl.store($(candidate_vbmm_ptr) + ((r2_group * $(16384)) + (r2_row[:, None] * $(128)) + r2_col[None, :]), r2_value)
    }
    r3_z = tl.arange($(131072))
    r3_value = (tl.load($(candidate_vbmm_ptr) + ((((r3_z / $(65536)) % $(2)) * $(65536)) + (((r3_z / $(16384)) % $(4)) * $(16384)) + (((r3_z / $(128)) % $(128)) * $(128)) + (r3_z % $(128))), dtype=tl.float32) * 0.07216878364870323)
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(131072)])
    tl.store($(candidate_vmul_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(16384))
    r4_value = (tl.cast((r4_z % $(128)), tl.int64) > tl.cast(((r4_z / $(128)) % $(128)), tl.int64))
    r4_value = tl.broadcast_to(r4_value, [$(16384)])
    tl.store($(candidate_vgt_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(131072))
    r5_value = tl.where(tl.load($(candidate_vgt_ptr) + ((((r5_z / $(128)) % $(128)) * $(128)) + (r5_z % $(128))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_ptr) + ((((r5_z / $(65536)) % $(2)) * $(65536)) + (((r5_z / $(16384)) % $(4)) * $(16384)) + (((r5_z / $(128)) % $(128)) * $(128)) + (r5_z % $(128))), dtype=tl.float32))
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(131072)])
    tl.store($(candidate_vmasked_fill_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(131072))
    r6_lane = tl.arange($(128))
    r6_row_values = tl.load($(candidate_vmasked_fill_ptr) + ((((r6_z[:, None] / $(65536)) % $(2)) * $(65536)) + (((r6_z[:, None] / $(16384)) % $(4)) * $(16384)) + (((r6_z[:, None] / $(128)) % $(128)) * $(128)) + (r6_lane[None, :] % $(128))), dtype=tl.float32)
    r6_maximum = tl.max(r6_row_values, axis=1, keep_dims=true)
    r6_weights = tl.exp(r6_row_values - r6_maximum)
    r6_total = tl.sum(r6_weights, axis=1)
    r6_maximum = tl.max(r6_row_values, axis=1)
    r6_value = tl.exp(tl.load($(candidate_vmasked_fill_ptr) + ((((r6_z / $(65536)) % $(2)) * $(65536)) + (((r6_z / $(16384)) % $(4)) * $(16384)) + (((r6_z / $(128)) % $(128)) * $(128)) + (r6_z % $(128))), dtype=tl.float32) - r6_maximum) / r6_total
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(131072)])
    tl.store($(candidate_vsoftmax_ptr) + r6_z, r6_value)
    r7_row = tl.arange($(128))
    r7_col = tl.arange($(128))
    r7_lane = tl.arange($(128))
    for r7_group in range($(8)) {
      r7_left = tl.load($(input3_ptr) + (((((((r7_group % $(8)) * $(16384)) + (r7_row[:, None] * $(128)) + r7_lane[None, :]) / $(65536)) % $(2)) * $(65536)) + ((((((r7_group % $(8)) * $(16384)) + (r7_row[:, None] * $(128)) + r7_lane[None, :]) / $(128)) % $(128)) * $(512)) + ((((((r7_group % $(8)) * $(16384)) + (r7_row[:, None] * $(128)) + r7_lane[None, :]) / $(16384)) % $(4)) * $(128)) + ((((r7_group % $(8)) * $(16384)) + (r7_row[:, None] * $(128)) + r7_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r7_right = tl.load($(input2_ptr) + (((((((r7_group % $(8)) * $(16384)) + (r7_col[None, :] * $(128)) + r7_lane[:, None]) / $(65536)) % $(2)) * $(65536)) + ((((((r7_group % $(8)) * $(16384)) + (r7_col[None, :] * $(128)) + r7_lane[:, None]) / $(16384)) % $(4)) * $(16384)) + ((((((r7_group % $(8)) * $(16384)) + (r7_col[None, :] * $(128)) + r7_lane[:, None]) / $(128)) % $(128)) * $(128)) + ((((r7_group % $(8)) * $(16384)) + (r7_col[None, :] * $(128)) + r7_lane[:, None]) % $(128))), dtype=tl.bfloat16)
      r7_value = tl.dot(r7_left, r7_right)
      r7_value = r7_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_2_ptr) + ((r7_group * $(16384)) + (r7_row[:, None] * $(128)) + r7_col[None, :]), r7_value)
    }
    r8_z = tl.arange($(131072))
    r8_value = tl.load($(candidate_vbmm_2_ptr) + ((((r8_z / $(65536)) % $(2)) * $(65536)) + (((r8_z / $(16384)) % $(4)) * $(16384)) + (((r8_z / $(128)) % $(128)) * $(128)) + (r8_z % $(128))), dtype=tl.bfloat16)
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(131072)])
    tl.store($(candidate_vto_copy_3_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(131072))
    r9_lane = tl.arange($(128))
    r9_row_values = tl.load($(candidate_vsoftmax_ptr) + ((((r9_z[:, None] / $(65536)) % $(2)) * $(65536)) + (((r9_z[:, None] / $(16384)) % $(4)) * $(16384)) + (((r9_z[:, None] / $(128)) % $(128)) * $(128)) + (r9_lane[None, :] % $(128))), dtype=tl.float32)
    r9_row_grad = tl.load($(candidate_vto_copy_3_ptr) + ((((r9_z[:, None] / $(65536)) % $(2)) * $(65536)) + (((r9_z[:, None] / $(16384)) % $(4)) * $(16384)) + (((r9_z[:, None] / $(128)) % $(128)) * $(128)) + (r9_lane[None, :] % $(128))), dtype=tl.float32)
    r9_weighted = r9_row_grad * r9_row_values
    r9_dot_grad = tl.sum(r9_weighted, axis=1)
    r9_value = (tl.load($(candidate_vto_copy_3_ptr) + ((((r9_z / $(65536)) % $(2)) * $(65536)) + (((r9_z / $(16384)) % $(4)) * $(16384)) + (((r9_z / $(128)) % $(128)) * $(128)) + (r9_z % $(128))), dtype=tl.float32) - r9_dot_grad) * tl.load($(candidate_vsoftmax_ptr) + ((((r9_z / $(65536)) % $(2)) * $(65536)) + (((r9_z / $(16384)) % $(4)) * $(16384)) + (((r9_z / $(128)) % $(128)) * $(128)) + (r9_z % $(128))), dtype=tl.float32)
    r9_value = r9_value.to(tl.float32)
    r9_value = tl.broadcast_to(r9_value, [$(131072)])
    tl.store($(candidate_vsoftmax_backward_data_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(131072))
    r10_value = tl.where(tl.load($(candidate_vgt_ptr) + ((((r10_z / $(128)) % $(128)) * $(128)) + (r10_z % $(128))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_ptr) + ((((r10_z / $(65536)) % $(2)) * $(65536)) + (((r10_z / $(16384)) % $(4)) * $(16384)) + (((r10_z / $(128)) % $(128)) * $(128)) + (r10_z % $(128))), dtype=tl.float32))
    r10_value = r10_value.to(tl.float32)
    r10_value = tl.broadcast_to(r10_value, [$(131072)])
    tl.store($(candidate_vmasked_fill_1_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(131072))
    r11_value = (tl.load($(candidate_vmasked_fill_1_ptr) + ((((r11_z / $(65536)) % $(2)) * $(65536)) + (((r11_z / $(16384)) % $(4)) * $(16384)) + (((r11_z / $(128)) % $(128)) * $(128)) + (r11_z % $(128))), dtype=tl.float32) * 0.07216878364870323)
    r11_value = r11_value.to(tl.float32)
    r11_value = tl.broadcast_to(r11_value, [$(131072)])
    tl.store($(candidate_vmul_1_ptr) + r11_z, r11_value)
    r12_row = tl.arange($(192))
    r12_col = tl.arange($(128))
    r12_lane = tl.arange($(128))
    for r12_group in range($(8)) {
      r12_left = tl.load($(candidate_vto_copy_ptr) + (((((((r12_group % $(8)) * $(24576)) + (r12_lane[None, :] * $(192)) + r12_row[:, None]) / $(98304)) % $(2)) * $(98304)) + ((((((r12_group % $(8)) * $(24576)) + (r12_lane[None, :] * $(192)) + r12_row[:, None]) / $(24576)) % $(4)) * $(24576)) + ((((((r12_group % $(8)) * $(24576)) + (r12_lane[None, :] * $(192)) + r12_row[:, None]) / $(192)) % $(128)) * $(192)) + ((((r12_group % $(8)) * $(24576)) + (r12_lane[None, :] * $(192)) + r12_row[:, None]) % $(192))), dtype=tl.float32)
      r12_right = tl.load($(candidate_vmul_1_ptr) + (((((((r12_group % $(8)) * $(16384)) + (r12_lane[:, None] * $(128)) + r12_col[None, :]) / $(65536)) % $(2)) * $(65536)) + ((((((r12_group % $(8)) * $(16384)) + (r12_lane[:, None] * $(128)) + r12_col[None, :]) / $(16384)) % $(4)) * $(16384)) + ((((((r12_group % $(8)) * $(16384)) + (r12_lane[:, None] * $(128)) + r12_col[None, :]) / $(128)) % $(128)) * $(128)) + ((((r12_group % $(8)) * $(16384)) + (r12_lane[:, None] * $(128)) + r12_col[None, :]) % $(128))), dtype=tl.float32)
      r12_value = tl.dot(r12_left, r12_right)
      r12_value = r12_value.to(tl.float32)
      tl.store($(candidate_vbmm_3_ptr) + ((r12_group * $(24576)) + (r12_row[:, None] * $(128)) + r12_col[None, :]), r12_value)
    }
    r13_z = tl.arange($(196608))
    r13_value = tl.load($(candidate_vbmm_3_ptr) + ((((r13_z / $(98304)) % $(2)) * $(98304)) + (((r13_z / $(24576)) % $(4)) * $(24576)) + ((r13_z % $(192)) * $(128)) + ((r13_z / $(192)) % $(128))), dtype=tl.float32)
    r13_value = r13_value.to(tl.bfloat16)
    r13_value = tl.broadcast_to(r13_value, [$(196608)])
    tl.store($(candidate_vto_copy_4_ptr) + r13_z, r13_value)
    z = tl.arange($(196608))
    value = tl.load($(candidate_vto_copy_4_ptr) + ((((z / $(98304)) % $(2)) * $(98304)) + (((z / $(24576)) % $(4)) * $(24576)) + (((z / $(192)) % $(128)) * $(192)) + (z % $(192))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(196608)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.te_fused_attn_mla_dk
