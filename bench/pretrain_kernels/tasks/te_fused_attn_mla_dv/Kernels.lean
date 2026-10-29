import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.te_fused_attn_mla_dv
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
  let vtriu_ptr : RegionName := "vtriu"
  let vmasked_fill_ptr : RegionName := "vmasked_fill"
  let vbmm_ptr : RegionName := "vbmm"
  let vmul_ptr : RegionName := "vmul"
  let vadd_ptr : RegionName := "vadd"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vsoftmax_ptr : RegionName := "vsoftmax"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let vbmm_2_ptr : RegionName := "vbmm_2"
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
    r7_z = tl.arange($(131072))
    r7_value = tl.load($(vsoftmax_ptr) + ((((r7_z / $(65536)) % $(2)) * $(65536)) + (((r7_z / $(16384)) % $(4)) * $(16384)) + (((r7_z / $(128)) % $(128)) * $(128)) + (r7_z % $(128))), dtype=tl.float32)
    r7_value = r7_value.to(tl.bfloat16)
    r7_value = tl.broadcast_to(r7_value, [$(131072)])
    tl.store($(vto_copy_1_ptr) + r7_z, r7_value)
    r8_row = tl.arange($(128))
    r8_col = tl.arange($(128))
    r8_lane = tl.arange($(128))
    for r8_group in range($(8)) {
      r8_left = tl.load($(vto_copy_1_ptr) + (((((((r8_group % $(8)) * $(16384)) + (r8_lane[None, :] * $(128)) + r8_row[:, None]) / $(65536)) % $(2)) * $(65536)) + ((((((r8_group % $(8)) * $(16384)) + (r8_lane[None, :] * $(128)) + r8_row[:, None]) / $(16384)) % $(4)) * $(16384)) + ((((((r8_group % $(8)) * $(16384)) + (r8_lane[None, :] * $(128)) + r8_row[:, None]) / $(128)) % $(128)) * $(128)) + ((((r8_group % $(8)) * $(16384)) + (r8_lane[None, :] * $(128)) + r8_row[:, None]) % $(128))), dtype=tl.bfloat16)
      r8_right = tl.load($(input2_ptr) + (((((((r8_group % $(8)) * $(16384)) + (r8_lane[:, None] * $(128)) + r8_col[None, :]) / $(65536)) % $(2)) * $(65536)) + ((((((r8_group % $(8)) * $(16384)) + (r8_lane[:, None] * $(128)) + r8_col[None, :]) / $(128)) % $(128)) * $(512)) + ((((((r8_group % $(8)) * $(16384)) + (r8_lane[:, None] * $(128)) + r8_col[None, :]) / $(16384)) % $(4)) * $(128)) + ((((r8_group % $(8)) * $(16384)) + (r8_lane[:, None] * $(128)) + r8_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r8_value = tl.dot(r8_left, r8_right)
      r8_value = r8_value.to(tl.bfloat16)
      tl.store($(vbmm_2_ptr) + ((r8_group * $(16384)) + (r8_row[:, None] * $(128)) + r8_col[None, :]), r8_value)
    }
    z = tl.arange($(131072))
    value = tl.load($(vbmm_2_ptr) + ((((z / $(65536)) % $(2)) * $(65536)) + (((z / $(16384)) % $(4)) * $(16384)) + (((z / $(128)) % $(128)) * $(128)) + (z % $(128))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(131072)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let input1_ptr : RegionName := "input1"
  let input2_ptr : RegionName := "input2"
  let out_ptr : RegionName := "out"
  let vmasked_fill_ptr : RegionName := "vmasked_fill"
  let vbmm_ptr : RegionName := "vbmm"
  let vmul_ptr : RegionName := "vmul"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vsoftmax_ptr : RegionName := "vsoftmax"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let vbmm_2_ptr : RegionName := "vbmm_2"
  let candidate_vto_copy_ptr : RegionName := "candidate_vto_copy"
  let candidate_vto_copy_1_ptr : RegionName := "candidate_vto_copy_1"
  let candidate_vbmm_ptr : RegionName := "candidate_vbmm"
  let candidate_vmul_ptr : RegionName := "candidate_vmul"
  let candidate_vgt_ptr : RegionName := "candidate_vgt"
  let candidate_vmasked_fill_ptr : RegionName := "candidate_vmasked_fill"
  let candidate_vsoftmax_ptr : RegionName := "candidate_vsoftmax"
  let candidate_vto_copy_2_ptr : RegionName := "candidate_vto_copy_2"
  let candidate_vbmm_2_ptr : RegionName := "candidate_vbmm_2"
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
    r7_z = tl.arange($(131072))
    r7_value = tl.load($(candidate_vsoftmax_ptr) + ((((r7_z / $(65536)) % $(2)) * $(65536)) + (((r7_z / $(16384)) % $(4)) * $(16384)) + (((r7_z / $(128)) % $(128)) * $(128)) + (r7_z % $(128))), dtype=tl.float32)
    r7_value = r7_value.to(tl.bfloat16)
    r7_value = tl.broadcast_to(r7_value, [$(131072)])
    tl.store($(candidate_vto_copy_2_ptr) + r7_z, r7_value)
    r8_row = tl.arange($(128))
    r8_col = tl.arange($(128))
    r8_lane = tl.arange($(128))
    for r8_group in range($(8)) {
      r8_left = tl.load($(candidate_vto_copy_2_ptr) + (((((((r8_group % $(8)) * $(16384)) + (r8_lane[None, :] * $(128)) + r8_row[:, None]) / $(65536)) % $(2)) * $(65536)) + ((((((r8_group % $(8)) * $(16384)) + (r8_lane[None, :] * $(128)) + r8_row[:, None]) / $(16384)) % $(4)) * $(16384)) + ((((((r8_group % $(8)) * $(16384)) + (r8_lane[None, :] * $(128)) + r8_row[:, None]) / $(128)) % $(128)) * $(128)) + ((((r8_group % $(8)) * $(16384)) + (r8_lane[None, :] * $(128)) + r8_row[:, None]) % $(128))), dtype=tl.bfloat16)
      r8_right = tl.load($(input2_ptr) + (((((((r8_group % $(8)) * $(16384)) + (r8_lane[:, None] * $(128)) + r8_col[None, :]) / $(65536)) % $(2)) * $(65536)) + ((((((r8_group % $(8)) * $(16384)) + (r8_lane[:, None] * $(128)) + r8_col[None, :]) / $(128)) % $(128)) * $(512)) + ((((((r8_group % $(8)) * $(16384)) + (r8_lane[:, None] * $(128)) + r8_col[None, :]) / $(16384)) % $(4)) * $(128)) + ((((r8_group % $(8)) * $(16384)) + (r8_lane[:, None] * $(128)) + r8_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r8_value = tl.dot(r8_left, r8_right)
      r8_value = r8_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_2_ptr) + ((r8_group * $(16384)) + (r8_row[:, None] * $(128)) + r8_col[None, :]), r8_value)
    }
    z = tl.arange($(131072))
    value = tl.load($(candidate_vbmm_2_ptr) + ((((z / $(65536)) % $(2)) * $(65536)) + (((z / $(16384)) % $(4)) * $(16384)) + (((z / $(128)) % $(128)) * $(128)) + (z % $(128))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(131072)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.te_fused_attn_mla_dv
