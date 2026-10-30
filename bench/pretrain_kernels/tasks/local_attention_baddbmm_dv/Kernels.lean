import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_attention_baddbmm_dv
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
  let vsum_1_ptr : RegionName := "vsum_1"
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
    r2_lane = tl.arange($(128))
    for r2_group in range($(32)) {
      r2_left = tl.load($(input0_ptr) + (((((((r2_group % $(32)) * $(16384)) + (r2_row[:, None] * $(128)) + r2_lane[None, :]) / $(262144)) % $(2)) * $(262144)) + ((((((r2_group % $(32)) * $(16384)) + (r2_row[:, None] * $(128)) + r2_lane[None, :]) / $(16384)) % $(16)) * $(16384)) + ((((((r2_group % $(32)) * $(16384)) + (r2_row[:, None] * $(128)) + r2_lane[None, :]) / $(128)) % $(128)) * $(128)) + ((((r2_group % $(32)) * $(16384)) + (r2_row[:, None] * $(128)) + r2_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r2_right = tl.load($(input1_ptr) + (((((((r2_group % $(32)) * $(16384)) + (r2_lane[:, None] * $(128)) + r2_col[None, :]) / $(262144)) % $(2)) * $(131072)) + ((((((((((r2_group % $(32)) * $(16384)) + (r2_lane[:, None] * $(128)) + r2_col[None, :]) / $(262144)) % $(2)) * $(262144)) + ((((((r2_group % $(32)) * $(16384)) + (r2_lane[:, None] * $(128)) + r2_col[None, :]) / $(16384)) % $(16)) * $(16384)) + (((((r2_group % $(32)) * $(16384)) + (r2_lane[:, None] * $(128)) + r2_col[None, :]) % $(128)) * $(128)) + (((((r2_group % $(32)) * $(16384)) + (r2_lane[:, None] * $(128)) + r2_col[None, :]) / $(128)) % $(128))) / $(32768)) % $(8)) * $(16384)) + (((((r2_group % $(32)) * $(16384)) + (r2_lane[:, None] * $(128)) + r2_col[None, :]) % $(128)) * $(128)) + (((((r2_group % $(32)) * $(16384)) + (r2_lane[:, None] * $(128)) + r2_col[None, :]) / $(128)) % $(128))), dtype=tl.bfloat16)
      r2_value = tl.dot(r2_left, r2_right)
      r2_value = r2_value.to(tl.bfloat16)
      tl.store($(vbmm_ptr) + ((r2_group * $(16384)) + (r2_row[:, None] * $(128)) + r2_col[None, :]), r2_value)
    }
    r3_z = tl.arange($(524288))
    r3_value = (tl.load($(vbmm_ptr) + ((((r3_z / $(262144)) % $(2)) * $(262144)) + (((r3_z / $(16384)) % $(16)) * $(16384)) + (((r3_z / $(128)) % $(128)) * $(128)) + (r3_z % $(128))), dtype=tl.bfloat16) * tl.cast(0.08838834764831845, tl.bfloat16))
    r3_value = r3_value.to(tl.bfloat16)
    r3_value = tl.broadcast_to(r3_value, [$(524288)])
    tl.store($(vmul_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(524288))
    r4_value = (tl.load($(vmul_ptr) + ((((r4_z / $(262144)) % $(2)) * $(262144)) + (((r4_z / $(16384)) % $(16)) * $(16384)) + (((r4_z / $(128)) % $(128)) * $(128)) + (r4_z % $(128))), dtype=tl.bfloat16) + tl.load($(vmasked_fill_ptr) + ((((r4_z / $(262144)) % $(2)) * $(16384)) + (((r4_z / $(128)) % $(128)) * $(128)) + (r4_z % $(128))), dtype=tl.bfloat16))
    r4_value = r4_value.to(tl.bfloat16)
    r4_value = tl.broadcast_to(r4_value, [$(524288)])
    tl.store($(vadd_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(524288))
    r5_value = tl.load($(vadd_ptr) + ((((r5_z / $(262144)) % $(2)) * $(262144)) + (((r5_z / $(16384)) % $(16)) * $(16384)) + (((r5_z / $(128)) % $(128)) * $(128)) + (r5_z % $(128))), dtype=tl.bfloat16)
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(524288)])
    tl.store($(vto_copy_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(524288))
    r6_lane = tl.arange($(128))
    r6_row_values = tl.load($(vto_copy_ptr) + ((((r6_z[:, None] / $(262144)) % $(2)) * $(262144)) + (((r6_z[:, None] / $(16384)) % $(16)) * $(16384)) + (((r6_z[:, None] / $(128)) % $(128)) * $(128)) + (r6_lane[None, :] % $(128))), dtype=tl.float32)
    r6_maximum = tl.max(r6_row_values, axis=1, keep_dims=true)
    r6_weights = tl.exp(r6_row_values - r6_maximum)
    r6_total = tl.sum(r6_weights, axis=1)
    r6_maximum = tl.max(r6_row_values, axis=1)
    r6_value = tl.exp(tl.load($(vto_copy_ptr) + ((((r6_z / $(262144)) % $(2)) * $(262144)) + (((r6_z / $(16384)) % $(16)) * $(16384)) + (((r6_z / $(128)) % $(128)) * $(128)) + (r6_z % $(128))), dtype=tl.float32) - r6_maximum) / r6_total
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(524288)])
    tl.store($(vsoftmax_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(524288))
    r7_value = tl.load($(vsoftmax_ptr) + ((((r7_z / $(262144)) % $(2)) * $(262144)) + (((r7_z / $(16384)) % $(16)) * $(16384)) + (((r7_z / $(128)) % $(128)) * $(128)) + (r7_z % $(128))), dtype=tl.float32)
    r7_value = r7_value.to(tl.bfloat16)
    r7_value = tl.broadcast_to(r7_value, [$(524288)])
    tl.store($(vto_copy_1_ptr) + r7_z, r7_value)
    r8_row = tl.arange($(128))
    r8_col = tl.arange($(128))
    r8_lane = tl.arange($(128))
    for r8_group in range($(32)) {
      r8_left = tl.load($(vto_copy_1_ptr) + (((((((r8_group % $(32)) * $(16384)) + (r8_lane[None, :] * $(128)) + r8_row[:, None]) / $(262144)) % $(2)) * $(262144)) + ((((((r8_group % $(32)) * $(16384)) + (r8_lane[None, :] * $(128)) + r8_row[:, None]) / $(16384)) % $(16)) * $(16384)) + ((((((r8_group % $(32)) * $(16384)) + (r8_lane[None, :] * $(128)) + r8_row[:, None]) / $(128)) % $(128)) * $(128)) + ((((r8_group % $(32)) * $(16384)) + (r8_lane[None, :] * $(128)) + r8_row[:, None]) % $(128))), dtype=tl.bfloat16)
      r8_right = tl.load($(input2_ptr) + (((((((r8_group % $(32)) * $(16384)) + (r8_lane[:, None] * $(128)) + r8_col[None, :]) / $(262144)) % $(2)) * $(262144)) + ((((((r8_group % $(32)) * $(16384)) + (r8_lane[:, None] * $(128)) + r8_col[None, :]) / $(128)) % $(128)) * $(2048)) + ((((((r8_group % $(32)) * $(16384)) + (r8_lane[:, None] * $(128)) + r8_col[None, :]) / $(16384)) % $(16)) * $(128)) + ((((r8_group % $(32)) * $(16384)) + (r8_lane[:, None] * $(128)) + r8_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r8_value = tl.dot(r8_left, r8_right)
      r8_value = r8_value.to(tl.bfloat16)
      tl.store($(vbmm_2_ptr) + ((r8_group * $(16384)) + (r8_row[:, None] * $(128)) + r8_col[None, :]), r8_value)
    }
    r9_z = tl.arange($(262144))
    r9_lane = tl.arange($(2))
    r9_value = tl.sum(tl.load($(vbmm_2_ptr) + ((((r9_z[:, None] / $(131072)) % $(2)) * $(262144)) + (((r9_z[:, None] / $(16384)) % $(8)) * $(32768)) + (r9_lane[None, :] * $(16384)) + (((r9_z[:, None] / $(128)) % $(128)) * $(128)) + (r9_z[:, None] % $(128))), dtype=tl.bfloat16), axis=1)
    r9_value = r9_value.to(tl.bfloat16)
    r9_value = tl.broadcast_to(r9_value, [$(262144)])
    tl.store($(vsum_1_ptr) + r9_z, r9_value)
    z = tl.arange($(262144))
    value = tl.load($(vsum_1_ptr) + ((((z / $(131072)) % $(2)) * $(131072)) + (((z / $(16384)) % $(8)) * $(16384)) + (((z / $(128)) % $(128)) * $(128)) + (z % $(128))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(262144)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let input1_ptr : RegionName := "input1"
  let input2_ptr : RegionName := "input2"
  let out_ptr : RegionName := "out"
  let vtriu_ptr : RegionName := "vtriu"
  let vmasked_fill_ptr : RegionName := "vmasked_fill"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vsoftmax_ptr : RegionName := "vsoftmax"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let vsum_1_ptr : RegionName := "vsum_1"
  let candidate_vbaddbmm_ptr : RegionName := "candidate_vbaddbmm"
  let candidate_vto_copy_ptr : RegionName := "candidate_vto_copy"
  let candidate_vtriu_ptr : RegionName := "candidate_vtriu"
  let candidate_vmasked_fill_ptr : RegionName := "candidate_vmasked_fill"
  let candidate_vsoftmax_ptr : RegionName := "candidate_vsoftmax"
  let candidate_vto_copy_1_ptr : RegionName := "candidate_vto_copy_1"
  let candidate_vbmm_1_ptr : RegionName := "candidate_vbmm_1"
  let candidate_vsum_1_ptr : RegionName := "candidate_vsum_1"
  triton {
    r0_row = tl.arange($(128))
    r0_col = tl.arange($(128))
    r0_lane = tl.arange($(128))
    for r0_group in range($(32)) {
      r0_left = tl.load($(input0_ptr) + (((((((r0_group % $(32)) * $(16384)) + (r0_row[:, None] * $(128)) + r0_lane[None, :]) / $(262144)) % $(2)) * $(262144)) + ((((((r0_group % $(32)) * $(16384)) + (r0_row[:, None] * $(128)) + r0_lane[None, :]) / $(16384)) % $(16)) * $(16384)) + ((((((r0_group % $(32)) * $(16384)) + (r0_row[:, None] * $(128)) + r0_lane[None, :]) / $(128)) % $(128)) * $(128)) + ((((r0_group % $(32)) * $(16384)) + (r0_row[:, None] * $(128)) + r0_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r0_right = tl.load($(input1_ptr) + (((((((r0_group % $(32)) * $(16384)) + (r0_col[None, :] * $(128)) + r0_lane[:, None]) / $(262144)) % $(2)) * $(131072)) + ((((((((((r0_group % $(32)) * $(16384)) + (r0_col[None, :] * $(128)) + r0_lane[:, None]) / $(262144)) % $(2)) * $(262144)) + ((((((r0_group % $(32)) * $(16384)) + (r0_col[None, :] * $(128)) + r0_lane[:, None]) / $(16384)) % $(16)) * $(16384)) + ((((((r0_group % $(32)) * $(16384)) + (r0_col[None, :] * $(128)) + r0_lane[:, None]) / $(128)) % $(128)) * $(128)) + ((((r0_group % $(32)) * $(16384)) + (r0_col[None, :] * $(128)) + r0_lane[:, None]) % $(128))) / $(32768)) % $(8)) * $(16384)) + ((((((r0_group % $(32)) * $(16384)) + (r0_col[None, :] * $(128)) + r0_lane[:, None]) / $(128)) % $(128)) * $(128)) + ((((r0_group % $(32)) * $(16384)) + (r0_col[None, :] * $(128)) + r0_lane[:, None]) % $(128))), dtype=tl.bfloat16)
      r0_value = tl.dot(r0_left, r0_right)
      r0_value = r0_value * 0.08838834764831845
      r0_value = r0_value.to(tl.bfloat16)
      tl.store($(candidate_vbaddbmm_ptr) + ((r0_group * $(16384)) + (r0_row[:, None] * $(128)) + r0_col[None, :]), r0_value)
    }
    r1_z = tl.arange($(524288))
    r1_value = tl.load($(candidate_vbaddbmm_ptr) + ((((r1_z / $(262144)) % $(2)) * $(262144)) + (((r1_z / $(16384)) % $(16)) * $(16384)) + (((r1_z / $(128)) % $(128)) * $(128)) + (r1_z % $(128))), dtype=tl.bfloat16)
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(524288)])
    tl.store($(candidate_vto_copy_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(16384))
    r2_value = tl.where((r2_z % $(128)) >= ((r2_z / $(128)) % $(128)) + $(1), true, false)
    r2_value = tl.broadcast_to(r2_value, [$(16384)])
    tl.store($(candidate_vtriu_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(524288))
    r3_value = tl.where(tl.load($(candidate_vtriu_ptr) + ((((r3_z / $(128)) % $(128)) * $(128)) + (r3_z % $(128))), dtype=tl.int1), -10000.0, tl.load($(candidate_vto_copy_ptr) + ((((r3_z / $(262144)) % $(2)) * $(262144)) + (((r3_z / $(16384)) % $(16)) * $(16384)) + (((r3_z / $(128)) % $(128)) * $(128)) + (r3_z % $(128))), dtype=tl.float32))
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(524288)])
    tl.store($(candidate_vmasked_fill_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(524288))
    r4_lane = tl.arange($(128))
    r4_row_values = tl.load($(candidate_vmasked_fill_ptr) + ((((r4_z[:, None] / $(262144)) % $(2)) * $(262144)) + (((r4_z[:, None] / $(16384)) % $(16)) * $(16384)) + (((r4_z[:, None] / $(128)) % $(128)) * $(128)) + (r4_lane[None, :] % $(128))), dtype=tl.float32)
    r4_maximum = tl.max(r4_row_values, axis=1, keep_dims=true)
    r4_weights = tl.exp(r4_row_values - r4_maximum)
    r4_total = tl.sum(r4_weights, axis=1)
    r4_maximum = tl.max(r4_row_values, axis=1)
    r4_value = tl.exp(tl.load($(candidate_vmasked_fill_ptr) + ((((r4_z / $(262144)) % $(2)) * $(262144)) + (((r4_z / $(16384)) % $(16)) * $(16384)) + (((r4_z / $(128)) % $(128)) * $(128)) + (r4_z % $(128))), dtype=tl.float32) - r4_maximum) / r4_total
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(524288)])
    tl.store($(candidate_vsoftmax_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(524288))
    r5_value = tl.load($(candidate_vsoftmax_ptr) + ((((r5_z / $(262144)) % $(2)) * $(262144)) + (((r5_z / $(16384)) % $(16)) * $(16384)) + (((r5_z / $(128)) % $(128)) * $(128)) + (r5_z % $(128))), dtype=tl.float32)
    r5_value = r5_value.to(tl.bfloat16)
    r5_value = tl.broadcast_to(r5_value, [$(524288)])
    tl.store($(candidate_vto_copy_1_ptr) + r5_z, r5_value)
    r6_row = tl.arange($(128))
    r6_col = tl.arange($(128))
    r6_lane = tl.arange($(128))
    for r6_group in range($(32)) {
      r6_left = tl.load($(candidate_vto_copy_1_ptr) + (((((((r6_group % $(32)) * $(16384)) + (r6_lane[None, :] * $(128)) + r6_row[:, None]) / $(262144)) % $(2)) * $(262144)) + ((((((r6_group % $(32)) * $(16384)) + (r6_lane[None, :] * $(128)) + r6_row[:, None]) / $(16384)) % $(16)) * $(16384)) + ((((((r6_group % $(32)) * $(16384)) + (r6_lane[None, :] * $(128)) + r6_row[:, None]) / $(128)) % $(128)) * $(128)) + ((((r6_group % $(32)) * $(16384)) + (r6_lane[None, :] * $(128)) + r6_row[:, None]) % $(128))), dtype=tl.bfloat16)
      r6_right = tl.load($(input2_ptr) + (((((((r6_group % $(32)) * $(16384)) + (r6_lane[:, None] * $(128)) + r6_col[None, :]) / $(262144)) % $(2)) * $(262144)) + ((((((r6_group % $(32)) * $(16384)) + (r6_lane[:, None] * $(128)) + r6_col[None, :]) / $(128)) % $(128)) * $(2048)) + ((((((r6_group % $(32)) * $(16384)) + (r6_lane[:, None] * $(128)) + r6_col[None, :]) / $(16384)) % $(16)) * $(128)) + ((((r6_group % $(32)) * $(16384)) + (r6_lane[:, None] * $(128)) + r6_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r6_value = tl.dot(r6_left, r6_right)
      r6_value = r6_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_1_ptr) + ((r6_group * $(16384)) + (r6_row[:, None] * $(128)) + r6_col[None, :]), r6_value)
    }
    r7_z = tl.arange($(262144))
    r7_lane = tl.arange($(2))
    r7_value = tl.sum(tl.load($(candidate_vbmm_1_ptr) + ((((r7_z[:, None] / $(131072)) % $(2)) * $(262144)) + (((r7_z[:, None] / $(16384)) % $(8)) * $(32768)) + (r7_lane[None, :] * $(16384)) + (((r7_z[:, None] / $(128)) % $(128)) * $(128)) + (r7_z[:, None] % $(128))), dtype=tl.bfloat16), axis=1)
    r7_value = r7_value.to(tl.bfloat16)
    r7_value = tl.broadcast_to(r7_value, [$(262144)])
    tl.store($(candidate_vsum_1_ptr) + r7_z, r7_value)
    z = tl.arange($(262144))
    value = tl.load($(candidate_vsum_1_ptr) + ((((z / $(131072)) % $(2)) * $(131072)) + (((z / $(16384)) % $(8)) * $(16384)) + (((z / $(128)) % $(128)) * $(128)) + (z % $(128))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(262144)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_attention_baddbmm_dv
