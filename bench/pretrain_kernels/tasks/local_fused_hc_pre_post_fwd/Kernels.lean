import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_fused_hc_pre_post_fwd
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
  let vto_copy_ptr : RegionName := "vto_copy"
  let vpow_1_ptr : RegionName := "vpow_1"
  let vmean_ptr : RegionName := "vmean"
  let vadd_ptr : RegionName := "vadd"
  let vrsqrt_ptr : RegionName := "vrsqrt"
  let vmm_ptr : RegionName := "vmm"
  let vmul_ptr : RegionName := "vmul"
  let vmul_2_ptr : RegionName := "vmul_2"
  let vadd_3_ptr : RegionName := "vadd_3"
  let vsigmoid_1_ptr : RegionName := "vsigmoid_1"
  let vmul_3_ptr : RegionName := "vmul_3"
  let vadd_4_ptr : RegionName := "vadd_4"
  triton {
    r0_z = tl.arange($(67108864))
    r0_value = tl.load($(input0_ptr) + ((((r0_z / $(33554432)) % $(2)) * $(33554432)) + (((r0_z / $(8192)) % $(4096)) * $(8192)) + (((((((r0_z / $(33554432)) % $(2)) * $(33554432)) + (((r0_z / $(8192)) % $(4096)) * $(8192)) + (r0_z % $(8192))) / $(2048)) % $(4)) * $(2048)) + (((((r0_z / $(33554432)) % $(2)) * $(33554432)) + (((r0_z / $(8192)) % $(4096)) * $(8192)) + (r0_z % $(8192))) % $(2048))), dtype=tl.bfloat16)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(67108864)])
    tl.store($(vto_copy_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(67108864))
    r1_value = ((tl.load($(vto_copy_ptr) + ((((r1_z / $(33554432)) % $(2)) * $(33554432)) + (((r1_z / $(8192)) % $(4096)) * $(8192)) + (r1_z % $(8192))), dtype=tl.float32)) * (tl.load($(vto_copy_ptr) + ((((r1_z / $(33554432)) % $(2)) * $(33554432)) + (((r1_z / $(8192)) % $(4096)) * $(8192)) + (r1_z % $(8192))), dtype=tl.float32)))
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(67108864)])
    tl.store($(vpow_1_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(8192))
    r2_lane = tl.arange($(8192))
    r2_value = (tl.sum(tl.load($(vpow_1_ptr) + ((((r2_z[:, None] / $(4096)) % $(2)) * $(33554432)) + ((r2_z[:, None] % $(4096)) * $(8192)) + r2_lane[None, :]), dtype=tl.float32), axis=1) / 8192.0)
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(8192)])
    tl.store($(vmean_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(8192))
    r3_value = (tl.load($(vmean_ptr) + ((((r3_z / $(4096)) % $(2)) * $(4096)) + (r3_z % $(4096))), dtype=tl.float32) + 0.000001)
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(8192)])
    tl.store($(vadd_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(8192))
    r4_value = tl.rsqrt(tl.load($(vadd_ptr) + ((((r4_z / $(4096)) % $(2)) * $(4096)) + (r4_z % $(4096))), dtype=tl.float32))
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(8192)])
    tl.store($(vrsqrt_ptr) + r4_z, r4_value)
    r5_row = tl.arange($(8192))
    r5_col = tl.arange($(8))
    r5_lane = tl.arange($(8192))
    r5_left = tl.load($(vto_copy_ptr) + ((((((r5_row[:, None] * $(8192)) + r5_lane[None, :]) / $(33554432)) % $(2)) * $(33554432)) + (((((r5_row[:, None] * $(8192)) + r5_lane[None, :]) / $(8192)) % $(4096)) * $(8192)) + (((r5_row[:, None] * $(8192)) + r5_lane[None, :]) % $(8192))), dtype=tl.float32)
    r5_right = tl.load($(input1_ptr) + (((r5_col[None, :] % $(8)) * $(8192)) + (r5_lane[:, None] % $(8192))), dtype=tl.float32)
    r5_value = tl.dot(r5_left, r5_right)
    r5_value = r5_value.to(tl.float32)
    tl.store($(vmm_ptr) + ((r5_row[:, None] * $(8)) + r5_col[None, :]), r5_value)
    r6_z = tl.arange($(65536))
    r6_value = (tl.load($(vmm_ptr) + ((((r6_z / $(32768)) % $(2)) * $(32768)) + (((r6_z / $(8)) % $(4096)) * $(8)) + (r6_z % $(8))), dtype=tl.float32) * tl.load($(vrsqrt_ptr) + ((((r6_z / $(32768)) % $(2)) * $(4096)) + ((r6_z / $(8)) % $(4096))), dtype=tl.float32))
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(65536)])
    tl.store($(vmul_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(32768))
    r7_value = (tl.load($(vmul_ptr) + ((((r7_z / $(16384)) % $(2)) * $(32768)) + (((r7_z / $(4)) % $(4096)) * $(8)) + (r7_z % $(4)) + $(4)), dtype=tl.float32) * tl.load($(input2_ptr) + $(1), dtype=tl.float32))
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(32768)])
    tl.store($(vmul_2_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(32768))
    r8_value = (tl.load($(vmul_2_ptr) + ((((r8_z / $(16384)) % $(2)) * $(16384)) + (((r8_z / $(4)) % $(4096)) * $(4)) + (r8_z % $(4))), dtype=tl.float32) + tl.load($(input3_ptr) + ((r8_z % $(4)) + $(4)), dtype=tl.float32))
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(32768)])
    tl.store($(vadd_3_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(32768))
    r9_value = tl.sigmoid(tl.load($(vadd_3_ptr) + ((((r9_z / $(16384)) % $(2)) * $(16384)) + (((r9_z / $(4)) % $(4096)) * $(4)) + (r9_z % $(4))), dtype=tl.float32))
    r9_value = r9_value.to(tl.float32)
    r9_value = tl.broadcast_to(r9_value, [$(32768)])
    tl.store($(vsigmoid_1_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(32768))
    r10_value = (tl.load($(vsigmoid_1_ptr) + ((((r10_z / $(16384)) % $(2)) * $(16384)) + (((r10_z / $(4)) % $(4096)) * $(4)) + (r10_z % $(4))), dtype=tl.float32) * 2.0)
    r10_value = r10_value.to(tl.float32)
    r10_value = tl.broadcast_to(r10_value, [$(32768)])
    tl.store($(vmul_3_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(32768))
    r11_value = (tl.load($(vmul_3_ptr) + ((((r11_z / $(16384)) % $(2)) * $(16384)) + (((r11_z / $(4)) % $(4096)) * $(4)) + (r11_z % $(4))), dtype=tl.float32) + 0.000001)
    r11_value = r11_value.to(tl.float32)
    r11_value = tl.broadcast_to(r11_value, [$(32768)])
    tl.store($(vadd_4_ptr) + r11_z, r11_value)
    z = tl.arange($(32768))
    value = tl.load($(vadd_4_ptr) + ((((z / $(16384)) % $(2)) * $(16384)) + (((z / $(4)) % $(4096)) * $(4)) + (z % $(4))), dtype=tl.float32)
    value = tl.broadcast_to(value, [$(32768)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let input1_ptr : RegionName := "input1"
  let input2_ptr : RegionName := "input2"
  let input3_ptr : RegionName := "input3"
  let out_ptr : RegionName := "out"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vpow_1_ptr : RegionName := "vpow_1"
  let vmean_ptr : RegionName := "vmean"
  let vadd_ptr : RegionName := "vadd"
  let vrsqrt_ptr : RegionName := "vrsqrt"
  let vmm_ptr : RegionName := "vmm"
  let vmul_ptr : RegionName := "vmul"
  let vmul_2_ptr : RegionName := "vmul_2"
  let vadd_3_ptr : RegionName := "vadd_3"
  let vsigmoid_1_ptr : RegionName := "vsigmoid_1"
  let vmul_3_ptr : RegionName := "vmul_3"
  let vadd_4_ptr : RegionName := "vadd_4"
  let candidate_vto_copy_ptr : RegionName := "candidate_vto_copy"
  let candidate_vpow_1_ptr : RegionName := "candidate_vpow_1"
  let candidate_vmean_ptr : RegionName := "candidate_vmean"
  let candidate_vadd_ptr : RegionName := "candidate_vadd"
  let candidate_vrsqrt_ptr : RegionName := "candidate_vrsqrt"
  let candidate_vmm_ptr : RegionName := "candidate_vmm"
  let candidate_vmul_ptr : RegionName := "candidate_vmul"
  let candidate_vmul_2_ptr : RegionName := "candidate_vmul_2"
  let candidate_vadd_3_ptr : RegionName := "candidate_vadd_3"
  let candidate_vsigmoid_1_ptr : RegionName := "candidate_vsigmoid_1"
  let candidate_vmul_3_ptr : RegionName := "candidate_vmul_3"
  let candidate_vadd_4_ptr : RegionName := "candidate_vadd_4"
  triton {
    r0_z = tl.arange($(67108864))
    r0_value = tl.load($(input0_ptr) + ((((r0_z / $(33554432)) % $(2)) * $(33554432)) + (((r0_z / $(8192)) % $(4096)) * $(8192)) + (((((((r0_z / $(33554432)) % $(2)) * $(33554432)) + (((r0_z / $(8192)) % $(4096)) * $(8192)) + (r0_z % $(8192))) / $(2048)) % $(4)) * $(2048)) + (((((r0_z / $(33554432)) % $(2)) * $(33554432)) + (((r0_z / $(8192)) % $(4096)) * $(8192)) + (r0_z % $(8192))) % $(2048))), dtype=tl.bfloat16)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(67108864)])
    tl.store($(candidate_vto_copy_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(67108864))
    r1_value = ((tl.load($(candidate_vto_copy_ptr) + ((((r1_z / $(33554432)) % $(2)) * $(33554432)) + (((r1_z / $(8192)) % $(4096)) * $(8192)) + (r1_z % $(8192))), dtype=tl.float32)) * (tl.load($(candidate_vto_copy_ptr) + ((((r1_z / $(33554432)) % $(2)) * $(33554432)) + (((r1_z / $(8192)) % $(4096)) * $(8192)) + (r1_z % $(8192))), dtype=tl.float32)))
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(67108864)])
    tl.store($(candidate_vpow_1_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(8192))
    r2_lane = tl.arange($(8192))
    r2_value = (tl.sum(tl.load($(candidate_vpow_1_ptr) + ((((r2_z[:, None] / $(4096)) % $(2)) * $(33554432)) + ((r2_z[:, None] % $(4096)) * $(8192)) + r2_lane[None, :]), dtype=tl.float32), axis=1) / 8192.0)
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(8192)])
    tl.store($(candidate_vmean_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(8192))
    r3_value = (tl.load($(candidate_vmean_ptr) + ((((r3_z / $(4096)) % $(2)) * $(4096)) + (r3_z % $(4096))), dtype=tl.float32) + 0.000001)
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(8192)])
    tl.store($(candidate_vadd_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(8192))
    r4_value = tl.rsqrt(tl.load($(candidate_vadd_ptr) + ((((r4_z / $(4096)) % $(2)) * $(4096)) + (r4_z % $(4096))), dtype=tl.float32))
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(8192)])
    tl.store($(candidate_vrsqrt_ptr) + r4_z, r4_value)
    r5_row = tl.arange($(8192))
    r5_col = tl.arange($(8))
    r5_lane = tl.arange($(8192))
    r5_left = tl.load($(candidate_vto_copy_ptr) + ((((((r5_row[:, None] * $(8192)) + r5_lane[None, :]) / $(33554432)) % $(2)) * $(33554432)) + (((((r5_row[:, None] * $(8192)) + r5_lane[None, :]) / $(8192)) % $(4096)) * $(8192)) + (((r5_row[:, None] * $(8192)) + r5_lane[None, :]) % $(8192))), dtype=tl.float32)
    r5_right = tl.load($(input1_ptr) + (((r5_col[None, :] % $(8)) * $(8192)) + (r5_lane[:, None] % $(8192))), dtype=tl.float32)
    r5_value = tl.dot(r5_left, r5_right)
    r5_value = r5_value.to(tl.float32)
    tl.store($(candidate_vmm_ptr) + ((r5_row[:, None] * $(8)) + r5_col[None, :]), r5_value)
    r6_z = tl.arange($(65536))
    r6_value = (tl.load($(candidate_vmm_ptr) + ((((r6_z / $(32768)) % $(2)) * $(32768)) + (((r6_z / $(8)) % $(4096)) * $(8)) + (r6_z % $(8))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_ptr) + ((((r6_z / $(32768)) % $(2)) * $(4096)) + ((r6_z / $(8)) % $(4096))), dtype=tl.float32))
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(65536)])
    tl.store($(candidate_vmul_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(32768))
    r7_value = (tl.load($(candidate_vmul_ptr) + ((((r7_z / $(16384)) % $(2)) * $(32768)) + (((r7_z / $(4)) % $(4096)) * $(8)) + (r7_z % $(4)) + $(4)), dtype=tl.float32) * tl.load($(input2_ptr) + $(1), dtype=tl.float32))
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(32768)])
    tl.store($(candidate_vmul_2_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(32768))
    r8_value = (tl.load($(candidate_vmul_2_ptr) + ((((r8_z / $(16384)) % $(2)) * $(16384)) + (((r8_z / $(4)) % $(4096)) * $(4)) + (r8_z % $(4))), dtype=tl.float32) + tl.load($(input3_ptr) + ((r8_z % $(4)) + $(4)), dtype=tl.float32))
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(32768)])
    tl.store($(candidate_vadd_3_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(32768))
    r9_value = tl.sigmoid(tl.load($(candidate_vadd_3_ptr) + ((((r9_z / $(16384)) % $(2)) * $(16384)) + (((r9_z / $(4)) % $(4096)) * $(4)) + (r9_z % $(4))), dtype=tl.float32))
    r9_value = r9_value.to(tl.float32)
    r9_value = tl.broadcast_to(r9_value, [$(32768)])
    tl.store($(candidate_vsigmoid_1_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(32768))
    r10_value = (tl.load($(candidate_vsigmoid_1_ptr) + ((((r10_z / $(16384)) % $(2)) * $(16384)) + (((r10_z / $(4)) % $(4096)) * $(4)) + (r10_z % $(4))), dtype=tl.float32) * 2.0)
    r10_value = r10_value.to(tl.float32)
    r10_value = tl.broadcast_to(r10_value, [$(32768)])
    tl.store($(candidate_vmul_3_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(32768))
    r11_value = (tl.load($(candidate_vmul_3_ptr) + ((((r11_z / $(16384)) % $(2)) * $(16384)) + (((r11_z / $(4)) % $(4096)) * $(4)) + (r11_z % $(4))), dtype=tl.float32) + 0.000001)
    r11_value = r11_value.to(tl.float32)
    r11_value = tl.broadcast_to(r11_value, [$(32768)])
    tl.store($(candidate_vadd_4_ptr) + r11_z, r11_value)
    z = tl.arange($(32768))
    value = tl.load($(candidate_vadd_4_ptr) + ((((z / $(16384)) % $(2)) * $(16384)) + (((z / $(4)) % $(4096)) * $(4)) + (z % $(4))), dtype=tl.float32)
    value = tl.broadcast_to(value, [$(32768)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_fused_hc_pre_post_fwd
