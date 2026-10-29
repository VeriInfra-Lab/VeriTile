import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_fused_hc_post_bda_dout
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
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  let vto_copy_5_ptr : RegionName := "vto_copy_5"
  let vmul_3_ptr : RegionName := "vmul_3"
  let vsum_1_ptr : RegionName := "vsum_1"
  let vmul_4_ptr : RegionName := "vmul_4"
  let vmul_5_ptr : RegionName := "vmul_5"
  let vto_copy_6_ptr : RegionName := "vto_copy_6"
  triton {
    r0_z = tl.arange($(16777216))
    r0_value = tl.where(tl.load($(input1_ptr) + ((((r0_z / $(8388608)) % $(2)) * $(8388608)) + (((r0_z / $(2048)) % $(4096)) * $(2048)) + (r0_z % $(2048))), dtype=tl.int1), 1.0, 0.0)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(16777216)])
    tl.store($(vto_copy_2_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(67108864))
    r1_value = tl.load($(input2_ptr) + ((((r1_z / $(33554432)) % $(2)) * $(33554432)) + (((r1_z / $(8192)) % $(4096)) * $(8192)) + (((r1_z / $(2048)) % $(4)) * $(2048)) + (r1_z % $(2048))), dtype=tl.bfloat16)
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(67108864)])
    tl.store($(vto_copy_5_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(67108864))
    r2_value = (tl.load($(vto_copy_5_ptr) + ((((r2_z / $(33554432)) % $(2)) * $(33554432)) + (((r2_z / $(8192)) % $(4096)) * $(8192)) + (((r2_z / $(2048)) % $(4)) * $(2048)) + (r2_z % $(2048))), dtype=tl.float32) * tl.load($(input0_ptr) + ((((r2_z / $(33554432)) % $(2)) * $(16384)) + (((r2_z / $(8192)) % $(4096)) * $(4)) + ((r2_z / $(2048)) % $(4))), dtype=tl.float32))
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(67108864)])
    tl.store($(vmul_3_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(16777216))
    r3_lane = tl.arange($(4))
    r3_value = tl.sum(tl.load($(vmul_3_ptr) + ((((r3_z[:, None] / $(8388608)) % $(2)) * $(33554432)) + (((r3_z[:, None] / $(2048)) % $(4096)) * $(8192)) + (r3_lane[None, :] * $(2048)) + (r3_z[:, None] % $(2048))), dtype=tl.float32), axis=1)
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(16777216)])
    tl.store($(vsum_1_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(16777216))
    r4_value = (tl.load($(vsum_1_ptr) + ((((r4_z / $(8388608)) % $(2)) * $(8388608)) + (((r4_z / $(2048)) % $(4096)) * $(2048)) + (r4_z % $(2048))), dtype=tl.float32) * 1.1111111111111112)
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(16777216)])
    tl.store($(vmul_4_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(16777216))
    r5_value = (tl.load($(vmul_4_ptr) + ((((r5_z / $(8388608)) % $(2)) * $(8388608)) + (((r5_z / $(2048)) % $(4096)) * $(2048)) + (r5_z % $(2048))), dtype=tl.float32) * tl.load($(vto_copy_2_ptr) + ((((r5_z / $(8388608)) % $(2)) * $(8388608)) + (((r5_z / $(2048)) % $(4096)) * $(2048)) + (r5_z % $(2048))), dtype=tl.float32))
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(16777216)])
    tl.store($(vmul_5_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(16777216))
    r6_value = tl.load($(vmul_5_ptr) + ((((r6_z / $(8388608)) % $(2)) * $(8388608)) + (((r6_z / $(2048)) % $(4096)) * $(2048)) + (r6_z % $(2048))), dtype=tl.float32)
    r6_value = r6_value.to(tl.bfloat16)
    r6_value = tl.broadcast_to(r6_value, [$(16777216)])
    tl.store($(vto_copy_6_ptr) + r6_z, r6_value)
    z = tl.arange($(16777216))
    value = tl.load($(vto_copy_6_ptr) + ((((z / $(8388608)) % $(2)) * $(8388608)) + (((z / $(2048)) % $(4096)) * $(2048)) + (z % $(2048))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(16777216)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let input1_ptr : RegionName := "input1"
  let input2_ptr : RegionName := "input2"
  let out_ptr : RegionName := "out"
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  let vto_copy_5_ptr : RegionName := "vto_copy_5"
  let vmul_3_ptr : RegionName := "vmul_3"
  let vsum_1_ptr : RegionName := "vsum_1"
  let vmul_4_ptr : RegionName := "vmul_4"
  let vmul_5_ptr : RegionName := "vmul_5"
  let vto_copy_6_ptr : RegionName := "vto_copy_6"
  let candidate_vto_copy_2_ptr : RegionName := "candidate_vto_copy_2"
  let candidate_vto_copy_5_ptr : RegionName := "candidate_vto_copy_5"
  let candidate_vmul_3_ptr : RegionName := "candidate_vmul_3"
  let candidate_vsum_1_ptr : RegionName := "candidate_vsum_1"
  let candidate_vmul_4_ptr : RegionName := "candidate_vmul_4"
  let candidate_vmul_5_ptr : RegionName := "candidate_vmul_5"
  let candidate_vto_copy_6_ptr : RegionName := "candidate_vto_copy_6"
  triton {
    r0_z = tl.arange($(16777216))
    r0_value = tl.where(tl.load($(input1_ptr) + ((((r0_z / $(8388608)) % $(2)) * $(8388608)) + (((r0_z / $(2048)) % $(4096)) * $(2048)) + (r0_z % $(2048))), dtype=tl.int1), 1.0, 0.0)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(16777216)])
    tl.store($(candidate_vto_copy_2_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(67108864))
    r1_value = tl.load($(input2_ptr) + ((((r1_z / $(33554432)) % $(2)) * $(33554432)) + (((r1_z / $(8192)) % $(4096)) * $(8192)) + (((r1_z / $(2048)) % $(4)) * $(2048)) + (r1_z % $(2048))), dtype=tl.bfloat16)
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(67108864)])
    tl.store($(candidate_vto_copy_5_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(67108864))
    r2_value = (tl.load($(candidate_vto_copy_5_ptr) + ((((r2_z / $(33554432)) % $(2)) * $(33554432)) + (((r2_z / $(8192)) % $(4096)) * $(8192)) + (((r2_z / $(2048)) % $(4)) * $(2048)) + (r2_z % $(2048))), dtype=tl.float32) * tl.load($(input0_ptr) + ((((r2_z / $(33554432)) % $(2)) * $(16384)) + (((r2_z / $(8192)) % $(4096)) * $(4)) + ((r2_z / $(2048)) % $(4))), dtype=tl.float32))
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(67108864)])
    tl.store($(candidate_vmul_3_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(16777216))
    r3_lane = tl.arange($(4))
    r3_value = tl.sum(tl.load($(candidate_vmul_3_ptr) + ((((r3_z[:, None] / $(8388608)) % $(2)) * $(33554432)) + (((r3_z[:, None] / $(2048)) % $(4096)) * $(8192)) + (r3_lane[None, :] * $(2048)) + (r3_z[:, None] % $(2048))), dtype=tl.float32), axis=1)
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(16777216)])
    tl.store($(candidate_vsum_1_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(16777216))
    r4_value = (tl.load($(candidate_vsum_1_ptr) + ((((r4_z / $(8388608)) % $(2)) * $(8388608)) + (((r4_z / $(2048)) % $(4096)) * $(2048)) + (r4_z % $(2048))), dtype=tl.float32) * 1.1111111111111112)
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(16777216)])
    tl.store($(candidate_vmul_4_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(16777216))
    r5_value = (tl.load($(candidate_vmul_4_ptr) + ((((r5_z / $(8388608)) % $(2)) * $(8388608)) + (((r5_z / $(2048)) % $(4096)) * $(2048)) + (r5_z % $(2048))), dtype=tl.float32) * tl.load($(candidate_vto_copy_2_ptr) + ((((r5_z / $(8388608)) % $(2)) * $(8388608)) + (((r5_z / $(2048)) % $(4096)) * $(2048)) + (r5_z % $(2048))), dtype=tl.float32))
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(16777216)])
    tl.store($(candidate_vmul_5_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(16777216))
    r6_value = tl.load($(candidate_vmul_5_ptr) + ((((r6_z / $(8388608)) % $(2)) * $(8388608)) + (((r6_z / $(2048)) % $(4096)) * $(2048)) + (r6_z % $(2048))), dtype=tl.float32)
    r6_value = r6_value.to(tl.bfloat16)
    r6_value = tl.broadcast_to(r6_value, [$(16777216)])
    tl.store($(candidate_vto_copy_6_ptr) + r6_z, r6_value)
    z = tl.arange($(16777216))
    value = tl.load($(candidate_vto_copy_6_ptr) + ((((z / $(8388608)) % $(2)) * $(8388608)) + (((z / $(2048)) % $(4096)) * $(2048)) + (z % $(2048))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(16777216)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_fused_hc_post_bda_dout
