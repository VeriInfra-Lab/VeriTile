import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_expert_bias_update
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Kernels

def original : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vsum_1_ptr : RegionName := "vsum_1"
  let vmul_ptr : RegionName := "vmul"
  let vsub_ptr : RegionName := "vsub"
  let vsign_ptr : RegionName := "vsign"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vadd_ptr : RegionName := "vadd"
  triton {
    r0_z = tl.arange($(4))
    r0_lane = tl.arange($(128))
    r0_value = tl.sum(tl.load($(input0_ptr) + ((r0_z[:, None] * $(128)) + r0_lane[None, :]), dtype=tl.float32), axis=1)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(4)])
    tl.store($(vsum_1_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(512))
    r1_value = (tl.load($(input0_ptr) + ((((r1_z / $(128)) % $(4)) * $(128)) + (r1_z % $(128))), dtype=tl.float32) * 128.0)
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(512)])
    tl.store($(vmul_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(512))
    r2_value = (tl.load($(vsum_1_ptr) + ((r2_z / $(128)) % $(4)), dtype=tl.float32) - tl.load($(vmul_ptr) + ((((r2_z / $(128)) % $(4)) * $(128)) + (r2_z % $(128))), dtype=tl.float32))
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(512)])
    tl.store($(vsub_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(512))
    r3_value = tl.where(tl.load($(vsub_ptr) + ((((r3_z / $(128)) % $(4)) * $(128)) + (r3_z % $(128))), dtype=tl.float32) > 0.0, 1.0, tl.where(tl.load($(vsub_ptr) + ((((r3_z / $(128)) % $(4)) * $(128)) + (r3_z % $(128))), dtype=tl.float32) < 0.0, -1.0, 0.0))
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(512)])
    tl.store($(vsign_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(512))
    r4_value = (tl.load($(vsign_ptr) + ((((r4_z / $(128)) % $(4)) * $(128)) + (r4_z % $(128))), dtype=tl.float32) * 0.001)
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(512)])
    tl.store($(vmul_1_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(512))
    r5_value = ((-0.05 + ((((r5_z / $(128)) % $(4)) * $(128)) + (r5_z % $(128))) * 0.00019569471624266145) + tl.load($(vmul_1_ptr) + ((((r5_z / $(128)) % $(4)) * $(128)) + (r5_z % $(128))), dtype=tl.float32))
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(512)])
    tl.store($(vadd_ptr) + r5_z, r5_value)
    z = tl.arange($(512))
    value = tl.load($(vadd_ptr) + ((((z / $(128)) % $(4)) * $(128)) + (z % $(128))), dtype=tl.float32)
    value = tl.broadcast_to(value, [$(512)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vsum_1_ptr : RegionName := "vsum_1"
  let vmul_ptr : RegionName := "vmul"
  let vsub_ptr : RegionName := "vsub"
  let vsign_ptr : RegionName := "vsign"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vadd_ptr : RegionName := "vadd"
  let candidate_vsum_1_ptr : RegionName := "candidate_vsum_1"
  let candidate_vmul_ptr : RegionName := "candidate_vmul"
  let candidate_vsub_ptr : RegionName := "candidate_vsub"
  let candidate_vsign_ptr : RegionName := "candidate_vsign"
  let candidate_vmul_1_ptr : RegionName := "candidate_vmul_1"
  let candidate_vadd_ptr : RegionName := "candidate_vadd"
  triton {
    r0_z = tl.arange($(4))
    r0_lane = tl.arange($(128))
    r0_value = tl.sum(tl.load($(input0_ptr) + ((r0_z[:, None] * $(128)) + r0_lane[None, :]), dtype=tl.float32), axis=1)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(4)])
    tl.store($(candidate_vsum_1_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(512))
    r1_value = (tl.load($(input0_ptr) + ((((r1_z / $(128)) % $(4)) * $(128)) + (r1_z % $(128))), dtype=tl.float32) * 128.0)
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(512)])
    tl.store($(candidate_vmul_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(512))
    r2_value = (tl.load($(candidate_vsum_1_ptr) + ((r2_z / $(128)) % $(4)), dtype=tl.float32) - tl.load($(candidate_vmul_ptr) + ((((r2_z / $(128)) % $(4)) * $(128)) + (r2_z % $(128))), dtype=tl.float32))
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(512)])
    tl.store($(candidate_vsub_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(512))
    r3_value = tl.where(tl.load($(candidate_vsub_ptr) + ((((r3_z / $(128)) % $(4)) * $(128)) + (r3_z % $(128))), dtype=tl.float32) > 0.0, 1.0, tl.where(tl.load($(candidate_vsub_ptr) + ((((r3_z / $(128)) % $(4)) * $(128)) + (r3_z % $(128))), dtype=tl.float32) < 0.0, -1.0, 0.0))
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(512)])
    tl.store($(candidate_vsign_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(512))
    r4_value = (tl.load($(candidate_vsign_ptr) + ((((r4_z / $(128)) % $(4)) * $(128)) + (r4_z % $(128))), dtype=tl.float32) * 0.001)
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(512)])
    tl.store($(candidate_vmul_1_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(512))
    r5_value = ((-0.05 + ((((r5_z / $(128)) % $(4)) * $(128)) + (r5_z % $(128))) * 0.00019569471624266145) + tl.load($(candidate_vmul_1_ptr) + ((((r5_z / $(128)) % $(4)) * $(128)) + (r5_z % $(128))), dtype=tl.float32))
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(512)])
    tl.store($(candidate_vadd_ptr) + r5_z, r5_value)
    z = tl.arange($(512))
    value = tl.load($(candidate_vadd_ptr) + ((((z / $(128)) % $(4)) * $(128)) + (z % $(128))), dtype=tl.float32)
    value = tl.broadcast_to(value, [$(512)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_expert_bias_update
