import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_fused_hc_post_dres
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Kernels

def original : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vto_copy_3_ptr : RegionName := "vto_copy_3"
  let vto_copy_4_ptr : RegionName := "vto_copy_4"
  triton {
    r0_z = tl.arange($(67108864))
    r0_value = tl.load($(input0_ptr) + ((((r0_z / $(33554432)) % $(2)) * $(33554432)) + (((r0_z / $(8192)) % $(4096)) * $(8192)) + (((r0_z / $(2048)) % $(4)) * $(2048)) + (r0_z % $(2048))), dtype=tl.bfloat16)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(67108864)])
    tl.store($(vto_copy_3_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(67108864))
    r1_value = tl.load($(vto_copy_3_ptr) + ((((r1_z / $(33554432)) % $(2)) * $(33554432)) + (((r1_z / $(8192)) % $(4096)) * $(8192)) + (((r1_z / $(2048)) % $(4)) * $(2048)) + (r1_z % $(2048))), dtype=tl.float32)
    r1_value = r1_value.to(tl.bfloat16)
    r1_value = tl.broadcast_to(r1_value, [$(67108864)])
    tl.store($(vto_copy_4_ptr) + r1_z, r1_value)
    z = tl.arange($(67108864))
    value = tl.load($(vto_copy_4_ptr) + ((((z / $(33554432)) % $(2)) * $(33554432)) + (((z / $(8192)) % $(4096)) * $(8192)) + (((z / $(2048)) % $(4)) * $(2048)) + (z % $(2048))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(67108864)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vto_copy_3_ptr : RegionName := "vto_copy_3"
  let vto_copy_4_ptr : RegionName := "vto_copy_4"
  let candidate_vto_copy_3_ptr : RegionName := "candidate_vto_copy_3"
  let candidate_vto_copy_4_ptr : RegionName := "candidate_vto_copy_4"
  triton {
    r0_z = tl.arange($(67108864))
    r0_value = tl.load($(input0_ptr) + ((((r0_z / $(33554432)) % $(2)) * $(33554432)) + (((r0_z / $(8192)) % $(4096)) * $(8192)) + (((r0_z / $(2048)) % $(4)) * $(2048)) + (r0_z % $(2048))), dtype=tl.bfloat16)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(67108864)])
    tl.store($(candidate_vto_copy_3_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(67108864))
    r1_value = tl.load($(candidate_vto_copy_3_ptr) + ((((r1_z / $(33554432)) % $(2)) * $(33554432)) + (((r1_z / $(8192)) % $(4096)) * $(8192)) + (((r1_z / $(2048)) % $(4)) * $(2048)) + (r1_z % $(2048))), dtype=tl.float32)
    r1_value = r1_value.to(tl.bfloat16)
    r1_value = tl.broadcast_to(r1_value, [$(67108864)])
    tl.store($(candidate_vto_copy_4_ptr) + r1_z, r1_value)
    z = tl.arange($(67108864))
    value = tl.load($(candidate_vto_copy_4_ptr) + ((((z / $(33554432)) % $(2)) * $(33554432)) + (((z / $(8192)) % $(4096)) * $(8192)) + (((z / $(2048)) % $(4)) * $(2048)) + (z % $(2048))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(67108864)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_fused_hc_post_dres
