import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_fused_hc_post_fwd
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
  let vto_copy_ptr : RegionName := "vto_copy"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let vmul_ptr : RegionName := "vmul"
  let vadd_ptr : RegionName := "vadd"
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  triton {
    r0_z = tl.arange($(16777216))
    r0_value = tl.load($(input0_ptr) + ((((r0_z / $(8388608)) % $(2)) * $(8388608)) + (((r0_z / $(2048)) % $(4096)) * $(2048)) + (r0_z % $(2048))), dtype=tl.bfloat16)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(16777216)])
    tl.store($(vto_copy_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(67108864))
    r1_value = tl.load($(input1_ptr) + ((((r1_z / $(33554432)) % $(2)) * $(33554432)) + (((r1_z / $(8192)) % $(4096)) * $(8192)) + (((r1_z / $(2048)) % $(4)) * $(2048)) + (r1_z % $(2048))), dtype=tl.bfloat16)
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(67108864)])
    tl.store($(vto_copy_1_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(67108864))
    r2_value = (tl.load($(input2_ptr) + ((((r2_z / $(33554432)) % $(2)) * $(16384)) + (((r2_z / $(8192)) % $(4096)) * $(4)) + ((r2_z / $(2048)) % $(4))), dtype=tl.float32) * tl.load($(vto_copy_ptr) + ((((r2_z / $(33554432)) % $(2)) * $(8388608)) + (((r2_z / $(8192)) % $(4096)) * $(2048)) + (r2_z % $(2048))), dtype=tl.float32))
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(67108864)])
    tl.store($(vmul_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(67108864))
    r3_value = (tl.load($(vmul_ptr) + ((((r3_z / $(33554432)) % $(2)) * $(33554432)) + (((r3_z / $(8192)) % $(4096)) * $(8192)) + (((r3_z / $(2048)) % $(4)) * $(2048)) + (r3_z % $(2048))), dtype=tl.float32) + tl.load($(vto_copy_1_ptr) + ((((r3_z / $(33554432)) % $(2)) * $(33554432)) + (((r3_z / $(8192)) % $(4096)) * $(8192)) + (((r3_z / $(2048)) % $(4)) * $(2048)) + (r3_z % $(2048))), dtype=tl.float32))
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(67108864)])
    tl.store($(vadd_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(67108864))
    r4_value = tl.load($(vadd_ptr) + ((((r4_z / $(33554432)) % $(2)) * $(33554432)) + (((r4_z / $(8192)) % $(4096)) * $(8192)) + (((r4_z / $(2048)) % $(4)) * $(2048)) + (r4_z % $(2048))), dtype=tl.float32)
    r4_value = r4_value.to(tl.bfloat16)
    r4_value = tl.broadcast_to(r4_value, [$(67108864)])
    tl.store($(vto_copy_2_ptr) + r4_z, r4_value)
    z = tl.arange($(67108864))
    value = tl.load($(vto_copy_2_ptr) + ((((z / $(33554432)) % $(2)) * $(33554432)) + (((z / $(8192)) % $(4096)) * $(8192)) + (((z / $(2048)) % $(4)) * $(2048)) + (z % $(2048))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(67108864)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let input1_ptr : RegionName := "input1"
  let input2_ptr : RegionName := "input2"
  let out_ptr : RegionName := "out"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let vmul_ptr : RegionName := "vmul"
  let vadd_ptr : RegionName := "vadd"
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  let candidate_vto_copy_ptr : RegionName := "candidate_vto_copy"
  let candidate_vto_copy_1_ptr : RegionName := "candidate_vto_copy_1"
  let candidate_vmul_ptr : RegionName := "candidate_vmul"
  let candidate_vadd_ptr : RegionName := "candidate_vadd"
  let candidate_vto_copy_2_ptr : RegionName := "candidate_vto_copy_2"
  triton {
    r0_z = tl.arange($(16777216))
    r0_value = tl.load($(input0_ptr) + ((((r0_z / $(8388608)) % $(2)) * $(8388608)) + (((r0_z / $(2048)) % $(4096)) * $(2048)) + (r0_z % $(2048))), dtype=tl.bfloat16)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(16777216)])
    tl.store($(candidate_vto_copy_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(67108864))
    r1_value = tl.load($(input1_ptr) + ((((r1_z / $(33554432)) % $(2)) * $(33554432)) + (((r1_z / $(8192)) % $(4096)) * $(8192)) + (((r1_z / $(2048)) % $(4)) * $(2048)) + (r1_z % $(2048))), dtype=tl.bfloat16)
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(67108864)])
    tl.store($(candidate_vto_copy_1_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(67108864))
    r2_value = (tl.load($(input2_ptr) + ((((r2_z / $(33554432)) % $(2)) * $(16384)) + (((r2_z / $(8192)) % $(4096)) * $(4)) + ((r2_z / $(2048)) % $(4))), dtype=tl.float32) * tl.load($(candidate_vto_copy_ptr) + ((((r2_z / $(33554432)) % $(2)) * $(8388608)) + (((r2_z / $(8192)) % $(4096)) * $(2048)) + (r2_z % $(2048))), dtype=tl.float32))
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(67108864)])
    tl.store($(candidate_vmul_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(67108864))
    r3_value = (tl.load($(candidate_vmul_ptr) + ((((r3_z / $(33554432)) % $(2)) * $(33554432)) + (((r3_z / $(8192)) % $(4096)) * $(8192)) + (((r3_z / $(2048)) % $(4)) * $(2048)) + (r3_z % $(2048))), dtype=tl.float32) + tl.load($(candidate_vto_copy_1_ptr) + ((((r3_z / $(33554432)) % $(2)) * $(33554432)) + (((r3_z / $(8192)) % $(4096)) * $(8192)) + (((r3_z / $(2048)) % $(4)) * $(2048)) + (r3_z % $(2048))), dtype=tl.float32))
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(67108864)])
    tl.store($(candidate_vadd_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(67108864))
    r4_value = tl.load($(candidate_vadd_ptr) + ((((r4_z / $(33554432)) % $(2)) * $(33554432)) + (((r4_z / $(8192)) % $(4096)) * $(8192)) + (((r4_z / $(2048)) % $(4)) * $(2048)) + (r4_z % $(2048))), dtype=tl.float32)
    r4_value = r4_value.to(tl.bfloat16)
    r4_value = tl.broadcast_to(r4_value, [$(67108864)])
    tl.store($(candidate_vto_copy_2_ptr) + r4_z, r4_value)
    z = tl.arange($(67108864))
    value = tl.load($(candidate_vto_copy_2_ptr) + ((((z / $(33554432)) % $(2)) * $(33554432)) + (((z / $(8192)) % $(4096)) * $(8192)) + (((z / $(2048)) % $(4)) * $(2048)) + (z % $(2048))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(67108864)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_fused_hc_post_fwd
