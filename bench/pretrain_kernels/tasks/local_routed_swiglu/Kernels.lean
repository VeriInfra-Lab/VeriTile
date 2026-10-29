import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_routed_swiglu
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Kernels

def original : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let input1_ptr : RegionName := "input1"
  let out_ptr : RegionName := "out"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vsilu_ptr : RegionName := "vsilu"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let vmul_ptr : RegionName := "vmul"
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  let vmul_1_ptr : RegionName := "vmul_1"
  triton {
    r0_z = tl.arange($(163840))
    r0_value = tl.load($(input0_ptr) + ((((r0_z / $(640)) % $(256)) * $(1280)) + (r0_z % $(640))), dtype=tl.bfloat16)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(163840)])
    tl.store($(vto_copy_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(163840))
    r1_value = (tl.load($(vto_copy_ptr) + ((((r1_z / $(640)) % $(256)) * $(640)) + (r1_z % $(640))), dtype=tl.float32) * tl.sigmoid(tl.load($(vto_copy_ptr) + ((((r1_z / $(640)) % $(256)) * $(640)) + (r1_z % $(640))), dtype=tl.float32)))
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(163840)])
    tl.store($(vsilu_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(163840))
    r2_value = tl.load($(input0_ptr) + ((((r2_z / $(640)) % $(256)) * $(1280)) + (r2_z % $(640)) + $(640)), dtype=tl.bfloat16)
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(163840)])
    tl.store($(vto_copy_1_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(163840))
    r3_value = (tl.load($(vsilu_ptr) + ((((r3_z / $(640)) % $(256)) * $(640)) + (r3_z % $(640))), dtype=tl.float32) * tl.load($(vto_copy_1_ptr) + ((((r3_z / $(640)) % $(256)) * $(640)) + (r3_z % $(640))), dtype=tl.float32))
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(163840)])
    tl.store($(vmul_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(163840))
    r4_value = tl.load($(vmul_ptr) + ((((r4_z / $(640)) % $(256)) * $(640)) + (r4_z % $(640))), dtype=tl.float32)
    r4_value = r4_value.to(tl.bfloat16)
    r4_value = tl.broadcast_to(r4_value, [$(163840)])
    tl.store($(vto_copy_2_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(163840))
    r5_value = (tl.load($(vto_copy_2_ptr) + ((((r5_z / $(640)) % $(256)) * $(640)) + (r5_z % $(640))), dtype=tl.bfloat16) * tl.load($(input1_ptr) + ((r5_z / $(640)) % $(256)), dtype=tl.bfloat16))
    r5_value = r5_value.to(tl.bfloat16)
    r5_value = tl.broadcast_to(r5_value, [$(163840)])
    tl.store($(vmul_1_ptr) + r5_z, r5_value)
    z = tl.arange($(163840))
    value = tl.load($(vmul_1_ptr) + ((((z / $(640)) % $(256)) * $(640)) + (z % $(640))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(163840)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let input1_ptr : RegionName := "input1"
  let out_ptr : RegionName := "out"
  let vsilu_ptr : RegionName := "vsilu"
  let vmul_ptr : RegionName := "vmul"
  let vmul_1_ptr : RegionName := "vmul_1"
  let candidate_vsilu_ptr : RegionName := "candidate_vsilu"
  let candidate_vmul_ptr : RegionName := "candidate_vmul"
  let candidate_vmul_1_ptr : RegionName := "candidate_vmul_1"
  triton {
    r0_z = tl.arange($(163840))
    r0_value = (tl.load($(input0_ptr) + ((((r0_z / $(640)) % $(256)) * $(1280)) + (r0_z % $(640))), dtype=tl.bfloat16) * tl.sigmoid(tl.load($(input0_ptr) + ((((r0_z / $(640)) % $(256)) * $(1280)) + (r0_z % $(640))), dtype=tl.bfloat16)))
    r0_value = r0_value.to(tl.bfloat16)
    r0_value = tl.broadcast_to(r0_value, [$(163840)])
    tl.store($(candidate_vsilu_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(163840))
    r1_value = (tl.load($(candidate_vsilu_ptr) + ((((r1_z / $(640)) % $(256)) * $(640)) + (r1_z % $(640))), dtype=tl.bfloat16) * tl.load($(input0_ptr) + ((((r1_z / $(640)) % $(256)) * $(1280)) + (r1_z % $(640)) + $(640)), dtype=tl.bfloat16))
    r1_value = r1_value.to(tl.bfloat16)
    r1_value = tl.broadcast_to(r1_value, [$(163840)])
    tl.store($(candidate_vmul_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(163840))
    r2_value = (tl.load($(candidate_vmul_ptr) + ((((r2_z / $(640)) % $(256)) * $(640)) + (r2_z % $(640))), dtype=tl.bfloat16) * tl.load($(input1_ptr) + ((r2_z / $(640)) % $(256)), dtype=tl.bfloat16))
    r2_value = r2_value.to(tl.bfloat16)
    r2_value = tl.broadcast_to(r2_value, [$(163840)])
    tl.store($(candidate_vmul_1_ptr) + r2_z, r2_value)
    z = tl.arange($(163840))
    value = tl.load($(candidate_vmul_1_ptr) + ((((z / $(640)) % $(256)) * $(640)) + (z % $(640))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(163840)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_routed_swiglu
