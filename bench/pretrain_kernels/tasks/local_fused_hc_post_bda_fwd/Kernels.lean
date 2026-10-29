import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_fused_hc_post_bda_fwd
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
  let input4_ptr : RegionName := "input4"
  let out_ptr : RegionName := "out"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let vadd_ptr : RegionName := "vadd"
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  let vmul_ptr : RegionName := "vmul"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vto_copy_3_ptr : RegionName := "vto_copy_3"
  let vmul_2_ptr : RegionName := "vmul_2"
  let vadd_1_ptr : RegionName := "vadd_1"
  let vto_copy_4_ptr : RegionName := "vto_copy_4"
  triton {
    r0_z = tl.arange($(16777216))
    r0_value = tl.load($(input0_ptr) + ((((r0_z / $(8388608)) % $(2)) * $(8388608)) + (((r0_z / $(2048)) % $(4096)) * $(2048)) + (r0_z % $(2048))), dtype=tl.bfloat16)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(16777216)])
    tl.store($(vto_copy_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(2048))
    r1_value = tl.load($(input1_ptr) + r1_z, dtype=tl.bfloat16)
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(2048)])
    tl.store($(vto_copy_1_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(16777216))
    r2_value = (tl.load($(vto_copy_ptr) + ((((r2_z / $(8388608)) % $(2)) * $(8388608)) + (((r2_z / $(2048)) % $(4096)) * $(2048)) + (r2_z % $(2048))), dtype=tl.float32) + tl.load($(vto_copy_1_ptr) + (r2_z % $(2048)), dtype=tl.float32))
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(16777216)])
    tl.store($(vadd_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(16777216))
    r3_value = tl.where(tl.load($(input4_ptr) + ((((r3_z / $(8388608)) % $(2)) * $(8388608)) + (((r3_z / $(2048)) % $(4096)) * $(2048)) + (r3_z % $(2048))), dtype=tl.int1), 1.0, 0.0)
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(16777216)])
    tl.store($(vto_copy_2_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(16777216))
    r4_value = (tl.load($(vadd_ptr) + ((((r4_z / $(8388608)) % $(2)) * $(8388608)) + (((r4_z / $(2048)) % $(4096)) * $(2048)) + (r4_z % $(2048))), dtype=tl.float32) * tl.load($(vto_copy_2_ptr) + ((((r4_z / $(8388608)) % $(2)) * $(8388608)) + (((r4_z / $(2048)) % $(4096)) * $(2048)) + (r4_z % $(2048))), dtype=tl.float32))
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(16777216)])
    tl.store($(vmul_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(16777216))
    r5_value = (tl.load($(vmul_ptr) + ((((r5_z / $(8388608)) % $(2)) * $(8388608)) + (((r5_z / $(2048)) % $(4096)) * $(2048)) + (r5_z % $(2048))), dtype=tl.float32) * 1.1111111111111112)
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(16777216)])
    tl.store($(vmul_1_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(67108864))
    r6_value = tl.load($(input2_ptr) + ((((r6_z / $(33554432)) % $(2)) * $(33554432)) + (((r6_z / $(8192)) % $(4096)) * $(8192)) + (((r6_z / $(2048)) % $(4)) * $(2048)) + (r6_z % $(2048))), dtype=tl.bfloat16)
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(67108864)])
    tl.store($(vto_copy_3_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(67108864))
    r7_value = (tl.load($(input3_ptr) + ((((r7_z / $(33554432)) % $(2)) * $(16384)) + (((r7_z / $(8192)) % $(4096)) * $(4)) + ((r7_z / $(2048)) % $(4))), dtype=tl.float32) * tl.load($(vmul_1_ptr) + ((((r7_z / $(33554432)) % $(2)) * $(8388608)) + (((r7_z / $(8192)) % $(4096)) * $(2048)) + (r7_z % $(2048))), dtype=tl.float32))
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(67108864)])
    tl.store($(vmul_2_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(67108864))
    r8_value = (tl.load($(vmul_2_ptr) + ((((r8_z / $(33554432)) % $(2)) * $(33554432)) + (((r8_z / $(8192)) % $(4096)) * $(8192)) + (((r8_z / $(2048)) % $(4)) * $(2048)) + (r8_z % $(2048))), dtype=tl.float32) + tl.load($(vto_copy_3_ptr) + ((((r8_z / $(33554432)) % $(2)) * $(33554432)) + (((r8_z / $(8192)) % $(4096)) * $(8192)) + (((r8_z / $(2048)) % $(4)) * $(2048)) + (r8_z % $(2048))), dtype=tl.float32))
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(67108864)])
    tl.store($(vadd_1_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(67108864))
    r9_value = tl.load($(vadd_1_ptr) + ((((r9_z / $(33554432)) % $(2)) * $(33554432)) + (((r9_z / $(8192)) % $(4096)) * $(8192)) + (((r9_z / $(2048)) % $(4)) * $(2048)) + (r9_z % $(2048))), dtype=tl.float32)
    r9_value = r9_value.to(tl.bfloat16)
    r9_value = tl.broadcast_to(r9_value, [$(67108864)])
    tl.store($(vto_copy_4_ptr) + r9_z, r9_value)
    z = tl.arange($(67108864))
    value = tl.load($(vto_copy_4_ptr) + ((((z / $(33554432)) % $(2)) * $(33554432)) + (((z / $(8192)) % $(4096)) * $(8192)) + (((z / $(2048)) % $(4)) * $(2048)) + (z % $(2048))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(67108864)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let input1_ptr : RegionName := "input1"
  let input2_ptr : RegionName := "input2"
  let input3_ptr : RegionName := "input3"
  let input4_ptr : RegionName := "input4"
  let out_ptr : RegionName := "out"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let vadd_ptr : RegionName := "vadd"
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  let vmul_ptr : RegionName := "vmul"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vto_copy_3_ptr : RegionName := "vto_copy_3"
  let vmul_2_ptr : RegionName := "vmul_2"
  let vadd_1_ptr : RegionName := "vadd_1"
  let vto_copy_4_ptr : RegionName := "vto_copy_4"
  let candidate_vto_copy_ptr : RegionName := "candidate_vto_copy"
  let candidate_vto_copy_1_ptr : RegionName := "candidate_vto_copy_1"
  let candidate_vadd_ptr : RegionName := "candidate_vadd"
  let candidate_vto_copy_2_ptr : RegionName := "candidate_vto_copy_2"
  let candidate_vmul_ptr : RegionName := "candidate_vmul"
  let candidate_vmul_1_ptr : RegionName := "candidate_vmul_1"
  let candidate_vto_copy_3_ptr : RegionName := "candidate_vto_copy_3"
  let candidate_vmul_2_ptr : RegionName := "candidate_vmul_2"
  let candidate_vadd_1_ptr : RegionName := "candidate_vadd_1"
  let candidate_vto_copy_4_ptr : RegionName := "candidate_vto_copy_4"
  triton {
    r0_z = tl.arange($(16777216))
    r0_value = tl.load($(input0_ptr) + ((((r0_z / $(8388608)) % $(2)) * $(8388608)) + (((r0_z / $(2048)) % $(4096)) * $(2048)) + (r0_z % $(2048))), dtype=tl.bfloat16)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(16777216)])
    tl.store($(candidate_vto_copy_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(2048))
    r1_value = tl.load($(input1_ptr) + r1_z, dtype=tl.bfloat16)
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(2048)])
    tl.store($(candidate_vto_copy_1_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(16777216))
    r2_value = (tl.load($(candidate_vto_copy_ptr) + ((((r2_z / $(8388608)) % $(2)) * $(8388608)) + (((r2_z / $(2048)) % $(4096)) * $(2048)) + (r2_z % $(2048))), dtype=tl.float32) + tl.load($(candidate_vto_copy_1_ptr) + (r2_z % $(2048)), dtype=tl.float32))
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(16777216)])
    tl.store($(candidate_vadd_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(16777216))
    r3_value = tl.where(tl.load($(input4_ptr) + ((((r3_z / $(8388608)) % $(2)) * $(8388608)) + (((r3_z / $(2048)) % $(4096)) * $(2048)) + (r3_z % $(2048))), dtype=tl.int1), 1.0, 0.0)
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(16777216)])
    tl.store($(candidate_vto_copy_2_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(16777216))
    r4_value = (tl.load($(candidate_vadd_ptr) + ((((r4_z / $(8388608)) % $(2)) * $(8388608)) + (((r4_z / $(2048)) % $(4096)) * $(2048)) + (r4_z % $(2048))), dtype=tl.float32) * tl.load($(candidate_vto_copy_2_ptr) + ((((r4_z / $(8388608)) % $(2)) * $(8388608)) + (((r4_z / $(2048)) % $(4096)) * $(2048)) + (r4_z % $(2048))), dtype=tl.float32))
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(16777216)])
    tl.store($(candidate_vmul_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(16777216))
    r5_value = (tl.load($(candidate_vmul_ptr) + ((((r5_z / $(8388608)) % $(2)) * $(8388608)) + (((r5_z / $(2048)) % $(4096)) * $(2048)) + (r5_z % $(2048))), dtype=tl.float32) * 1.1111111111111112)
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(16777216)])
    tl.store($(candidate_vmul_1_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(67108864))
    r6_value = tl.load($(input2_ptr) + ((((r6_z / $(33554432)) % $(2)) * $(33554432)) + (((r6_z / $(8192)) % $(4096)) * $(8192)) + (((r6_z / $(2048)) % $(4)) * $(2048)) + (r6_z % $(2048))), dtype=tl.bfloat16)
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(67108864)])
    tl.store($(candidate_vto_copy_3_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(67108864))
    r7_value = (tl.load($(input3_ptr) + ((((r7_z / $(33554432)) % $(2)) * $(16384)) + (((r7_z / $(8192)) % $(4096)) * $(4)) + ((r7_z / $(2048)) % $(4))), dtype=tl.float32) * tl.load($(candidate_vmul_1_ptr) + ((((r7_z / $(33554432)) % $(2)) * $(8388608)) + (((r7_z / $(8192)) % $(4096)) * $(2048)) + (r7_z % $(2048))), dtype=tl.float32))
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(67108864)])
    tl.store($(candidate_vmul_2_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(67108864))
    r8_value = (tl.load($(candidate_vmul_2_ptr) + ((((r8_z / $(33554432)) % $(2)) * $(33554432)) + (((r8_z / $(8192)) % $(4096)) * $(8192)) + (((r8_z / $(2048)) % $(4)) * $(2048)) + (r8_z % $(2048))), dtype=tl.float32) + tl.load($(candidate_vto_copy_3_ptr) + ((((r8_z / $(33554432)) % $(2)) * $(33554432)) + (((r8_z / $(8192)) % $(4096)) * $(8192)) + (((r8_z / $(2048)) % $(4)) * $(2048)) + (r8_z % $(2048))), dtype=tl.float32))
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(67108864)])
    tl.store($(candidate_vadd_1_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(67108864))
    r9_value = tl.load($(candidate_vadd_1_ptr) + ((((r9_z / $(33554432)) % $(2)) * $(33554432)) + (((r9_z / $(8192)) % $(4096)) * $(8192)) + (((r9_z / $(2048)) % $(4)) * $(2048)) + (r9_z % $(2048))), dtype=tl.float32)
    r9_value = r9_value.to(tl.bfloat16)
    r9_value = tl.broadcast_to(r9_value, [$(67108864)])
    tl.store($(candidate_vto_copy_4_ptr) + r9_z, r9_value)
    z = tl.arange($(67108864))
    value = tl.load($(candidate_vto_copy_4_ptr) + ((((z / $(33554432)) % $(2)) * $(33554432)) + (((z / $(8192)) % $(4096)) * $(8192)) + (((z / $(2048)) % $(4)) * $(2048)) + (z % $(2048))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(67108864)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_fused_hc_post_bda_fwd
