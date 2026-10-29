import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_triton_matmul
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Kernels

def original : ComputeKernel :=
  let parameter0_ptr : RegionName := "parameter0"
  let parameter1_ptr : RegionName := "parameter1"
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vmul_ptr : RegionName := "vmul"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vaddmm_ptr : RegionName := "vaddmm"
  triton {
    r0_z = tl.arange($(16705))
    r0_value = (tl.load($(parameter0_ptr) + ((((r0_z / $(257)) % $(65)) * $(257)) + (r0_z % $(257))), dtype=tl.bfloat16) * tl.cast(0.02, tl.bfloat16))
    r0_value = r0_value.to(tl.bfloat16)
    r0_value = tl.broadcast_to(r0_value, [$(16705)])
    tl.store($(vmul_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(65))
    r1_value = (tl.load($(parameter1_ptr) + r1_z, dtype=tl.bfloat16) * tl.cast(0.02, tl.bfloat16))
    r1_value = r1_value.to(tl.bfloat16)
    r1_value = tl.broadcast_to(r1_value, [$(65)])
    tl.store($(vmul_1_ptr) + r1_z, r1_value)
    r2_row = tl.arange($(65))
    r2_col = tl.arange($(65))
    r2_lane = tl.arange($(257))
    r2_left = tl.load($(input0_ptr) + (((r2_row[:, None] % $(65)) * $(257)) + (r2_lane[None, :] % $(257))), dtype=tl.bfloat16)
    r2_right = tl.load($(vmul_ptr) + (((r2_col[None, :] % $(65)) * $(257)) + (r2_lane[:, None] % $(257))), dtype=tl.bfloat16)
    r2_value = tl.dot(r2_left, r2_right)
    r2_value = r2_value + tl.cast(tl.load($(vmul_1_ptr) + (r2_col[None, :] % $(65)), dtype=tl.bfloat16), tl.float32) * 1.0
    r2_value = r2_value.to(tl.bfloat16)
    tl.store($(vaddmm_ptr) + ((r2_row[:, None] * $(65)) + r2_col[None, :]), r2_value)
    z = tl.arange($(4225))
    value = tl.load($(vaddmm_ptr) + ((((z / $(65)) % $(65)) * $(65)) + (z % $(65))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(4225)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let parameter0_ptr : RegionName := "parameter0"
  let parameter1_ptr : RegionName := "parameter1"
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vmul_ptr : RegionName := "vmul"
  let vmul_1_ptr : RegionName := "vmul_1"
  let candidate_vmul_ptr : RegionName := "candidate_vmul"
  let candidate_vmul_1_ptr : RegionName := "candidate_vmul_1"
  let candidate_vmm_ptr : RegionName := "candidate_vmm"
  let candidate_vto_copy_ptr : RegionName := "candidate_vto_copy"
  let candidate_vadd_ptr : RegionName := "candidate_vadd"
  let candidate_vto_copy_1_ptr : RegionName := "candidate_vto_copy_1"
  triton {
    r0_z = tl.arange($(16705))
    r0_value = (tl.load($(parameter0_ptr) + ((((r0_z / $(257)) % $(65)) * $(257)) + (r0_z % $(257))), dtype=tl.bfloat16) * tl.cast(0.02, tl.bfloat16))
    r0_value = r0_value.to(tl.bfloat16)
    r0_value = tl.broadcast_to(r0_value, [$(16705)])
    tl.store($(candidate_vmul_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(65))
    r1_value = (tl.load($(parameter1_ptr) + r1_z, dtype=tl.bfloat16) * tl.cast(0.02, tl.bfloat16))
    r1_value = r1_value.to(tl.bfloat16)
    r1_value = tl.broadcast_to(r1_value, [$(65)])
    tl.store($(candidate_vmul_1_ptr) + r1_z, r1_value)
    r2_row = tl.arange($(65))
    r2_col = tl.arange($(65))
    r2_lane = tl.arange($(257))
    r2_left = tl.load($(input0_ptr) + (((r2_row[:, None] % $(65)) * $(257)) + (r2_lane[None, :] % $(257))), dtype=tl.bfloat16)
    r2_right = tl.load($(candidate_vmul_ptr) + (((r2_col[None, :] % $(65)) * $(257)) + (r2_lane[:, None] % $(257))), dtype=tl.bfloat16)
    r2_value = tl.dot(r2_left, r2_right)
    r2_value = r2_value.to(tl.float32)
    tl.store($(candidate_vmm_ptr) + ((r2_row[:, None] * $(65)) + r2_col[None, :]), r2_value)
    r3_z = tl.arange($(65))
    r3_value = tl.load($(candidate_vmul_1_ptr) + r3_z, dtype=tl.bfloat16)
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(65)])
    tl.store($(candidate_vto_copy_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(4225))
    r4_value = (tl.load($(candidate_vmm_ptr) + ((((r4_z / $(65)) % $(65)) * $(65)) + (r4_z % $(65))), dtype=tl.float32) + tl.load($(candidate_vto_copy_ptr) + (r4_z % $(65)), dtype=tl.float32))
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(4225)])
    tl.store($(candidate_vadd_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(4225))
    r5_value = tl.load($(candidate_vadd_ptr) + ((((r5_z / $(65)) % $(65)) * $(65)) + (r5_z % $(65))), dtype=tl.float32)
    r5_value = r5_value.to(tl.bfloat16)
    r5_value = tl.broadcast_to(r5_value, [$(4225)])
    tl.store($(candidate_vto_copy_1_ptr) + r5_z, r5_value)
    z = tl.arange($(4225))
    value = tl.load($(candidate_vto_copy_1_ptr) + ((((z / $(65)) % $(65)) * $(65)) + (z % $(65))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(4225)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_triton_matmul
