import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_deepgemm_grouped_accum_wgrad
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
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  let vto_copy_3_ptr : RegionName := "vto_copy_3"
  let vmm_2_ptr : RegionName := "vmm_2"
  let vselect_backward_ptr : RegionName := "vselect_backward"
  let vmm_3_ptr : RegionName := "vmm_3"
  let vselect_backward_1_ptr : RegionName := "vselect_backward_1"
  let vadd_ptr : RegionName := "vadd"
  triton {
    r0_z = tl.arange($(524288))
    r0_value = tl.load($(input0_ptr) + ((((r0_z / $(2048)) % $(256)) * $(2048)) + (r0_z % $(2048))), dtype=tl.bfloat16)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(524288)])
    tl.store($(vto_copy_2_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(163840))
    r1_value = tl.load($(input1_ptr) + ((((r1_z / $(640)) % $(256)) * $(640)) + (r1_z % $(640))), dtype=tl.bfloat16)
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(163840)])
    tl.store($(vto_copy_3_ptr) + r1_z, r1_value)
    r2_row = tl.arange($(640))
    r2_col = tl.arange($(2048))
    r2_lane = tl.arange($(128))
    r2_left = tl.load($(vto_copy_3_ptr) + ((((r2_lane[None, :] + $(128)) % $(256)) * $(640)) + (r2_row[:, None] % $(640))), dtype=tl.float32)
    r2_right = tl.load($(vto_copy_2_ptr) + ((((r2_lane[:, None] + $(128)) % $(256)) * $(2048)) + (r2_col[None, :] % $(2048))), dtype=tl.float32)
    r2_value = tl.dot(r2_left, r2_right)
    r2_value = r2_value.to(tl.float32)
    tl.store($(vmm_2_ptr) + ((r2_row[:, None] * $(2048)) + r2_col[None, :]), r2_value)
    r3_z = tl.arange($(2621440))
    r3_value = tl.where((((r3_z / $(1310720)) % $(2)) == $(1)), tl.load($(vmm_2_ptr) + ((((r3_z / $(2048)) % $(640)) * $(2048)) + (r3_z % $(2048))), dtype=tl.float32), 0.0)
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(2621440)])
    tl.store($(vselect_backward_ptr) + r3_z, r3_value)
    r4_row = tl.arange($(640))
    r4_col = tl.arange($(2048))
    r4_lane = tl.arange($(128))
    r4_left = tl.load($(vto_copy_3_ptr) + (((r4_lane[None, :] % $(256)) * $(640)) + (r4_row[:, None] % $(640))), dtype=tl.float32)
    r4_right = tl.load($(vto_copy_2_ptr) + (((r4_lane[:, None] % $(256)) * $(2048)) + (r4_col[None, :] % $(2048))), dtype=tl.float32)
    r4_value = tl.dot(r4_left, r4_right)
    r4_value = r4_value.to(tl.float32)
    tl.store($(vmm_3_ptr) + ((r4_row[:, None] * $(2048)) + r4_col[None, :]), r4_value)
    r5_z = tl.arange($(2621440))
    r5_value = tl.where((((r5_z / $(1310720)) % $(2)) == $(0)), tl.load($(vmm_3_ptr) + ((((r5_z / $(2048)) % $(640)) * $(2048)) + (r5_z % $(2048))), dtype=tl.float32), 0.0)
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(2621440)])
    tl.store($(vselect_backward_1_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(2621440))
    r6_value = (tl.load($(vselect_backward_ptr) + ((((r6_z / $(1310720)) % $(2)) * $(1310720)) + (((r6_z / $(2048)) % $(640)) * $(2048)) + (r6_z % $(2048))), dtype=tl.float32) + tl.load($(vselect_backward_1_ptr) + ((((r6_z / $(1310720)) % $(2)) * $(1310720)) + (((r6_z / $(2048)) % $(640)) * $(2048)) + (r6_z % $(2048))), dtype=tl.float32))
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(2621440)])
    tl.store($(vadd_ptr) + r6_z, r6_value)
    z = tl.arange($(2621440))
    value = tl.load($(vadd_ptr) + ((((z / $(1310720)) % $(2)) * $(1310720)) + (((z / $(2048)) % $(640)) * $(2048)) + (z % $(2048))), dtype=tl.float32)
    value = tl.broadcast_to(value, [$(2621440)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let input1_ptr : RegionName := "input1"
  let parameter0_ptr : RegionName := "parameter0"
  let out_ptr : RegionName := "out"
  let vmm_2_ptr : RegionName := "vmm_2"
  let vmm_3_ptr : RegionName := "vmm_3"
  let vadd_ptr : RegionName := "vadd"
  let candidate_vmul_ptr : RegionName := "candidate_vmul"
  let candidate_vmm_2_ptr : RegionName := "candidate_vmm_2"
  let candidate_vmm_3_ptr : RegionName := "candidate_vmm_3"
  let candidate_vstack_ptr : RegionName := "candidate_vstack"
  let candidate_vadd_ptr : RegionName := "candidate_vadd"
  triton {
    r0_z = tl.arange($(2621440))
    r0_value = (tl.load($(parameter0_ptr) + ((((r0_z / $(1310720)) % $(2)) * $(1310720)) + (((r0_z / $(2048)) % $(640)) * $(2048)) + (r0_z % $(2048))), dtype=tl.bfloat16) * tl.cast(0.02, tl.bfloat16))
    r0_value = r0_value.to(tl.bfloat16)
    r0_value = tl.broadcast_to(r0_value, [$(2621440)])
    tl.store($(candidate_vmul_ptr) + r0_z, r0_value)
    r1_row = tl.arange($(640))
    r1_col = tl.arange($(2048))
    r1_lane = tl.arange($(128))
    r1_left = tl.load($(input1_ptr) + ((((((r1_lane[None, :] * $(640)) + r1_row[:, None]) / $(640)) % $(256)) * $(640)) + (((r1_lane[None, :] * $(640)) + r1_row[:, None]) % $(640))), dtype=tl.bfloat16)
    r1_right = tl.load($(input0_ptr) + (((r1_lane[:, None] % $(256)) * $(2048)) + (r1_col[None, :] % $(2048))), dtype=tl.bfloat16)
    r1_value = tl.dot(r1_left, r1_right)
    r1_value = r1_value.to(tl.float32)
    tl.store($(candidate_vmm_2_ptr) + ((r1_row[:, None] * $(2048)) + r1_col[None, :]), r1_value)
    r2_row = tl.arange($(640))
    r2_col = tl.arange($(2048))
    r2_lane = tl.arange($(128))
    r2_left = tl.load($(input1_ptr) + ((((((r2_lane[None, :] * $(640)) + r2_row[:, None] + $(81920)) / $(640)) % $(256)) * $(640)) + (((r2_lane[None, :] * $(640)) + r2_row[:, None] + $(81920)) % $(640))), dtype=tl.bfloat16)
    r2_right = tl.load($(input0_ptr) + ((((r2_lane[:, None] + $(128)) % $(256)) * $(2048)) + (r2_col[None, :] % $(2048))), dtype=tl.bfloat16)
    r2_value = tl.dot(r2_left, r2_right)
    r2_value = r2_value.to(tl.float32)
    tl.store($(candidate_vmm_3_ptr) + ((r2_row[:, None] * $(2048)) + r2_col[None, :]), r2_value)
    r3_z = tl.arange($(2621440))
    r3_value = tl.where(((r3_z / $(1310720)) % $(2)) < $(1), tl.load($(candidate_vmm_2_ptr) + ((((r3_z / $(2048)) % $(640)) * $(2048)) + (r3_z % $(2048))), dtype=tl.float32), tl.load($(candidate_vmm_3_ptr) + ((((r3_z / $(2048)) % $(640)) * $(2048)) + (r3_z % $(2048))), dtype=tl.float32))
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(2621440)])
    tl.store($(candidate_vstack_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(2621440))
    r4_value = (0.0 + tl.load($(candidate_vstack_ptr) + ((((r4_z / $(1310720)) % $(2)) * $(1310720)) + (((r4_z / $(2048)) % $(640)) * $(2048)) + (r4_z % $(2048))), dtype=tl.float32))
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(2621440)])
    tl.store($(candidate_vadd_ptr) + r4_z, r4_value)
    z = tl.arange($(2621440))
    value = tl.load($(candidate_vadd_ptr) + ((((z / $(1310720)) % $(2)) * $(1310720)) + (((z / $(2048)) % $(640)) * $(2048)) + (z % $(2048))), dtype=tl.float32)
    value = tl.broadcast_to(value, [$(2621440)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_deepgemm_grouped_accum_wgrad
