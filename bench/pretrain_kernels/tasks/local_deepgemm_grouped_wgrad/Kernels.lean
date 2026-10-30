import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_deepgemm_grouped_wgrad
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
  let vmm_2_ptr : RegionName := "vmm_2"
  let vselect_backward_ptr : RegionName := "vselect_backward"
  let vmm_3_ptr : RegionName := "vmm_3"
  let vselect_backward_1_ptr : RegionName := "vselect_backward_1"
  let vadd_ptr : RegionName := "vadd"
  triton {
    r0_row = tl.arange($(640))
    r0_col = tl.arange($(2048))
    r0_lane = tl.arange($(128))
    r0_left = tl.load($(input1_ptr) + ((((r0_lane[None, :] + $(128)) % $(256)) * $(640)) + (r0_row[:, None] % $(640))), dtype=tl.bfloat16)
    r0_right = tl.load($(input0_ptr) + ((((r0_lane[:, None] + $(128)) % $(256)) * $(2048)) + (r0_col[None, :] % $(2048))), dtype=tl.bfloat16)
    r0_value = tl.dot(r0_left, r0_right)
    r0_value = r0_value.to(tl.bfloat16)
    tl.store($(vmm_2_ptr) + ((r0_row[:, None] * $(2048)) + r0_col[None, :]), r0_value)
    r1_z = tl.arange($(2621440))
    r1_value = tl.where((((r1_z / $(1310720)) % $(2)) == $(1)), tl.load($(vmm_2_ptr) + ((((r1_z / $(2048)) % $(640)) * $(2048)) + (r1_z % $(2048))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r1_value = r1_value.to(tl.bfloat16)
    r1_value = tl.broadcast_to(r1_value, [$(2621440)])
    tl.store($(vselect_backward_ptr) + r1_z, r1_value)
    r2_row = tl.arange($(640))
    r2_col = tl.arange($(2048))
    r2_lane = tl.arange($(128))
    r2_left = tl.load($(input1_ptr) + (((r2_lane[None, :] % $(256)) * $(640)) + (r2_row[:, None] % $(640))), dtype=tl.bfloat16)
    r2_right = tl.load($(input0_ptr) + (((r2_lane[:, None] % $(256)) * $(2048)) + (r2_col[None, :] % $(2048))), dtype=tl.bfloat16)
    r2_value = tl.dot(r2_left, r2_right)
    r2_value = r2_value.to(tl.bfloat16)
    tl.store($(vmm_3_ptr) + ((r2_row[:, None] * $(2048)) + r2_col[None, :]), r2_value)
    r3_z = tl.arange($(2621440))
    r3_value = tl.where((((r3_z / $(1310720)) % $(2)) == $(0)), tl.load($(vmm_3_ptr) + ((((r3_z / $(2048)) % $(640)) * $(2048)) + (r3_z % $(2048))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r3_value = r3_value.to(tl.bfloat16)
    r3_value = tl.broadcast_to(r3_value, [$(2621440)])
    tl.store($(vselect_backward_1_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(2621440))
    r4_value = (tl.load($(vselect_backward_ptr) + ((((r4_z / $(1310720)) % $(2)) * $(1310720)) + (((r4_z / $(2048)) % $(640)) * $(2048)) + (r4_z % $(2048))), dtype=tl.bfloat16) + tl.load($(vselect_backward_1_ptr) + ((((r4_z / $(1310720)) % $(2)) * $(1310720)) + (((r4_z / $(2048)) % $(640)) * $(2048)) + (r4_z % $(2048))), dtype=tl.bfloat16))
    r4_value = r4_value.to(tl.bfloat16)
    r4_value = tl.broadcast_to(r4_value, [$(2621440)])
    tl.store($(vadd_ptr) + r4_z, r4_value)
    z = tl.arange($(2621440))
    value = tl.load($(vadd_ptr) + ((((z / $(1310720)) % $(2)) * $(1310720)) + (((z / $(2048)) % $(640)) * $(2048)) + (z % $(2048))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(2621440)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let input1_ptr : RegionName := "input1"
  let out_ptr : RegionName := "out"
  let vmm_2_ptr : RegionName := "vmm_2"
  let vmm_3_ptr : RegionName := "vmm_3"
  let candidate_vmm_2_ptr : RegionName := "candidate_vmm_2"
  let candidate_vmm_3_ptr : RegionName := "candidate_vmm_3"
  let candidate_vstack_ptr : RegionName := "candidate_vstack"
  triton {
    r0_row = tl.arange($(640))
    r0_col = tl.arange($(2048))
    r0_lane = tl.arange($(128))
    r0_left = tl.load($(input1_ptr) + ((((((r0_lane[None, :] * $(640)) + r0_row[:, None]) / $(640)) % $(256)) * $(640)) + (((r0_lane[None, :] * $(640)) + r0_row[:, None]) % $(640))), dtype=tl.bfloat16)
    r0_right = tl.load($(input0_ptr) + (((r0_lane[:, None] % $(256)) * $(2048)) + (r0_col[None, :] % $(2048))), dtype=tl.bfloat16)
    r0_value = tl.dot(r0_left, r0_right)
    r0_value = r0_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_2_ptr) + ((r0_row[:, None] * $(2048)) + r0_col[None, :]), r0_value)
    r1_row = tl.arange($(640))
    r1_col = tl.arange($(2048))
    r1_lane = tl.arange($(128))
    r1_left = tl.load($(input1_ptr) + ((((((r1_lane[None, :] * $(640)) + r1_row[:, None] + $(81920)) / $(640)) % $(256)) * $(640)) + (((r1_lane[None, :] * $(640)) + r1_row[:, None] + $(81920)) % $(640))), dtype=tl.bfloat16)
    r1_right = tl.load($(input0_ptr) + ((((r1_lane[:, None] + $(128)) % $(256)) * $(2048)) + (r1_col[None, :] % $(2048))), dtype=tl.bfloat16)
    r1_value = tl.dot(r1_left, r1_right)
    r1_value = r1_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_3_ptr) + ((r1_row[:, None] * $(2048)) + r1_col[None, :]), r1_value)
    r2_z = tl.arange($(2621440))
    r2_value = tl.where(((r2_z / $(1310720)) % $(2)) < $(1), tl.load($(candidate_vmm_2_ptr) + ((((r2_z / $(2048)) % $(640)) * $(2048)) + (r2_z % $(2048))), dtype=tl.bfloat16), tl.load($(candidate_vmm_3_ptr) + ((((r2_z / $(2048)) % $(640)) * $(2048)) + (r2_z % $(2048))), dtype=tl.bfloat16))
    r2_value = r2_value.to(tl.bfloat16)
    r2_value = tl.broadcast_to(r2_value, [$(2621440)])
    tl.store($(candidate_vstack_ptr) + r2_z, r2_value)
    z = tl.arange($(2621440))
    value = tl.load($(candidate_vstack_ptr) + ((((z / $(1310720)) % $(2)) * $(1310720)) + (((z / $(2048)) % $(640)) * $(2048)) + (z % $(2048))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(2621440)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_deepgemm_grouped_wgrad
