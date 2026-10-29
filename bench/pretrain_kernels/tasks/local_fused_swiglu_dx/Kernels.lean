import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_fused_swiglu_dx
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
  let vto_copy_3_ptr : RegionName := "vto_copy_3"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vmul_2_ptr : RegionName := "vmul_2"
  let vto_copy_4_ptr : RegionName := "vto_copy_4"
  let vsilu_backward_ptr : RegionName := "vsilu_backward"
  let vto_copy_5_ptr : RegionName := "vto_copy_5"
  let vcat_ptr : RegionName := "vcat"
  triton {
    r0_z = tl.arange($(2621440))
    r0_value = tl.load($(input0_ptr) + ((((r0_z / $(640)) % $(4096)) * $(1280)) + (r0_z % $(640))), dtype=tl.bfloat16)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(2621440)])
    tl.store($(vto_copy_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(2621440))
    r1_value = (tl.load($(vto_copy_ptr) + ((((r1_z / $(640)) % $(4096)) * $(640)) + (r1_z % $(640))), dtype=tl.float32) * tl.sigmoid(tl.load($(vto_copy_ptr) + ((((r1_z / $(640)) % $(4096)) * $(640)) + (r1_z % $(640))), dtype=tl.float32)))
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(2621440)])
    tl.store($(vsilu_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(2621440))
    r2_value = tl.load($(input0_ptr) + ((((r2_z / $(640)) % $(4096)) * $(1280)) + (r2_z % $(640)) + $(640)), dtype=tl.bfloat16)
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(2621440)])
    tl.store($(vto_copy_1_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(2621440))
    r3_value = tl.load($(input1_ptr) + ((((r3_z / $(640)) % $(4096)) * $(640)) + (r3_z % $(640))), dtype=tl.bfloat16)
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(2621440)])
    tl.store($(vto_copy_3_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(2621440))
    r4_value = (tl.load($(vto_copy_3_ptr) + ((((r4_z / $(640)) % $(4096)) * $(640)) + (r4_z % $(640))), dtype=tl.float32) * tl.load($(vsilu_ptr) + ((((r4_z / $(640)) % $(4096)) * $(640)) + (r4_z % $(640))), dtype=tl.float32))
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(2621440)])
    tl.store($(vmul_1_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(2621440))
    r5_value = (tl.load($(vto_copy_3_ptr) + ((((r5_z / $(640)) % $(4096)) * $(640)) + (r5_z % $(640))), dtype=tl.float32) * tl.load($(vto_copy_1_ptr) + ((((r5_z / $(640)) % $(4096)) * $(640)) + (r5_z % $(640))), dtype=tl.float32))
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(2621440)])
    tl.store($(vmul_2_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(2621440))
    r6_value = tl.load($(vmul_1_ptr) + ((((r6_z / $(640)) % $(4096)) * $(640)) + (r6_z % $(640))), dtype=tl.float32)
    r6_value = r6_value.to(tl.bfloat16)
    r6_value = tl.broadcast_to(r6_value, [$(2621440)])
    tl.store($(vto_copy_4_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(2621440))
    r7_value = (tl.load($(vmul_2_ptr) + ((((r7_z / $(640)) % $(4096)) * $(640)) + (r7_z % $(640))), dtype=tl.float32) * tl.sigmoid(tl.load($(vto_copy_ptr) + ((((r7_z / $(640)) % $(4096)) * $(640)) + (r7_z % $(640))), dtype=tl.float32)) * (1.0 + tl.load($(vto_copy_ptr) + ((((r7_z / $(640)) % $(4096)) * $(640)) + (r7_z % $(640))), dtype=tl.float32) * (1.0 - tl.sigmoid(tl.load($(vto_copy_ptr) + ((((r7_z / $(640)) % $(4096)) * $(640)) + (r7_z % $(640))), dtype=tl.float32)))))
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(2621440)])
    tl.store($(vsilu_backward_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(2621440))
    r8_value = tl.load($(vsilu_backward_ptr) + ((((r8_z / $(640)) % $(4096)) * $(640)) + (r8_z % $(640))), dtype=tl.float32)
    r8_value = r8_value.to(tl.bfloat16)
    r8_value = tl.broadcast_to(r8_value, [$(2621440)])
    tl.store($(vto_copy_5_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(5242880))
    r9_value = tl.where((r9_z % $(1280)) < $(640), tl.load($(vto_copy_5_ptr) + ((((r9_z / $(1280)) % $(4096)) * $(640)) + (((r9_z % $(1280)) - $(0)) % $(640))), dtype=tl.bfloat16), tl.load($(vto_copy_4_ptr) + ((((r9_z / $(1280)) % $(4096)) * $(640)) + (((r9_z % $(1280)) - $(640)) % $(640))), dtype=tl.bfloat16))
    r9_value = r9_value.to(tl.bfloat16)
    r9_value = tl.broadcast_to(r9_value, [$(5242880)])
    tl.store($(vcat_ptr) + r9_z, r9_value)
    z = tl.arange($(5242880))
    value = tl.load($(vcat_ptr) + ((((z / $(1280)) % $(4096)) * $(1280)) + (z % $(1280))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(5242880)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let input1_ptr : RegionName := "input1"
  let out_ptr : RegionName := "out"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vsilu_ptr : RegionName := "vsilu"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let vto_copy_3_ptr : RegionName := "vto_copy_3"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vmul_2_ptr : RegionName := "vmul_2"
  let vto_copy_4_ptr : RegionName := "vto_copy_4"
  let vsilu_backward_ptr : RegionName := "vsilu_backward"
  let vto_copy_5_ptr : RegionName := "vto_copy_5"
  let vcat_ptr : RegionName := "vcat"
  let candidate_vto_copy_ptr : RegionName := "candidate_vto_copy"
  let candidate_vsilu_ptr : RegionName := "candidate_vsilu"
  let candidate_vto_copy_1_ptr : RegionName := "candidate_vto_copy_1"
  let candidate_vto_copy_3_ptr : RegionName := "candidate_vto_copy_3"
  let candidate_vmul_1_ptr : RegionName := "candidate_vmul_1"
  let candidate_vmul_2_ptr : RegionName := "candidate_vmul_2"
  let candidate_vto_copy_4_ptr : RegionName := "candidate_vto_copy_4"
  let candidate_vsilu_backward_ptr : RegionName := "candidate_vsilu_backward"
  let candidate_vto_copy_5_ptr : RegionName := "candidate_vto_copy_5"
  let candidate_vcat_ptr : RegionName := "candidate_vcat"
  triton {
    r0_z = tl.arange($(2621440))
    r0_value = tl.load($(input0_ptr) + ((((r0_z / $(640)) % $(4096)) * $(1280)) + (r0_z % $(640))), dtype=tl.bfloat16)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(2621440)])
    tl.store($(candidate_vto_copy_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(2621440))
    r1_value = (tl.load($(candidate_vto_copy_ptr) + ((((r1_z / $(640)) % $(4096)) * $(640)) + (r1_z % $(640))), dtype=tl.float32) * tl.sigmoid(tl.load($(candidate_vto_copy_ptr) + ((((r1_z / $(640)) % $(4096)) * $(640)) + (r1_z % $(640))), dtype=tl.float32)))
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(2621440)])
    tl.store($(candidate_vsilu_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(2621440))
    r2_value = tl.load($(input0_ptr) + ((((r2_z / $(640)) % $(4096)) * $(1280)) + (r2_z % $(640)) + $(640)), dtype=tl.bfloat16)
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(2621440)])
    tl.store($(candidate_vto_copy_1_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(2621440))
    r3_value = tl.load($(input1_ptr) + ((((r3_z / $(640)) % $(4096)) * $(640)) + (r3_z % $(640))), dtype=tl.bfloat16)
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(2621440)])
    tl.store($(candidate_vto_copy_3_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(2621440))
    r4_value = (tl.load($(candidate_vto_copy_3_ptr) + ((((r4_z / $(640)) % $(4096)) * $(640)) + (r4_z % $(640))), dtype=tl.float32) * tl.load($(candidate_vsilu_ptr) + ((((r4_z / $(640)) % $(4096)) * $(640)) + (r4_z % $(640))), dtype=tl.float32))
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(2621440)])
    tl.store($(candidate_vmul_1_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(2621440))
    r5_value = (tl.load($(candidate_vto_copy_3_ptr) + ((((r5_z / $(640)) % $(4096)) * $(640)) + (r5_z % $(640))), dtype=tl.float32) * tl.load($(candidate_vto_copy_1_ptr) + ((((r5_z / $(640)) % $(4096)) * $(640)) + (r5_z % $(640))), dtype=tl.float32))
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(2621440)])
    tl.store($(candidate_vmul_2_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(2621440))
    r6_value = tl.load($(candidate_vmul_1_ptr) + ((((r6_z / $(640)) % $(4096)) * $(640)) + (r6_z % $(640))), dtype=tl.float32)
    r6_value = r6_value.to(tl.bfloat16)
    r6_value = tl.broadcast_to(r6_value, [$(2621440)])
    tl.store($(candidate_vto_copy_4_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(2621440))
    r7_value = (tl.load($(candidate_vmul_2_ptr) + ((((r7_z / $(640)) % $(4096)) * $(640)) + (r7_z % $(640))), dtype=tl.float32) * tl.sigmoid(tl.load($(candidate_vto_copy_ptr) + ((((r7_z / $(640)) % $(4096)) * $(640)) + (r7_z % $(640))), dtype=tl.float32)) * (1.0 + tl.load($(candidate_vto_copy_ptr) + ((((r7_z / $(640)) % $(4096)) * $(640)) + (r7_z % $(640))), dtype=tl.float32) * (1.0 - tl.sigmoid(tl.load($(candidate_vto_copy_ptr) + ((((r7_z / $(640)) % $(4096)) * $(640)) + (r7_z % $(640))), dtype=tl.float32)))))
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(2621440)])
    tl.store($(candidate_vsilu_backward_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(2621440))
    r8_value = tl.load($(candidate_vsilu_backward_ptr) + ((((r8_z / $(640)) % $(4096)) * $(640)) + (r8_z % $(640))), dtype=tl.float32)
    r8_value = r8_value.to(tl.bfloat16)
    r8_value = tl.broadcast_to(r8_value, [$(2621440)])
    tl.store($(candidate_vto_copy_5_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(5242880))
    r9_value = tl.where((r9_z % $(1280)) < $(640), tl.load($(candidate_vto_copy_5_ptr) + ((((r9_z / $(1280)) % $(4096)) * $(640)) + (((r9_z % $(1280)) - $(0)) % $(640))), dtype=tl.bfloat16), tl.load($(candidate_vto_copy_4_ptr) + ((((r9_z / $(1280)) % $(4096)) * $(640)) + (((r9_z % $(1280)) - $(640)) % $(640))), dtype=tl.bfloat16))
    r9_value = r9_value.to(tl.bfloat16)
    r9_value = tl.broadcast_to(r9_value, [$(5242880)])
    tl.store($(candidate_vcat_ptr) + r9_z, r9_value)
    z = tl.arange($(5242880))
    value = tl.load($(candidate_vcat_ptr) + ((((z / $(1280)) % $(4096)) * $(1280)) + (z % $(1280))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(5242880)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_fused_swiglu_dx
