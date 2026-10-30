import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_mla_rope_fused_dkv
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Kernels

def original : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vcat_8_ptr : RegionName := "vcat_8"
  triton {
    r0_z = tl.arange($(98304))
    r0_value = tl.where((r0_z % $(32)) < $(16), tl.load($(input0_ptr) + ((((r0_z / $(49152)) % $(2)) * $(98304)) + (((r0_z / $(4096)) % $(12)) * $(8192)) + (((r0_z / $(32)) % $(128)) * $(64)) + ((((r0_z % $(32)) - $(0)) + $(24)) % $(64))), dtype=tl.bfloat16), tl.load($(input0_ptr) + ((((r0_z / $(49152)) % $(2)) * $(98304)) + (((r0_z / $(4096)) % $(12)) * $(8192)) + (((r0_z / $(32)) % $(128)) * $(64)) + ((((r0_z % $(32)) - $(16)) + $(48)) % $(64))), dtype=tl.bfloat16))
    r0_value = r0_value.to(tl.bfloat16)
    r0_value = tl.broadcast_to(r0_value, [$(98304)])
    tl.store($(vcat_8_ptr) + r0_z, r0_value)
    z = tl.arange($(98304))
    value = tl.load($(vcat_8_ptr) + ((((z / $(49152)) % $(2)) * $(49152)) + (((z / $(4096)) % $(12)) * $(4096)) + (((z / $(32)) % $(128)) * $(32)) + (z % $(32))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(98304)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let candidate_vslice_backward_ptr : RegionName := "candidate_vslice_backward"
  let candidate_vslice_backward_1_ptr : RegionName := "candidate_vslice_backward_1"
  let candidate_vadd_2_ptr : RegionName := "candidate_vadd_2"
  triton {
    r0_z = tl.arange($(98304))
    r0_value = tl.where((((r0_z % $(32)) >= $(16)) & ((r0_z % $(32)) < $(32)) & (((r0_z % $(32)) - $(16)) % $(1) == $(0))), tl.load($(input0_ptr) + ((((r0_z / $(384)) % $(2)) * $(98304)) + (((r0_z / $(32)) % $(12)) * $(8192)) + (((r0_z / $(768)) % $(128)) * $(64)) + ((((r0_z % $(32)) - $(16)) + $(48)) % $(64))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r0_value = r0_value.to(tl.bfloat16)
    r0_value = tl.broadcast_to(r0_value, [$(98304)])
    tl.store($(candidate_vslice_backward_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(98304))
    r1_value = tl.where((((r1_z % $(32)) >= $(0)) & ((r1_z % $(32)) < $(16)) & (((r1_z % $(32)) - $(0)) % $(1) == $(0))), tl.load($(input0_ptr) + ((((r1_z / $(384)) % $(2)) * $(98304)) + (((r1_z / $(32)) % $(12)) * $(8192)) + (((r1_z / $(768)) % $(128)) * $(64)) + ((((r1_z % $(32)) - $(0)) + $(24)) % $(64))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r1_value = r1_value.to(tl.bfloat16)
    r1_value = tl.broadcast_to(r1_value, [$(98304)])
    tl.store($(candidate_vslice_backward_1_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(98304))
    r2_value = (tl.load($(candidate_vslice_backward_ptr) + ((((r2_z / $(768)) % $(128)) * $(768)) + (((r2_z / $(384)) % $(2)) * $(384)) + (((r2_z / $(32)) % $(12)) * $(32)) + (r2_z % $(32))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_1_ptr) + ((((r2_z / $(768)) % $(128)) * $(768)) + (((r2_z / $(384)) % $(2)) * $(384)) + (((r2_z / $(32)) % $(12)) * $(32)) + (r2_z % $(32))), dtype=tl.bfloat16))
    r2_value = r2_value.to(tl.bfloat16)
    r2_value = tl.broadcast_to(r2_value, [$(98304)])
    tl.store($(candidate_vadd_2_ptr) + r2_z, r2_value)
    z = tl.arange($(98304))
    value = tl.load($(candidate_vadd_2_ptr) + ((((z / $(32)) % $(128)) * $(768)) + (((z / $(49152)) % $(2)) * $(384)) + (((z / $(4096)) % $(12)) * $(32)) + (z % $(32))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(98304)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_mla_rope_fused_dkv
