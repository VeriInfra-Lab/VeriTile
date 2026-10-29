import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_mtp_roll_fwd
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Kernels

def original : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vcat_ptr : RegionName := "vcat"
  triton {
    r0_z = tl.arange($(256))
    r0_value = tl.where((r0_z % $(128)) < $(127), tl.load($(input0_ptr) + ((((r0_z / $(128)) % $(2)) * $(128)) + ((((r0_z % $(128)) - $(0)) + $(1)) % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r0_value = r0_value.to(tl.bfloat16)
    r0_value = tl.broadcast_to(r0_value, [$(256)])
    tl.store($(vcat_ptr) + r0_z, r0_value)
    z = tl.arange($(256))
    value = tl.load($(vcat_ptr) + ((((z / $(128)) % $(2)) * $(128)) + (z % $(128))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(256)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let candidate_vfill_ptr : RegionName := "candidate_vfill"
  let candidate_vselect_scatter_ptr : RegionName := "candidate_vselect_scatter"
  triton {
    r0_z = tl.arange($(2))
    r0_value = tl.cast(0.0, tl.bfloat16)
    r0_value = r0_value.to(tl.bfloat16)
    r0_value = tl.broadcast_to(r0_value, [$(2)])
    tl.store($(candidate_vfill_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(256))
    r1_value = tl.where(((r1_z % $(128)) == $(127)), tl.load($(candidate_vfill_ptr) + ((r1_z / $(128)) % $(2)), dtype=tl.bfloat16), tl.load($(input0_ptr) + ((((r1_z / $(128)) % $(2)) * $(128)) + (((r1_z % $(128)) + $(1)) % $(128))), dtype=tl.bfloat16))
    r1_value = r1_value.to(tl.bfloat16)
    r1_value = tl.broadcast_to(r1_value, [$(256)])
    tl.store($(candidate_vselect_scatter_ptr) + r1_z, r1_value)
    z = tl.arange($(256))
    value = tl.load($(candidate_vselect_scatter_ptr) + ((((z / $(128)) % $(2)) * $(128)) + (z % $(128))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(256)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_mtp_roll_fwd
