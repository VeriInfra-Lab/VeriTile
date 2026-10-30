import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_mtp_roll_packed_fwd
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
  let vcat_1_ptr : RegionName := "vcat_1"
  let vcat_2_ptr : RegionName := "vcat_2"
  let vcat_3_ptr : RegionName := "vcat_3"
  let vcat_4_ptr : RegionName := "vcat_4"
  triton {
    r0_z = tl.arange($(32))
    r0_value = tl.where((r0_z % $(16)) < $(15), tl.load($(input0_ptr) + ((((r0_z / $(16)) % $(2)) * $(128)) + ((((r0_z % $(16)) - $(0)) + $(1)) % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r0_value = r0_value.to(tl.bfloat16)
    r0_value = tl.broadcast_to(r0_value, [$(32)])
    tl.store($(vcat_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(64))
    r1_value = tl.where((r1_z % $(32)) < $(31), tl.load($(input0_ptr) + ((((r1_z / $(32)) % $(2)) * $(128)) + ((((r1_z % $(32)) - $(0)) + $(17)) % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r1_value = r1_value.to(tl.bfloat16)
    r1_value = tl.broadcast_to(r1_value, [$(64)])
    tl.store($(vcat_1_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(64))
    r2_value = tl.where((r2_z % $(32)) < $(31), tl.load($(input0_ptr) + ((((r2_z / $(32)) % $(2)) * $(128)) + ((((r2_z % $(32)) - $(0)) + $(49)) % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r2_value = r2_value.to(tl.bfloat16)
    r2_value = tl.broadcast_to(r2_value, [$(64)])
    tl.store($(vcat_2_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(96))
    r3_value = tl.where((r3_z % $(48)) < $(47), tl.load($(input0_ptr) + ((((r3_z / $(48)) % $(2)) * $(128)) + ((((r3_z % $(48)) - $(0)) + $(81)) % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r3_value = r3_value.to(tl.bfloat16)
    r3_value = tl.broadcast_to(r3_value, [$(96)])
    tl.store($(vcat_3_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(256))
    r4_value = tl.where((r4_z % $(128)) < $(16), tl.load($(vcat_ptr) + ((((r4_z / $(128)) % $(2)) * $(16)) + (((r4_z % $(128)) - $(0)) % $(16))), dtype=tl.bfloat16), tl.where((r4_z % $(128)) < $(48), tl.load($(vcat_1_ptr) + ((((r4_z / $(128)) % $(2)) * $(32)) + (((r4_z % $(128)) - $(16)) % $(32))), dtype=tl.bfloat16), tl.where((r4_z % $(128)) < $(80), tl.load($(vcat_2_ptr) + ((((r4_z / $(128)) % $(2)) * $(32)) + (((r4_z % $(128)) - $(48)) % $(32))), dtype=tl.bfloat16), tl.load($(vcat_3_ptr) + ((((r4_z / $(128)) % $(2)) * $(48)) + (((r4_z % $(128)) - $(80)) % $(48))), dtype=tl.bfloat16))))
    r4_value = r4_value.to(tl.bfloat16)
    r4_value = tl.broadcast_to(r4_value, [$(256)])
    tl.store($(vcat_4_ptr) + r4_z, r4_value)
    z = tl.arange($(256))
    value = tl.load($(vcat_4_ptr) + ((((z / $(128)) % $(2)) * $(128)) + (z % $(128))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(256)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let constant53d3e3c65e_ptr : RegionName := "constant53d3e3c65e"
  let out_ptr : RegionName := "out"
  let candidate_vsub_ptr : RegionName := "candidate_vsub"
  let candidate_vremainder_ptr : RegionName := "candidate_vremainder"
  let candidate_vindex_fill_ptr : RegionName := "candidate_vindex_fill"
  triton {
    r0_z = tl.arange($(4))
    r0_value = (tl.load($(constant53d3e3c65e_ptr) + (r0_z + $(1)), dtype=tl.int64) - tl.full([], $(1), dtype=tl.int64))
    r0_value = tl.broadcast_to(r0_value, [$(4)])
    tl.store($(candidate_vsub_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(4))
    r1_value = (tl.load($(candidate_vsub_ptr) + r1_z, dtype=tl.int64) % tl.full([], $(128), dtype=tl.int64))
    r1_value = tl.broadcast_to(r1_value, [$(4)])
    tl.store($(candidate_vremainder_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(256))
    r2_value = tl.broadcast_to(tl.load($(input0_ptr) + ((((r2_z / $(128)) % $(2)) * $(128)) + (((r2_z % $(128)) + $(1)) % $(128))), dtype=tl.bfloat16), [$(256)])
    for r2_lane in range($(4)) {
      r2_value = tl.where(tl.load($(candidate_vremainder_ptr) + (r2_lane % $(4)), dtype=tl.int64) == tl.cast((r2_z % $(128)), tl.int64), tl.cast(0.0, tl.bfloat16), r2_value)
    }
    r2_value = r2_value.to(tl.bfloat16)
    r2_value = tl.broadcast_to(r2_value, [$(256)])
    tl.store($(candidate_vindex_fill_ptr) + r2_z, r2_value)
    z = tl.arange($(256))
    value = tl.load($(candidate_vindex_fill_ptr) + ((((z / $(128)) % $(2)) * $(128)) + (z % $(128))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(256)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_mtp_roll_packed_fwd
