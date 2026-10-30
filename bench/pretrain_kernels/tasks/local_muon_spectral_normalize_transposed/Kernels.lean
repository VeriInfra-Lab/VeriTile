import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_muon_spectral_normalize_transposed
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Kernels

def original : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vlinalg_vector_norm_ptr : RegionName := "vlinalg_vector_norm"
  let vclamp_ptr : RegionName := "vclamp"
  let vdiv_ptr : RegionName := "vdiv"
  triton {
    r0_z = tl.arange($(3))
    r0_lane = tl.arange($(128733))
    r0_value = tl.sqrt(tl.sum(tl.load($(input0_ptr) + ((r0_z[:, None] * $(128733)) + ((r0_lane[None, :] % $(517)) * $(249)) + ((r0_lane[None, :] / $(517)) % $(249))), dtype=tl.float32) * tl.load($(input0_ptr) + ((r0_z[:, None] * $(128733)) + ((r0_lane[None, :] % $(517)) * $(249)) + ((r0_lane[None, :] / $(517)) % $(249))), dtype=tl.float32), axis=1))
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(3)])
    tl.store($(vlinalg_vector_norm_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(3))
    r1_value = tl.maximum(tl.load($(vlinalg_vector_norm_ptr) + r1_z, dtype=tl.float32), 0.0000001)
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(3)])
    tl.store($(vclamp_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(386199))
    r2_value = (tl.load($(input0_ptr) + ((((r2_z / $(128733)) % $(3)) * $(128733)) + ((r2_z % $(517)) * $(249)) + ((r2_z / $(517)) % $(249))), dtype=tl.float32) / tl.load($(vclamp_ptr) + ((r2_z / $(128733)) % $(3)), dtype=tl.float32))
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(386199)])
    tl.store($(vdiv_ptr) + r2_z, r2_value)
    z = tl.arange($(386199))
    value = tl.load($(vdiv_ptr) + ((((z / $(128733)) % $(3)) * $(128733)) + (((z / $(517)) % $(249)) * $(517)) + (z % $(517))), dtype=tl.float32)
    value = tl.broadcast_to(value, [$(386199)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vlinalg_vector_norm_ptr : RegionName := "vlinalg_vector_norm"
  let vclamp_ptr : RegionName := "vclamp"
  let vdiv_ptr : RegionName := "vdiv"
  let candidate_vlinalg_vector_norm_ptr : RegionName := "candidate_vlinalg_vector_norm"
  let candidate_vclamp_ptr : RegionName := "candidate_vclamp"
  let candidate_vdiv_ptr : RegionName := "candidate_vdiv"
  triton {
    r0_z = tl.arange($(3))
    r0_lane = tl.arange($(128733))
    r0_value = tl.sqrt(tl.sum(tl.load($(input0_ptr) + ((r0_z[:, None] * $(128733)) + ((r0_lane[None, :] % $(517)) * $(249)) + ((r0_lane[None, :] / $(517)) % $(249))), dtype=tl.float32) * tl.load($(input0_ptr) + ((r0_z[:, None] * $(128733)) + ((r0_lane[None, :] % $(517)) * $(249)) + ((r0_lane[None, :] / $(517)) % $(249))), dtype=tl.float32), axis=1))
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(3)])
    tl.store($(candidate_vlinalg_vector_norm_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(3))
    r1_value = tl.maximum(tl.load($(candidate_vlinalg_vector_norm_ptr) + r1_z, dtype=tl.float32), 0.0000001)
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(3)])
    tl.store($(candidate_vclamp_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(386199))
    r2_value = (tl.load($(input0_ptr) + ((((r2_z / $(128733)) % $(3)) * $(128733)) + ((r2_z % $(517)) * $(249)) + ((r2_z / $(517)) % $(249))), dtype=tl.float32) / tl.load($(candidate_vclamp_ptr) + ((r2_z / $(128733)) % $(3)), dtype=tl.float32))
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(386199)])
    tl.store($(candidate_vdiv_ptr) + r2_z, r2_value)
    z = tl.arange($(386199))
    value = tl.load($(candidate_vdiv_ptr) + ((((z / $(128733)) % $(3)) * $(128733)) + (((z / $(517)) % $(249)) * $(517)) + (z % $(517))), dtype=tl.float32)
    value = tl.broadcast_to(value, [$(386199)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_muon_spectral_normalize_transposed
