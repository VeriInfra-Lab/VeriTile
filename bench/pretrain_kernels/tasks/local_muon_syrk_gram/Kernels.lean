import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_muon_syrk_gram
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Kernels

def original : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vbmm_ptr : RegionName := "vbmm"
  triton {
    r0_row = tl.arange($(1544))
    r0_col = tl.arange($(1544))
    r0_lane = tl.arange($(520))
    for r0_group in range($(3)) {
      r0_left = tl.load($(input0_ptr) + (((r0_group % $(3)) * $(802880)) + ((r0_row[:, None] % $(1544)) * $(520)) + (r0_lane[None, :] % $(520))), dtype=tl.bfloat16)
      r0_right = tl.load($(input0_ptr) + (((r0_group % $(3)) * $(802880)) + ((r0_col[None, :] % $(1544)) * $(520)) + (r0_lane[:, None] % $(520))), dtype=tl.bfloat16)
      r0_value = tl.dot(r0_left, r0_right)
      r0_value = r0_value.to(tl.bfloat16)
      tl.store($(vbmm_ptr) + ((r0_group * $(2383936)) + (r0_row[:, None] * $(1544)) + r0_col[None, :]), r0_value)
    }
    z = tl.arange($(7151808))
    value = tl.load($(vbmm_ptr) + ((((z / $(2383936)) % $(3)) * $(2383936)) + (((z / $(1544)) % $(1544)) * $(1544)) + (z % $(1544))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(7151808)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vbmm_ptr : RegionName := "vbmm"
  let candidate_vbmm_ptr : RegionName := "candidate_vbmm"
  let candidate_vmul_ptr : RegionName := "candidate_vmul"
  let candidate_vle_ptr : RegionName := "candidate_vle"
  let candidate_vwhere_ptr : RegionName := "candidate_vwhere"
  triton {
    r0_row = tl.arange($(1544))
    r0_col = tl.arange($(1544))
    r0_lane = tl.arange($(520))
    for r0_group in range($(3)) {
      r0_left = tl.load($(input0_ptr) + (((r0_group % $(3)) * $(802880)) + ((r0_row[:, None] % $(1544)) * $(520)) + (r0_lane[None, :] % $(520))), dtype=tl.bfloat16)
      r0_right = tl.load($(input0_ptr) + (((r0_group % $(3)) * $(802880)) + ((r0_col[None, :] % $(1544)) * $(520)) + (r0_lane[:, None] % $(520))), dtype=tl.bfloat16)
      r0_value = tl.dot(r0_left, r0_right)
      r0_value = r0_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_ptr) + ((r0_group * $(2383936)) + (r0_row[:, None] * $(1544)) + r0_col[None, :]), r0_value)
    }
    r1_z = tl.arange($(7151808))
    r1_value = (tl.load($(candidate_vbmm_ptr) + ((((r1_z / $(2383936)) % $(3)) * $(2383936)) + (((r1_z / $(1544)) % $(1544)) * $(1544)) + (r1_z % $(1544))), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r1_value = r1_value.to(tl.bfloat16)
    r1_value = tl.broadcast_to(r1_value, [$(7151808)])
    tl.store($(candidate_vmul_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(2383936))
    r2_value = (tl.cast(((r2_z / $(1544)) % $(1544)), tl.int64) <= tl.cast((r2_z % $(1544)), tl.int64))
    r2_value = tl.broadcast_to(r2_value, [$(2383936)])
    tl.store($(candidate_vle_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(7151808))
    r3_value = tl.where(tl.load($(candidate_vle_ptr) + ((((r3_z / $(1544)) % $(1544)) * $(1544)) + (r3_z % $(1544))), dtype=tl.int1), tl.load($(candidate_vmul_ptr) + ((((r3_z / $(2383936)) % $(3)) * $(2383936)) + (((r3_z / $(1544)) % $(1544)) * $(1544)) + (r3_z % $(1544))), dtype=tl.bfloat16), tl.load($(candidate_vmul_ptr) + ((((r3_z / $(2383936)) % $(3)) * $(2383936)) + ((r3_z % $(1544)) * $(1544)) + ((r3_z / $(1544)) % $(1544))), dtype=tl.bfloat16))
    r3_value = r3_value.to(tl.bfloat16)
    r3_value = tl.broadcast_to(r3_value, [$(7151808)])
    tl.store($(candidate_vwhere_ptr) + r3_z, r3_value)
    z = tl.arange($(7151808))
    value = tl.load($(candidate_vwhere_ptr) + ((((z / $(2383936)) % $(3)) * $(2383936)) + (((z / $(1544)) % $(1544)) * $(1544)) + (z % $(1544))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(7151808)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_muon_syrk_gram
