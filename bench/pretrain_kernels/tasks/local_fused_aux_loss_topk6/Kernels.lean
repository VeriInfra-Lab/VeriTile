import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_fused_aux_loss_topk6
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
  let vsum_1_ptr : RegionName := "vsum_1"
  let vmul_ptr : RegionName := "vmul"
  let vsum_2_ptr : RegionName := "vsum_2"
  let vmul_1_ptr : RegionName := "vmul_1"
  triton {
    r0_z = tl.arange($(128))
    r0_lane = tl.arange($(512))
    r0_value = tl.sum(tl.load($(input0_ptr) + ((r0_lane[None, :] * $(128)) + r0_z[:, None]), dtype=tl.float32), axis=1)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(128)])
    tl.store($(vsum_1_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(128))
    r1_value = (tl.load($(vsum_1_ptr) + r1_z, dtype=tl.float32) * tl.load($(input1_ptr) + r1_z, dtype=tl.float32))
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(128)])
    tl.store($(vmul_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(1))
    r2_lane = tl.arange($(128))
    r2_value = tl.sum(tl.load($(vmul_ptr) + r2_lane[None, :], dtype=tl.float32), axis=1)
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(1)])
    tl.store($(vsum_2_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(1))
    r3_value = (tl.load($(vsum_2_ptr) + $(0), dtype=tl.float32) * 0.0000008138020833333334)
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(1)])
    tl.store($(vmul_1_ptr) + r3_z, r3_value)
    z = tl.arange($(1))
    value = tl.load($(vmul_1_ptr) + $(0), dtype=tl.float32)
    value = tl.broadcast_to(value, [$(1)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let input1_ptr : RegionName := "input1"
  let configurationSmCount_ptr : RegionName := "configurationSmCount"
  let out_ptr : RegionName := "out"
  let candidate_vaux_forward_partials_ptr : RegionName := "candidate_vaux_forward_partials"
  let candidate_vaux_forward_ptr : RegionName := "candidate_vaux_forward"
  triton {
    r0_grid = (tl.load($(configurationSmCount_ptr), dtype=tl.uint32) * $(2))
    r0_col = tl.arange($(128))
    for r0_block in range($(512)) {
      if r0_block < r0_grid {
        r0_colsum = tl.full([$(128)], 0.0, dtype=tl.float32)
        for r0_row in range(r0_block, $(512), r0_grid) {
          r0_colsum = r0_colsum + tl.load($(input0_ptr) + (((r0_row % $(512)) * $(128)) + (r0_col % $(128))), dtype=tl.float32)
        }
        r0_partial = tl.sum(r0_colsum * tl.load($(input1_ptr) + (r0_col % $(128)), dtype=tl.float32), axis=0) * 0.0000008138020833333334
        tl.store($(candidate_vaux_forward_partials_ptr) + r0_block, r0_partial)
      } else {
        tl.store($(candidate_vaux_forward_partials_ptr) + r0_block, 0.0)
      }
    }
    r1_z = tl.arange($(1))
    r1_lane = tl.arange($(32))
    r1_acc = tl.full([$(32)], 0.0, dtype=tl.float64)
    r1_grid = (tl.load($(configurationSmCount_ptr), dtype=tl.uint32) * $(2))
    for r1_step in range($(16)) {
      r1_part = r1_lane + r1_step * $(32)
      r1_value = tl.load($(candidate_vaux_forward_partials_ptr) + r1_part, mask=(r1_part < $(512)) & (r1_part < r1_grid), other=0, dtype=tl.float32)
      r1_value = r1_value.to(tl.float64)
      r1_acc = r1_acc + r1_value
    }
    r1_value = tl.sum(r1_acc, axis=0)
    r1_value = r1_value.to(tl.float32)
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(1)])
    tl.store($(candidate_vaux_forward_ptr) + r1_z, r1_value)
    z = tl.arange($(1))
    value = tl.load($(candidate_vaux_forward_ptr) + $(0), dtype=tl.float32)
    value = tl.broadcast_to(value, [$(1)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_fused_aux_loss_topk6
