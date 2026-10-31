import bench.tritonbench_g.destindex_copy_kv2.Proofs.Tile
import VeriTile.Triton.Float.ProfiledMemory

namespace VeriTile.Bench.Repairs.DestindexCopyKv2.Rewrite
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- The selected typed source succeeds for any floating primitive backend. -/
def execution {α : Type} [Inhabited α] (M : Algebra α) (p : Layout) (s : State α) :
    {t : State α // FP.Structural.exec M (kernel p) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [kernel,
    ComputeKernel.surfaceBody, FP.Structural.exec, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, evalOffsets, evalDeltas, Option.bind, natLt,
    ComputeDType.eraseDType, numeric, bop, ofFloat, toFloat, Region.cast,
    FP.Structural.store, TileShape.dropInsertedIndex]
  rfl

/-- Integer multiplication commutation changes neither execution nor memory;
floating payloads and inactive lanes are arbitrary. -/
theorem exec_eq {α : Type} [Inhabited α] (M : Algebra α) (p : Layout) (s : State α) :
    FP.Structural.exec M (kernel p) s =
      FP.Structural.exec M (indexCommutedKernel p) s := by
  have hmul :
      FP.Structural.bop (fun a b : Nat => a*b) Broadcast.nil
        (fun (_ : TileIndex []) => s.pids 0) (fun _ => p.stride_k_bs) =
      FP.Structural.bop (fun a b : Nat => a*b) Broadcast.nil
        (fun (_ : TileIndex []) => p.stride_k_bs) (fun _ => s.pids 0) := by
    funext i
    exact Nat.mul_comm _ _
  simp [kernel, indexCommutedKernel,
    ComputeKernel.surfaceBody, FP.Structural.exec, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, evalOffsets, evalDeltas, Option.bind, natLt,
    ComputeDType.eraseDType, numeric, bop, ofFloat, toFloat, Region.cast,
    FP.Structural.store, TileShape.dropInsertedIndex, hmul, Nat.mul_comm]

#axiomsClean execution
#axiomsClean exec_eq
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DestindexCopyKv2.Rewrite
