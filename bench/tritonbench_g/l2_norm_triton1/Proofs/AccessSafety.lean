import bench.tritonbench_g.l2_norm_triton1.Proofs.Tile
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.L2NormTriton1.Tile
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Actual source accesses are the masked X/Y row windows. The runtime epsilon
register changes arithmetic only and needs no numerical finiteness premise. -/
theorem source_traceSafe {α : Type} [Inhabited α] (M : Algebra α)
    (N B row : Nat) (eps : α) (bounds : RegionName → Nat) (s : State α)
    (input : ∀ j : Fin B, j.val < N → row * N + j.val < bounds "X")
    (output : ∀ j : Fin B, j.val < N → row * N + j.val < bounds "Y") :
    FP.Structural.TraceSafe M bounds (source N B) (rowState row eps s) := by
  have index_toNat (lane : Nat) :
      ((row : Int) * (N : Int) + (lane : Int)).toNat = row * N + lane := by
    rw [← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast]
  have index_lt (lane : Nat) : ((lane : Int) < (N : Int)) ↔ lane < N := by omega
  simp (maxSteps := 1400000) [FP.Structural.TraceSafe, source, template,
    ComputeKernel.surfaceBody, Safety.traceSafeList_unfold, Safety.stepSafe_unfold,
    Safety.safeExpr_unfold, Safety.safeCompute_unfold, Safety.safeOp_unfold,
    Safety.safeAddress_unfold, Safety.safeMask_unfold, Safety.addressSafe_unfold,
    Safety.maskActive_unfold, step, FP.Structural.evalExpr, evalComputeOp,
    evalOp_unfold, evalOffsets, evalDeltas, store, Option.bind,
    ComputeDType.eraseDType, numeric, numericLt, bop, natLt, rowState,
    ofFloat, toFloat, TileShape.axisDim, Region.cast, State.setReg, index_toNat,
    index_lt]
  exact ⟨input, output⟩

/-- The existing integer-product candidate has the same actual access windows. -/
theorem commuted_traceSafe {α : Type} [Inhabited α] (M : Algebra α)
    (N B row : Nat) (eps : α) (bounds : RegionName → Nat) (s : State α)
    (input : ∀ j : Fin B, j.val < N → row * N + j.val < bounds "X")
    (output : ∀ j : Fin B, j.val < N → row * N + j.val < bounds "Y") :
    FP.Structural.TraceSafe M bounds (indexCommuted N B) (rowState row eps s) := by
  have index_toNat (lane : Nat) :
      ((row : Int) * (N : Int) + (lane : Int)).toNat = row * N + lane := by
    rw [← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast]
  have index_lt (lane : Nat) : ((lane : Int) < (N : Int)) ↔ lane < N := by omega
  simp (maxSteps := 1400000) [FP.Structural.TraceSafe, indexCommuted, commutedTemplate,
    ComputeKernel.surfaceBody, Safety.traceSafeList_unfold, Safety.stepSafe_unfold,
    Safety.safeExpr_unfold, Safety.safeCompute_unfold, Safety.safeOp_unfold,
    Safety.safeAddress_unfold, Safety.safeMask_unfold, Safety.addressSafe_unfold,
    Safety.maskActive_unfold, step, FP.Structural.evalExpr, evalComputeOp,
    evalOp_unfold, evalOffsets, evalDeltas, store, Option.bind,
    ComputeDType.eraseDType, numeric, numericLt, bop, natLt, rowState,
    ofFloat, toFloat, TileShape.axisDim, Region.cast, State.setReg, index_toNat,
    index_lt]
  simpa only [Int.toNat_mul, Int.toNat_natCast, Nat.mul_comm] using And.intro input output

#axiomsClean source_traceSafe
#axiomsClean commuted_traceSafe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton1.Tile
