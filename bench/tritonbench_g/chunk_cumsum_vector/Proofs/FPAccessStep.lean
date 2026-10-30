import bench.tritonbench_g.chunk_cumsum_vector.Proofs.FPStep
import bench.tritonbench_g.chunk_cumsum_vector.Proofs.CandidateProgram
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.FPAccess
open VeriTile.Triton FP.Structural IndexProgram FPProgram FPValues
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem assign_alg (M : Algebra α) (bounds : RegionBounds)
    (d : TileDType) (sh : TileShape) (name : RegName)
    (e : Op d sh) (v : Values α d sh) (rest : List ComputeStmt) (u : State α)
    (safe : Safety.safeOp M none bounds u e)
    (evaluated : FP.Structural.evalOp M none e u = some v)
    (tailSafe : Safety.traceSafeList M bounds rest (u.setReg name d sh v)) :
    Safety.traceSafeList M bounds (.assign d sh name (.alg e) :: rest) u := by
  apply Safety.traceSafeList_cons_of_step (t := u.setReg name d sh v)
  · simpa only [Safety.stepSafe_unfold, Safety.safeExpr_unfold] using safe
  · simp [step, evalExpr, evaluated]
  · exact tailSafe

theorem assign_compute (M : Algebra α) (bounds : RegionBounds)
    (sh : TileShape) (name : RegName)
    (e : Op .real sh) (v : Values α .real sh) (rest : List ComputeStmt) (u : State α)
    (safe : Safety.safeOp M (some .fp32) bounds u e)
    (evaluated : FP.Structural.evalOp M (some .fp32) e u = some v)
    (tailSafe : Safety.traceSafeList M bounds rest (u.setReg name .real sh v)) :
    Safety.traceSafeList M bounds (.assign .real sh name (.compute (.alg .fp32 e)) :: rest) u := by
  apply Safety.traceSafeList_cons_of_step (t := u.setReg name .real sh v)
  · simpa only [Safety.stepSafe_unfold, Safety.safeExpr_unfold,
      Safety.safeCompute_unfold] using safe
  · simp only [step, evalExpr, evalComputeOp]
    change (FP.Structural.evalOp M (some .fp32) e u).bind
      (fun value => some (u.setReg name .real sh value)) = _
    rw [evaluated]
    rfl
  · exact tailSafe

theorem load_safe (M : Algebra α) (bounds : RegionBounds) (h : Call) (c : Config)
    (feature row q : Nat) (u : State α)
    (hp : u.regs .ptr [c.BT,BS] "p_s" = some (pointer h c feature row q h.input h.inputStrides))
    (hm : u.regs .bool [c.BT,BS] "active" = some (mask h c feature q))
    (bounded : ∀ i : TileIndex [c.BT,BS], Geometry.active h c feature q i.1.val i.2.1.val →
      h.inputStrides.address h.H row (Geometry.time h c q i.1.val)
        (Geometry.column h feature i.2.1.val) < bounds h.input) :
    Safety.safeOp M (some .fp32) bounds u (FPProgram.loadOp h c) := by
  simpa [FPProgram.loadOp, Safety.safeOp_unfold, Safety.safeAddress_unfold,
    Safety.safeMask_unfold, Safety.addressSafe_unfold, Safety.maskActive_unfold,
    evalOp_unfold, hp, hm, pointer, mask] using bounded

theorem store_safe (M : Algebra α) (bounds : RegionBounds) (h : Call) (c : Config)
    (feature row q : Nat) (u : State α)
    (hp : u.regs .ptr [c.BT,BS] "p_z" = some (pointer h c feature row q h.output h.outputStrides))
    (hm : u.regs .bool [c.BT,BS] "active" = some (mask h c feature q))
    (bounded : ∀ i : TileIndex [c.BT,BS], Geometry.active h c feature q i.1.val i.2.1.val →
      h.outputStrides.address h.H row (Geometry.time h c q i.1.val)
        (Geometry.column h feature i.2.1.val) < bounds h.output) :
    Safety.stepSafe M bounds (FPProgram.storeOp h c) u := by
  simpa [FPProgram.storeOp, Safety.stepSafe_unfold, Safety.safeExpr_unfold,
    Safety.safeOp_unfold, Safety.safeAddress_unfold, Safety.safeMask_unfold,
    Safety.addressSafe_unfold, Safety.maskActive_unfold, evalOp_unfold, hp, hm,
    pointer, mask] using bounded

#axiomsClean load_safe
#axiomsClean store_safe
end
end VeriTile.Bench.Repairs.ChunkCumsumVector.FPAccess
