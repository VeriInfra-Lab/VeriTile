import bench.tritonbench_g.chunk_delta_fwd.IO
import bench.tritonbench_g.chunk_delta_fwd.Proofs.IOImplements

namespace VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.LogicalTensors
open VeriTile Triton FP.Structural ChunkDeltaFwd
open VeriTile.Bench.TritonBenchG.ChunkDeltaFwd
open scoped BigOperators
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Complete original judgment, in both directions, for the original
canonical formula after bijective input/output tensor coordinates. -/
theorem full_canonical_iff (c : Parameters) (schedule : List PID) :
    (io c schedule ⊨ fun values => outputs c ((Reference.realFormula c) ((inputs c).symm values))) ↔
      (Interface.io Real.algebra Real.dot c schedule id id).Implements (Reference.realFormula c) := by
  change (Interface.io Real.algebra Real.dot c schedule id id).Implements (fun raw => (outputs c).symm (outputs c ((Reference.realFormula c) ((inputs c).symm (inputs c raw))))) ↔ (Interface.io Real.algebra Real.dot c schedule id id).Implements (Reference.realFormula c)
  simp only [Equiv.symm_apply_apply]


theorem correct (c : Parameters) (schedule : List PID) (shape : Geometry.ShapeValid c) (order : schedule.Perm (grid c)) :
    io c schedule ⊨ fun values => outputs c ((Reference.realFormula c) ((inputs c).symm values)) :=
  (full_canonical_iff c schedule).mpr (IOImplements.real c shape schedule order)

/-- Generic independent recurrence: the reader supplies only its two defining
equations; induction and complete tensor/output transport stay in Proofs. -/
theorem correct_recurrence (p : Parameters) (schedule : List PID)
    (shape : Geometry.ShapeValid p) (order : schedule.Perm (grid p))
    (S : RealMath.Inputs → Nat → Nat → Nat → Nat → Bool → Nat → Nat → Nat → ℝ)
    (initial : ∀ xs T K stride width seed e j,
      S xs T K stride width seed 0 e j = if seed then xs.seed e j else 0)
    (step : ∀ xs T K stride width seed c e j,
      S xs T K stride width seed (c+1) e j = S xs T K stride width seed c e j +
        ∑ t : Fin width, if c*stride+t.val < T then xs.key (c*stride+t.val) e *
          RealMath.corrected xs K (S xs T K stride width seed c) (c*stride+t.val) j else 0) :
    io p schedule ⊨ (fun xs =>
      let state := fun head => S (realInputs p xs head)
        p.T p.K p.BT (((p.BT+p.BC-1)/p.BC)*p.BC) p.useSeed
      { history := fun head chunk key value => state head.val chunk.val key.val value.val
        corrected := fun head time value => RealMath.corrected (realInputs p xs head.val) p.K
          (state head.val (time.val/p.BT)) time.val value.val
        final := fun head key value => state head.val p.NT key.val value.val }) := by
  have state_eq (xs : RealMath.Inputs) (T K stride width : Nat) (seed : Bool) (c : Nat) :
      S xs T K stride width seed c = Real.state xs T K stride width seed c := by
    induction c with
    | zero => funext e j; exact initial xs T K stride width seed e j
    | succ c ih =>
      funext e j
      rw [step]
      simp only [Real.state, ih]
  simpa only [state_eq] using correct p schedule shape order

#axiomsClean full_canonical_iff
#axiomsClean correct
end
end VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.LogicalTensors
