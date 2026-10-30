import bench.tritonbench_g.fused_recurrent_retention.IO
import bench.tritonbench_g.fused_recurrent_retention.Proofs.MathResult

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.PublicData
open VeriTile.Triton FP.Structural Quoted
open scoped BigOperators
noncomputable section

def tensorExpected (s : Shape) (seed final : Bool) (xs : TensorInputs (Staged.Node ℝ) s) : TensorOutputs ℝ s :=
  let x := tensorData s xs
  { output := fun part h t c => ∑ j∈Finset.range s.BK, if part.val*s.BK+j<s.DK then
      mathState s x seed h.val (part.val*s.BK+j) c.val (t.val+1)*
        (x.q h.val t.val (part.val*s.BK+j)*realArithmetic.scale s.DK) else 0
    dq := fun part h t c => (∑ j∈Finset.range s.BV, if part.val*s.BV+j<s.DV then
      mathState s x seed h.val c.val (part.val*s.BV+j) (t.val+1)*
        x.do_ h.val t.val (part.val*s.BV+j) else 0)*realArithmetic.scale s.DK
    dk := fun part h t c => ∑ j∈Finset.range s.BV, if part.val*s.BV+j<s.DV then
      mathAdjoint s x final h.val t.val c.val (part.val*s.BV+j)*x.v h.val t.val (part.val*s.BV+j) else 0
    dv := fun part h t c => ∑ j∈Finset.range s.BK, if part.val*s.BK+j<s.DK then
      mathAdjoint s x final h.val t.val (part.val*s.BK+j) c.val*x.k h.val t.val (part.val*s.BK+j) else 0
    finalState := fun h k v => mathState s x seed h.val k.val v.val s.T }

/-- Equality to the previous independent target for every raw tagged input and
all five complete output windows, before applying any program judgment. -/
theorem tensor_lift_eq (mode : Mode) (s : Shape) (p : PlanBinding s)
    (incoming finalGradient : Fin 4 → Nat) (seed final : Bool) :
    (logicalIO mode s p incoming finalGradient seed final).lift (tensorExpected s seed final) =
      mathExpected s seed final := by
  funext xs o j
  fin_cases o <;>
    dsimp [LogicalIO.lift, logicalIO, outputCoordinates, tensorExpected, tensorData,
      partialValues, stateValues, partialIndex, stateIndex, mathExpected, mathCell]
  all_goals rw [(inputCoordinates (Staged.Node ℝ) s).symm_apply_apply]
  all_goals rfl

theorem logical_iff (mode : Mode) (s : Shape) (p : PlanBinding s)
    (incoming finalGradient : Fin 4 → Nat) (seed final : Bool) :
    (logicalIO mode s p incoming finalGradient seed final).Implements (tensorExpected s seed final) ↔
    (io mode realArithmetic s p incoming finalGradient seed final).Observes
      (fun _ _ n => n.value) (mathExpected s seed final) := by
  change (io mode realArithmetic s p incoming finalGradient seed final).Observes _ _ ↔ _
  rw [tensor_lift_eq]

theorem logical_real_complete (mode : Mode) (s : Shape) (shape : s.Admitted)
    (p : PlanBinding s) (incoming finalGradient : Fin 4 → Nat) (seed final : Bool) :
    (logicalIO mode s p incoming finalGradient seed final).Implements (tensorExpected s seed final) :=
  (logical_iff mode s p incoming finalGradient seed final).mpr
    (real_complete mode s shape p incoming finalGradient seed final)
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.PublicData
