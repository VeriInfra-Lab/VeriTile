import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.Proofs.FPTraceBounds
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.SourceSafety
open VeriTile Triton FP.Structural Safety
open SoftmaxFlaggemsFiniteSeed FPReference FPLoop
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α β : Type} [Inhabited α]

@[simp] theorem option_true {γ : Type} (value : Option γ) :
    (match value with | some _ => True | none => True) := by
  cases value <;> trivial

/-- Lift existing loop-step invariants to the actual Structural safety trace.
Every successor is the same successful state used by the source evaluator. -/
theorem range_safe (M : Algebra α) (bounds : RegionName → Nat)
    (idx : RegName) (body : List ComputeStmt) (Inv : Nat → β → State α → Prop)
    (f : Nat → β → β) (stop stride : Nat)
    (safe : ∀ cur value s, cur<stop → Inv cur value s →
      traceSafeList M bounds body (s.setReg idx .nat [] (fun _ => cur)))
    (next : ∀ cur value s, cur<stop → Inv cur value s →
      ∃ t,run M body (s.setReg idx .nat [] (fun _ => cur))=some t ∧
        Inv (cur+stride) (f cur value) t)
    (cur : Nat) (value : β) (s : State α) (hs : Inv cur value s) :
    rangeSafe M bounds idx cur stop stride body s := by
  rw [rangeSafe_unfold]
  by_cases hz : stride=0
  · simp only [hz,↓reduceIte]
  by_cases hc : cur<stop
  · simp only [hz,hc,↓reduceIte]
    obtain ⟨mid,ran,ready⟩ := next cur value s hc hs
    rw [ran]
    exact ⟨safe cur value s hc hs,
      range_safe M bounds idx body Inv f stop stride safe next (cur+stride) (f cur value) mid ready⟩
  · simp only [hz,hc,↓reduceIte]
termination_by stop-cur

end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.SourceSafety
