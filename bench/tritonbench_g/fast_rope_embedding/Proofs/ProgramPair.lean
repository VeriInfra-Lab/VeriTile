import bench.tritonbench_g.fast_rope_embedding.PairIO
import bench.tritonbench_g.fast_rope_embedding.Proofs.SourcePair
import bench.tritonbench_g.fast_rope_embedding.IO

namespace VeriTile.Bench.Repairs.FastRopeEmbedding.ProgramPair
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.RopeEmbedding
open scoped VeriTile.Spec
noncomputable section

/-- Source and actual launch order form the program being compared. -/
def Equivalent (lhs rhs : Program) : Prop :=
  ∀ (α : Type) [Inhabited α] (M : Algebra α) (s : State α),
    runSchedule (schedule M lhs) s = runSchedule (schedule M rhs) s ∧
    ∀ bounds, ScheduleSafe bounds (schedule M lhs) s ↔ ScheduleSafe bounds (schedule M rhs) s

instance : Spec.ProgramSyntax Program where
  Statement := ComputeStmt
  Signature := (List RegionName × List RegionName) × List (Nat × Nat)
  signature p := ((p.kernel.inputs,p.kernel.outputs),p.launches)
  body p := p.kernel.surfaceBody
  structural := some Equivalent
  sameContext := fun lhs rhs => lhs = rhs

def assumptions : Spec.Assumptions ComputeStmt := []

/-- Python launches rows × ceil(heads/4); zero half-width launches no work. -/
theorem job_run {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (r : Request) (QR CR SR : Nat) (backward : Bool) (launch : Nat × Nat) :
    (job M (original r QR CR SR backward []).kernel launch).run s =
      (job M (candidate r QR CR SR backward []).kernel launch).run s := by
  exact SourcePair.source_exec M _ _ _ _ _ _ _ _ _ _ _ _

theorem job_safe {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (r : Request) (QR CR SR : Nat) (backward : Bool) (launch : Nat × Nat) (bounds : RegionName → Nat) :
    (job M (original r QR CR SR backward []).kernel launch).Safe bounds s ↔
      (job M (candidate r QR CR SR backward []).kernel launch).Safe bounds s := by
  exact SourcePair.source_safe M _ bounds _ _ _ _ _ _ _ _ _ _ _

theorem schedules {α : Type} [Inhabited α] (M : Algebra α)
    (r : Request) (QR CR SR : Nat) (backward : Bool) (launches : List (Nat × Nat)) (s : State α) :
    runSchedule (schedule M (original r QR CR SR backward launches)) s =
      runSchedule (schedule M (candidate r QR CR SR backward launches)) s ∧
    ∀ bounds, ScheduleSafe bounds (schedule M (original r QR CR SR backward launches)) s ↔
      ScheduleSafe bounds (schedule M (candidate r QR CR SR backward launches)) s := by
  induction launches generalizing s with
  | nil => exact ⟨rfl, fun _ => Iff.rfl⟩
  | cons launch rest ih =>
    constructor
    · change ((job M (original r QR CR SR backward []).kernel launch).run s).bind _ = _
      rw [job_run]
      congr 1
      funext t
      exact (ih t).1
    · intro bounds
      change ((job M (original r QR CR SR backward []).kernel launch).Safe bounds s ∧
          ∀ t, (job M (original r QR CR SR backward []).kernel launch).run s = some t →
            ScheduleSafe bounds (schedule M (original r QR CR SR backward rest)) t) ↔
        ((job M (candidate r QR CR SR backward []).kernel launch).Safe bounds s ∧
          ∀ t, (job M (candidate r QR CR SR backward []).kernel launch).run s = some t →
            ScheduleSafe bounds (schedule M (candidate r QR CR SR backward rest)) t)
      simp only [job_safe,job_run]
      exact and_congr Iff.rfl (forall_congr' fun t => imp_congr_right fun _ => (ih t).2 bounds)

theorem equivalent (r : Request) (QR CR SR : Nat) (backward : Bool) (launches : List (Nat × Nat)) :
    original r QR CR SR backward launches ≡[assumptions] candidate r QR CR SR backward launches := by
  apply Spec.FloatingPoint.ofStructural (structural := Equivalent) rfl
  · cases backward <;> rfl
  exact fun α _ M s => schedules M r QR CR SR backward launches s

/-- The singleton schedule is exactly the retained logical IO execution. -/
theorem original_io_schedule {α : Type} (op : Operations α) (r : Request)
    (QR CR SR : Nat) (backward : Bool) (row group : Nat) (values : Interface.Inputs α r) :
    (Interface.programIO op r QR CR SR backward row group).schedule values =
      schedule op.backend (original r QR CR SR backward [(row,group)]) := rfl

/-- Candidate shares every logical channel, mask, observation and memory frame;
only the complete invoked kernel changes to the preserved candidate. -/
theorem io_execution {α : Type} [Inhabited α] (op : Operations α) (r : Request)
    (QR CR SR : Nat) (backward : Bool) (row group : Nat)
    (values : Interface.Inputs α r) (s : State α) :
    runSchedule ((Interface.programIO op r QR CR SR backward row group).schedule values) s =
      runSchedule ((candidateIO op r QR CR SR backward row group).schedule values) s ∧
    ∀ bounds, ScheduleSafe bounds ((Interface.programIO op r QR CR SR backward row group).schedule values) s ↔
      ScheduleSafe bounds ((candidateIO op r QR CR SR backward row group).schedule values) s :=
  schedules op.backend r QR CR SR backward [(row,group)] s

end
end VeriTile.Bench.Repairs.FastRopeEmbedding.ProgramPair
