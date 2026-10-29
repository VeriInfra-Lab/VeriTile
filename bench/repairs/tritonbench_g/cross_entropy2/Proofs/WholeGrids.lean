import bench.repairs.tritonbench_g.cross_entropy2.Proofs.Host
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.BackwardGrid
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.Native

namespace VeriTile.Bench.Repairs.CrossEntropy2.Host
open VeriTile.Triton Source
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World Step bind)
noncomputable section

/-- The host's direct memory fold is the already-proved complete backward
schedule; this is an execution identity, not an output premise. -/
theorem backward_models {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters) (m : Memory α) (count : Nat) :
    backwardGrid M p m count=BackwardGrid.backward M p m count := by
  have fold : ∀ schedule m,
      schedule.foldl (fun current q => Source.backward M p current q.1 q.2) m=
        VeriTile.Bench.Repairs.CrossEntropy1.Grid.unflat
          (VeriTile.Bench.Repairs.CrossEntropy1.Schedule.run (BackwardGrid.backwardStep M p) schedule
            (VeriTile.Bench.Repairs.CrossEntropy1.Grid.flat m)) := by
    intro schedule
    induction schedule with
    | nil => intro m; rfl
    | cons q tail ih => intro m; exact ih (Source.backward M p m q.1 q.2)
  exact fold _ m

inductive LaunchTrace {α σ : Type} (rows : Nat)
    (setup : World α σ → Step (World α σ) Unit) (kernel : Memory α → Memory α)
    (initial : World α σ) : World α σ → Type
  | empty (zero : rows=0) : LaunchTrace rows setup kernel initial initial
  | dispatched (nonzero : rows≠0) (ready : World α σ) (call : setup initial=.ok () ready) :
      LaunchTrace rows setup kernel initial {ready with memory:=kernel ready.memory}

theorem launch_trace {α σ : Type} (rows : Nat)
    (setup : World α σ → Step (World α σ) Unit) (kernel : Memory α → Memory α)
    (initial final : World α σ)
    (success : (if rows=0 then .ok () initial else bind (setup initial) fun _ ready =>
      .ok () {ready with memory:=kernel ready.memory})=.ok () final) :
    Nonempty (LaunchTrace rows setup kernel initial final) := by
  by_cases zero : rows=0
  · rw [if_pos zero] at success
    cases success
    exact ⟨.empty zero⟩
  · rw [if_neg zero] at success
    obtain ⟨u,ready,hcall,hreturn⟩ := (bind_success _ _ _ _).mp success
    cases u
    cases hreturn
    exact ⟨.dispatched zero ready hcall⟩

#axiomsClean backward_models
#axiomsClean launch_trace
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Host

namespace VeriTile.Bench.Repairs.CrossEntropy2.Physical
open VeriTile.Triton Source
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World)
open VeriTile.Bench.Repairs.CrossEntropy1.Physical (Layout Word bytes Covers)
noncomputable section

/-- Every physical byte outside the three actual scalar-store windows is
unchanged. Partial trailing words and mixed int32/FP32 aliases are included. -/
theorem forward_bytes {α σ : Type} [Inhabited α] (layout : Layout σ) (encode : α → Word)
    (M : FP32 α) (p : Parameters) (s : World α σ) (count device address : Nat) (sep : Grid.Separation p)
    (outside : ∀ q∈VeriTile.Bench.Repairs.CrossEntropy1.Grid.tasks p.rows count, ∀ r i,
      Grid.writes p q (r,i) → ¬Covers layout s.native r i device address) :
    bytes layout encode {s with memory:=Grid.forward M p s.memory count} device address=
      bytes layout encode s device address := by
  unfold bytes
  cases hloc : layout.locate s.native device address with
  | none => rfl
  | some loc =>
    rcases loc with ⟨r,i,j⟩
    have found := layout.located s.native device address r i j hloc
    have frame := Grid.complete_frame M p s.memory count sep r i (by
      intro q hq hw
      exact outside q hq r i hw ⟨found.2.1,by omega,by have hj := j.isLt; omega⟩)
    dsimp only
    rw [frame]

theorem backward_bytes {α σ : Type} [Inhabited α] (layout : Layout σ) (encode : α → Word)
    (M : FP32 α) (p : Parameters) (s : World α σ) (count device address : Nat)
    (sep : BackwardGrid.BackwardSeparation p)
    (outside : ∀ q∈VeriTile.Bench.Repairs.CrossEntropy1.Grid.tasks p.rows count, ∀ r i,
      BackwardGrid.backwardWrites p q (r,i) → ¬Covers layout s.native r i device address) :
    bytes layout encode {s with memory:=Host.backwardGrid M p s.memory count} device address=
      bytes layout encode s device address := by
  rw [Host.backward_models]
  unfold bytes
  cases hloc : layout.locate s.native device address with
  | none => rfl
  | some loc =>
    rcases loc with ⟨r,i,j⟩
    have found := layout.located s.native device address r i j hloc
    have frame := BackwardGrid.backward_complete_frame M p s.memory count sep r i (by
      intro q hq hw
      exact outside q hq r i hw ⟨found.2.1,by omega,by have hj := j.isLt; omega⟩)
    dsimp only
    rw [frame]

#axiomsClean forward_bytes
#axiomsClean backward_bytes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Physical
