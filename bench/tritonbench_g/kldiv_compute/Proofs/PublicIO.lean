import bench.tritonbench_g.kldiv_compute.IO
import bench.tritonbench_g.kldiv_compute.Proofs.LogicalExecution
import VeriTile.Triton.Float.ProgramIO

namespace VeriTile.Bench.Repairs.KldivCompute.PublicIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.VectorAddition
open ValueProofs (gridCount)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000


theorem schedules_eq {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (jobs : List Nat) (s : State α) :
    runSchedule (jobs.map (invocation M p)) s =
      PointwisePrograms.runSchedule (fun pid current => (invocation M p pid).run current) jobs s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons pid rest ih =>
      simp only [List.map_cons, runSchedule, PointwisePrograms.runSchedule]
      cases h : (invocation M p pid).run s <;> simp [ih]

theorem complete {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (positive : 0 < p.block) (s : State α) :
    ∃ final, runSchedule ((List.range (gridCount p)).map (invocation M p)) s = some final ∧
      final.mem = PointwisePrograms.finalMemory (Program.resultCell M p) p s.mem := by
  rw [schedules_eq]
  apply PointwisePrograms.complete_schedule (Program.resultCell M p) ?_ p _ ?_
    s positive _ (List.Perm.refl _)
  · intro a b o h
    simp only [Program.resultCell, h]
  · intro pid current
    let initial := (invocation M p pid).initial current
    obtain ⟨final, executed, memory⟩ := Program.complete_correct M p initial
    refine ⟨final, executed, memory.trans ?_⟩
    funext r o
    simpa only [initial, invocation, Invocation.initial, bindRegisters,
      List.foldl_nil, Option.getD_some, ↓reduceIte, PointwisePrograms.oneMemory] using
        Program.expected_cell M p initial r o

theorem safe {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (bounds : RegionName → Nat) (hx : p.n ≤ bounds p.x) (hy : p.n ≤ bounds p.y)
    (ho : p.n ≤ bounds p.output) (jobs : List Nat) (s : State α) :
    ScheduleSafe bounds (jobs.map (invocation M p)) s := by
  induction jobs generalizing s with
  | nil => trivial
  | cons pid rest ih =>
      refine ⟨?_, fun next _ => ih next⟩
      exact LogicalExecution.source_safe M p ((invocation M p pid).initial s) bounds hx hy ho





def formula (f : α → α → α) (p : Parameters)
    (xs : ChannelValues α (inputChannels p)) : ChannelValues α (outputChannels p) := fun o =>
  match o with
  | ⟨0, _⟩ => fun j => f (xs ⟨0, by simp⟩ j) (xs ⟨1, by simp⟩ j)
  | ⟨_+1, h⟩ => False.elim (by simpa using h)

theorem implements {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (positive : 0 < p.block) (f : α → α → α) (scalar : ∀ x y, Program.scalar M x y = f x y) :
    (io M p).Implements (formula f p) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds bounded s _
    have capacity {n b : Nat} (h : ∀ i : Fin n, i.val < b) : n ≤ b := by
      by_contra hn
      exact (Nat.lt_irrefl b) (h ⟨b, by omega⟩)
    exact safe M p bounds
      (capacity (fun j => bounded.2.1 ⟨0, by simp [io]⟩ (j, PUnit.unit) trivial))
      (capacity (fun j => bounded.2.1 ⟨1, by simp [io]⟩ (j, PUnit.unit) trivial))
      (capacity (fun j => bounded.2.2.1 ⟨0, by simp [io]⟩ (j, PUnit.unit) trivial)) _ s
  · intro vals s binding
    obtain ⟨final, executed, memory⟩ := complete M p positive s
    refine ⟨final, executed, ?_, ?_⟩
    · intro o j _
      fin_cases o
      change mapCell id (final.mem p.output j.1.val) =
        Cell.mk .real (f (vals 0 j) (vals 1 j))
      rw [memory]
      simp only [PointwisePrograms.finalMemory]
      rw [if_pos (by simpa using j.1.isLt)]
      change Cell.mk .real (Program.scalar M ((s.mem p.x j.1.val).read .real)
        ((s.mem p.y j.1.val).read .real)) = Cell.mk .real (f (vals 0 j) (vals 1 j))
      rw [scalar]
      have hx := binding ⟨0, by simp [io]⟩ j trivial
      have hy := binding ⟨1, by simp [io]⟩ j trivial
      exact congrArg₂ (fun a b => Cell.mk .real (f a b)) hx hy
    · intro r offset outside
      rw [memory]
      simp only [PointwisePrograms.finalMemory]
      rw [if_neg]
      rintro ⟨same, active⟩
      have h := outside.1 ⟨0, by simp [io]⟩ (⟨offset, active⟩, PUnit.unit) trivial
      exact h.elim (fun h => h same) (fun h => h rfl)

#axiomsClean complete
#axiomsClean safe
#axiomsClean implements
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivCompute.PublicIO
