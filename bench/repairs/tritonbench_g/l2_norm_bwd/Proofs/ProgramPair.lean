import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.PairIO
import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.Rewrite
import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.IOGrid

namespace VeriTile.Bench.Repairs.L2NormBwd.ProgramPair
open VeriTile Triton FP.Structural FPEquiv
open scoped VeriTile.Spec
noncomputable section

theorem invocation_run_eq {A : Type} [Inhabited A] (M : Algebra A)
    (layout : Strides) (N block row : Nat) (eps : A) (initial : State A) :
    (PublicIO.invocation M layout N block row eps).run initial =
      (candidateInvocation M layout N block row eps).run initial :=
  Rewrite.execution_eq M layout N block _

theorem invocation_safe_iff {A : Type} [Inhabited A] (M : Algebra A)
    (layout : Strides) (N block row : Nat) (eps : A)
    (bounds : RegionName → Nat) (initial : State A) :
    (PublicIO.invocation M layout N block row eps).Safe bounds initial ↔
      (candidateInvocation M layout N block row eps).Safe bounds initial :=
  Rewrite.source_safe_iff M layout N block bounds _

private theorem jobs_run_eq {A : Type} [Inhabited A] (M : Algebra A)
    (layout : Strides) (N block : Nat) (eps : A) (jobs : List Nat) (initial : State A) :
    runSchedule (jobs.map (fun row => PublicIO.invocation M layout N block row eps)) initial =
      runSchedule (jobs.map (fun row => candidateInvocation M layout N block row eps)) initial := by
  induction jobs generalizing initial with
  | nil => rfl
  | cons row jobs ih =>
    simp only [List.map_cons,runSchedule,invocation_run_eq]
    cases (candidateInvocation M layout N block row eps).run initial with
    | none => rfl
    | some final => exact ih final

theorem schedule_run_eq {A : Type} [Inhabited A] (M : Algebra A)
    (layout : Strides) (rows N block : Nat) (eps : A) (initial : State A) :
    runSchedule (PublicIO.schedule M layout rows N block eps) initial =
      runSchedule (candidateSchedule M layout rows N block eps) initial :=
  jobs_run_eq M layout N block eps (List.range rows) initial

private theorem jobs_safe_iff {A : Type} [Inhabited A] (M : Algebra A)
    (layout : Strides) (N block : Nat) (eps : A) (jobs : List Nat)
    (bounds : RegionName → Nat) (initial : State A) :
    ScheduleSafe bounds (jobs.map (fun row => PublicIO.invocation M layout N block row eps)) initial ↔
      ScheduleSafe bounds (jobs.map (fun row => candidateInvocation M layout N block row eps)) initial := by
  induction jobs generalizing initial with
  | nil => exact Iff.rfl
  | cons row jobs ih =>
    simp only [List.map_cons,ScheduleSafe,invocation_safe_iff,invocation_run_eq,ih]

theorem schedule_safe_iff {A : Type} [Inhabited A] (M : Algebra A)
    (layout : Strides) (rows N block : Nat) (eps : A)
    (bounds : RegionName → Nat) (initial : State A) :
    ScheduleSafe bounds (PublicIO.schedule M layout rows N block eps) initial ↔
      ScheduleSafe bounds (candidateSchedule M layout rows N block eps) initial :=
  jobs_safe_iff M layout N block eps (List.range rows) bounds initial

/-- Start from the existing constructive full-grid source execution. Exact
local index rewrites give the same complete candidate state, and transport
actual trace safety. Success, outputs and both frames are conclusions. -/
theorem equivalent {A : Type} [Inhabited A] (M : Algebra A) (layout : Strides)
    (rows N block : Nat) (cover : N≤block) (unique : PublicIO.OutputUnique layout rows N)
    (eps : A) :
    (originalIO M layout rows N block eps).Equivalent (candidateIO M layout rows N block eps) := by
  intro xs _ bounds hb initial _
  have safe := PublicIO.io_safe M id (fun _ => id) layout rows N block eps xs bounds hb initial
  obtain ⟨final,ran,_,frame⟩ :=
    PublicIO.grid_execution M layout rows N block cover unique eps initial
  have candidateRan : runSchedule (candidateSchedule M layout rows N block eps) initial=some final := by
    rw [←schedule_run_eq]
    exact ran
  have framed : (originalIO M layout rows N block eps).Frame xs initial final := by
    intro name offset outside
    exact frame name offset
      (PublicIO.outside_of_io M id (fun _ => id) layout rows N block eps xs name offset outside)
  exact ⟨safe,(schedule_safe_iff M layout rows N block eps bounds initial).mp safe,
    final,final,ran,candidateRan,fun _ _ _ => rfl,framed,framed⟩

theorem grid_equiv {A : Type} [Inhabited A] (M : Algebra A) (layout : Strides)
    (rows N block : Nat) (cover : N≤block) (unique : PublicIO.OutputUnique layout rows N)
    (eps : A) :
    original M layout rows N block eps ≡[[]] candidate M layout rows N block eps :=
  ProgramIO.Implementation.of_equivalent (candidateIO M layout rows N block eps)
    (equivalent M layout rows N block cover unique eps)

#axiomsClean equivalent
#axiomsClean grid_equiv
end
end VeriTile.Bench.Repairs.L2NormBwd.ProgramPair
