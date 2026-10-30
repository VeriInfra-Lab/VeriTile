import bench.tritonbench_g.matmul_dequant_int4.IO
import bench.tritonbench_g.matmul_dequant_int4.Proofs.AccessSafety

namespace VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.PublicIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

variable {α : Type} [Inhabited α]

theorem schedule_run (f : Source.Profile) (M : Algebra α) (p : Source.Parameters)
    (ps : List Nat) (s : State α) :
    runSchedule (ps.map (invocation f M p)) s=
      GridExecution.sequence (KernelGrid.one f M p) ps s := by
  induction ps generalizing s with
  | nil => rfl
  | cons pid ps ih =>
    change ((invocation f M p pid).run s).bind _ = _
    have exactSource : (invocation f M p pid).run s=KernelGrid.one f M p pid s := rfl
    rw [exactSource]
    simp only [GridExecution.sequence]
    congr 1
    funext next
    exact ih next

private theorem safe_list (bounds : RegionName → Nat) (jobs : List (Invocation α))
    (each : ∀ job∈jobs,∀ s,job.Safe bounds s) (s : State α) : ScheduleSafe bounds jobs s := by
  induction jobs generalizing s with
  | nil => trivial
  | cons job rest ih =>
    exact ⟨each job (by simp) s,fun t _ => ih (fun next h => each next (by simp [h])) t⟩

theorem io_safe (f : Source.Profile) (M : Algebra α) (p : Source.Parameters)
    (valid : KernelGrid.Valid p) (xs : Inputs α f p) (bounds : RegionName → Nat)
    (bounded : (dequantIO f M p).InBounds bounds xs) (s : State α) :
    ScheduleSafe bounds ((dequantIO f M p).schedule xs) s := by
  apply safe_list
  intro job member before
  obtain ⟨pid,_,rfl⟩ := List.mem_map.mp member
  apply Source.trace_safe f M p (KernelGrid.atProgram p pid before) bounds
  · intro i active
    exact bounded.2.1 ⟨0,by simp [inputs]⟩
      (⟨Source.row p (KernelGrid.atProgram p pid before) i,active.1⟩,
       ⟨Source.column p (KernelGrid.atProgram p pid before) i,active.2⟩,PUnit.unit) trivial
  · intro i active
    exact bounded.2.1 ⟨1,by simp [inputs]⟩
      (⟨Source.row p (KernelGrid.atProgram p pid before) i,active.1⟩,
       ⟨Source.column p (KernelGrid.atProgram p pid before) i,active.2⟩,PUnit.unit) trivial
  · intro i active
    exact bounded.2.1 ⟨2,by simp [inputs]⟩
      (⟨Source.row p (KernelGrid.atProgram p pid before) i,active.1⟩,
       ⟨Source.column p (KernelGrid.atProgram p pid before) i,active.2⟩,PUnit.unit) trivial
  · intro i active
    have h := bounded.2.2.1 ⟨0,by simp [outputs]⟩
      (⟨Source.row p (KernelGrid.atProgram p pid before) i,active.1⟩,
       ⟨Source.column p (KernelGrid.atProgram p pid before) i,active.2⟩,PUnit.unit) trivial
    simpa [Source.outputOffset,valid.outputRow,valid.outputColumn] using h

def scalarFormula (f : Source.Profile) (M : Algebra α) (p : Source.Parameters)
    (xs : Inputs α f p) (x : KernelGrid.Coordinate p) : α :=
  M.binary none f.dtype .mul
    (M.cast (some .fp32) .real f.dtype (M.fromInt (some .fp32) (difference xs x)))
    (scale f xs x)

theorem kernel_result (f : Source.Profile) (M : Algebra α) (p : Source.Parameters)
    (xs : Inputs α f p) (s : State α) (pins : (dequantIO f M p).InputBinding xs s)
    (x : KernelGrid.Coordinate p) :
    KernelGrid.result f M p s x=scalarFormula f M p xs x := by
  have weight := pins ⟨0,by simp [inputs]⟩ (x.1,x.2,PUnit.unit) trivial
  have zero := pins ⟨1,by simp [inputs]⟩ (x.1,x.2,PUnit.unit) trivial
  have scale := pins ⟨2,by simp [inputs]⟩ (x.1,x.2,PUnit.unit) trivial
  cases f
  all_goals
    change (s.mem p.weight (x.1.val/8*p.weightRowStride+x.2.val*p.weightColumnStride)).read .int =
      packed xs Bool.false x at weight
    change (s.mem p.zero (x.1.val/p.groupSize*p.zeroRowStride+x.2.val/8*p.zeroColumnStride)).read .int =
      packed xs Bool.true x at zero
    simp only [KernelGrid.result,scalarFormula,difference,weight,zero]
    exact congrArg (fun z => M.binary none _ .mul _ z) scale

/-- Whole-grid IO proof reuses the existing grid run/output/frame theorem;
source trace safety is connected separately to every declared active window. -/
theorem implements (f : Source.Profile) (M : Algebra α) (p : Source.Parameters)
    (valid : KernelGrid.Valid p) :
    (dequantIO f M p).Implements (fun xs => result f p (scalarFormula f M p xs)) := by
  apply ProgramIO.Implements.intro
  · intro xs bounds bounded initial _
    exact io_safe f M p valid xs bounds bounded initial
  · intro xs initial pins
    obtain ⟨final,ran,values,frame⟩ := KernelGrid.whole f M p valid initial
    refine ⟨final,?_,?_,?_⟩
    · rw [schedule_run]
      exact ran
    · intro o j _
      have ho : o=⟨0,by simp [outputs]⟩ := by
        apply Fin.ext
        change o.val=0
        have bound : o.val<1 := o.isLt
        omega
      subst o
      let x : KernelGrid.Coordinate p := (j.1,j.2.1)
      have value := values x trivial
      change final.mem p.output (KernelGrid.address p x)=Source.cell f (KernelGrid.result f M p initial x) at value
      rw [kernel_result f M p xs initial pins x] at value
      change mapCell id (final.mem p.output (KernelGrid.address p x))=_
      rw [value]
      cases f <;> rfl
    · intro r offset outside
      apply frame r offset
      by_cases other : r≠p.output
      · exact Or.inl other
      · right
        intro x _
        have excluded := outside.1 ⟨0,by simp [outputs]⟩ (x.1,x.2,PUnit.unit) trivial
        exact excluded.resolve_left other

#axiomsClean implements
end
end VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.PublicIO
