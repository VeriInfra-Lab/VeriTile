import bench.tritonbench_g.triton_argmax.Proofs.JITNativeDimensionGrid

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITNativeDimensionIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
noncomputable section
set_option maxHeartbeats 5000000

theorem grid_execution (rest : Algebra Word) (c : DimConfig) (index64 : Bool)
    (nonempty : 0 < c.extent) (jobs : List (Nat × Nat))
    (valid : ∀ job, job ∈ jobs → job.2 < c.inner)
    (s : State Word) (input : Nat → Nat → Nat → Word)
    (reads : ∀ r col inner, r < c.outer → col < c.extent → inner < c.inner →
      (s.mem BoundSource.X (r*c.extent*c.inner+col*c.inner+inner)).read .real = input r inner col) :
    ∃ t, runSchedule (jobs.map (launch rest c index64)) s = some t ∧
      (∀ row inner, row < c.outer → inner < c.inner →
        t.mem (Region.cast BoundSource.O) (row*c.inner+inner) =
          if (row/128,inner) ∈ jobs then .mk .int (formula c nonempty input row inner)
          else s.mem (Region.cast BoundSource.O) (row*c.inner+inner)) ∧
      (∀ r o, (r ≠ Region.cast BoundSource.O ∨ ∀ row inner, row < c.outer → inner < c.inner →
        o ≠ row*c.inner+inner) → t.mem r o = s.mem r o) := by
  induction jobs generalizing s with
  | nil => exact ⟨s,rfl,by simp,by simp⟩
  | cons job jobs ih =>
    have hi := valid job (by simp)
    obtain ⟨a, ran, output, frame⟩ := program rest c index64 nonempty job hi s input reads
    have nextReads : ∀ r col inner, r < c.outer → col < c.extent → inner < c.inner →
        (a.mem BoundSource.X (r*c.extent*c.inner+col*c.inner+inner)).read .real = input r inner col := by
      intro r col inner hr hc hk
      rw [frame _ _ (Or.inl (by decide))]
      exact reads r col inner hr hc hk
    obtain ⟨t, runs, outputs, frames⟩ := ih (fun j hj => valid j (by simp [hj])) a nextReads
    refine ⟨t, ?_, ?_, ?_⟩
    · simp only [List.map_cons, runSchedule, ran, Option.bind_some]
      exact runs
    · intro row inner hr hk
      rw [outputs row inner hr hk]
      by_cases later : (row/128,inner) ∈ jobs
      · simp [later]
      · simp only [if_neg later, List.mem_cons]
        by_cases owner : (row/128,inner) = job
        · simp only [owner, true_or, if_true]
          have ownerRow : row/128 = job.1 := congrArg Prod.fst owner
          have ownerInner : inner = job.2 := congrArg Prod.snd owner
          have rowEq : job.1*128+row%128 = row := by
            rw [←ownerRow]
            omega
          have stored := output ⟨row%128, Nat.mod_lt _ (by decide)⟩ (by simpa [rowEq] using hr)
          simpa only [Fin.val_mk, rowEq, ←ownerInner] using stored
        · simp only [owner, later, false_or, if_false]
          apply frame
          right
          intro lane _ equal
          obtain ⟨sameRow,sameInner⟩ := coordinates c.inner row (job.1*128+lane.val) inner job.2 hk hi equal
          have rowDiv : (job.1*128+lane.val)/128 = job.1 := by omega
          apply owner
          exact Prod.ext (by rw [sameRow, rowDiv]) sameInner
    · intro r o outside
      rw [frames r o outside]
      apply frame
      rcases outside with region | out
      · exact Or.inl region
      · exact Or.inr (fun lane hr => out (job.1*128+lane.val) job.2 hr hi)

/-- Logical input and independent dimension result; every output is active. -/
def input (c : DimConfig) (xs : ChannelValues Word [⟨.real,[c.outer,c.extent,c.inner],BoundSource.X⟩])
    (r i col : Nat) : Word :=
  if hr : r < c.outer then if hi : i < c.inner then if hc : col < c.extent then
    xs 0 (⟨r,hr⟩,⟨col,hc⟩,⟨i,hi⟩,PUnit.unit) else 0 else 0 else 0

def mathematical (rest : Algebra Word) (c : DimConfig) (nonempty : 0 < c.extent) (index64 : Bool)
    (xs : ChannelValues Word [⟨.real,[c.outer,c.extent,c.inner],BoundSource.X⟩]) :
    ChannelValues Word [⟨.int,[c.outer,c.inner],Region.cast BoundSource.O⟩] :=
  Fin.cases (fun j => formula c nonempty (input c xs) j.1.val j.2.1.val) (fun i => Fin.elim0 i)

theorem schedule_safe (rest : Algebra Word) (c : DimConfig) (index64 : Bool)
    (bounds : RegionName → Nat)
    (inputBounds : ∀ row col inner, row < c.outer → col < c.extent → inner < c.inner →
      row*c.extent*c.inner+col*c.inner+inner < bounds BoundSource.X)
    (outputBounds : ∀ row inner, row < c.outer → inner < c.inner →
      row*c.inner+inner < bounds (Region.cast BoundSource.O)) (s : State Word) :
    ScheduleSafe bounds (JITIO.nativeDimensionSchedule rest c index64) s := by
  have allSafe : ∀ (jobs : List (Nat × Nat)), (∀ job, job ∈ jobs → job.2 < c.inner) → ∀ before,
      ScheduleSafe bounds (jobs.map (launch rest c index64)) before := by
    intro jobs
    induction jobs with
    | nil => intro _ _; trivial
    | cons job jobs ih =>
      intro valid before
      have hi := valid job (by simp)
      constructor
      · let initial := (JITIO.atProgram (Typed.backend rest) (JITSource.dimension BoundSource.X BoundSource.O 128 128 index64) (job.1*c.inner+job.2)).initial before
        have hp : initial.pids 0 = job.1*c.inner+job.2 := by simp [initial, launch]
        change TraceSafe (Typed.backend rest) bounds
          (JITSource.dimension BoundSource.X BoundSource.O 128 128 index64) (JITDimension.arguments initial c.outer c.extent c.inner 0)
        apply JITNativeDimensionSafety.source_safe rest initial BoundSource.X BoundSource.O
          c.outer c.extent c.inner 128 128 0 index64 (by decide) bounds
        · intro row col hr hc
          simpa only [hp, Nat.add_zero, remainder c job hi] using inputBounds row col job.2 hr hc hi
        · intro lane hr
          simp only [hp, Nat.add_zero, quotient c job hi, remainder c job hi] at hr ⊢
          exact outputBounds _ _ hr hi
      · intro next _
        exact ih (fun j hj => valid j (by simp [hj])) next
  exact allSafe (BoundSource.groups c) (fun job hj => BoundSource.group_inner c job hj) s

theorem implements (rest : Algebra Word) (c : DimConfig) (nonempty : 0 < c.extent) (index64 : Bool) :
    (JITIO.nativeDimension rest c index64).Implements (mathematical rest c nonempty index64) := by
  intro vals _ bounds bounded s binding
  let xs := input c vals
  have reads : ∀ r col inner, r < c.outer → col < c.extent → inner < c.inner →
      (s.mem BoundSource.X (r*c.extent*c.inner+col*c.inner+inner)).read .real = xs r inner col := by
    intro r col inner hr hc hi
    have h := binding 0 (⟨r,hr⟩,⟨col,hc⟩,⟨inner,hi⟩,PUnit.unit) trivial
    simpa [JITIO.nativeDimension, mapValue, xs, input, hr, hc, hi] using h
  refine ⟨schedule_safe rest c index64 bounds
    (fun r col inner hr hc hi => bounded.2.1 0 (⟨r,hr⟩,⟨col,hc⟩,⟨inner,hi⟩,PUnit.unit) trivial)
    (fun r inner hr hi => bounded.2.2.1 0 (⟨r,hr⟩,⟨inner,hi⟩,PUnit.unit) trivial) s, ?_⟩
  obtain ⟨final, ran, output, frame⟩ := grid_execution rest c index64 nonempty (BoundSource.groups c)
    (fun job hj => BoundSource.group_inner c job hj) s xs reads
  refine ⟨final, ran, ?_, ?_⟩
  · intro o j _
    have ho : o = 0 := Fin.eq_zero o
    subst o
    have member : (j.1.val/128,j.2.1.val) ∈ BoundSource.groups c := by
      simp only [BoundSource.groups, List.mem_flatMap, List.mem_map, List.mem_range]
      exact ⟨j.1.val/128, by have h := j.1.isLt; omega, j.2.1.val, j.2.1.isLt, rfl⟩
    have result := output j.1.val j.2.1.val j.1.isLt j.2.1.isLt
    rw [if_pos member] at result
    change mapCell id (final.mem (Region.cast BoundSource.O) (j.1.val*c.inner+j.2.1.val)) = _
    rw [result]
    rfl
  · intro r o outside
    apply frame r o
    by_cases region : r = Region.cast BoundSource.O
    · right
      intro row inner hr hi
      have miss := outside.1 0 (⟨row,hr⟩,⟨inner,hi⟩,PUnit.unit) trivial
      change r ≠ Region.cast BoundSource.O ∨ o ≠ row*c.inner+inner at miss
      exact miss.resolve_left (not_not.mpr region)
    · exact Or.inl region

#axiomsClean grid_execution
#axiomsClean implements
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITNativeDimensionIO
