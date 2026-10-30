import bench.tritonbench_g.rmsnorm_fused.IO
import bench.tritonbench_g.rmsnorm_fused.Proofs.KernelSafety
import bench.tritonbench_g.rmsnorm_fused.Proofs.TypedGrid

namespace VeriTile.Bench.Repairs.RmsnormFused.Interface
open VeriTile Triton FP.Structural
noncomputable section
variable {α β : Type} [Inhabited α] [Inhabited β]

@[simp] theorem invocation_initial (A : Algebra β) (shape : Shape) (epsilon : β)
    (row : Nat) (s : State β) :
    (invocation A shape epsilon row).initial s=Runtime.bind (Runtime.setup shape.rows row s) epsilon := rfl

@[simp] theorem schedule_run (A : Algebra β) (shape : Shape) (epsilon : β)
    (rows : List Nat) (s : State β) :
    runSchedule (rows.map (invocation A shape epsilon)) s=
      Runtime.run A "X" "Y" "W" shape.rows shape.width shape.block epsilon rows s := by
  induction rows generalizing s with
  | nil => rfl
  | cons row rows ih =>
    change (exec A (source "X" "Y" "W" shape.width shape.block)
      (Runtime.bind (Runtime.setup shape.rows row s) epsilon)).bind
        (runSchedule (rows.map (invocation A shape epsilon))) =
      (exec A (source "X" "Y" "W" shape.width shape.block)
      (Runtime.bind (Runtime.setup shape.rows row s) epsilon)).bind
        (Runtime.run A "X" "Y" "W" shape.rows shape.width shape.block epsilon rows)
    rw [show runSchedule (rows.map (invocation A shape epsilon)) =
      Runtime.run A "X" "Y" "W" shape.rows shape.width shape.block epsilon rows from funext ih]

theorem schedule_safe (A : Algebra β) (shape : Shape) (valid : shape.Valid) (epsilon : β)
    (rows : List Nat) (members : ∀ row∈rows,row<shape.rows) (bounds : RegionName → Nat)
    (hx : ∀ i,i<shape.rows*shape.width → i<bounds "X")
    (hw : ∀ i,i<shape.width → i<bounds "W")
    (hy : ∀ i,i<shape.rows*shape.width → i<bounds "Y") (s : State β) :
    ScheduleSafe bounds (rows.map (invocation A shape epsilon)) s := by
  induction rows generalizing s with
  | nil => trivial
  | cons row rows ih =>
    refine ⟨?_,fun t _ => ih (fun r hr => members r (by simp [hr])) t⟩
    change FP.Structural.TraceSafe A bounds (source "X" "Y" "W" shape.width shape.block)
      (Runtime.bind (Runtime.setup shape.rows row s) epsilon)
    apply Runtime.source_trace_safe A bounds (Runtime.setup shape.rows row s)
      "X" "Y" "W" shape.width shape.block epsilon valid.1 valid.2.1
    · intro j active
      apply hx
      exact offset_bounded shape row j.val (members row (by simp)) active
    · intro j active
      exact hw j.val active
    · intro j active
      apply hy
      exact offset_bounded shape row j.val (members row (by simp)) active

/-- Source safety and IO assembly are shared by mathematical/native entries.
The result premise supplies typed logical values; execution stays the actual
source run already related to the shared schedule above. -/
theorem implements (A : Algebra β) (shape : Shape) (valid : shape.Valid) (epsilon : β)
    (rows : List Nat) (all : rows.Perm shape.grid) (encode : α → β) (observe : Nat → β → α)
    (f : Inputs α shape → Fin (shape.rows*shape.width) → α)
    (computed : ∀ (values : Inputs α shape) (initial : State β),
      (∀ i, i<shape.rows*shape.width → Runtime.read initial "X" i=encode (inputAt values i)) →
      (∀ i, i<shape.width → Runtime.read initial "W" i=encode (weightAt values i)) →
      ∃ final, Runtime.run A "X" "Y" "W" shape.rows shape.width shape.block epsilon rows initial=some final ∧
        (∀ i : Fin (shape.rows*shape.width),
          mapCell (observe i.val) (final.mem "Y" i.val)=Cell.mk .real (f values i)) ∧
        (∀ (region : RegionName) offset,region≠"Y" ∨ shape.rows*shape.width≤offset →
          final.mem region offset=initial.mem region offset)) :
    (programIO A shape epsilon rows encode observe).Implements (fun values => result (f values)) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds hb initial _
    apply schedule_safe A shape valid epsilon rows
      (fun row member => List.mem_range.mp (all.mem_iff.mp member)) bounds
    · intro i hi
      exact hb.2.1 ⟨0,by simp [inputs,programIO,memory]⟩ (⟨i,hi⟩,PUnit.unit) trivial
    · intro i hi
      exact hb.2.1 ⟨1,by simp [inputs,programIO,memory]⟩ (⟨i,hi⟩,PUnit.unit) trivial
    · intro i hi
      exact hb.2.2.1 ⟨0,by simp [outputs,programIO,memory]⟩ (⟨i,hi⟩,PUnit.unit) trivial
  · intro vals initial hin
    have hx (i : Nat) (hi : i<shape.rows*shape.width) :
        Runtime.read initial "X" i=encode (inputAt vals i) := by
      simpa only [programIO,memory,inputs,ProgramIO.InputBinding,mapValue,Runtime.read,
        inputOffset,inputAt,dif_pos hi,input] using
        hin ⟨0,by simp [programIO,memory,inputs]⟩ (⟨i,hi⟩,PUnit.unit) trivial
    have hw (i : Nat) (hi : i<shape.width) :
        Runtime.read initial "W" i=encode (weightAt vals i) := by
      simpa only [programIO,memory,inputs,ProgramIO.InputBinding,mapValue,Runtime.read,
        inputOffset,weightAt,dif_pos hi,weight] using
        hin ⟨1,by simp [programIO,memory,inputs]⟩ (⟨i,hi⟩,PUnit.unit) trivial
    obtain ⟨final,ran,values,frame⟩ := computed vals initial hx hw
    refine ⟨final,?_,?_,?_⟩
    · exact (schedule_run A shape epsilon rows initial).trans ran
    · intro o j _
      have ho : o.val<1 := o.isLt
      rcases o with ⟨o,ho⟩
      have same : o=0 := by simpa [outputs] using Nat.eq_zero_of_le_zero (Nat.le_of_lt_succ ho)
      subst o
      exact values j.1
    · intro r offset outside
      apply frame r offset
      by_cases different : r≠"Y"
      · exact .inl different
      · have same : r="Y" := Classical.not_not.mp different
        subst r
        right
        by_contra h
        have bound : offset<shape.rows*shape.width := Nat.lt_of_not_ge h
        have no := outside.1 ⟨0,by simp [programIO,memory,outputs]⟩ (⟨offset,bound⟩,PUnit.unit) trivial
        exact no.elim (fun bad => bad rfl) (fun bad => bad rfl)
end
end VeriTile.Bench.Repairs.RmsnormFused.Interface
