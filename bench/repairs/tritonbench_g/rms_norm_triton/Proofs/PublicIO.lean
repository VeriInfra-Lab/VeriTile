import bench.repairs.tritonbench_g.rms_norm_triton.Proofs.SourceSafety
import bench.tritonbench_g.rms_norm_triton.Proofs.WholeMath
import bench.tritonbench_g.rms_norm_triton.Proofs.Observed

namespace VeriTile.Bench.Repairs.RmsNormTriton.GridPublic
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.RmsNormTriton Whole Observed
noncomputable section

private theorem whole_frame {α β : Type} (M : Algebra β) (encode : α → β)
    (observe : Nat → β → α) (c : Layout) (eps : ℝ) (rows : List Nat) (epsilon : β)
    (xs : (io M encode observe c eps rows epsilon).Inputs) (s t : State β)
    (frame : ∀ r a,r≠c.Y ∨ c.M*c.N≤a → t.mem r a=s.mem r a) :
    (io M encode observe c eps rows epsilon).Frame xs s t := by
  intro r a outside
  apply frame
  by_cases same : r=c.Y
  · right
    by_contra h
    have small : a<c.M*c.N := by omega
    have pos : 0<c.N := by
      by_contra h
      have zero : c.N=0 := by omega
      simp [zero] at small
    let row : Fin c.M := ⟨a/c.N,(Nat.div_lt_iff_lt_mul pos).2 small⟩
    let col : Fin c.N := ⟨a%c.N,Nat.mod_lt a pos⟩
    have eq : a=row.val*c.N+col.val := by
      dsimp [row,col]
      simpa [Nat.mul_comm,Nat.add_comm] using (Nat.mod_add_div a c.N).symm
    exact (outside.1 0 (row,col,()) trivial).elim (fun ne => ne same) (fun ne => ne eq)
  · exact Or.inl same

private theorem schedule_safe {α β : Type} [Inhabited β] (M : Algebra β)
    (encode : α → β) (observe : Nat → β → α) (c : Layout) (eps : ℝ) (epsilon : β)
    (rows : List Nat) (schedule : rows.Perm (List.range c.M))
    (xs : (io M encode observe c eps rows epsilon).Inputs) (bounds : RegionName → Nat)
    (bound : (io M encode observe c eps rows epsilon).InBounds bounds xs) (s : State β) :
    ScheduleSafe bounds ((io M encode observe c eps rows epsilon).schedule xs) s := by
  apply KernelSafety.schedule_safe M c eps rows s bounds (epsilon := epsilon)
  · intro pid hp i hi
    exact bound.2.1 0 (⟨pid,List.mem_range.mp (schedule.mem_iff.mp hp)⟩,⟨i.val,hi⟩,()) trivial
  · intro i hi
    exact bound.2.1 1 (⟨i.val,hi⟩,()) trivial
  · intro pid hp i hi
    exact bound.2.2.1 0 (⟨pid,List.mem_range.mp (schedule.mem_iff.mp hp)⟩,⟨i.val,hi⟩,()) trivial

/-- The actual-source complete-grid proof supplies execution and values. The
single IO description adds source safety and binds logical inputs; the scalar
premise interprets the explicitly supplied runtime epsilon argument. -/
theorem implements_argument {α β : Type} [Inhabited α] [Inhabited β] (M : Algebra β)
    (encode : α → β) (observe : Nat → β → α) (c : Layout) (eps : ℝ) (epsilon : β)
    (rows : List Nat) (schedule : rows.Perm (List.range c.M))
    (xSeparate : c.X≠c.Y) (wSeparate : c.W≠c.Y)
    (f : (Nat → α) → (Nat → α) → Nat → α)
    (scalar : ∀ xs ws i, observe i (ActualRow.rowValue M c epsilon
      (fun j => encode (xs j)) (fun j => encode (ws j)) i)=f xs ws i) :
    (io M encode observe c eps rows epsilon).Implements (target M encode observe c eps rows f) := by
  apply ProgramIO.Implements.intro
  · intro xs bounds hb s _
    exact schedule_safe M encode observe c eps epsilon rows schedule xs bounds hb s
  · intro xs s input
    obtain ⟨t,ran,values,frame⟩ := ActualGrid.grid_result M c epsilon xSeparate wSeparate rows schedule s
    refine ⟨t,?_,?_,whole_frame M encode observe c eps rows epsilon xs s t frame⟩
    · exact (KernelSafety.schedule_runs M c eps rows s epsilon).trans ran
    · intro o
      fin_cases o
      intro j _
      change mapCell (observe j.2.1.val) (t.mem c.Y (j.1.val*c.N+j.2.1.val)) =
        Cell.mk .real (f (inputRow M encode observe c eps rows xs j.1)
          (weights M encode observe c eps rows xs) j.2.1.val)
      rw [values j.1 j.2.1]
      change Cell.mk .real (observe j.2.1.val (ActualRow.rowValue M c epsilon _ _ j.2.1.val)) = _
      rw [ActualRow.rowValue_congr M c epsilon _ _
        (fun k => encode (inputRow M encode observe c eps rows xs j.1 k))
        (fun k => encode (weights M encode observe c eps rows xs k))
        j.2.1.val j.2.1.isLt]
      · rw [scalar]
      · intro k hk
        have read := input 0 (j.1,⟨k,hk⟩,()) trivial
        change (s.mem c.X (j.1.val*c.N+k)).read .real=encode (xs 0 (j.1,⟨k,hk⟩,())) at read
        simpa only [inputRow,dif_pos hk] using read
      · intro k hk
        have read := input 1 (⟨k,hk⟩,()) trivial
        change (s.mem c.W k).read .real=encode (xs 1 (⟨k,hk⟩,())) at read
        simpa only [weights,dif_pos hk] using read

theorem implements {α β : Type} [Inhabited α] [Inhabited β] (M : Algebra β)
    (encode : α → β) (observe : Nat → β → α) (c : Layout) (eps : ℝ)
    (rows : List Nat) (schedule : rows.Perm (List.range c.M))
    (xSeparate : c.X≠c.Y) (wSeparate : c.W≠c.Y)
    (f : (Nat → α) → (Nat → α) → Nat → α)
    (scalar : ∀ xs ws i, observe i (rowValue M c eps
      (fun j => encode (xs j)) (fun j => encode (ws j)) i)=f xs ws i) :
    (io M encode observe c eps rows).Implements (target M encode observe c eps rows f) := by
  apply implements_argument M encode observe c eps (ActualRow.argument M eps)
    rows schedule xSeparate wSeparate f
  simpa only [ActualRow.rowValue_argument] using scalar

#axiomsClean implements_argument
#axiomsClean implements
end
end VeriTile.Bench.Repairs.RmsNormTriton.GridPublic
