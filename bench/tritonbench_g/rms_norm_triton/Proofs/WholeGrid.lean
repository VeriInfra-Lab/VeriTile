import bench.tritonbench_g.rms_norm_triton.Proofs.WholeTile

namespace VeriTile.Bench.TritonBenchG.RmsNormTriton.Whole
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000

def rowExec [Inhabited α] (mdl : Algebra α) (c : Layout) (eps : ℝ)
    (pid : Nat) (s : State α) : Option (State α) :=
  run (engine mdl) (c.source eps).surfaceBody (rowState pid s)

def runRows [Inhabited α] (mdl : Algebra α) (c : Layout) (eps : ℝ) :
    List Nat → State α → Option (State α)
  | [], s => some s
  | pid::rest, s => (rowExec mdl c eps pid s).bind (runRows mdl c eps rest)

theorem row_column_unique {N row other i j : Nat} (hi : i < N) (hj : j < N)
    (equal : row*N+i = other*N+j) : row = other ∧ i = j := by
  have positive : 0 < N := Nat.zero_lt_of_lt hi
  have quotient (r k : Nat) (hk : k < N) : (r*N+k)/N = r := by
    simpa [Nat.mul_comm, Nat.div_eq_of_lt hk] using Nat.mul_add_div positive r k
  have rows : row = other := by
    simpa only [quotient row i hi,quotient other j hj] using congrArg (fun n => n/N) equal
  subst other
  exact ⟨rfl,Nat.add_left_cancel equal⟩

theorem rows_result [Inhabited α] (mdl : Algebra α) (c : Layout) (eps : ℝ)
    (xSeparate : c.X ≠ c.Y) (wSeparate : c.W ≠ c.Y) (rows : List Nat) (s : State α) :
    ∃ final, runRows mdl c eps rows s = some final ∧
      (∀ row ∈ rows, ∀ i : Fin c.N, final.mem c.Y (row*c.N+i.val) =
        Cell.mk .real (rowValue mdl c eps
          (fun j => (s.mem c.X (row*c.N+j)).read .real)
          (fun j => (s.mem c.W j).read .real) i.val)) ∧
      (∀ region a, (∀ row ∈ rows, Outside c row region a) → final.mem region a = s.mem region a) := by
  induction rows generalizing s with
  | nil =>
    refine ⟨s,rfl,?_,?_⟩
    · intro row h; simp at h
    · intro region a _; rfl
  | cons pid rest ih =>
    let middle := (sourceRun mdl c eps pid s).val
    have ran : rowExec mdl c eps pid s = some middle := (sourceRun mdl c eps pid s).property
    obtain ⟨final,tail,values,frame⟩ := ih middle
    have xf (a : Nat) : middle.mem c.X a = s.mem c.X a :=
      row_frame mdl c eps pid s c.X a (Or.inl xSeparate)
    have wf (a : Nat) : middle.mem c.W a = s.mem c.W a :=
      row_frame mdl c eps pid s c.W a (Or.inl wSeparate)
    refine ⟨final,?_,?_,?_⟩
    · simp only [runRows,ran,Option.bind_some,tail]
    · intro row covered i
      by_cases hr : row ∈ rest
      · rw [values row hr i]
        simp only [xf,wf]
      · have roweq : row = pid := (List.mem_cons.mp covered).resolve_right hr
        subst row
        rw [frame c.Y (pid*c.N+i.val) (fun q hq => Or.inr (fun j hj equal =>
          hr ((row_column_unique i.isLt hj equal).1 ▸ hq)))]
        exact row_readback mdl c eps pid s ⟨i.val,Nat.lt_of_lt_of_le i.isLt c.complete⟩ i.isLt
    · intro region a outside
      rw [frame region a (fun q hq => outside q (List.mem_cons_of_mem pid hq))]
      exact row_frame mdl c eps pid s region a (outside pid List.mem_cons_self)

/-- Original one-program-per-row launch in any complete order; no autotuning
exists in this source. Includes M=0 and empty normalized axes. -/
theorem grid_result [Inhabited α] (mdl : Algebra α) (c : Layout) (eps : ℝ)
    (xSeparate : c.X ≠ c.Y) (wSeparate : c.W ≠ c.Y)
    (rows : List Nat) (schedule : rows.Perm (List.range c.M)) (s : State α) :
    ∃ final, runRows mdl c eps rows s = some final ∧
      (∀ row : Fin c.M, ∀ i : Fin c.N, final.mem c.Y (row.val*c.N+i.val) =
        Cell.mk .real (rowValue mdl c eps
          (fun j => (s.mem c.X (row.val*c.N+j)).read .real)
          (fun j => (s.mem c.W j).read .real) i.val)) ∧
      (∀ region a, (region ≠ c.Y ∨ c.M*c.N ≤ a) → final.mem region a = s.mem region a) := by
  obtain ⟨final,ran,values,frame⟩ := rows_result mdl c eps xSeparate wSeparate rows s
  refine ⟨final,ran,?_,?_⟩
  · intro row i
    exact values row.val (schedule.mem_iff.mpr (List.mem_range.mpr row.isLt)) i
  · intro region a outside
    apply frame
    intro row hr
    rcases outside with h | h
    · exact Or.inl h
    · right
      intro i hi equal
      have rowBound : row < c.M := List.mem_range.mp (schedule.mem_iff.mp hr)
      have small : row*c.N+i.val < c.M*c.N := by
        calc
          row*c.N+i.val < row*c.N+c.N := Nat.add_lt_add_left hi _
          _ = (row+1)*c.N := by ring
          _ ≤ c.M*c.N := Nat.mul_le_mul_right c.N rowBound
      omega

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsNormTriton.Whole
