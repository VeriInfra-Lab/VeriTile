import bench.repairs.tritonbench_g.softmax_optimize.Proofs.SourceIO

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.GridReal
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.SoftmaxOptimize
open scoped BigOperators
noncomputable section

def atRow (state : BlockState) (M row : Nat) : BlockState :=
  { state with pids := fun dimension => if dimension = 0 then row else 0
               numPids := fun dimension => if dimension = 0 then M else 1 }

def rowSpec (initial : BlockState) (X : RegionName) (N row : Nat) (column : Fin N) : ℝ :=
  Real.exp (initial.readMem X (row * N + column.val)) /
    ∑ j : Fin N, Real.exp (initial.readMem X (row * N + j.val))

/-- An explicit serial schedule of actual per-program executions. This
constructs the mathematical grid execution; native launch/PID compatibility is
kept separate. Each block receives its own source program id and grid metadata. -/
def runRows (Y X : RegionName) (M N : Nat) (start : Nat) : Nat → BlockState → Option BlockState
  | 0, state => some state
  | count + 1, state =>
    (exec (kernel Y X M N (Config.tile N)).eraseDType.toAlgKernel (atRow state M start)).bind
      (runRows Y X M N (start + 1) count)

def Frame (Y : RegionName) (N start count : Nat) (before after : BlockState) : Prop :=
  ∀ region offset, (region ≠ Y ∨ offset < start * N ∨ (start + count) * N ≤ offset) →
    after.mem region offset = before.mem region offset

theorem input_readback {Y X : RegionName} {N start count : Nat} {before after : BlockState}
    (different : Y ≠ X) (frame : Frame Y N start count before after)
    (undef : after.undef = before.undef) (offset : Nat) :
    after.readMem X offset = before.readMem X offset := by
  unfold BlockState.readMem
  rw [frame X offset (Or.inl different.symm)]

theorem rowSpec_preserved {Y X : RegionName} {N start count : Nat} {before after : BlockState}
    (different : Y ≠ X) (frame : Frame Y N start count before after)
    (undef : after.undef = before.undef) (row : Nat) (column : Fin N) :
    rowSpec after X N row column = rowSpec before X N row column := by
  unfold rowSpec
  simp_rw [input_readback different frame undef]

theorem runRows_io (Y X : RegionName) (M N : Nat) (different : Y ≠ X) (hN : 0 < N)
    (start count : Nat) (initial : BlockState) :
    ∃ finalState, runRows Y X M N start count initial = some finalState ∧
      (∀ i : Fin count, ∀ j : Fin N, finalState.readMem Y ((start + i.val) * N + j.val) =
        rowSpec initial X N (start + i.val) j) ∧
      Frame Y N start count initial finalState ∧ finalState.undef = initial.undef := by
  induction count generalizing start initial with
  | zero =>
    refine ⟨initial, rfl, ?_, ?_, rfl⟩
    · intro i; exact Fin.elim0 i
    · intro region offset outside; rfl
  | succ count ih =>
    obtain ⟨first, hfirst, firstValues, firstFrame, _, _, firstUndef⟩ :=
      SourceIO.selected_tile_io Y X M N different hN (atRow initial M start)
    have firstFrame' : Frame Y N start 1 initial first := by
      intro region offset outside
      apply firstFrame region offset
      simpa only [atRow, BlockState.pid, ↓reduceIte, Nat.add_mul, Nat.one_mul] using outside
    have firstUndef' : first.undef = initial.undef := firstUndef
    have firstValues' : ∀ j : Fin N, first.readMem Y (start * N + j.val) =
        rowSpec initial X N start j := by
      intro j
      exact firstValues j
    obtain ⟨last, hlast, lastValues, lastFrame, lastUndef⟩ := ih (start + 1) first
    refine ⟨last, ?_, ?_, ?_, lastUndef.trans firstUndef'⟩
    · simp only [runRows, hfirst, Option.bind_some, hlast]
    · intro i j
      by_cases hi : i.val = 0
      · have addressLow : (start + i.val) * N + j.val < (start + 1) * N := by
          rw [hi, Nat.add_zero, Nat.add_mul, Nat.one_mul]
          omega
        have sameCell := lastFrame Y ((start + i.val) * N + j.val) (Or.inr (Or.inl addressLow))
        have sameRead : last.readMem Y ((start + i.val) * N + j.val) =
            first.readMem Y ((start + i.val) * N + j.val) := by
          unfold BlockState.readMem
          rw [sameCell]
        rw [sameRead, hi, Nat.add_zero]
        exact firstValues' j
      · let previous : Fin count := ⟨i.val - 1, by omega⟩
        have rowEq : start + 1 + previous.val = start + i.val := by dsimp [previous]; omega
        have value := lastValues previous j
        rw [rowEq] at value
        exact value.trans (rowSpec_preserved different firstFrame' firstUndef' _ _)
    · intro region offset outside
      have tailOutside : region ≠ Y ∨ offset < (start + 1) * N ∨
          (start + 1 + count) * N ≤ offset := by
        rcases outside with outside | outside | outside
        · exact Or.inl outside
        · right; left
          have : start * N ≤ (start + 1) * N := Nat.mul_le_mul_right N (by omega)
          omega
        · right; right
          convert outside using 1 <;> congr 1 <;> omega
      have headOutside : region ≠ Y ∨ offset < start * N ∨ (start + 1) * N ≤ offset := by
        rcases outside with outside | outside | outside
        · exact Or.inl outside
        · exact Or.inr (Or.inl outside)
        · right; right
          have : (start + 1) * N ≤ (start + (count + 1)) * N := Nat.mul_le_mul_right N (by omega)
          omega
      exact (lastFrame region offset tailOutside).trans (firstFrame' region offset headOutside)

def launch (Y X : RegionName) (M N : Nat) (initial : BlockState) : Option BlockState :=
  if M = 0 ∨ N = 0 then some initial else runRows Y X M N 0 M initial

/-- Every logical output in the complete grid is the independent original-row
softmax. Empty dimensions execute no block. The frame covers all raw cells,
including cells beyond the output's logical extent. -/
theorem launch_io (Y X : RegionName) (M N : Nat) (different : Y ≠ X) (initial : BlockState) :
    ∃ finalState, launch Y X M N initial = some finalState ∧
      (∀ row : Fin M, ∀ column : Fin N,
        finalState.readMem Y (row.val * N + column.val) = rowSpec initial X N row.val column) ∧
      (∀ region offset, (region ≠ Y ∨ M * N ≤ offset) → finalState.mem region offset = initial.mem region offset) ∧
      finalState.undef = initial.undef := by
  by_cases empty : M = 0 ∨ N = 0
  · refine ⟨initial, by simp only [launch, if_pos empty], ?_, ?_, rfl⟩
    · intro row column
      rcases empty with h | h
      · have := row.isLt; omega
      · have := column.isLt; omega
    · intro region offset outside; rfl
  · have hN : 0 < N := by omega
    obtain ⟨finalState, run, values, frame, undef⟩ := runRows_io Y X M N different hN 0 M initial
    refine ⟨finalState, by simp only [launch, if_neg empty, run], ?_, ?_, undef⟩
    · simpa only [Nat.zero_add] using values
    · intro region offset outside
      apply frame region offset
      simpa only [Nat.zero_add] using outside.imp_right Or.inr

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxOptimize.GridReal
