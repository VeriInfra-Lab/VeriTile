import bench.tritonbench_g.mean_reduction.Proofs.Mathematical_SourceDefinitions
import bench.tritonbench_g.mean_reduction.Proofs.Mathematical_Real

namespace VeriTile.Bench.TritonBenchG.MeanReduction.Whole
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000

/-- Independent mathematical mean of a finite logical row. -/
def rowMean (N : Nat) (x : Nat → ℝ) (row : Nat) : ℝ :=
  (∑ j : Fin N, x (row*N+j.val)) / (N : ℝ)

def gridSize (M : Nat) : Nat := (M+7)/8

def tileState (pid : Nat) (s : BlockState) : BlockState :=
  { s with pids := fun ax => if ax = 0 then pid else s.pids ax }

def runTiles (X Mean : RegionName) (M N : Nat) : List Nat → BlockState → Option BlockState
  | [],s => some s
  | pid::rest,s => (exec (source X Mean M N).eraseDType.toAlgKernel (tileState pid s)).bind
      (runTiles X Mean M N rest)

def Covered (tiles : List Nat) (i : Nat) : Prop :=
  ∃ pid, pid ∈ tiles ∧ ∃ lane : Fin 8, i = pid*8+lane.val

theorem tiles_result (X Mean : RegionName) (M N : Nat) (separate : X ≠ Mean)
    (tiles : List Nat) (s : BlockState) :
    ∃ final, runTiles X Mean M N tiles s = some final ∧
      (∀ i, i < M → Covered tiles i → final.readMem Mean i = rowMean N (s.readMem X) i) ∧
      (∀ r o, (r ≠ Mean ∨ ∀ pid ∈ tiles, ∀ lane : Fin 8,
        pid*8+lane.val < M → o ≠ pid*8+lane.val) → final.mem r o = s.mem r o) := by
  induction tiles generalizing s with
  | nil =>
    refine ⟨s,rfl,?_,?_⟩
    · intro i hi covered
      rcases covered with ⟨pid,hpid,_⟩
      simp at hpid
    · intro r o _; rfl
  | cons pid rest ih =>
    obtain ⟨middle,ran,values,frame⟩ := default_source_tile_run X Mean M N (tileState pid s)
    obtain ⟨final,tail,valuesTail,frameTail⟩ := ih middle
    have inputFrame (i : Nat) : middle.readMem X i = s.readMem X i := by
      simp only [BlockState.readMem]
      rw [frame X i (Or.inl separate)]
      rfl
    refine ⟨final,?_,?_,?_⟩
    · change (exec (source X Mean M N).eraseDType.toAlgKernel (tileState pid s)).bind _ = _
      rw [show exec (source X Mean M N).eraseDType.toAlgKernel (tileState pid s) = some middle from ran]
      exact tail
    · intro i hi covered
      by_cases hc : Covered rest i
      · rw [valuesTail i hi hc]
        unfold rowMean
        simp only [inputFrame]
      · have preserve : final.mem Mean i = middle.mem Mean i := by
          apply frameTail Mean i
          right
          intro q hq lane active equal
          exact hc ⟨q,hq,lane,equal⟩
        have readPreserve : final.readMem Mean i = middle.readMem Mean i := by
          simp only [BlockState.readMem,preserve]
        rw [readPreserve]
        rcases covered with ⟨q,hq,lane,equal⟩
        rcases List.mem_cons.mp hq with head | tail
        · subst q
          subst i
          simpa only [meanOutOffset,meanSpec,meanInpElem,meanRowActive,tileState,if_pos rfl]
            using values lane (by simpa [meanRowActive,meanOutOffset,tileState] using hi)
        · exact False.elim (hc ⟨q,tail,lane,equal⟩)
    · intro r o outside
      rw [frameTail r o (by
        rcases outside with hr | ho
        · exact Or.inl hr
        · exact Or.inr fun q hq lane active => ho q (List.mem_cons_of_mem pid hq) lane active)]
      apply frame r o
      rcases outside with hr | ho
      · exact Or.inl hr
      · right
        intro lane active
        exact ho pid List.mem_cons_self lane active

theorem grid_covers (M i : Nat) (hi : i < M) : Covered (List.range (gridSize M)) i := by
  refine ⟨i/8,List.mem_range.mpr ?_,⟨i%8,by omega⟩,?_⟩
  · unfold gridSize; omega
  · change i = i/8*8+i%8
    omega

/-- Every original program executes, including the empty grid. The supplied
schedule may be any permutation of the complete source grid. -/
theorem grid_result (X Mean : RegionName) (M N : Nat) (separate : X ≠ Mean)
    (tiles : List Nat) (schedule : tiles.Perm (List.range (gridSize M))) (s : BlockState) :
    ∃ final, runTiles X Mean M N tiles s = some final ∧
      (∀ i, i < M → final.readMem Mean i = rowMean N (s.readMem X) i) ∧
      (∀ r o, (r ≠ Mean ∨ M ≤ o) → final.mem r o = s.mem r o) := by
  obtain ⟨final,ran,values,frame⟩ := tiles_result X Mean M N separate tiles s
  refine ⟨final,ran,?_,?_⟩
  · intro i hi
    apply values i hi
    obtain ⟨pid,hpid,lane,equal⟩ := grid_covers M i hi
    exact ⟨pid,schedule.mem_iff.mpr hpid,lane,equal⟩
  · intro r o outside
    apply frame r o
    rcases outside with hr | ho
    · exact Or.inl hr
    · right; intro pid hpid lane active; omega

/-- The row ownership map is injective across the complete grid, so disjoint
blocks cannot overwrite each other's output, even on the masked tail. -/
theorem output_ownership (pid q : Nat) (i j : Fin 8)
    (h : pid*8+i.val = q*8+j.val) : pid = q ∧ i = j := by
  constructor
  · omega
  · apply Fin.ext; omega

/-- Active read addresses fit the entire compressed input allocation. -/
theorem active_read_bound (M N pid t : Nat) (l : Fin 64)
    (active : meanReadActive M N 8 8 pid t l) :
    meanReadAddr N 8 8 pid t l < M*N := by
  rcases active with ⟨row,col⟩
  unfold meanReadAddr
  nlinarith

/-- Boundedness applies to the actual complete source trace, not merely to a
hand-written list of output addresses. -/
theorem source_trace_safe (X Mean : RegionName) (M N : Nat) (s : BlockState)
    (bounds : RegionBounds) (hi : M*N ≤ bounds X) (ho : M ≤ bounds Mean) :
    (source X Mean M N).eraseDType.toAlgKernel.TraceSafeR RoundingModel.triv bounds s := by
  apply default_source_tile_safe bounds X Mean M N s
  · intro t l active
    exact Nat.lt_of_lt_of_le (active_read_bound M N (s.pids 0) t.val l active) hi
  · intro i active
    exact Nat.lt_of_lt_of_le active ho

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MeanReduction.Whole
