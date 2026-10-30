import bench.tritonbench_g.batched_vecmat_mult.Proofs.Mathematical_SourceGraphComplete

namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult.SourceGraph
open VeriTile.Triton FP.Structural
open FP.Equational (ReductionPlan)
noncomputable section
local instance : Inhabited Word := ⟨Expr.zero⟩

def gridM (p : Parameters) := p.M / p.BM
def gridN (p : Parameters) := p.N / p.BN
def gridCount (p : Parameters) := gridM p * gridN p

/-- Actual wrapper grid, in an explicit pid0-fastest logical schedule. -/
def programState (p : Parameters) (initial : State Word) (mem : Memory) (k : Nat) : State Word :=
  { initial with
    mem := mem
    regs := fun _ _ _ => none
    pids := fun ax => if ax = 0 then k % gridM p else if ax = 1 then k / gridM p else 0
    numPids := fun ax => if ax = 0 then gridM p else if ax = 1 then gridN p else 1 }

def runGridPrefix (rest : Algebra Word) (plans : PlanSelector)
    (p : Parameters) (initial : State Word) : Nat → Option (State Word)
  | 0 => some initial
  | k+1 => (runGridPrefix rest plans p initial k).bind fun previous =>
      FP.Structural.exec (engine rest plans) (source p) (programState p initial previous.mem k)

def runGrid (rest : Algebra Word) (plans : PlanSelector)
    (p : Parameters) (initial : State Word) := runGridPrefix rest plans p initial (gridCount p)

/-- Independent ordered references read prior reference memory, preserving
any program-to-program aliases instead of merging isolated executions. -/
def referenceGridPrefix (plans : PlanSelector)
    (p : Parameters) (initial : State Word) : Nat → Memory
  | 0 => initial.mem
  | k+1 => expectedMemory plans p
      (programState p initial (referenceGridPrefix plans p initial k) k)

def gridExpectedMemory (plans : PlanSelector)
    (p : Parameters) (initial : State Word) := referenceGridPrefix plans p initial (gridCount p)

theorem grid_prefix_correct (rest : Algebra Word) (plans : PlanSelector)
    (p : Parameters) (initial : State Word) (k : Nat) :
    ∃ final, runGridPrefix rest plans p initial k = some final ∧
      final.mem = referenceGridPrefix plans p initial k := by
  induction k with
  | zero => exact ⟨initial, rfl, rfl⟩
  | succ k ih =>
    obtain ⟨previous, hr, hm⟩ := ih
    obtain ⟨final, he, hmem, _⟩ := complete_correct rest plans p
      (programState p initial previous.mem k)
    refine ⟨final, ?_, ?_⟩
    · simp only [runGridPrefix, hr, Option.bind_some]
      exact he
    · rw [hmem, hm]
      rfl

def GridOutside (p : Parameters) (r : RegionName) (o : Nat) : Prop :=
  r ≠ p.output ∨ ∀ m : Fin (gridM p), ∀ n : Fin (gridN p),
    ∀ i : TileIndex [p.BM,p.BN],
      o ≠ (m.val*p.BM+i.1.val)*p.N + (n.val*p.BN+i.2.1.val)

theorem coordinate_bounds (p : Parameters) (k : Nat) (hk : k < gridCount p) :
    k % gridM p < gridM p ∧ k / gridM p < gridN p := by
  have hm : 0 < gridM p := by
    by_contra h
    have hz : gridM p = 0 := by omega
    simp [gridCount, hz] at hk
  refine ⟨Nat.mod_lt _ hm, ?_⟩
  apply (Nat.div_lt_iff_lt_mul hm).2
  simpa [gridCount, Nat.mul_comm] using hk

theorem coordinates_cover (p : Parameters) (m : Fin (gridM p)) (n : Fin (gridN p)) :
    n.val*gridM p+m.val < gridCount p ∧
      (n.val*gridM p+m.val) % gridM p = m.val ∧
      (n.val*gridM p+m.val) / gridM p = n.val := by
  have hm : 0 < gridM p := Nat.zero_lt_of_lt m.isLt
  have bound := Nat.mul_le_mul_right (gridM p) (Nat.succ_le_of_lt n.isLt)
  rw [Nat.succ_mul] at bound
  have hmi := m.isLt
  refine ⟨?_, ?_, ?_⟩
  · unfold gridCount
    rw [Nat.mul_comm (gridM p) (gridN p)]
    omega
  · simp [Nat.add_mod, Nat.mod_eq_of_lt m.isLt]
  · rw [Nat.add_comm, Nat.add_mul_div_right _ _ hm, Nat.div_eq_of_lt m.isLt]
    omega

theorem outside_program (p : Parameters) (initial : State Word) (mem : Memory)
    (k : Nat) (hk : k < gridCount p) (r : RegionName) (o : Nat) (h : GridOutside p r o) :
    Outside p (programState p initial mem k) r o := by
  rcases h with h | h
  · exact Or.inl h
  · right
    intro i
    obtain ⟨hm,hn⟩ := coordinate_bounds p k hk
    simpa [outputOffset, row, col, programState] using
      h ⟨k % gridM p,hm⟩ ⟨k / gridM p,hn⟩ i

theorem grid_reference_frame (plans : PlanSelector)
    (p : Parameters) (initial : State Word) (k : Nat) (hk : k ≤ gridCount p)
    (r : RegionName) (o : Nat) (ho : GridOutside p r o) :
    referenceGridPrefix plans p initial k r o = initial.mem r o := by
  induction k with
  | zero => rfl
  | succ k ih =>
    rw [referenceGridPrefix, expectedMemory_frame plans p
      (programState p initial (referenceGridPrefix plans p initial k) k) r o
      (outside_program p initial _ k (by omega) r o ho)]
    exact ih (by omega)

theorem grid_correct (rest : Algebra Word) (plans : PlanSelector)
    (p : Parameters) (initial : State Word) :
    ∃ final, runGrid rest plans p initial = some final ∧
      final.mem = gridExpectedMemory plans p initial ∧
      (∀ r o, GridOutside p r o → final.mem r o = initial.mem r o) := by
  obtain ⟨final, he, hm⟩ := grid_prefix_correct rest plans p initial (gridCount p)
  refine ⟨final, he, hm, ?_⟩
  intro r o ho
  rw [hm]
  exact grid_reference_frame plans p initial (gridCount p) (by rfl) r o ho

#specNonCircular gridExpectedMemory avoiding [runGrid, runGridPrefix, FP.Structural.exec, source]
#axiomsClean grid_prefix_correct
#axiomsClean coordinate_bounds
#axiomsClean coordinates_cover
#axiomsClean outside_program
#axiomsClean grid_reference_frame
#axiomsClean grid_correct
end
end VeriTile.Bench.TritonBenchG.BatchedVecmatMult.SourceGraph
