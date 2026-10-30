import bench.repairs.tritonbench_g.fused_activation.Proofs.Source
import bench.repairs.tritonbench_g.triton_mul2.Proofs.Scheduling

/-! Complete ordered grid for typed floating source materializations. Prior
stores cannot change later tile input addresses. The disjoint cyclic bias has a
distinct region; current work/input reads use only the still-unwritten tile.
This does not assert native GPU scheduling, physical layout or allocation. -/
namespace VeriTile.Bench.Repairs.FusedActivation
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs (offset gridCount programState grid_covers)
noncomputable section

def runGridPrefix {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (weights : Nat) (multiplier : ℝ) (activation : Bool) (initial : State α) : Nat → Option (State α)
  | 0 => some initial
  | k+1 => (runGridPrefix M p weights multiplier activation initial k).bind fun previous =>
      FP.Structural.exec M (candidateKernel p weights multiplier activation) (programState p initial previous.mem k)

def runGrid {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (weights : Nat) (multiplier : ℝ) (activation : Bool) (initial : State α) :=
  runGridPrefix M p weights multiplier activation initial (gridCount p)

/-- Independent whole-vector prefix formula, not a fold over executions. -/
def prefixMemory {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (weights : Nat) (multiplier : ℝ) (activation : Bool) (initial : State α) (k : Nat) : RegionName → Nat → Cell α :=
  fun r o => if r = p.output ∧ o < k*p.block ∧ o < p.n then
    resultCell M p weights multiplier activation initial.mem o else initial.mem r o

def gridExpectedMemory {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (weights : Nat) (multiplier : ℝ) (activation : Bool) (initial : State α) : RegionName → Nat → Cell α :=
  fun r o => if r = p.output ∧ o < p.n then resultCell M p weights multiplier activation initial.mem o else initial.mem r o

/-- Earlier output stores do not touch any input address in the next tile,
including when work and additional input share a logical region. The cyclic
bias region is separate; the input contract rejects overlap with the grad write window. -/
theorem next_tile_inputs {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (weights : Nat) (multiplier : ℝ) (activation : Bool) (initial : State α) (k o : Nat) (ho : k*p.block  ≤  o) (separate : p.y ≠ p.output) :
    resultCell M p weights multiplier activation (prefixMemory M p weights multiplier activation initial k) o = resultCell M p weights multiplier activation initial.mem o := by
  simp [resultCell, prefixMemory, Nat.not_lt.mpr ho,separate]

theorem reference_step {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (weights : Nat) (multiplier : ℝ) (activation : Bool) (initial : State α) (k : Nat) (separate : p.y ≠ p.output) :
    expectedMemory M p weights multiplier activation (programState p initial (prefixMemory M p weights multiplier activation initial k) k) =
      prefixMemory M p weights multiplier activation initial (k+1) := by
  funext r o
  rw [expected_cell]
  simp only [programState, ↓reduceIte]
  by_cases before : o < k*p.block
  · have later : o < (k+1)*p.block := by rw [Nat.add_mul]; simp only [Nat.one_mul]; omega
    simp [prefixMemory, before, later, Nat.not_le.mpr before]
  · have lower : k*p.block  ≤  o := Nat.le_of_not_gt before
    rw [next_tile_inputs M p weights multiplier activation initial k o lower separate]
    simp [prefixMemory, before, lower, Nat.add_mul]

theorem grid_prefix_correct {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (weights : Nat) (multiplier : ℝ) (activation : Bool) (initial : State α) (k : Nat) (separate : p.y ≠ p.output) :
    ∃ final, runGridPrefix M p weights multiplier activation initial k = some final ∧ final.mem = prefixMemory M p weights multiplier activation initial k := by
  induction k with
  | zero => exact ⟨initial, rfl, by funext r o; simp [prefixMemory]⟩
  | succ k ih =>
    obtain ⟨previous, hr, hm⟩ := ih
    obtain ⟨final, he, hf⟩ := complete_correct M p weights multiplier activation (programState p initial previous.mem k)
    refine ⟨final, ?_, ?_⟩
    · simp only [runGridPrefix, hr, Option.bind_some]
      exact he
    · rw [hf, hm, reference_step M p weights multiplier activation initial k separate]

theorem grid_reference_complete {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (weights : Nat) (multiplier : ℝ) (activation : Bool) (initial : State α) (positive : 0 < p.block) :
    prefixMemory M p weights multiplier activation initial (gridCount p) = gridExpectedMemory M p weights multiplier activation initial := by
  have covers := grid_covers p positive
  funext r o
  by_cases ho : o < p.n
  · have hgrid : o < gridCount p * p.block := lt_of_lt_of_le ho covers
    simp [prefixMemory, gridExpectedMemory, ho, hgrid]
  · simp [prefixMemory, gridExpectedMemory, ho]

theorem grid_correct {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (weights : Nat) (multiplier : ℝ) (activation : Bool) (initial : State α) (positive : 0 < p.block) (separate : p.y ≠ p.output) :
    ∃ final, runGrid M p weights multiplier activation initial = some final ∧ final.mem = gridExpectedMemory M p weights multiplier activation initial := by
  obtain ⟨final, he, hm⟩ := grid_prefix_correct M p weights multiplier activation initial (gridCount p) separate
  exact ⟨final, he, hm.trans (grid_reference_complete M p weights multiplier activation initial positive)⟩

/-- One complete tile's source-derived memory update, with all loads taken
before its stores. This also names its complete active read/write dependency. -/
def tileMemory {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (weights : Nat) (multiplier : ℝ) (activation : Bool) (pid : Nat)
    (memory : RegionName → Nat → Cell α) : RegionName → Nat → Cell α :=
  fun r i => if r=p.output ∧ pid*p.block ≤ i ∧ i<(pid+1)*p.block ∧ i<p.n then
    resultCell M p weights multiplier activation memory i else memory r i

theorem tile_source {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (weights : Nat) (multiplier : ℝ) (activation : Bool) (pid : Nat) (initial : State α) :
    expectedMemory M p weights multiplier activation (programState p initial initial.mem pid)=
      tileMemory M p weights multiplier activation pid initial.mem := by
  funext r i
  rw [expected_cell]
  simp only [programState,tileMemory,Nat.add_mul,Nat.one_mul,↓reduceIte]

theorem tile_other_inputs {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (weights : Nat) (multiplier : ℝ) (activation : Bool) (pid q i : Nat)
    (memory : RegionName → Nat → Cell α) (separateBias : p.y≠p.output) (separateInput : p.x≠p.output)
    (different : pid≠q) (low : q*p.block ≤ i) (high : i<(q+1)*p.block) :
    resultCell M p weights multiplier activation (tileMemory M p weights multiplier activation pid memory) i=
      resultCell M p weights multiplier activation memory i := by
  have outside : ¬(pid*p.block ≤ i ∧ i<(pid+1)*p.block) := by
    rintro ⟨lo,hi⟩
    exact different (TritonMul2.Scheduling.tile_unique p.block pid q i lo hi low high)
  simp only [resultCell,tileMemory,separateBias,separateInput,false_and,↓reduceIte,true_and]
  have skip : ¬(pid*p.block ≤ i ∧ i<(pid+1)*p.block ∧ i<p.n) := fun h => outside ⟨h.1,h.2.1⟩
  simp only [skip,↓reduceIte]

theorem tiles_commute {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (weights : Nat) (multiplier : ℝ) (activation : Bool) (pid q : Nat)
    (memory : RegionName → Nat → Cell α) (separateBias : p.y≠p.output) (separateInput : p.x≠p.output) :
    tileMemory M p weights multiplier activation pid (tileMemory M p weights multiplier activation q memory)=
    tileMemory M p weights multiplier activation q (tileMemory M p weights multiplier activation pid memory) := by
  by_cases same : pid=q
  · subst q; rfl
  funext r i
  by_cases hp : r=p.output ∧ pid*p.block ≤ i ∧ i<(pid+1)*p.block ∧ i<p.n
  · have hq : ¬(r=p.output ∧ q*p.block ≤ i ∧ i<(q+1)*p.block ∧ i<p.n) := by
      intro h
      exact same (TritonMul2.Scheduling.tile_unique p.block pid q i hp.2.1 hp.2.2.1 h.2.1 h.2.2.1)
    simp only [tileMemory]
    rw [if_pos hp,if_neg hq,if_pos hp]
    exact tile_other_inputs M p weights multiplier activation q pid i memory separateBias separateInput (Ne.symm same) hp.2.1 hp.2.2.1
  · by_cases hq : r=p.output ∧ q*p.block ≤ i ∧ i<(q+1)*p.block ∧ i<p.n
    · simp only [tileMemory]
      rw [if_neg hp,if_pos hq,if_pos hq]
      exact (tile_other_inputs M p weights multiplier activation pid q i memory separateBias separateInput same hq.2.1 hq.2.2.1).symm
    · simp only [tileMemory]
      rw [if_neg hp,if_neg hq,if_neg hp,if_neg hq]

/-- Every permutation of complete program phases has the same source memory.
The preceding read/write separation also excludes cross-program data races. -/
theorem schedule_independent {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (weights : Nat) (multiplier : ℝ) (activation : Bool)
    (memory : RegionName → Nat → Cell α) (separateBias : p.y≠p.output) (separateInput : p.x≠p.output)
    (left right : List Nat) (permutation : left.Perm right) :
    left.foldl (fun mem pid => tileMemory M p weights multiplier activation pid mem) memory=
    right.foldl (fun mem pid => tileMemory M p weights multiplier activation pid mem) memory := by
  letI : RightCommutative (fun mem pid => tileMemory M p weights multiplier activation pid mem) :=
    ⟨fun mem pid q => tiles_commute M p weights multiplier activation q pid mem separateBias separateInput⟩
  exact permutation.foldl_eq memory

#axiomsClean tile_source
#axiomsClean tile_other_inputs
#axiomsClean tiles_commute
#axiomsClean schedule_independent

#specNonCircular gridExpectedMemory avoiding [candidateKernel, runGrid, runGridPrefix, FP.Structural.exec]
#axiomsClean next_tile_inputs
#axiomsClean reference_step
#axiomsClean grid_prefix_correct
#axiomsClean grid_reference_complete
#axiomsClean grid_correct
end
end VeriTile.Bench.Repairs.FusedActivation
