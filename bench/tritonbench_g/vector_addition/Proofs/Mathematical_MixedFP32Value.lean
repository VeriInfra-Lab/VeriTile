import bench.tritonbench_g.vector_addition.Proofs.Mathematical_MixedFP32Source

namespace VeriTile.Bench.TritonBenchG.VectorAddition.MixedFP32
open VeriTile.Triton FP.Structural FP.IntegerConversion
open ValueProofs (Parameters offset)
open IntegerSource (Storage)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def expectedMemory (cfg : FP.Config) (us : Nat → Word → Int) (uu : Nat → Word → Nat)
    (integerFirst : Bool) (d : Storage) (p : Parameters) (s : State Word) :
    RegionName → Nat → Cell Word :=
  (TileShape.allIndices [p.block]).foldl (fun mem i =>
    if offset p s i < p.n then fun r o =>
      if r = p.output ∧ o = offset p s i then
        resultCell cfg us uu integerFirst d p s.mem (offset p s i) else mem r o
    else mem) s.mem

/-- Actual totalized source execution, including arbitrary unspecified masked
load values and undefined conversion policies. Success is constructed. -/
def execution (rest : Algebra Word) (cfg : FP.Config)
    (us : Nat → Word → Int) (uu : Nat → Word → Nat)
    (integerFirst : Bool) (d : Storage) (p : Parameters) (s : State Word) :
    {t : State Word // FP.Structural.exec (backend rest cfg us uu)
      (source integerFirst d p) s = some t} := by
  cases integerFirst <;> cases d
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp [source, FP.Structural.exec, ComputeKernel.surfaceBody,
      run, step, evalExpr, evalComputeOp, eval_integerToFloat,
      IntegerSource.eval_loadOp, storeInteger,
      FP.Structural.evalOp, Option.bind, numeric, numericLt, natLt, bop,
      store, Region.cast, backend_add, backend_signed, backend_unsigned, ofFloat, toFloat]
    rfl

theorem execution_memory (rest : Algebra Word) (cfg : FP.Config)
    (us : Nat → Word → Int) (uu : Nat → Word → Nat)
    (integerFirst : Bool) (d : Storage) (p : Parameters) (s : State Word) :
    (execution rest cfg us uu integerFirst d p s).val.mem =
      expectedMemory cfg us uu integerFirst d p s := by
  cases integerFirst <;> cases d
  all_goals simp [execution, expectedMemory, offset, resultCell, sumValue,
    convertedCell, IntegerSource.loaded, IntegerSource.loadValue,
    bop, Broadcast.leftIndex, Broadcast.rightIndex,
    ValueProofs.masked_scatter_memory, State.setReg]
  all_goals congr 1
  all_goals funext mem i
  all_goals by_cases hi : s.pids 0 * p.block + i.1.val < p.n
  all_goals simp only [hi, ↓reduceIte]
  all_goals rfl

theorem expectedMemory_eq_scatter (cfg : FP.Config) (us : Nat → Word → Int) (uu : Nat → Word → Nat)
    (integerFirst : Bool) (d : Storage) (p : Parameters) (s : State Word) :
    expectedMemory cfg us uu integerFirst d p s =
      ((TileShape.allIndices [p.block]).foldl (fun t i =>
        if offset p s i < p.n then t.write p.output (offset p s i)
          (resultCell cfg us uu integerFirst d p s.mem (offset p s i)) else t) s).mem := by
  rw [ValueProofs.masked_scatter_memory]
  rfl

theorem expected_readback (cfg : FP.Config) (us : Nat → Word → Int) (uu : Nat → Word → Nat)
    (integerFirst : Bool) (d : Storage) (p : Parameters) (s : State Word) (i : TileIndex [p.block]) (hi : offset p s i < p.n) :
    expectedMemory cfg us uu integerFirst d p s p.output (offset p s i) = resultCell cfg us uu integerFirst d p s.mem (offset p s i) := by
  rw [expectedMemory_eq_scatter]
  exact State.masked_scatter_readback p.output (offset p s)
    (fun i => resultCell cfg us uu integerFirst d p s.mem (offset p s i))
    (fun i => offset p s i < p.n) s i hi (ValueProofs.offset_injective p s)

theorem expected_frame (cfg : FP.Config) (us : Nat → Word → Int) (uu : Nat → Word → Nat)
    (integerFirst : Bool) (d : Storage) (p : Parameters) (s : State Word) (r : RegionName) (o : Nat) (h : ValueProofs.Outside p s r o) :
    expectedMemory cfg us uu integerFirst d p s r o = s.mem r o := by
  rw [expectedMemory_eq_scatter]
  apply State.masked_scatter_frame
  exact h.imp id (fun h i _ hi => h i hi)

/-- Closed address formula for one whole source program. -/
theorem expected_cell (cfg : FP.Config) (us : Nat → Word → Int) (uu : Nat → Word → Nat)
    (integerFirst : Bool) (d : Storage) (p : Parameters) (s : State Word) (r : RegionName) (o : Nat) :
    expectedMemory cfg us uu integerFirst d p s r o =
      if r = p.output ∧ s.pids 0 * p.block ≤ o ∧
          o < s.pids 0 * p.block + p.block ∧ o < p.n then
        resultCell cfg us uu integerFirst d p s.mem o else s.mem r o := by
  by_cases h : r = p.output ∧ s.pids 0 * p.block ≤ o ∧
      o < s.pids 0 * p.block + p.block ∧ o < p.n
  · rw [if_pos h]
    let i : TileIndex [p.block] := (⟨o - s.pids 0 * p.block, by omega⟩, PUnit.unit)
    have ho : offset p s i = o := by simp only [offset, i]; omega
    rw [h.1, ← ho]
    exact expected_readback cfg us uu integerFirst d p s i (by rw [ho]; exact h.2.2.2)
  · rw [if_neg h]
    apply expected_frame
    by_cases hr : r = p.output
    · right
      intro i hi ho
      apply h
      have bound := i.1.isLt
      unfold offset at hi ho
      exact ⟨hr, by omega, by omega, by omega⟩
    · exact Or.inl hr

theorem complete_correct (rest : Algebra Word) (cfg : FP.Config) (us : Nat → Word → Int) (uu : Nat → Word → Nat)
    (integerFirst : Bool) (d : Storage) (p : Parameters) (s : State Word) :
    ∃ t, FP.Structural.exec (backend rest cfg us uu) (source integerFirst d p) s = some t ∧ t.mem = expectedMemory cfg us uu integerFirst d p s :=
  ⟨(execution rest cfg us uu integerFirst d p s).val, (execution rest cfg us uu integerFirst d p s).property, execution_memory rest cfg us uu integerFirst d p s⟩

#specNonCircular expectedMemory avoiding [source, FP.Structural.exec]
#axiomsClean execution
#axiomsClean execution_memory
#axiomsClean expected_cell
#axiomsClean complete_correct
end
end VeriTile.Bench.TritonBenchG.VectorAddition.MixedFP32
