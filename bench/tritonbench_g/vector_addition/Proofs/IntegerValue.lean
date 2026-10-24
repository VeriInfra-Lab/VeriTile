import bench.tritonbench_g.vector_addition.Proofs.IntegerSource

namespace VeriTile.Bench.TritonBenchG.VectorAddition.IntegerSource
open VeriTile.Triton FP.Structural
open ValueProofs (Parameters offset)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Independent initial-cell bitvector formula, not an interpreter fold. -/
def expectedMemory {α : Type} [Inhabited α] (x y : Storage)
    (p : Parameters) (s : State α) : RegionName → Nat → Cell α :=
  (TileShape.allIndices [p.block]).foldl (fun mem i =>
    if offset p s i < p.n then fun r o =>
      if r = p.output ∧ o = offset p s i then resultCell x y p s.mem (offset p s i) else mem r o
    else mem) s.mem

/-- Every source statement actually executes. There is no floating callback
premise and no assumed successful execution; fixed widths are explicit. -/
def execution {α : Type} [Inhabited α] (M : Algebra α) (x y : Storage)
    (p : Parameters) (s : State α) :
    {t : State α // FP.Structural.exec M (source x y p) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [source, FP.Structural.exec, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, eval_castOp, eval_loadOp, eval_storeStmt,
    FP.Structural.evalOp, Option.bind, numeric, numericLt, numericEq, natLt, bop,
    Region.cast, IntegerExpr.same_left, IntegerExpr.same_right]
  rfl

/-- Readback is the bitvector formula, including i1 wrapping before bool i8
storage. Both operand loads precede all output stores. -/
theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α) (x y : Storage)
    (p : Parameters) (s : State α) :
    (execution M x y p s).val.mem = expectedMemory x y p s := by
  cases x <;> cases y
  all_goals simp [execution, expectedMemory, offset, resultCell, stored, loaded, loadValue,
    bop, Broadcast.leftIndex, Broadcast.rightIndex,
    ValueProofs.masked_scatter_memory, State.setReg, unsigned_cast_eq_bitvec, -BitVec.toNat_ofInt, -BitVec.toInt_ofInt]
  all_goals congr 1
  all_goals funext mem i
  all_goals by_cases hi : s.pids 0 * p.block + i.1.val < p.n
  all_goals simp only [hi, ↓reduceIte]
  all_goals rfl

theorem expectedMemory_eq_scatter {α : Type} [Inhabited α] (x y : Storage) (p : Parameters) (s : State α) :
    expectedMemory x y p s =
      ((TileShape.allIndices [p.block]).foldl (fun t i =>
        if offset p s i < p.n then t.write p.output (offset p s i)
          (resultCell x y p s.mem (offset p s i)) else t) s).mem := by
  rw [ValueProofs.masked_scatter_memory]
  rfl

theorem expected_readback {α : Type} [Inhabited α] (x y : Storage) (p : Parameters) (s : State α) (i : TileIndex [p.block]) (hi : offset p s i < p.n) :
    expectedMemory x y p s p.output (offset p s i) = resultCell x y p s.mem (offset p s i) := by
  rw [expectedMemory_eq_scatter]
  exact State.masked_scatter_readback p.output (offset p s)
    (fun i => resultCell x y p s.mem (offset p s i))
    (fun i => offset p s i < p.n) s i hi (ValueProofs.offset_injective p s)

theorem expected_frame {α : Type} [Inhabited α] (x y : Storage) (p : Parameters) (s : State α) (r : RegionName) (o : Nat) (h : ValueProofs.Outside p s r o) :
    expectedMemory x y p s r o = s.mem r o := by
  rw [expectedMemory_eq_scatter]
  apply State.masked_scatter_frame
  exact h.imp id (fun h i _ hi => h i hi)

/-- Closed address formula for one whole source program. -/
theorem expected_cell {α : Type} [Inhabited α] (x y : Storage) (p : Parameters) (s : State α) (r : RegionName) (o : Nat) :
    expectedMemory x y p s r o =
      if r = p.output ∧ s.pids 0 * p.block ≤ o ∧
          o < s.pids 0 * p.block + p.block ∧ o < p.n then
        resultCell x y p s.mem o else s.mem r o := by
  by_cases h : r = p.output ∧ s.pids 0 * p.block ≤ o ∧
      o < s.pids 0 * p.block + p.block ∧ o < p.n
  · rw [if_pos h]
    let i : TileIndex [p.block] := (⟨o - s.pids 0 * p.block, by omega⟩, PUnit.unit)
    have ho : offset p s i = o := by simp only [offset, i]; omega
    rw [h.1, ← ho]
    exact expected_readback x y p s i (by rw [ho]; exact h.2.2.2)
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

theorem complete_correct {α : Type} [Inhabited α] (M : Algebra α) (x y : Storage) (p : Parameters) (s : State α) :
    ∃ t, FP.Structural.exec M (source x y p) s = some t ∧ t.mem = expectedMemory x y p s :=
  ⟨(execution M x y p s).val, (execution M x y p s).property, execution_memory M x y p s⟩

#specNonCircular expectedMemory avoiding [source, FP.Structural.exec]
#axiomsClean execution
#axiomsClean execution_memory
#axiomsClean expected_cell
#axiomsClean complete_correct
end
end VeriTile.Bench.TritonBenchG.VectorAddition.IntegerSource
