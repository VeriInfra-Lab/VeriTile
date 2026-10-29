import bench.repairs.tritonbench_g.dropout_triton.Kernels
import bench.tritonbench_g.vector_addition.Proofs.TypedFloat

namespace VeriTile.Bench.Repairs.DropoutTriton.Typed
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.VectorAddition
open ValueProofs (Parameters offset)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

abbrev castBackend := @TypedFloat.castBackend

def source (q : Parameters) (p : ℝ) : ComputeKernel :=
  sourceKernel q.x q.y q.output q.n q.block p

def resultCell {α : Type} [Inhabited α] (M : Algebra α)
    (q : Parameters) (p : ℝ) (mem : RegionName → Nat → Cell α) (o : Nat) : Cell α :=
  .mk .real (if (mem q.y o).read .int ≠ 0 then
    M.binary (some .fp32) .real .div (M.fp32Load ((mem q.x o).read .real))
      (M.binary (some .fp32) .real .sub (M.literal (some .fp32) .real 1) (M.literal (some .fp32) .real p))
    else M.literal (some .fp32) .real 0)

def expectedMemory {α : Type} [Inhabited α] (M : Algebra α)
    (q : Parameters) (p : ℝ) (s : State α) : RegionName → Nat → Cell α :=
  (TileShape.allIndices [q.block]).foldl (fun mem i =>
    if offset q s i < q.n then fun r o =>
      if r = q.output ∧ o = offset q s i then
        resultCell M q p s.mem (offset q s i) else mem r o
    else mem) s.mem

def execution {α : Type} [Inhabited α] (M : Algebra α)
    (q : Parameters) (p : ℝ) (s : State α) :
    {t : State α // FP.Structural.exec (castBackend M) (source q p) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [source, sourceKernel,
    FP.Structural.exec, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    numericLt, numericEq, natLt, bop, store, Region.cast, castBackend,
    TypedFloat.castBackend, numeric, ofFloat, toFloat, ComputeDType.eraseDType]
  rfl

theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α)
    (q : Parameters) (p : ℝ) (s : State α) :
    (execution M q p s).val.mem = expectedMemory M q p s := by
  simp [execution, expectedMemory, offset, resultCell, bop,
    Broadcast.leftIndex, Broadcast.rightIndex,
    ValueProofs.masked_scatter_memory, State.setReg, castBackend,
    TypedFloat.castBackend, numeric]
  congr 1
  funext mem i
  by_cases hi : s.pids 0 * q.block + i.1.val < q.n
  · simp only [hi, ↓reduceIte]
    norm_num
  · simp only [hi, ↓reduceIte]

theorem expectedMemory_eq_scatter {α : Type} [Inhabited α] (M : Algebra α)
    (q : Parameters) (p : ℝ) (s : State α) :
    expectedMemory M q p s =
      ((TileShape.allIndices [q.block]).foldl (fun t i =>
        if offset q s i < q.n then t.write q.output (offset q s i)
          (resultCell M q p s.mem (offset q s i)) else t) s).mem := by
  rw [ValueProofs.masked_scatter_memory]
  rfl

theorem expected_readback {α : Type} [Inhabited α] (M : Algebra α)
    (q : Parameters) (p : ℝ) (s : State α) (i : TileIndex [q.block]) (hi : offset q s i < q.n) :
    expectedMemory M q p s q.output (offset q s i) = resultCell M q p s.mem (offset q s i) := by
  rw [expectedMemory_eq_scatter]
  exact State.masked_scatter_readback q.output (offset q s)
    (fun i => resultCell M q p s.mem (offset q s i))
    (fun i => offset q s i < q.n) s i hi (ValueProofs.offset_injective q s)

theorem expected_frame {α : Type} [Inhabited α] (M : Algebra α)
    (q : Parameters) (p : ℝ) (s : State α) (r : RegionName) (o : Nat) (h : ValueProofs.Outside q s r o) :
    expectedMemory M q p s r o = s.mem r o := by
  rw [expectedMemory_eq_scatter]
  apply State.masked_scatter_frame
  exact h.imp id (fun h i _ hi => h i hi)

/-- Closed address formula for one whole source program. -/
theorem expected_cell {α : Type} [Inhabited α] (M : Algebra α)
    (q : Parameters) (p : ℝ) (s : State α) (r : RegionName) (o : Nat) :
    expectedMemory M q p s r o =
      if r = q.output ∧ s.pids 0 * q.block ≤ o ∧
          o < s.pids 0 * q.block + q.block ∧ o < q.n then
        resultCell M q p s.mem o else s.mem r o := by
  by_cases h : r = q.output ∧ s.pids 0 * q.block ≤ o ∧
      o < s.pids 0 * q.block + q.block ∧ o < q.n
  · rw [if_pos h]
    let i : TileIndex [q.block] := (⟨o - s.pids 0 * q.block, by omega⟩, PUnit.unit)
    have ho : offset q s i = o := by simp only [offset, i]; omega
    rw [h.1, ← ho]
    exact expected_readback M q p s i (by rw [ho]; exact h.2.2.2)
  · rw [if_neg h]
    apply expected_frame
    by_cases hr : r = q.output
    · right
      intro i hi ho
      apply h
      have bound := i.1.isLt
      unfold offset at hi ho
      exact ⟨hr, by omega, by omega, by omega⟩
    · exact Or.inl hr

theorem complete_correct {α : Type} [Inhabited α] (M : Algebra α)
    (q : Parameters) (p : ℝ) (s : State α) :
    ∃ t, FP.Structural.exec (castBackend M) (source q p) s = some t ∧ t.mem = expectedMemory M q p s :=
  ⟨(execution M q p s).val, (execution M q p s).property, execution_memory M q p s⟩


#specNonCircular resultCell avoiding [source, FP.Structural.exec]
#axiomsClean execution_memory
#axiomsClean complete_correct
#guard_msgs (drop info) in
#auditModuleAxioms

end
end VeriTile.Bench.Repairs.DropoutTriton.Typed
