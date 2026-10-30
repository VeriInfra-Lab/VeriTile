import bench.repairs.tritonbench_g.masked_add_cuda.Proofs.Source
import bench.tritonbench_g.vector_addition.Proofs.TypedFloat

namespace VeriTile.Bench.Repairs.MaskedAddCuda
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs (offset programState gridCount grid_covers)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def active {α : Type} [Inhabited α] (p : Parameters) (s : State α) (i : TileIndex [p.block]) : Prop :=
  offset p s i < p.n ∧ (s.mem p.y (offset p s i)).read .int = 0

instance {α : Type} [Inhabited α] (p : Parameters) (s : State α) (i : TileIndex [p.block]) :
    Decidable (active p s i) := by unfold active; infer_instance

def alphaValue {α : Type} (M : Algebra α) (alpha : ℝ) : α :=
  M.cast (some .fp32) .real .real (M.literal (some .fp32) .real alpha)

def updateValue {α : Type} (M : Algebra α) (alpha : ℝ) (grad data : α) : α :=
  M.binary (some .fp32) .real .add grad
    (M.binary (some .fp32) .real .mul data (alphaValue M alpha))

def resultCell {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (alpha : ℝ)
    (mem : RegionName → Nat → Cell α) (o : Nat) : Cell α :=
  .mk .real (updateValue M alpha (M.fp32Load ((mem p.output o).read .real)) (M.fp32Load ((mem p.x o).read .real)))

def expectedMemory {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (alpha : ℝ) (s : State α) :=
  (TileShape.allIndices [p.block]).foldl (fun mem i =>
    if active p s i then fun r o =>
      if r=p.output ∧ o=offset p s i then resultCell M p alpha s.mem (offset p s i) else mem r o
    else mem) s.mem

/-- Actual full original typed body, instantiated at the bounded integer
projection. Source success is constructed for arbitrary scalar operations. -/
def execution {α : Type} [Inhabited α] (M : Algebra α) (storage : MaskStorage)
    (p : Parameters) (alpha : ℝ) (s : State α) :
    {t : State α // FP.Structural.exec M (candidateKernel storage p alpha) s = some t} := by
  cases storage
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp [candidateKernel,kernel,VeriTile.Bench.TritonBenchG.MaskedAddCuda.integer_kernel,FP.Structural.exec,ComputeKernel.surfaceBody,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      numericLt,numericEq,natLt,bop,store,Region.cast,ComputeDType.eraseDType]
    rfl

theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α) (storage : MaskStorage)
    (p : Parameters) (alpha : ℝ) (s : State α) :
    (execution M storage p alpha s).val.mem = expectedMemory M p alpha s := by
  cases storage
  all_goals
    simp [execution,expectedMemory,offset,active,resultCell,updateValue,alphaValue,bop,
      Broadcast.leftIndex,Broadcast.rightIndex,VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.masked_scatter_memory,State.setReg]
    congr 1
    funext mem i
    have regionEta (r : RegionName) : ({ name := r.name } : RegionName) = r := by cases r; rfl
    by_cases bound : s.pids 0 * p.block + i.1.val < p.n
    · by_cases zero : (s.mem p.y (s.pids 0 * p.block + i.1.val)).read .int = 0
      · simp [bound,zero,numeric,ofFloat,toFloat]
      · simp [bound,zero]
    · simp [bound]

theorem expectedMemory_eq_scatter {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (alpha : ℝ) (s : State α) :
    expectedMemory M p alpha s =
      ((TileShape.allIndices [p.block]).foldl (fun t i =>
        if active p s i then t.write p.output (offset p s i)
          (resultCell M p alpha s.mem (offset p s i)) else t) s).mem := by
  rw [VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.masked_scatter_memory]
  rfl

theorem expected_readback {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (alpha : ℝ) (s : State α) (i : TileIndex [p.block]) (hi : active p s i) :
    expectedMemory M p alpha s p.output (offset p s i) = resultCell M p alpha s.mem (offset p s i) := by
  rw [expectedMemory_eq_scatter]
  exact State.masked_scatter_readback p.output (offset p s)
    (fun i => resultCell M p alpha s.mem (offset p s i))
    (active p s) s i hi (VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.offset_injective p s)

theorem expected_frame {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (alpha : ℝ) (s : State α) (r : RegionName) (o : Nat)
    (h : r ≠ p.output ∨ ∀ i, active p s i → o ≠ offset p s i) :
    expectedMemory M p alpha s r o = s.mem r o := by
  rw [expectedMemory_eq_scatter]
  apply State.masked_scatter_frame
  exact h.imp id (fun h i _ hi => h i hi)

theorem expected_cell {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (alpha : ℝ) (s : State α) (r : RegionName) (o : Nat) :
    expectedMemory M p alpha s r o =
      if r = p.output ∧ s.pids 0 * p.block ≤ o ∧
          o < s.pids 0 * p.block + p.block ∧ o < p.n ∧ (s.mem p.y o).read .int = 0 then
        resultCell M p alpha s.mem o else s.mem r o := by
  by_cases h : r = p.output ∧ s.pids 0 * p.block ≤ o ∧
      o < s.pids 0 * p.block + p.block ∧ o < p.n ∧ (s.mem p.y o).read .int = 0
  · rw [if_pos h]
    let i : TileIndex [p.block] := (⟨o - s.pids 0 * p.block, by omega⟩, PUnit.unit)
    have ho : offset p s i = o := by simp only [offset, i]; omega
    rw [h.1, ← ho]
    exact expected_readback M p alpha s i (by unfold active; rw [ho]; exact h.2.2.2)
  · rw [if_neg h]
    apply expected_frame
    by_cases hr : r = p.output
    · right
      intro i hi ho
      apply h
      have bound := i.1.isLt
      have hz : (s.mem p.y o).read .int = 0 := by rw [ho]; exact hi.2
      have hb := hi.1
      unfold offset at hb ho
      exact ⟨hr, by omega, by omega, by omega, hz⟩
    · exact Or.inl hr

theorem complete_correct {α : Type} [Inhabited α] (M : Algebra α)
    (storage : MaskStorage) (p : Parameters) (alpha : ℝ) (s : State α) :
    ∃ t, FP.Structural.exec M (candidateKernel storage p alpha) s = some t ∧
      t.mem = expectedMemory M p alpha s :=
  ⟨(execution M storage p alpha s).val, (execution M storage p alpha s).property,
    execution_memory M storage p alpha s⟩

#specNonCircular expectedMemory avoiding [candidateKernel,kernel, FP.Structural.exec]
#axiomsClean expected_cell
#axiomsClean complete_correct
#axiomsClean execution
#axiomsClean execution_memory
end
end VeriTile.Bench.Repairs.MaskedAddCuda
