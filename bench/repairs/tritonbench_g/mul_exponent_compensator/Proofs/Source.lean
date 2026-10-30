import bench.repairs.tritonbench_g.mul_exponent_compensator.Kernels
import bench.tritonbench_g.vector_addition.Proofs.TypedFloat

namespace VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous
open VeriTile.Triton FP.Structural
abbrev Parameters := VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.Parameters
open VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs (offset programState gridCount grid_covers)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def active {α : Type} [Inhabited α] (p : Parameters) (s : State α) (i : TileIndex [p.block]) : Prop :=
  offset p s i < p.n

instance {α : Type} [Inhabited α] (p : Parameters) (s : State α) (i : TileIndex [p.block]) :
    Decidable (active p s i) := by unfold active; infer_instance

def scalar {α : Type} (M : Algebra α) (x : α) : α :=
  M.binary (some .fp32) .real .mul (M.fp32Load x) (M.literal none .real exponentCompensator)

def resultCell {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (mem : RegionName → Nat → Cell α) (o : Nat) : Cell α :=
  .mk .real (scalar M ((mem p.x o).read .real))

def expectedMemory {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :=
  (TileShape.allIndices [p.block]).foldl (fun mem i =>
    if active p s i then fun r o =>
      if r=p.output ∧ o=offset p s i then resultCell M p s.mem (offset p s i) else mem r o
    else mem) s.mem

/-- Actual full repaired typed body, instantiated at the repaired integer
projection. Source success is constructed for arbitrary scalar operations. -/
def execution {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) :
    {t : State α // FP.Structural.exec M (candidateKernel p.x p.output p.n p.block) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [candidateKernel,FP.Structural.exec,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    numericLt,natLt,bop,store,Region.cast,ComputeDType.eraseDType]
  rfl

theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) :
    (execution M p s).val.mem = expectedMemory M p s := by
  simp [execution,expectedMemory,offset,active,resultCell,scalar,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.masked_scatter_memory,State.setReg]
  congr 1
  funext mem i
  by_cases bound : s.pids 0 * p.block + i.1.val < p.n
  · simp [bound,numeric,ofFloat,toFloat]
  · simp [bound]

theorem expectedMemory_eq_scatter {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) :
    expectedMemory M p s =
      ((TileShape.allIndices [p.block]).foldl (fun t i =>
        if active p s i then t.write p.output (offset p s i)
          (resultCell M p s.mem (offset p s i)) else t) s).mem := by
  rw [VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.masked_scatter_memory]
  rfl

theorem expected_readback {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) (i : TileIndex [p.block]) (hi : active p s i) :
    expectedMemory M p s p.output (offset p s i) = resultCell M p s.mem (offset p s i) := by
  rw [expectedMemory_eq_scatter]
  exact State.masked_scatter_readback p.output (offset p s)
    (fun i => resultCell M p s.mem (offset p s i))
    (active p s) s i hi (VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.offset_injective p s)

theorem expected_frame {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) (r : RegionName) (o : Nat)
    (h : r ≠ p.output ∨ ∀ i, active p s i → o ≠ offset p s i) :
    expectedMemory M p s r o = s.mem r o := by
  rw [expectedMemory_eq_scatter]
  apply State.masked_scatter_frame
  exact h.imp id (fun h i _ hi => h i hi)

theorem expected_cell {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) (r : RegionName) (o : Nat) :
    expectedMemory M p s r o =
      if r = p.output ∧ s.pids 0 * p.block ≤ o ∧
          o < s.pids 0 * p.block + p.block ∧ o < p.n then
        resultCell M p s.mem o else s.mem r o := by
  by_cases h : r = p.output ∧ s.pids 0 * p.block ≤ o ∧
      o < s.pids 0 * p.block + p.block ∧ o < p.n
  · rw [if_pos h]
    let i : TileIndex [p.block] := (⟨o - s.pids 0 * p.block, by omega⟩, PUnit.unit)
    have ho : offset p s i = o := by simp only [offset, i]; omega
    rw [h.1, ← ho]
    exact expected_readback M p s i (by unfold active; rw [ho]; exact h.2.2.2)
  · rw [if_neg h]
    apply expected_frame
    by_cases hr : r = p.output
    · right
      intro i hi ho
      apply h
      have bound := i.1.isLt
      have hb := hi
      unfold active offset at hb
      unfold offset at ho
      exact ⟨hr, by omega, by omega, by omega⟩
    · exact Or.inl hr

theorem complete_correct {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) :
    ∃ t, FP.Structural.exec M (candidateKernel p.x p.output p.n p.block) s = some t ∧
      t.mem = expectedMemory M p s :=
  ⟨(execution M p s).val, (execution M p s).property,
    execution_memory M p s⟩

#specNonCircular expectedMemory avoiding [candidateKernel, FP.Structural.exec]
#axiomsClean expected_cell
#axiomsClean complete_correct
#axiomsClean execution
#axiomsClean execution_memory
end
end VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous
