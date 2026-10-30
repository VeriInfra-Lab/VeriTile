import bench.repairs.tritonbench_g.kldiv_compute.Kernels
import bench.tritonbench_g.vector_addition.Proofs.PointwiseSchedule
import bench.tritonbench_g.kldiv_compute.Proofs.FP
namespace VeriTile.Bench.Repairs.KldivCompute.Program
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs
  (offset masked_scatter_memory offset_injective)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def scalar {α : Type} (M : Algebra α) (x y : α) : α :=
  let a := M.fp32Load x
  let b := M.fp32Load y
  M.binary (some .fp32) .real .mul a
    (M.unary (some .fp32) .log (M.binary (some .fp32) .real .div a b))

def resultCell {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (mem : RegionName → Nat → Cell α) (o : Nat) : Cell α :=
  .mk .real (scalar M ((mem p.x o).read .real) ((mem p.y o).read .real))

def expectedMemory {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) : RegionName → Nat → Cell α :=
  (TileShape.allIndices [p.block]).foldl (fun mem i =>
    if offset p s i < p.n then fun r o =>
      if r = p.output ∧ o = offset p s i then resultCell M p s.mem (offset p s i) else mem r o
    else mem) s.mem

def Outside {α : Type} (p : Parameters) (s : State α) (r : RegionName) (o : Nat) : Prop :=
  r ≠ p.output ∨ ∀ i : TileIndex [p.block], offset p s i < p.n → o ≠ offset p s i

theorem source_admitted (p : Parameters) :
    (source p).toAlgorithm? = .ok (source p).toAlgKernel := by rfl

def execution {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    {t : State α // FP.Structural.exec M (source p) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [source, VeriTile.Bench.TritonBenchG.KldivCompute.TypedSource.kernel,
    VeriTile.Bench.TritonBenchG.KldivCompute.TypedSource.kernel_fp32_fp32,
    FP.Structural.exec, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, Algebra.withDefaultPrecision,
    resolvePrecision, ComputeDType.eraseDType, State.setReg, numeric, numericLt, natLt, bop, store, Region.cast]
  rfl

/-- The reference is the exact memory effect, even when input and output
region names coincide. The loads precede the original final store. -/
theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) :
    (execution M p s).val.mem = expectedMemory M p s := by
  simp [execution, expectedMemory, offset, resultCell, scalar, bop,
    Broadcast.leftIndex, Broadcast.rightIndex, TileShape.dropInsertedIndex,
    masked_scatter_memory, State.setReg]
  congr 1
  funext mem i
  by_cases hi : s.pids 0 * p.block + i.1.val < p.n
  · simp only [hi, ↓reduceIte]
  · simp only [hi, ↓reduceIte]

theorem expectedMemory_eq_scatter {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) :
    expectedMemory M p s =
      ((TileShape.allIndices [p.block]).foldl (fun t i =>
        if offset p s i < p.n then t.write p.output (offset p s i)
          (resultCell M p s.mem (offset p s i)) else t) s).mem := by
  rw [masked_scatter_memory]
  rfl

theorem expected_readback {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) (i : TileIndex [p.block]) (hi : offset p s i < p.n) :
    expectedMemory M p s p.output (offset p s i) = resultCell M p s.mem (offset p s i) := by
  rw [expectedMemory_eq_scatter]
  exact State.masked_scatter_readback p.output (offset p s)
    (fun i => resultCell M p s.mem (offset p s i))
    (fun i => offset p s i < p.n) s i hi (offset_injective p s)

theorem expected_frame {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) (r : RegionName) (o : Nat) (h : Outside p s r o) :
    expectedMemory M p s r o = s.mem r o := by
  rw [expectedMemory_eq_scatter]
  apply State.masked_scatter_frame
  exact h.imp id (fun h i _ hi => h i hi)

/-- Closed address formula for one whole source program. -/
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
    exact expected_readback M p s i (by rw [ho]; exact h.2.2.2)
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

theorem complete_correct {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) :
    ∃ t, FP.Structural.exec M (source p) s = some t ∧ t.mem = expectedMemory M p s :=
  ⟨(execution M p s).val, (execution M p s).property, execution_memory M p s⟩


open VeriTile.Bench.TritonBenchG.VectorAddition
open VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs (gridCount programState)

def grid {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (initial : State α) : Option (State α) :=
  PointwisePrograms.runSchedule (fun pid current =>
    FP.Structural.exec M (source p) (programState p initial current.mem pid))
      (List.range (gridCount p)) initial

theorem grid_complete {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (initial : State α) (positive : 0 < p.block) :
    ∃ final, grid M p initial = some final ∧
      final.mem = PointwisePrograms.finalMemory (resultCell M p) p initial.mem := by
  apply PointwisePrograms.complete_schedule (resultCell M p) ?_ p _ ?_
    initial positive _ (List.Perm.refl _)
  · intro a b o h
    simp only [resultCell, h]
  · intro pid current
    obtain ⟨final, he, hm⟩ := complete_correct M p (programState p initial current.mem pid)
    refine ⟨final, he, hm.trans ?_⟩
    funext r o
    simpa only [programState, ↓reduceIte, PointwisePrograms.oneMemory] using
      expected_cell M p (programState p initial current.mem pid) r o

#axiomsClean execution
#axiomsClean execution_memory
#axiomsClean complete_correct
#axiomsClean grid_complete
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivCompute.Program
