import bench.repairs.tritonbench_g.masked_add_cuda.Proofs.Graph

/-! The genuine addition-order variant uses the same selected fused graph.
Both runs are constructed from their complete canonical source. Raw FP32 input
words are leaves; this is a representation binding, not a finiteness premise. -/
namespace VeriTile.Bench.Repairs.MaskedAddCuda.Variant
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs (offset)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def execution {α : Type} [Inhabited α] (M : Algebra α) (storage : MaskStorage)
    (p : Parameters) (alpha : ℝ) (s : State α) :
    {t : State α // FP.Structural.exec M
      (commutedKernel storage p.output p.x p.y p.n alpha p.block) s = some t} := by
  cases storage
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp [commutedKernel, FP.Structural.exec, ComputeKernel.surfaceBody,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
      numericLt, numericEq, natLt, bop, store, Region.cast, ComputeDType.eraseDType]
    rfl

theorem execution_memory (rest : Algebra Graph.Expr) (word : Graph.Word)
    (storage : MaskStorage) (p : Parameters) (alpha : ℝ) (s : State Graph.Expr)
    (grad : Nat → Graph.Word)
    (hg : ∀ i : TileIndex [p.block], active p s i →
      (s.mem p.output (offset p s i)).read .real = .leaf (grad (offset p s i))) :
    (execution (Graph.nativeAlgebra rest word) storage p alpha s).val.mem =
      expectedMemory (Graph.nativeAlgebra rest word) p alpha s := by
  cases storage
  all_goals
    simp [execution, expectedMemory, offset, active, resultCell, updateValue, alphaValue, bop,
      Broadcast.leftIndex, Broadcast.rightIndex,
      VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.masked_scatter_memory, State.setReg]
    congr 1
    funext mem i
    by_cases bound : s.pids 0 * p.block + i.1.val < p.n
    · by_cases zero : (s.mem p.y (s.pids 0 * p.block + i.1.val)).read .int = 0
      · have leaf := hg i ⟨bound, zero⟩
        simp only [offset] at leaf
        simp [bound, zero, numeric, ofFloat, toFloat, Graph.nativeAlgebra,
          Graph.algebra, Graph.binary, leaf]
      · simp [bound, zero]
    · simp [bound]

theorem equivalent (rest : Algebra Graph.Expr) (word : Graph.Word)
    (storage : MaskStorage) (p : Parameters) (alpha : ℝ) (s : State Graph.Expr)
    (grad : Nat → Graph.Word)
    (hg : ∀ i : TileIndex [p.block], active p s i →
      (s.mem p.output (offset p s i)).read .real = .leaf (grad (offset p s i))) :
    ∃ originalFinal commutedFinal,
      FP.Structural.exec (Graph.nativeAlgebra rest word)
        (kernel storage p.output p.x p.y p.n alpha p.block) s = some originalFinal ∧
      FP.Structural.exec (Graph.nativeAlgebra rest word)
        (commutedKernel storage p.output p.x p.y p.n alpha p.block) s = some commutedFinal ∧
      originalFinal.mem = commutedFinal.mem := by
  refine ⟨(MaskedAddCuda.execution (Graph.nativeAlgebra rest word) storage p alpha s).val,
    (execution (Graph.nativeAlgebra rest word) storage p alpha s).val,
    (MaskedAddCuda.execution (Graph.nativeAlgebra rest word) storage p alpha s).property,
    (execution (Graph.nativeAlgebra rest word) storage p alpha s).property, ?_⟩
  rw [MaskedAddCuda.execution_memory, execution_memory rest word storage p alpha s grad hg]

#axiomsClean execution
#axiomsClean execution_memory
#axiomsClean equivalent
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MaskedAddCuda.Variant
