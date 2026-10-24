import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardReal

namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward
open VeriTile.Triton
open scoped VeriTile.Triton.Masked2DKernelIO₂
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

theorem source_flattenOk (GradIn GradOut Out : RegionName)
    (bm bn gm gn om on K D : Nat) (log causal fp16 : Bool) :
    (source GradIn GradOut Out bm bn gm gn om on K D log causal fp16).toAlgKernel.FlattenOk := by
  cases log <;> cases causal <;> cases fp16 <;>
    dsimp only [source, ksoftmax_backward_surface, ComputeKernel.toAlgKernel,
      ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, Except.pure,
      Bind.bind, Pure.pure]
  all_goals simp [Kernel.FlattenOk, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

theorem source_traceSafe (GradIn GradOut Out : RegionName)
    (bm bn gm gn om on K D : Nat) (log causal fp16 : Bool)
    (bounds : RegionBounds) (s : BlockState)
    (hg : ∀ j : Fin D, readActive K (s.pids 1) causal j.val = Bool.true →
      s.pids 0 * gm + s.pids 1 * gn + j.val < bounds GradOut)
    (ho : ∀ j : Fin D, readActive K (s.pids 1) causal j.val = Bool.true →
      s.pids 0 * om + s.pids 1 * on + j.val < bounds Out)
    (hb : ∀ j : Fin D, j.val < K →
      s.pids 0 * bm + s.pids 1 * bn + j.val < bounds GradIn) :
    Kernel.TraceSafe bounds
      (source GradIn GradOut Out bm bn gm gn om on K D log causal fp16).toAlgKernel s := by
  cases log <;> cases causal <;> cases fp16
  all_goals
    simp [readActive] at hg ho
    dsimp only [source, ksoftmax_backward_surface, ComputeKernel.toAlgKernel,
      ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, Except.pure,
      Bind.bind, Pure.pure]
    unfold Kernel.TraceSafe
    simp [Stmt.TraceSafeList, Stmt.TraceSafe,
      Op.SafeAt.eq_def, MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, stepStmts, evalOp.eq_def,
      MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MaskOpt.Active,
      Op.PointerAddressesSafeOn, Op.MemorySafe, BlockState.setReg, tile_elementwise]
    exact ⟨hg, ho, hb⟩

/-- Three independently strided buffers on the original 2D program grid.
Causal masking restricts reads, while all lanes below K are written. -/
def io (GradIn GradOut Out : RegionName)
    (bm bn gm gn om on K D : Nat) (log causal fp16 : Bool) : Masked2DKernelIO₂ where
  kernel := source GradIn GradOut Out bm bn gm gn om on K D log causal fp16
  projection := source_projects GradIn GradOut Out bm bn gm gn om on K D log causal fp16
  in1 := GradOut
  in2 := Out
  out := GradIn
  B := D
  read1 := fun m n j => m * gm + n * gn + j.val
  read2 := fun m n j => m * om + n * on + j.val
  write := fun m n j => m * bm + n * bn + j.val
  mask := fun _ n j => readActive K n causal j.val = Bool.true
  writeMask := fun _ _ j => j.val < K

/-- Complete backward real IO: successful flat execution, the independent
finite-sum result on every written lane, and all unwritten cells preserved.
The launch shape and machine-width address range are external obligations. -/
specification ksoftmax_backward_correctness (GradIn GradOut Out : RegionName)
    (bm bn gm gn om on K D : Nat) (log causal fp16 : Bool) :
    io GradIn GradOut Out bm bn gm gn om on K D log causal fp16 ⊨
      fun _ row g o i => formula K row log causal g o i := by
  refine Masked2DKernelIO₂.Implements.intro _
    (source_flattenOk GradIn GradOut Out bm bn gm gn om on K D log causal fp16) ?_ ?_
  · intro bounds s hg ho hb _
    exact source_traceSafe GradIn GradOut Out bm bn gm gn om on K D log causal fp16 bounds s hg ho hb
  · intro s g o hg ho
    obtain ⟨s', hrun, hvalue, hframe⟩ :=
      source_region_run GradIn GradOut Out bm bn gm gn om on K D log causal fp16 s g o hg ho
    exact ⟨s', hrun, hvalue, fun r offset hmiss _ => hframe r offset hmiss⟩

#axiomsClean source_flattenOk
#axiomsClean source_traceSafe
#axiomsClean ksoftmax_backward_correctness
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward
