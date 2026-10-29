import bench.tritonbench_g.matrix_reduction.Proofs.KernelSafety

namespace VeriTile.Bench.TritonBenchG.MatrixReduction.Candidate
open VeriTile.Triton FP.Structural TypedSource ConcreteCorrect
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

/-- Direct affine addressing supplies the identical ordered native graph. No
floating arithmetic, reduction, promotion or store law is rewritten. -/
theorem source_graph {α : Type} [Inhabited α] (A : Algebra α) (d : Storage)
    (s : State α) (M N : Nat) (hN : 0 < N) :
    ∃ t, exec A (candidateKernel d "x" "out" N 1 1 M N) s = some t ∧
      t.mem = KernelFP.written A d M N s := by
  cases d <;> simp [candidateKernel, ComputeKernel.surfaceBody, FP.Structural.exec,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, store, State.setReg,
    TileShape.axisDim, hN, TileShape.insertAxisIndex, TileShape.eraseAxis,
    ofFloat, toFloat, Region.cast, ComputeDType.eraseDType,
    KernelFP.written, KernelFP.formula, KernelFP.sourceValues, channel,
    fold_memory, numeric, bop, Option.bind]

/-- Both concrete address lowerings access exactly the declared whole tile
and every output row, including the vacuous M=0 case. -/
theorem trace {α : Type} [Inhabited α] (A : Algebra α) (d : Storage)
    (M N : Nat) (positive : 0 < N) (bounds : RegionName → Nat) (s : State α)
    (input : ∀ i : Fin M, ∀ j : Fin N, i.val * N + j.val < bounds "x")
    (output : ∀ i : Fin M, i.val < bounds "out") :
    TraceSafe A bounds (candidateKernel d "x" "out" N 1 1 M N) s := by
  cases d <;>
    simp [FP.Structural.TraceSafe, candidateKernel, ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold, Safety.stepSafe_unfold, Safety.safeExpr_unfold,
      Safety.safeCompute_unfold, Safety.safeOp_unfold, Safety.safeAddress_unfold,
      Safety.safeMask_unfold, Safety.addressSafe_unfold, Safety.maskActive_unfold,
      step, evalExpr, evalComputeOp, evalOp_unfold, State.setReg, store, Option.bind,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      Region.cast, ofFloat, toFloat, positive, numeric, bop]
  all_goals exact ⟨input, output⟩

#axiomsClean source_graph
#axiomsClean trace
end
end VeriTile.Bench.TritonBenchG.MatrixReduction.Candidate
