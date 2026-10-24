import bench.tritonbench_g.matrix_reduction.Proofs.KernelFP
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.TritonBenchG.MatrixReduction.KernelSafety
open VeriTile.Triton FP.Structural TypedSource
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 16000

/-- The actual typed block-pointer load and row stores only access the declared
matrix/output windows. No floating property is used to certify addresses. -/
theorem trace {α : Type} [Inhabited α] (A : Algebra α) (d : Storage)
    (M N : Nat) (positive : 0 < N) (bounds : RegionName → Nat) (s : State α)
    (input : ∀ i : Fin M, ∀ j : Fin N, i.val * N + j.val < bounds "x")
    (output : ∀ i : Fin M, i.val < bounds "out") :
    FP.Structural.TraceSafe A bounds (kernel d "x" "out" N 1 1 M N) s := by
  cases d <;>
    simp [FP.Structural.TraceSafe, kernel, source16, source32, ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold, Safety.stepSafe_unfold, Safety.safeExpr_unfold,
      Safety.safeCompute_unfold, Safety.safeOp_unfold, Safety.safeAddress_unfold,
      Safety.safeMask_unfold, Safety.addressSafe_unfold, Safety.maskActive_unfold,
      step, evalExpr, evalComputeOp, evalOp_unfold, State.setReg, store, Option.bind,
      BlockPtr.inBounds, BlockPtr.address, TileShape.indexToList,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      Region.cast, ofFloat, toFloat, positive, ComputeDType.eraseDType]
  all_goals
    refine ⟨?_, fun i => output i⟩
    intro i j
    have address := BlockPtr.address_2d_zero_offsets "x" 0 M N M N N 1 i.val j.val
    have backed : BlockPtr.address
        { region := "x", baseOffset := 0, parentShape := [M, N], blockShape := [M, N],
          strides := [N, 1], offsets := [0, 0] } [i.val, j.val] < bounds "x" := by
      rw [address]
      simpa using input i j
    simpa only [BlockPtr.address, Nat.zero_add] using backed

#axiomsClean trace
end
end VeriTile.Bench.TritonBenchG.MatrixReduction.KernelSafety
