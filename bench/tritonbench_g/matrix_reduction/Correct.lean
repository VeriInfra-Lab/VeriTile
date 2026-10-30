import bench.tritonbench_g.matrix_reduction.Proofs.LogicalIO

namespace VeriTile.Bench.TritonBenchG.MatrixReduction
open VeriTile.Triton TypedSource
open scoped VeriTile.Triton.LogicalIO
noncomputable section


/-- Every output row is the greatest element of its nonempty logical input
row. The old output-column-zero mask and complete flat-memory frame remain. -/
specification matrix_reduction_kernel_correct (storage : Storage) (M N : Nat) (hN : 0<N) :
    VeriTile.Spec.Real (rowMaximumIO storage M N ⊨ fun xs row _ =>
      (Finset.univ : Finset (Fin N)).sup'
        ⟨⟨0,hN⟩,Finset.mem_univ _⟩ (fun col => xs row col)) :=
  LogicalIOProof.implements storage M N hN

#specNonCircular KernelMath.maximumValue avoiding [kernel,exec]
#stmtSurfaceSubset matrix_reduction_kernel_correct ⊆
  [VeriTile.Spec.Real,rowMaximumIO,LogicalIO.Implements,Storage,TileIndex,MaskedTileKernelIO₁.Implements,matrixReductionIO,MaskedTileKernelIO₁.shape]
#axiomsClean matrix_reduction_kernel_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatrixReduction
