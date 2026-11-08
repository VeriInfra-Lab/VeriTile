import bench.tritonbench_g.relu_strided_buffer.Proofs.LogicalReader

namespace VeriTile.Bench.Repairs.ReluStridedBuffer
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
open TritonBenchG.ReluStridedBuffer HostPlan SignedView Whole
noncomputable section

/-- The complete original capped grid implements mathematical ReLU on the
explicit signed-view IO. Both source branches have proved access safety,
successful execution and all logical outputs, with the full exterior frame.
The original signed view, capacity and footprint conditions are retained. -/
specification relu_strided_buffer_kernel_correctness (rest : Algebra ℝ) (x y : SignedView.View)
    (n inputCapacity outputCapacity : Nat) (hn : 0<n)
    (hin : x.Valid n inputCapacity) (hout : y.Valid n outputCapacity) (hstride : y.stride≠0)
    (hsep : SignedGrid.FootprintsDisjoint x y n) :
    Spec.Real (LogicalView.reluIO rest x y n ⊨ (fun xs i => max 0 (xs i))) :=
  (LogicalReader.judgment_iff rest x y n).mpr
    (ReaderBefore.relu_strided_buffer_kernel_correctness rest x y n inputCapacity outputCapacity hn hin hout hstride hsep)

#specNonCircular ReluIO.mathematical avoiding [kernel,foldedKernel,FP.Structural.exec,runGrid]
#axiomsClean relu_strided_buffer_kernel_correctness
end
end VeriTile.Bench.Repairs.ReluStridedBuffer
