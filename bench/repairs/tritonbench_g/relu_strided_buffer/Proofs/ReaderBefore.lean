import bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.PublicIO

namespace VeriTile.Bench.Repairs.ReluStridedBuffer.ReaderBefore
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.ProgramIO
open TritonBenchG.ReluStridedBuffer HostPlan SignedView Whole
noncomputable section

/-- The complete original capped grid implements mathematical ReLU on the
explicit signed-view IO. Both source branches have proved access safety,
successful execution and all logical outputs, with the full exterior frame.
The original signed view, capacity and footprint conditions are retained. -/
theorem relu_strided_buffer_kernel_correctness (rest : Algebra ℝ) (x y : SignedView.View)
    (n inputCapacity outputCapacity : Nat) (hn : 0<n)
    (hin : x.Valid n inputCapacity) (hout : y.Valid n outputCapacity) (hstride : y.stride≠0)
    (hsep : SignedGrid.FootprintsDisjoint x y n) :
    Spec.Real (ReluIO.kernelIO (realBackend rest) Bool.false x y n ⊨
      ReluIO.pointwise (fun z => max 0 z) x y n) :=
  ReluIO.real_implements rest x y n inputCapacity outputCapacity hn hin hout hstride hsep

end
end VeriTile.Bench.Repairs.ReluStridedBuffer.ReaderBefore
