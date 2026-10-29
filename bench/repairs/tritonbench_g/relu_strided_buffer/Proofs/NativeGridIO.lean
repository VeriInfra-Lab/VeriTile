import bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.PublicIO

namespace VeriTile.Bench.Repairs.ReluStridedBuffer
open VeriTile.Triton FP.Structural
open TritonBenchG.ReluStridedBuffer HostPlan SignedView Whole
noncomputable section
local instance gridWordInhabited : Inhabited (FP.Value .fp32) := ⟨FP.Value.zero .fp32⟩

/-- Full-grid original/candidate equivalence and independent ordered-ReLU
words on the explicit signed FP32 IO. Both actual schedules are safe,
successful and preserve the exterior frame. Comparison, identity casts and
positive zero use the existing software primitive model; no floating rewrite
law, finiteness condition or numerical experiment is assumed. -/
theorem relu_strided_buffer_grid_equiv (rest : Algebra (FP.Value .fp32))
    (x y : SignedView.View) (n inputCapacity outputCapacity : Nat) (hn : 0<n)
    (hin : x.Valid n inputCapacity) (hout : y.Valid n outputCapacity) (hstride : y.stride≠0)
    (hsep : SignedGrid.FootprintsDisjoint x y n) :
    (ReluIO.kernelIO (fp32Backend rest) Bool.false x y n).Implements (ReluIO.nativeTarget x y n) ∧
      (ReluIO.kernelIO (fp32Backend rest) Bool.false x y n).Equivalent
        (ReluIO.candidate (fp32Backend rest) x y n) :=
  ⟨ReluIO.native_implements rest Bool.false x y n inputCapacity outputCapacity hn hin hout hstride hsep,
   ReluIO.native_equivalent rest x y n inputCapacity outputCapacity hn hin hout hstride hsep⟩

#stmtSurfaceSubset relu_strided_buffer_grid_equiv ⊆
  [FP.Structural.Algebra,FP.Value,FP.Format.fp32,gridWordInhabited,SignedView.View,SignedView.View.Valid,SignedView.View.stride,
   SignedGrid.FootprintsDisjoint,fp32Backend,ReluIO.kernelIO,ReluIO.nativeTarget,ReluIO.candidate,
   ProgramIO.Implements,ProgramIO.Equivalent]
#specNonCircular ReluIO.nativeTarget avoiding [kernel,foldedKernel,FP.Structural.exec,runGrid]
#axiomsClean relu_strided_buffer_grid_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ReluStridedBuffer
