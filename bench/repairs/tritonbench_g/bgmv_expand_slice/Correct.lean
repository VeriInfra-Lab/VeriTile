import bench.repairs.tritonbench_g.bgmv_expand_slice.Kernels
import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.Direct.CheckedReal
import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.Direct.Bounds

namespace VeriTile.Bench.Repairs.BgmvExpandSlice.Correct
open VeriTile.Triton Direct
noncomputable section

/-- Complete direct source grid, independent K-term GEMV, and exact logical
frame from executable wrapper checks and local metadata-load observations. -/
theorem checked_real (x w y i : Contract.TensorInfo) (scalars : Contract.Scalars) (indices : Nat → Int)
    (ok : Contract.Admitted x w y i scalars indices) (initial : BlockState)
    (metadata : ∀ b,b < i.numel → initial.readMemValue .int "Indices" b=indices b) :
    let h:=Contract.call x w y i scalars indices ok
    ∃ final,RealGrid.run h (Direct.Geometry.programs h) initial=some final ∧
      (∀ p : Direct.Geometry.Program h,indices p.batch≠-1 →
        final.readMem "Y" (Direct.Geometry.outputOffset h p.batch p.row 0)=
          RealExecution.formula h p initial (indices p.batch).toNat) ∧
      (∀ r off,¬RealGrid.Writes h indices (Direct.Geometry.programs h) r off → final.mem r off=initial.mem r off) :=
  CheckedReal.complete x w y i scalars indices ok initial metadata

#axiomsClean checked_real
#axiomsClean Bounds.active
#axiomsClean Bounds.formed
#axiomsClean Bounds.signed_offsets
#specNonCircular RealExecution.formula avoiding [RealGrid.run,VeriTile.Triton.exec,Direct.Kernels.source]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BgmvExpandSlice.Correct
