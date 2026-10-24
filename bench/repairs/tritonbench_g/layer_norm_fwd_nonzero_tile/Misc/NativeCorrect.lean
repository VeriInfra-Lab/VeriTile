import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.NativeIO
import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.NativeImplementation

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Bench.Repairs.LayerNormFwd
noncomputable section

/-- The same complete ordinary source grid implements its independent ordered
primitive reference, including casts, masked padding and both dropout streams.
The algebra/comparison boundary is explicit; no FP=Real or optimization law is
asserted. Native compiler-artifact and physical-store results are preserved in
 Misc/RawNativeResults.lean and their Tensor applications in Misc/TensorBridge.lean. -/
theorem quoted_fp {α : Type} [Inhabited α] (A : Algebra α) (cmp : TypedSource.Comparisons A)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat) (eps p : ℝ)
    (width : widthCheck N=Bool.true) (sep : Domain.ReadWriteSeparated f r)
    (geometry : NativeGeometry.Geometry f r stride M N) :
    (NativeInterface.io A f r stride M N eps p).Implements
      (NativeInterface.target A cmp f r stride M N eps p ((widthCheck_iff N).mp width)) :=
  NativeInterface.implements A cmp f r stride M N eps p width sep geometry

#axiomsClean quoted_fp
#stmtSurfaceSubset quoted_fp ⊆ [OrdinaryProgramIO.Implements,
  NativeInterface.io,NativeInterface.target,FP.Structural.Algebra,TypedSource.Comparisons,
  Flags,Regions,Strides,widthCheck,widthCheck_iff,tile,
  Domain.ReadWriteSeparated,NativeGeometry.Geometry]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile
