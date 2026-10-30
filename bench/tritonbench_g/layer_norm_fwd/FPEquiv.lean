import bench.tritonbench_g.layer_norm_fwd.Proofs.GridPair

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Bench.Repairs.LayerNormFwd
open GridPair
open scoped VeriTile.Spec
noncomputable section

/-- Complete ordinary-grid equivalence of the actual FP32 sources: commute
only X's initial integer row/stride product. All inputs, flags, casts, ordered
FP operations, Philox streams, outputs, access safety and frames are retained,
including empty grids and zero-column statistics. No numerical rule is used. -/
specification index_commuted_equiv {α : Type} [Inhabited α] (A : Algebra α)
    (cmp : TypedSource.Comparisons A) (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (width : widthCheck N = Bool.true)
    (sep : Domain.ReadWriteSeparated f r)
    (geometry : NativeGeometry.Geometry f r stride M N) :
    original A f r stride M N eps p ≡[[]] indexCommuted A f r stride M N eps p :=
  OrdinaryEquivalence.Implementation.of_equivalent _ _
    (GridPair.equivalent A cmp f r stride M N eps p width sep geometry)

#axiomsClean index_commuted_equiv
#print_fp_assumptions index_commuted_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile
