import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeSafetyCapacity
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeSafetyInterComplete
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeSafetyDiagonalComplete
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeSafetyMergeComplete
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeSafetyOutputComplete

/-! Actual source trace safety for every invocation in the five original stage
launches. Buffer extents and grid coordinates imply all access obligations;
there is no numerical, initialization, or typed-cell hypothesis. -/
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativeSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite
noncomputable section
variable {α : Type} [Inhabited α]

/-- All loads and stores in the reached source trace lie within the supplied
finite buffer bounds. The invocation preserves the incoming memory and `undef`;
only launch registers and program coordinates are installed by `initial`. -/
theorem source_safe (M : Algebra α) (D : FPOutput.DotPrimitive α)
    (stage : FPStages.Stage) (idx : GridIndex stage.grid)
    (bounds : RegionName → Nat) (hb : NativeLaunch.Bounds stage bounds)
    (s : State α) : (NativeLaunch.job M D stage idx).Safe bounds s := by
  change TraceSafe (engine (stage.algebra M D)) bounds stage.kernel
    ((NativeLaunch.job M D stage idx).initial s)
  rw [NativeLaunch.initial_eq]
  cases stage with
  | inter c scale =>
    exact inter_source M D c scale bounds _ (inter_capacity c scale idx bounds hb s)
  | intra c BK scale =>
    exact diagonal_source M Bool.false (FPStages.diagonalLayout c BK) scale bounds _
      (intra_capacity c BK scale idx bounds hb s)
  | split c scale =>
    exact diagonal_source M Bool.true (FPStages.diagonalLayout c c.BK) scale bounds _
      (split_capacity c scale idx bounds hb s)
  | merge c =>
    exact merge_source M (FPStages.mergeLayout c) bounds _ (merge_capacity c idx bounds hb s)
  | output c scale =>
    exact output_source M D c scale bounds _ (output_capacity c scale idx bounds hb s)

#axiomsClean source_safe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativeSafety
