import bench.tritonbench_g.rbe_triton_transform.Proofs.Pair

namespace VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch.FPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RbeTritonTransform
open scoped VeriTile.Spec
noncomputable section

/-- Complete FP32 source pair: reuse existing input masks for output stores.
Every floating operation, cast, barrier and two-axis launch remains. The full
input/output windows, odd tails, arbitrary complete legal schedule, successful
execution, actual source safety and both exterior frames are preserved. -/
specification mask_reuse_equiv {α β : Type} [Inhabited α] [Inhabited β]
    (ops : Operations β) (r : Request) (X Y : RegionName) (different : X≠Y)
    (qs : List Runtime.Program) (all : qs.Perm (Runtime.programs r))
    (encode : α → β) (observe : β → α) :
    Pair.original ops r X Y qs encode observe ≡[[]] Pair.reuseMasks ops r X Y qs encode observe :=
  ProgramIO.Implementation.of_equivalent (Pair.variant ops r X Y qs encode observe)
    (Pair.equivalent ops r X Y different qs all encode observe)

#axiomsClean mask_reuse_equiv
end
end VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch.FPEquiv
