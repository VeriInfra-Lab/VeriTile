import bench.tritonbench_g.layer_norm_fwd.OrdinaryIO

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.OrdinaryEquivalence
open VeriTile.Triton FP.Structural
noncomputable section
namespace Implementation
variable {α β : Type} {io : OrdinaryProgramIO α β}

theorem of_equivalent [Inhabited β] (left right : Implementation io)
    (proof : left.Equivalent right) :
    Spec.Equiv left right ([] : Spec.Assumptions (Implementation io)) :=
  Spec.FloatingPoint.ofStructural rfl rfl proof

#axiomsClean of_equivalent
end Implementation
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.OrdinaryEquivalence
