import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.PairIO

namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.FPEquiv
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.LightningAttention.Native (Instructions Plan)
open scoped VeriTile.Spec
noncomputable section

/-- Complete actual forward schedules, including overlapping padded windows
and every per-CTA materialization. Only the duplicate integer base-offset
calculation is replaced by reuse of v_offset. No numerical rule is used. -/
specification forward_equiv (d : Host.Dimensions) (ops : Instructions) (plan : Plan) :
    PairIO.Forward.original ops plan d ≡[[]] PairIO.Forward.candidate ops plan d :=
  Spec.FloatingPoint.ofStructural rfl rfl (PairIO.Forward.equivalent ops plan d)

/-- Both complete backward grids and actual intra-to-inter dataflow, arbitrary
original dimensions/words/ISA operations and reduction plans, every output cell,
source safety/success and both complete exterior frames. -/
specification backward_equiv (d : Host.Dimensions) (ops : Instructions) (plan : Plan) :
    PairIO.Backward.original ops plan d ≡[[]] PairIO.Backward.candidate ops plan d :=
  Spec.FloatingPoint.ofStructural rfl rfl (PairIO.Backward.equivalent ops plan d)

#axiomsClean forward_equiv
#axiomsClean backward_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.FPEquiv
