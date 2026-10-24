import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.ReadableSource
open Lean VeriTile Triton

/-- Complete source functions for all arguments and both source branches. -/
theorem forward_noninner_preserved :
    forward_noninner=SoftmaxFlaggemsFiniteSeed.Source.forward_noninner := rfl
theorem forward_inner_preserved :
    forward_inner=SoftmaxFlaggemsFiniteSeed.Source.forward_inner := rfl
theorem backward_noninner_preserved :
    backward_noninner=SoftmaxFlaggemsFiniteSeed.Source.softmax_backward_kernel_non_inner_fp32 := rfl
theorem backward_inner_preserved :
    backward_inner=SoftmaxFlaggemsFiniteSeed.Source.softmax_backward_kernel_inner_fp32 := rfl

/-- Equality to exactly the pre-existing candidate macro expansions, without
changing the retained generic legacy source or creating a new candidate. -/
theorem candidate_forward_noninner_preserved : AddressCandidate.forward_noninner=
    finiteSeedFlaggems0%[tl.float32, tl.float32, address_commuted] := rfl
theorem candidate_forward_inner_preserved : AddressCandidate.forward_inner=
    finiteSeedFlaggems1%[tl.float32, tl.float32, address_commuted] := rfl
theorem candidate_backward_noninner_preserved : AddressCandidate.backward_noninner=
    finiteSeedFlaggems2%[tl.float32, tl.float32, address_commuted] := rfl
theorem candidate_backward_inner_preserved : AddressCandidate.backward_inner=
    finiteSeedFlaggems3%[tl.float32, tl.float32, address_commuted] := rfl

#axiomsClean forward_noninner_preserved
#axiomsClean forward_inner_preserved
#axiomsClean backward_noninner_preserved
#axiomsClean backward_inner_preserved
#axiomsClean candidate_forward_noninner_preserved
#axiomsClean candidate_forward_inner_preserved
#axiomsClean candidate_backward_noninner_preserved
#axiomsClean candidate_backward_inner_preserved
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.ReadableSource
