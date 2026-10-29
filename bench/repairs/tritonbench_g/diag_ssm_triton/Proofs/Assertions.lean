import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.Host

namespace VeriTile.Bench.Repairs.DiagSsmTriton.Assertions
open Kernels Tensors Host Protocol
set_option autoImplicit false

theorem input_contiguity_failure (api : Services) (kernels : KernelActions) (s x transition : Tensor)
    (before : State) (good : s.contiguous = Bool.true) (bad : x.contiguous = Bool.false) :
    Host.forward api kernels s x transition before =
      .failure .assertion before [.contiguousCheck .initial, .contiguousCheck .input] := by
  simp [Host.forward, requireContiguous, good, bad, call, Protocol.bind, Protocol.pure, raise, Result.prepend]

theorem transition_contiguity_failure (api : Services) (kernels : KernelActions) (s x transition : Tensor)
    (before : State) (sgood : s.contiguous = Bool.true) (xgood : x.contiguous = Bool.true)
    (bad : transition.contiguous = Bool.false) :
    Host.forward api kernels s x transition before =
      .failure .assertion before [.contiguousCheck .initial, .contiguousCheck .input, .contiguousCheck .transition] := by
  simp [Host.forward, requireContiguous, sgood, xgood, bad, call, Protocol.bind, Protocol.pure, raise, Result.prepend]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DiagSsmTriton.Assertions
