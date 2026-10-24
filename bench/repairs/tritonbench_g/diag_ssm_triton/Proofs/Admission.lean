import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.InputBinding
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.PublicIO

/-! Optional composition of kernel admission with the historical host API. -/
namespace VeriTile.Bench.Repairs.DiagSsmTriton.Admission
open Kernels Tensors Execution Host LibraryContracts TensorLaws
set_option autoImplicit false

theorem forward_domain (smeta xmeta lmeta : Contract.Metadata) (s x transition : Tensor)
    (before : State) (h : Contract.InputContract smeta xmeta lmeta)
    (sb : Binding smeta s) (xb : Binding xmeta x) (lb : Binding lmeta transition)
    (slive : Resident before s) (xlive : Resident before x) (llive : Resident before transition) :
    PublicIO.Forward x.route (dimensions xmeta) s x transition before := by
  obtain ⟨sl,xl,ll⟩ := layouts smeta xmeta lmeta s x transition h sb xb lb
  exact ⟨sl,xl,ll,slive,xlive,llive⟩

/-- The saved context invariant supplied by the already completed forward.
It concerns saved descriptor shapes and ownership, not the cotangent domain. -/
structure SavedContext (route : Route) (d : Dimensions) (saved : Saved) (before : State) : Prop where
  present : before.saved = some saved
  initial : Layout route [d.batch, d.dim] saved.initial
  output : Layout route [d.length, d.batch, d.dim] saved.output
  parameter : Layout route [d.dim] saved.transition
  initialLive : Resident before saved.initial
  outputLive : Resident before saved.output
  parameterLive : Resident before saved.transition

theorem backward_domain (gmeta ymeta : Contract.Metadata) (route : Route) (d : Dimensions)
    (saved : Saved) (gy : Tensor) (before : State) (h : Contract.GradientContract gmeta ymeta)
    (gb : Binding gmeta gy) (yb : Binding ymeta saved.output)
    (context : SavedContext route d saved before) (glive : Resident before gy) :
    PublicIO.Backward route d saved gy before := by
  have routeEq := dtype_injective (gb.dtype.symm.trans (h.dtype.trans yb.dtype))
  exact ⟨context.present,context.initial,context.output,context.parameter,
    routeEq.trans context.output.dtype,
    gb.shape.symm.trans (h.shape.trans (yb.shape.trans context.output.shape)),
    glive,context.outputLive,context.initialLive,context.parameterLive⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DiagSsmTriton.Admission
