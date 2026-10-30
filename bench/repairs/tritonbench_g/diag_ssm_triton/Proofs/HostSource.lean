import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.Admission
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.BackwardWrapper

/-! Checked host entry and concrete input/return identity. Rejection has no
allocation or kernel effect. The accepted path reuses the full byte/grid proof,
with logical-view admission now an identity check rather than a copy. -/
namespace VeriTile.Bench.Repairs.DiagSsmTriton.HostSource
open Kernels ByteMemory Execution Protocol Tensors Host LibraryContracts HostPaths
set_option autoImplicit false

def forward (api : Services) (kernels : KernelActions)
    (smeta xmeta lmeta : Contract.Metadata) (s x transition : Tensor) : Program State Tensor :=
  match Contract.checkInputs smeta xmeta lmeta with
  | .ok _ => Host.forward api kernels s x transition
  | .error _ => raise .dtype

def backward (api : Services) (kernels : KernelActions)
    (gmeta ymeta : Contract.Metadata) (gy : Tensor) : Program State (Tensor × Tensor × Tensor) :=
  match Contract.checkGradient gmeta ymeta with
  | .ok _ => Host.backward api kernels gy
  | .error _ => raise .dtype

theorem forward_admitted (api : Services) (kernels : KernelActions)
    (smeta xmeta lmeta : Contract.Metadata) (s x transition : Tensor)
    (h : Contract.InputContract smeta xmeta lmeta) :
    forward api kernels smeta xmeta lmeta s x transition = Host.forward api kernels s x transition := by
  simp [forward, (Contract.checkInputs_iff smeta xmeta lmeta).mpr h]

theorem backward_admitted (api : Services) (kernels : KernelActions)
    (gmeta ymeta : Contract.Metadata) (gy : Tensor) (h : Contract.GradientContract gmeta ymeta) :
    backward api kernels gmeta ymeta gy = Host.backward api kernels gy := by
  simp [backward, (Contract.checkGradient_iff gmeta ymeta).mpr h]

theorem forward_original_inputs (codec : Codec Word) (graph : Graph) (ops : Operations Word)
    (api : Services) (choices : Choices) (route : Route) (d : Dimensions)
    (s x transition returned : Tensor) (before after : State)
    (report : ForwardWrapper.Report codec graph ops api choices route d s x transition returned before after) :
    report.path.initial = s ∧ report.path.input = x ∧ report.path.parameter = transition := by
  exact ⟨(normalize_identity api .initial s report.path.initial before report.path.afterInitial
    report.path.normalizeInitial).1,
    (normalize_identity api .input x report.path.input report.path.afterInitial report.path.afterInput
    report.path.normalizeInput).1,
    (normalize_identity api .transition transition report.path.parameter report.path.afterInput report.path.afterParameter
    report.path.normalizeParameter).1⟩

theorem forward_return_metadata (codec : Codec Word) (graph : Graph) (ops : Operations Word)
    (api : Services) (contracts : Contracts codec ops api) (choices : Choices) (route : Route) (d : Dimensions)
    (s x transition returned : Tensor) (before after : State)
    (report : ForwardWrapper.Report codec graph ops api choices route d s x transition returned before after) :
    returned.route = route ∧ returned.shape = [d.length,d.batch,d.dim] ∧ returned.device = x.device ∧
    (0 < returned.count → returned.view.pointer % 4 = 0) := by
  have allocation := (contracts.zeros report.path.input).success report.path.afterParameter
    returned report.path.allocated report.path.allocate
  have inputs := forward_original_inputs codec graph ops api choices route d s x transition returned before after report
  exact ⟨report.outputLayout.dtype,report.outputLayout.shape,
    allocation.device.trans (congrArg Tensor.device inputs.2.1),allocation.aligned⟩

theorem saved_original_inputs (codec : Codec Word) (graph : Graph) (ops : Operations Word)
    (api : Services) (choices : Choices) (route : Route) (d : Dimensions)
    (s x transition returned : Tensor) (before after : State)
    (report : ForwardWrapper.Report codec graph ops api choices route d s x transition returned before after) :
    after.saved = some ⟨s,returned,transition,report.path.launched.heap⟩ := by
  have inputs := forward_original_inputs codec graph ops api choices route d s x transition returned before after report
  simpa only [inputs.1, inputs.2.2] using report.saved

/-- Shapes, source precision and devices of every returned gradient Tensor. -/
theorem backward_return_metadata (codec : Codec Word) (graph : Graph) (ops : Operations Word)
    (api : Services) (contracts : Contracts codec ops api) (choices : Choices) (route : Route) (d : Dimensions)
    (saved : Saved) (gy : Tensor) (returned : Tensor × Tensor × Tensor) (before after : State)
    (context : Admission.SavedContext route d saved before)
    (report : BackwardWrapper.Report codec graph ops api choices route d saved gy returned before after) :
    (returned.1.route = route ∧ returned.1.shape = [d.batch,d.dim] ∧ returned.1.device = saved.initial.device) ∧
    (returned.2.1.route = route ∧ returned.2.1.shape = [d.length,d.batch,d.dim] ∧ returned.2.1.device = gy.device) ∧
    (returned.2.2.route = route ∧ returned.2.2.shape = [d.dim] ∧ returned.2.2.device = saved.initial.device) := by
  let path := report.path
  obtain ⟨restored⟩ := contracts.restore.success before path.restored path.afterRestore path.restore
  have savedEq : restored.saved = saved := Option.some.inj (restored.present.symm.trans context.present)
  have initialDevice : path.restored.1.device = saved.initial.device := by
    simpa [savedEq] using restored.initialDevice
  have initialShape : path.restored.1.shape = [d.batch,d.dim] :=
    report.restoredInitial.2.2.2.2.1.trans context.initial.shape
  have initialRoute : path.restored.1.route = route :=
    report.restoredInitial.2.2.2.1.trans context.initial.dtype
  have allocS := (contracts.empty path.restored.1).success path.afterContiguous path.gs path.afterS path.allocateS
  have allocX := (contracts.empty path.contiguousGradient).success path.afterS path.gx path.afterX path.allocateX
  have allocL := (contracts.empty path.restored.1).success path.afterX path.gl path.afterL path.allocateL
  have cont := (contracts.contiguous path.normalized).success path.afterNormalize path.contiguousGradient
    path.afterContiguous path.contiguous
  have gradientEq := (normalize_identity api .gradOutput gy path.normalized
    path.afterRestore path.afterNormalize path.normalize).1
  refine ⟨?_, ?_, ?_⟩
  · rw [report.initialIdentity]
    exact ⟨allocS.same.1.trans initialRoute, allocS.same.2.trans initialShape,
      allocS.device.trans initialDevice⟩
  · rw [report.inputIdentity]
    exact ⟨allocX.same.1.trans report.gradientLayout.dtype,
      allocX.same.2.trans report.gradientLayout.shape,
      allocX.device.trans (cont.1.device.trans (congrArg Tensor.device gradientEq))⟩
  · exact ⟨report.sum.route.trans report.sumRoute, by simpa [report.sumDim] using report.sum.shape,
      report.sum.device.trans (allocL.device.trans initialDevice)⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DiagSsmTriton.HostSource
