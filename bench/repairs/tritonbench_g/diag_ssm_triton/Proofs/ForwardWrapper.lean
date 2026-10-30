import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.Dispatch
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.TensorLaws

/-! Whole successful forward wrapper. Normalization, allocation, dispatch,
source grid, saved rounded y, actual returned storage and every caller byte
are composed from ordinary local contracts and checked source execution. -/
namespace VeriTile.Bench.Repairs.DiagSsmTriton.ForwardWrapper
open Kernels ByteMemory Execution Protocol Tensors Host LibraryContracts HostPaths Dispatch TensorLaws
set_option autoImplicit false
set_option maxHeartbeats 2000000

structure Report (codec : Codec Word) (graph : Graph) (ops : Operations Word) (api : Services)
    (choices : Choices) (route : Route) (d : Dimensions)
    (s x transition returned : Tensor) (before after : State) where
  path : ForwardPath api (implementation codec graph ops choices) d s x transition returned before after
  dispatch : ForwardResult codec graph ops route d (forwardViews path.initial path.input path.parameter returned)
    path.allocated path.launched
  initialLayout : Layout route [d.batch, d.dim] path.initial
  inputLayout : Layout route [d.length, d.batch, d.dim] path.input
  parameterLayout : Layout route [d.dim] path.parameter
  outputLayout : Layout route [d.length, d.batch, d.dim] returned
  initialResolved : Resolved path.initial
  inputResolved : Resolved path.input
  parameterResolved : Resolved path.parameter
  outputResolved : Resolved returned
  initialMeaning : SameLogical codec ops.flipSign s path.initial before.heap dispatch.start.heap
  inputMeaning : SameLogical codec ops.flipSign x path.input before.heap dispatch.start.heap
  parameterMeaning : SameLogical codec ops.flipSign transition path.parameter before.heap dispatch.start.heap
  sizes : ForwardSizes route d (forwardViews path.initial path.input path.parameter returned)
  fresh : ForwardFresh (forwardViews path.initial path.input path.parameter returned)
  resident : Resident after returned
  saved : after.saved = some ⟨path.initial, returned, path.parameter, path.launched.heap⟩
  callerFrame : ∀ backing ∈ before.live, Frame before.heap after.heap backing
  readback : ∀ t, t < d.length → ∀ owner (j : Lane), active d.width owner j →
    ∀ k, k ∈ componentList route →
      load codec after.heap (returned.view.address (route.components * timeIndex d t owner j + k)) =
        getComponent (forwardValues codec route graph ops d
          (forwardViews path.initial path.input path.parameter returned) owner dispatch.start.heap (t + 1) j) k
  initialFrame : Frame dispatch.start.heap after.heap path.initial.view.backing
  parameterFrame : Frame dispatch.start.heap after.heap path.parameter.view.backing
  initialResident : Resident after path.initial
  parameterResident : Resident after path.parameter

/-- Read-only inputs may alias each other and use nonzero offsets. No finite
value predicate or small-shape restriction is part of this exact word theorem. -/
theorem success (codec : Codec Word) (graph : Graph) (ops : Operations Word) (api : Services)
    (contracts : Contracts codec ops api) (choices : Choices) (route : Route) (d : Dimensions)
    (s x transition returned : Tensor) (before after : State)
    (sl : Layout route [d.batch, d.dim] s) (xl : Layout route [d.length, d.batch, d.dim] x)
    (ll : Layout route [d.dim] transition)
    (sLive : Resident before s) (xLive : Resident before x) (lLive : Resident before transition)
    (run : Succeeded (Host.forward api (implementation codec graph ops choices) s x transition) before returned after) :
    Nonempty (Report codec graph ops api choices route d s x transition returned before after) := by
  obtain ⟨path⟩ := forward_path api (implementation codec graph ops choices) d s x transition returned before after
    sl.dense xl.dense ll.dense xl.shape run
  have sn := normalize_success codec ops api contracts .initial s path.initial before path.afterInitial sl.dense sLive path.normalizeInitial
  have xn := normalize_success codec ops api contracts .input x path.input path.afterInitial path.afterInput xl.dense (sn.1.resident xLive) path.normalizeInput
  have ln := normalize_success codec ops api contracts .transition transition path.parameter path.afterInput path.afterParameter ll.dense ((sn.1.trans xn.1).resident lLive) path.normalizeParameter
  have alloc := (contracts.zeros path.input).success path.afterParameter returned path.allocated path.allocate
  have allocExt := (contracts.zeros path.input).extension path.allocate
  have snLayout := sl.transport sn.2.1.1 sn.2.2.1
  have xnLayout := xl.transport xn.2.1.1 xn.2.2.1
  have lnLayout := ll.transport ln.2.1.1 ln.2.2.1
  have yLayout := xnLayout.transport alloc.same alloc.dense
  obtain ⟨dispatch⟩ := Dispatch.forward_success codec graph ops api contracts choices route d
    path.initial path.input path.parameter returned path.allocated path.launched
    snLayout.dtype xnLayout.dtype lnLayout.dtype path.dispatch
  have sToStart := ((xn.1.trans ln.1).trans allocExt).trans dispatch.extension
  have xToStart := (ln.1.trans allocExt).trans dispatch.extension
  have lToStart := allocExt.trans dispatch.extension
  have beginning := sn.1.trans xn.1
  have setup := ((beginning.trans ln.1).trans allocExt).trans dispatch.extension
  have sMeaning := TensorLaws.SameLogical.reframe sn.2.1 (Frame.refl before.heap s.view.backing)
    (sToStart.frame _ sn.2.2.2.2)
  have xMeaning := TensorLaws.SameLogical.reframe xn.2.1 (sn.1.frame _ xLive)
    (xToStart.frame _ xn.2.2.2.2)
  have lMeaning := TensorLaws.SameLogical.reframe ln.2.1 (beginning.frame _ lLive)
    (lToStart.frame _ ln.2.2.2.2)
  have sizes := forward_sizes route d path.initial path.input path.parameter returned snLayout xnLayout lnLayout yLayout
  have fresh : ForwardFresh (forwardViews path.initial path.input path.parameter returned) :=
    ⟨alloc.fresh _ ((xn.1.trans ln.1).resident sn.2.2.2.2),
      alloc.fresh _ (ln.1.resident xn.2.2.2.2), alloc.fresh _ ln.2.2.2.2⟩
  have yStart := dispatch.extension.resident alloc.resident
  have yLaunched : Resident path.launched returned := by simpa [dispatch.actual] using yStart
  have saved := (contracts.save path.initial returned path.parameter).success path.launched () after path.save
  have saveExt := (contracts.save path.initial returned path.parameter).extension path.save
  have savedInput (tensor : Tensor) (live : Resident dispatch.start tensor)
      (separate : returned.view.backing.disjoint tensor.view.backing) :
      Frame dispatch.start.heap after.heap tensor.view.backing ∧ Resident after tensor := by
    have liveLaunched : Resident path.launched tensor := by simpa [dispatch.actual] using live
    have kernelFrame : Frame dispatch.start.heap path.launched.heap tensor.view.backing := by
      simpa [dispatch.actual] using forwardGrid_caller_frame codec route graph ops d
        (forwardViews path.initial path.input path.parameter returned) dispatch.schedule.order
        dispatch.start.heap sizes tensor.view.backing separate
    exact ⟨kernelFrame.trans (saveExt.frame _ liveLaunched), saveExt.resident liveLaunched⟩
  have savedInitial := savedInput path.initial (sToStart.resident sn.2.2.2.2) fresh.initial
  have savedParameter := savedInput path.parameter (lToStart.resident ln.2.2.2.2) fresh.transition
  refine ⟨⟨path, dispatch, snLayout, xnLayout, lnLayout, yLayout,
    sn.2.2.2.1, xn.2.2.2.1, ln.2.2.2.1, alloc.resolved,
    sMeaning, xMeaning, lMeaning, sizes, fresh, saveExt.resident yLaunched, saved.saved, ?_, ?_,
    savedInitial.1, savedParameter.1, savedInitial.2, savedParameter.2⟩⟩
  · intro backing member
    have freshCaller := alloc.fresh backing ((beginning.trans ln.1).live backing member)
    have kernelFrame := forwardGrid_caller_frame codec route graph ops d
      (forwardViews path.initial path.input path.parameter returned) dispatch.schedule.order dispatch.start.heap
      sizes backing freshCaller
    have kernelFrame : Frame dispatch.start.heap path.launched.heap backing := by
      simpa [dispatch.actual] using kernelFrame
    have liveKernel : backing ∈ path.launched.live := by
      simpa [dispatch.actual] using setup.live backing member
    exact (setup.frame backing member).trans (kernelFrame.trans (saveExt.frame backing liveKernel))
  · intro t ht owner j hj k hk
    have bound : route.components * timeIndex d t owner j + k < returned.view.words := by
      have fits := time_fits route returned.view d t owner sizes.output ht j hj
      have kc := component_member route k hk
      omega
    rw [Frame.load codec returned.view (saveExt.frame _ yLaunched) _ bound]
    have heapEq := congrArg State.heap dispatch.actual
    dsimp only at heapEq
    rw [heapEq]
    exact forwardGrid_output codec route graph ops d
      (forwardViews path.initial path.input path.parameter returned) dispatch.schedule.order owner t
      dispatch.start.heap sizes fresh (SourceActions.owner_in_schedule d dispatch.schedule owner j hj) ht j hj k hk

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DiagSsmTriton.ForwardWrapper
