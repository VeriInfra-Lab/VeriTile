import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.BackwardWrapper

/-! All three restored tensors are bound to the preceding forward. The saved
initial state and parameter retain their normalized logical input values, and
y retains the rounded forward output. If the caller leaves the saved backings
unchanged between invocations, backward loads precisely those words. Mutation/version failures remain
explicit in LibraryContracts.invalid_saved_version; no ideal-y substitution is
made by the backward interpreter. -/
namespace VeriTile.Bench.Repairs.DiagSsmTriton.SavedComposition
open Kernels ByteMemory Execution Protocol Tensors Host LibraryContracts HostPaths Dispatch
set_option autoImplicit false

/-- Local composition of normalization, retained saved storage and a backing
frame. This uses no kernel-result assumption. -/
theorem restored_normalized_word (codec : Codec Word) (flip : Word → Word)
    (caller normalized restored : Tensor) (before normalizedHeap restoredHeap : Heap)
    (logical : SameLogical codec flip caller normalized before normalizedHeap)
    (dense : Dense normalized) (resolved : Resolved normalized)
    (same : SameSavedStorage normalized restored)
    (frame : Frame normalizedHeap restoredHeap normalized.view.backing)
    (q : Nat) (hq : q < caller.count) (k : Nat) (hk : k < caller.route.components) :
    load codec restoredHeap (restored.view.address (normalized.route.components * q + k)) =
      logicalWord codec flip caller before q k := by
  have hqn : q < normalized.count := by simpa only [Tensor.count, logical.1.2] using hq
  have hkn : k < normalized.route.components := by simpa only [logical.1.1] using hk
  have bound := normalized.bounded q hqn k hkn
  rw [dense.2.2 q hqn] at bound
  rw [same.2.2.1, Frame.load codec normalized.view frame _ bound]
  exact normalized_load codec flip caller normalized before normalizedHeap logical dense resolved q hq k hk

theorem saved_initial_word (codec : Codec Word) (graph : Graph) (ops : Operations Word)
    (api : Services) (choices : Choices) (route : Route) (d : Dimensions)
    (s x transition y gy : Tensor) (gradients : Tensor × Tensor × Tensor)
    (forwardBefore forwardAfter backwardBefore backwardAfter : State) (saved : Saved)
    (forward : ForwardWrapper.Report codec graph ops api choices route d s x transition y forwardBefore forwardAfter)
    (backward : BackwardWrapper.Report codec graph ops api choices route d saved gy gradients backwardBefore backwardAfter)
    (sameContext : backwardBefore.saved = forwardAfter.saved) (present : backwardBefore.saved = some saved)
    (between : Frame forwardAfter.heap backwardBefore.heap forward.path.initial.view.backing)
    (q : Nat) (hq : q < s.count) (k : Nat) (hk : k < s.route.components) :
    load codec backward.dispatch.start.heap (backward.path.restored.1.view.address (route.components * q + k)) =
      logicalWord codec ops.flipSign s forwardBefore.heap q k := by
  have savedEq : saved = ⟨forward.path.initial, y, forward.path.parameter, forward.path.launched.heap⟩ :=
    Option.some.inj (present.symm.trans (sameContext.trans forward.saved))
  have initialEq : saved.initial = forward.path.initial := congrArg Saved.initial savedEq
  have same : SameSavedStorage forward.path.initial backward.path.restored.1 := by
    simpa [initialEq] using backward.restoredInitial
  have during : Frame backwardBefore.heap backward.dispatch.start.heap forward.path.initial.view.backing := by
    simpa [initialEq] using backward.savedInitialFrame
  simpa only [forward.initialLayout.dtype] using restored_normalized_word codec ops.flipSign
    s forward.path.initial backward.path.restored.1 forwardBefore.heap forward.dispatch.start.heap backward.dispatch.start.heap
    forward.initialMeaning forward.initialLayout.dense forward.initialResolved same
    (forward.initialFrame.trans (between.trans during)) q hq k hk

theorem saved_parameter_word (codec : Codec Word) (graph : Graph) (ops : Operations Word)
    (api : Services) (choices : Choices) (route : Route) (d : Dimensions)
    (s x transition y gy : Tensor) (gradients : Tensor × Tensor × Tensor)
    (forwardBefore forwardAfter backwardBefore backwardAfter : State) (saved : Saved)
    (forward : ForwardWrapper.Report codec graph ops api choices route d s x transition y forwardBefore forwardAfter)
    (backward : BackwardWrapper.Report codec graph ops api choices route d saved gy gradients backwardBefore backwardAfter)
    (sameContext : backwardBefore.saved = forwardAfter.saved) (present : backwardBefore.saved = some saved)
    (between : Frame forwardAfter.heap backwardBefore.heap forward.path.parameter.view.backing)
    (q : Nat) (hq : q < transition.count) (k : Nat) (hk : k < transition.route.components) :
    load codec backward.dispatch.start.heap (backward.path.restored.2.2.view.address (route.components * q + k)) =
      logicalWord codec ops.flipSign transition forwardBefore.heap q k := by
  have savedEq : saved = ⟨forward.path.initial, y, forward.path.parameter, forward.path.launched.heap⟩ :=
    Option.some.inj (present.symm.trans (sameContext.trans forward.saved))
  have parameterEq : saved.transition = forward.path.parameter := congrArg Saved.transition savedEq
  have same : SameSavedStorage forward.path.parameter backward.path.restored.2.2 := by
    simpa [parameterEq] using backward.restoredParameter
  have during : Frame backwardBefore.heap backward.dispatch.start.heap forward.path.parameter.view.backing := by
    simpa [parameterEq] using backward.savedParameterFrame
  simpa only [forward.parameterLayout.dtype] using restored_normalized_word codec ops.flipSign
    transition forward.path.parameter backward.path.restored.2.2 forwardBefore.heap forward.dispatch.start.heap backward.dispatch.start.heap
    forward.parameterMeaning forward.parameterLayout.dense forward.parameterResolved same
    (forward.parameterFrame.trans (between.trans during)) q hq k hk

theorem saved_output_word (codec : Codec Word) (graph : Graph) (ops : Operations Word)
    (api : Services) (choices : Choices) (route : Route) (d : Dimensions)
    (s x transition y gy : Tensor) (gradients : Tensor × Tensor × Tensor)
    (forwardBefore forwardAfter backwardBefore backwardAfter : State) (saved : Saved)
    (forward : ForwardWrapper.Report codec graph ops api choices route d s x transition y forwardBefore forwardAfter)
    (backward : BackwardWrapper.Report codec graph ops api choices route d saved gy gradients backwardBefore backwardAfter)
    (sameContext : backwardBefore.saved = forwardAfter.saved) (present : backwardBefore.saved = some saved)
    (between : Frame forwardAfter.heap backwardBefore.heap y.view.backing)
    (t : Nat) (ht : t < d.length) (pid : Nat) (j : Lane) (hj : active d.width pid j)
    (k : Nat) (hk : k ∈ componentList route) :
    load codec backward.dispatch.start.heap
      (backward.path.restored.2.1.view.address (route.components * timeIndex d t pid j + k)) =
      getComponent (forwardValues codec route graph ops d
        (forwardViews forward.path.initial forward.path.input forward.path.parameter y)
        pid forward.dispatch.start.heap (t + 1) j) k := by
  have savedEq : saved = ⟨forward.path.initial, y, forward.path.parameter, forward.path.launched.heap⟩ :=
    Option.some.inj (present.symm.trans (sameContext.trans forward.saved))
  have outputEq : saved.output = y := congrArg Saved.output savedEq
  have viewEq : backward.path.restored.2.1.view = y.view := by
    simpa [outputEq] using backward.restoredOutput.2.2.1
  have during : Frame backwardBefore.heap backward.dispatch.start.heap y.view.backing := by
    simpa [outputEq] using backward.savedOutputFrame
  have size := forward.sizes.output
  have bound : route.components * timeIndex d t pid j + k < y.view.words := by
    have fits := time_fits route y.view d t pid size ht j hj
    have kc := component_member route k hk
    omega
  rw [viewEq, Frame.load codec y.view (between.trans during) _ bound]
  exact forward.readback t ht pid j hj k hk

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DiagSsmTriton.SavedComposition
