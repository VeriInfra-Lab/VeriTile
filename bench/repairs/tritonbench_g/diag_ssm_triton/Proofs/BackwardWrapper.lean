import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.ForwardWrapper
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.BatchContributions

/-! Whole successful backward wrapper. Restored saved descriptors, arbitrary
grad_y layout, three fresh allocations, source-grid readback, actual batch sum,
returned object identities and full caller byte frames are composed here. -/
namespace VeriTile.Bench.Repairs.DiagSsmTriton.BackwardWrapper
open Kernels ByteMemory Execution Protocol Tensors Host LibraryContracts HostPaths Dispatch TensorLaws
set_option autoImplicit false
set_option maxHeartbeats 3000000

structure Report (codec : Codec Word) (graph : Graph) (ops : Operations Word) (api : Services)
    (choices : Choices) (route : Route) (d : Dimensions) (saved : Saved)
    (gy : Tensor) (returned : Tensor × Tensor × Tensor) (before after : State) where
  path : BackwardPath api (implementation codec graph ops choices) gy returned before after
  dimensions : path.d = d
  restoredInitial : SameSavedStorage saved.initial path.restored.1
  restoredOutput : SameSavedStorage saved.output path.restored.2.1
  restoredParameter : SameSavedStorage saved.transition path.restored.2.2
  dispatch : BackwardResult codec graph ops route d
    (backwardViews path.restored.1 path.restored.2.1 path.restored.2.2 path.contiguousGradient path.gs path.gx path.gl)
    path.afterL path.launched
  gradientLayout : Layout route [d.length, d.batch, d.dim] path.contiguousGradient
  gradientResolved : Resolved path.contiguousGradient
  gradientMeaning : SameLogical codec ops.flipSign gy path.contiguousGradient before.heap dispatch.start.heap
  sizes : BackwardSizes route d
    (backwardViews path.restored.1 path.restored.2.1 path.restored.2.2 path.contiguousGradient path.gs path.gx path.gl)
  fresh : BackwardAllFresh
    (backwardViews path.restored.1 path.restored.2.1 path.restored.2.2 path.contiguousGradient path.gs path.gx path.gl)
  sum : SumReply ops codec path.gl returned.2.2 path.launched after
  sumBatch : sum.batch = d.batch
  sumDim : sum.dim = d.dim
  sumRoute : path.gl.route = route
  initialIdentity : returned.1 = path.gs
  inputIdentity : returned.2.1 = path.gx
  callerFrame : ∀ backing ∈ before.live, Frame before.heap after.heap backing
  savedOutputFrame : Frame before.heap dispatch.start.heap saved.output.view.backing
  savedInitialFrame : Frame before.heap dispatch.start.heap saved.initial.view.backing
  savedParameterFrame : Frame before.heap dispatch.start.heap saved.transition.view.backing
  initialReadback : ∀ owner (j : Lane), active d.width owner j → ∀ k, k ∈ componentList route →
    load codec after.heap (returned.1.view.address (route.components * columnIndex owner j + k)) =
      getComponent (terminalGradient route ops (backwardValues codec route graph ops d
        (backwardViews path.restored.1 path.restored.2.1 path.restored.2.2 path.contiguousGradient path.gs path.gx path.gl)
        owner dispatch.start.heap d.length j).carry) k
  inputReadback : ∀ i, i < d.length → ∀ owner (j : Lane), active d.width owner j → ∀ k, k ∈ componentList route →
    load codec after.heap (returned.2.1.view.address (route.components * timeIndex d (d.length - 1 - i) owner j + k)) =
      getComponent (backwardInputValues codec route graph ops d
        (backwardViews path.restored.1 path.restored.2.1 path.restored.2.2 path.contiguousGradient path.gs path.gx path.gl)
        owner dispatch.start.heap i j) k
  transitionReadback : ∀ q, q < route.components * d.dim →
    load codec after.heap (returned.2.2.view.address q) = (sum.plan.tree q).evaluate ops.zero ops.add
      (BatchContributions.value codec route graph ops d
        (backwardViews path.restored.1 path.restored.2.1 path.restored.2.2 path.contiguousGradient path.gs path.gx path.gl)
        dispatch.start.heap q)

theorem restored_layout (route : Route) (shape : List Nat) (saved restored : Tensor)
    (layout : Layout route shape saved) (same : SameSavedStorage saved restored)
    (dense : Dense restored) : Layout route shape restored :=
  ⟨same.2.2.2.1.trans layout.dtype, same.2.2.2.2.1.trans layout.shape, dense⟩

theorem dimensions_equal (a b : Dimensions)
    (shape : [a.length, a.batch, a.dim] = [b.length, b.batch, b.dim]) : a = b := by
  cases a
  cases b
  simp only [List.cons.injEq, and_true] at shape
  rcases shape with ⟨rfl, rfl, rfl⟩
  rfl

theorem success (codec : Codec Word) (graph : Graph) (ops : Operations Word) (api : Services)
    (contracts : Contracts codec ops api) (choices : Choices) (route : Route) (d : Dimensions)
    (saved : Saved) (gy : Tensor) (returned : Tensor × Tensor × Tensor) (before after : State)
    (present : before.saved = some saved)
    (sl : Layout route [d.batch, d.dim] saved.initial)
    (yl : Layout route [d.length, d.batch, d.dim] saved.output)
    (ll : Layout route [d.dim] saved.transition)
    (gyRoute : gy.route = route) (gyShape : gy.shape = [d.length, d.batch, d.dim])
    (gyLive : Resident before gy) (yLive : Resident before saved.output)
    (sLive : Resident before saved.initial) (lLive : Resident before saved.transition)
    (run : Succeeded (Host.backward api (implementation codec graph ops choices) gy) before returned after) :
    Nonempty (Report codec graph ops api choices route d saved gy returned before after) := by
  obtain ⟨path⟩ := backward_path api (implementation codec graph ops choices) gy returned before after run
  obtain ⟨restored⟩ := contracts.restore.success before path.restored path.afterRestore path.restore
  have restoredEq : restored.saved = saved := Option.some.inj (restored.present.symm.trans present)
  have sr : SameSavedStorage saved.initial path.restored.1 := by simpa [restoredEq] using restored.initial
  have yr : SameSavedStorage saved.output path.restored.2.1 := by simpa [restoredEq] using restored.output
  have lr : SameSavedStorage saved.transition path.restored.2.2 := by simpa [restoredEq] using restored.transition
  have rs := restored_layout route _ saved.initial path.restored.1 sl sr
    (restored.initialDense (by simpa [restoredEq] using sl.dense))
  have ry := restored_layout route _ saved.output path.restored.2.1 yl yr
    (restored.outputDense (by simpa [restoredEq] using yl.dense))
  have rl := restored_layout route _ saved.transition path.restored.2.2 ll lr
    (restored.transitionDense (by simpa [restoredEq] using ll.dense))
  have dimEq := dimensions_equal path.d d
    ((shape_success .output path.restored.2.1 path.afterRestore path.afterRestore path.d path.shape).1.symm.trans ry.shape)
  have restoreExt := contracts.restore.extension path.restore
  have norm := normalize_any_success codec ops api contracts .gradOutput gy path.normalized
    path.afterRestore path.afterNormalize (restoreExt.resident gyLive) path.normalize
  have cont := (contracts.contiguous path.normalized).success path.afterNormalize path.contiguousGradient path.afterContiguous path.contiguous
  have contExt := (contracts.contiguous path.normalized).extension path.contiguous
  have gySame := norm.2.1.trans cont.1.logical
  have gradientLayout : Layout route [d.length, d.batch, d.dim] path.contiguousGradient :=
    ⟨gySame.1.1.trans gyRoute, gySame.1.2.trans gyShape, cont.1.dense⟩
  have gradientResolved := cont.1.resolved norm.2.2.1
  have sa := (contracts.empty path.restored.1).success path.afterContiguous path.gs path.afterS path.allocateS
  have xa := (contracts.empty path.contiguousGradient).success path.afterS path.gx path.afterX path.allocateX
  have la := (contracts.empty path.restored.1).success path.afterX path.gl path.afterL path.allocateL
  have se := (contracts.empty path.restored.1).extension path.allocateS
  have xe := (contracts.empty path.contiguousGradient).extension path.allocateX
  have le := (contracts.empty path.restored.1).extension path.allocateL
  have gsLayout := rs.transport sa.same sa.dense
  have gxLayout := gradientLayout.transport xa.same xa.dense
  have glLayout := rs.transport la.same la.dense
  have dispatchRun : Succeeded (Host.backwardDispatch api (implementation codec graph ops choices) d
      path.restored.1 path.restored.2.1 path.restored.2.2 path.contiguousGradient path.gs path.gx path.gl)
      path.afterL () path.launched := by simpa [dimEq] using path.dispatch
  obtain ⟨dispatch⟩ := Dispatch.backward_success codec graph ops api contracts choices route d
    path.restored.1 path.restored.2.1 path.restored.2.2 path.contiguousGradient path.gs path.gx path.gl
    path.afterL path.launched rl.dtype dispatchRun
  let views := backwardViews path.restored.1 path.restored.2.1 path.restored.2.2 path.contiguousGradient path.gs path.gx path.gl
  have sizes : BackwardSizes route d views := backward_sizes route d _ _ _ _ _ _ _ rs ry rl gradientLayout gsLayout gxLayout glLayout
  have toContiguous := norm.1.trans contExt
  have allocations := (se.trans xe).trans le
  have toStart := allocations.trans dispatch.extension
  have setup := ((restoreExt.trans norm.1).trans contExt).trans toStart
  have gradientMeaning := TensorLaws.SameLogical.reframe gySame (restoreExt.frame _ gyLive)
    (toStart.frame _ cont.2)
  have freshFor (backing : Backing) (live : backing ∈ path.afterContiguous.live) : OutputsFreshFor views backing :=
    ⟨xa.fresh backing (se.live backing live), sa.fresh backing live,
      la.fresh backing ((se.trans xe).live backing live)⟩
  have fresh : BackwardAllFresh views :=
    ⟨freshFor _ (toContiguous.resident restored.initialResident),
      freshFor _ (toContiguous.resident restored.transitionResident),
      freshFor _ (toContiguous.resident restored.outputResident), freshFor _ cont.2,
      ⟨(xa.fresh _ sa.resident).symm, la.fresh _ xa.resident, la.fresh _ (xe.resident sa.resident)⟩⟩
  have gsLive : Resident path.launched path.gs := by
    simpa [dispatch.actual] using dispatch.extension.resident (le.resident (xe.resident sa.resident))
  have gxLive : Resident path.launched path.gx := by
    simpa [dispatch.actual] using dispatch.extension.resident (le.resident xa.resident)
  have glLive : Resident path.launched path.gl := by
    simpa [dispatch.actual] using dispatch.extension.resident la.resident
  obtain ⟨sum⟩ := (contracts.sum path.gl).success path.launched returned.2.2 after path.sum
  have sumExt := (contracts.sum path.gl).extension path.sum
  have sumShape := sum.inputShape.symm.trans glLayout.shape
  have sumSizes : sum.batch = d.batch ∧ sum.dim = d.dim := by simpa using sumShape
  have heapEq := congrArg State.heap dispatch.actual
  dsimp only at heapEq
  refine ⟨{
    path := path
    dimensions := dimEq
    restoredInitial := sr
    restoredOutput := yr
    restoredParameter := lr
    dispatch := dispatch
    gradientLayout := gradientLayout
    gradientResolved := gradientResolved
    gradientMeaning := gradientMeaning
    sizes := sizes
    fresh := fresh
    sum := sum
    sumBatch := sumSizes.1
    sumDim := sumSizes.2
    sumRoute := glLayout.dtype
    initialIdentity := path.initialIdentity
    inputIdentity := path.inputIdentity
    callerFrame := ?_
    savedOutputFrame := setup.frame _ yLive
    savedInitialFrame := setup.frame _ sLive
    savedParameterFrame := setup.frame _ lLive
    initialReadback := ?_
    inputReadback := ?_
    transitionReadback := ?_ }⟩
  · intro backing member
    have early := (restoreExt.trans norm.1).trans contExt
    have freshCaller := freshFor backing (early.live backing member)
    have kernelFrame := backwardGrid_caller_frame codec route graph ops d views dispatch.schedule.order dispatch.start.heap
      sizes backing freshCaller.1 freshCaller.2.1 freshCaller.2.2
    have kernelFrame : Frame dispatch.start.heap path.launched.heap backing := by simpa [heapEq] using kernelFrame
    have liveKernel : backing ∈ path.launched.live := by simpa [dispatch.actual] using setup.live backing member
    exact (setup.frame backing member).trans (kernelFrame.trans (sumExt.frame backing liveKernel))
  · intro owner j hj k hk
    rw [path.initialIdentity]
    have bound : route.components * columnIndex owner j + k < path.gs.view.words := by
      have fits := static_fits route path.gs.view d.width owner sizes.gradInitial j hj
      have kc := component_member route k hk
      omega
    rw [Frame.load codec path.gs.view (sumExt.frame _ gsLive) _ bound, heapEq]
    exact backwardGrid_initial codec route graph ops d views dispatch.schedule.order owner dispatch.start.heap sizes fresh
      (SourceActions.owner_in_schedule d dispatch.schedule owner j hj) j hj k hk
  · intro i hi owner j hj k hk
    rw [path.inputIdentity]
    have bound : route.components * timeIndex d (d.length - 1 - i) owner j + k < path.gx.view.words := by
      have fits := time_fits route path.gx.view d (d.length - 1 - i) owner sizes.gradInput (by omega) j hj
      have kc := component_member route k hk
      omega
    rw [Frame.load codec path.gx.view (sumExt.frame _ gxLive) _ bound, heapEq]
    exact backwardGrid_input codec route graph ops d views dispatch.schedule.order owner dispatch.start.heap sizes fresh
      (SourceActions.owner_in_schedule d dispatch.schedule owner j hj) i hi j hj k hk
  · intro q hq
    apply sum_returned codec ops path.gl returned.2.2 path.launched after sum glLive glLayout.dense
      (BatchContributions.value codec route graph ops d views dispatch.start.heap)
    · intro q hq b hb
      rw [glLayout.dtype, sumSizes.2] at hq
      rw [sumSizes.1] at hb
      rw [glLayout.dtype, sumSizes.2, heapEq]
      exact BatchContributions.grid_readback codec route graph ops d views dispatch.schedule dispatch.start.heap sizes fresh q b hq hb
    · simpa [glLayout.dtype, sumSizes.2] using hq

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DiagSsmTriton.BackwardWrapper
