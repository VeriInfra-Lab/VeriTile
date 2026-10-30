import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.HostPaths

/-! Actual wrapper dispatch binding. Kernel preparation and complex real-view
calls precede the source action in Python argument order. Their local replies
identify the actual views; source execution determines all written bytes. -/
namespace VeriTile.Bench.Repairs.DiagSsmTriton.Dispatch
open Kernels ByteMemory Execution Protocol Tensors Host LibraryContracts HostPaths
set_option autoImplicit false

def forwardViews (s x transition y : Tensor) : ForwardViews := ⟨s.view, x.view, transition.view, y.view⟩
def backwardViews (s y transition gy gs gx gl : Tensor) : BackwardViews :=
  ⟨s.view, transition.view, y.view, gs.view, gx.view, gl.view, gy.view⟩

structure ForwardResult (codec : Codec Word) (graph : Graph) (ops : Operations Word)
    (route : Route) (d : Dimensions) (views : ForwardViews) (before after : State) where
  start : State
  schedule : SourceActions.Schedule d
  extension : Extension before start
  actual : after = {start with heap := forwardGrid codec route graph ops d views schedule.order start.heap}

structure BackwardResult (codec : Codec Word) (graph : Graph) (ops : Operations Word)
    (route : Route) (d : Dimensions) (views : BackwardViews) (before after : State) where
  start : State
  schedule : SourceActions.Schedule d
  extension : Extension before start
  actual : after = {start with heap := backwardGrid codec route graph ops d views schedule.order start.heap}

theorem forward_success (codec : Codec Word) (graph : Graph) (ops : Operations Word)
    (api : Services) (contracts : Contracts codec ops api) (choices : Choices)
    (route : Route) (d : Dimensions) (s x transition y : Tensor) (before after : State)
    (sr : s.route = route) (xr : x.route = route) (lr : transition.route = route)
    (run : Succeeded (Host.forwardDispatch api (implementation codec graph ops choices) d s x transition y) before () after) :
    Nonempty (ForwardResult codec graph ops route d (forwardViews s x transition y) before after) := by
  cases route with
  | real32 =>
    simp [Host.forwardDispatch, lr] at run
    obtain ⟨u, a, prepareRun, kernelRun⟩ := run
    cases u
    obtain ⟨schedule, actual⟩ := source_forward_success codec graph ops choices .real32 d _ a after kernelRun
    exact ⟨⟨a, schedule, (contracts.prepare Bool.false .real32 x.device).extension prepareRun, actual⟩⟩
  | complex64 =>
    simp [Host.forwardDispatch, lr, Host.requireComplex, sr, xr] at run
    obtain ⟨u, a, prepareRun, sv, b, sRun, xv, c, xRun, yv, e, yRun, lv, f, lRun, kernelRun⟩ := run
    cases u
    have hs := (contracts.realView s).success a sv b sRun
    have hx := (contracts.realView x).success b xv c xRun
    have hy := (contracts.realView y).success c yv e yRun
    have hl := (contracts.realView transition).success e lv f lRun
    subst sv
    subst xv
    subst yv
    subst lv
    obtain ⟨schedule, actual⟩ := source_forward_success codec graph ops choices .complex64 d _ f after kernelRun
    have extension := ((((contracts.prepare Bool.false .complex64 x.device).extension prepareRun).trans
      ((contracts.realView s).extension sRun)).trans ((contracts.realView x).extension xRun)).trans
      ((contracts.realView y).extension yRun)
    exact ⟨⟨f, schedule, extension.trans ((contracts.realView transition).extension lRun), actual⟩⟩

theorem backward_success (codec : Codec Word) (graph : Graph) (ops : Operations Word)
    (api : Services) (contracts : Contracts codec ops api) (choices : Choices)
    (route : Route) (d : Dimensions) (s y transition gy gs gx gl : Tensor) (before after : State)
    (lr : transition.route = route)
    (run : Succeeded (Host.backwardDispatch api (implementation codec graph ops choices) d s y transition gy gs gx gl) before () after) :
    Nonempty (BackwardResult codec graph ops route d (backwardViews s y transition gy gs gx gl) before after) := by
  cases route with
  | real32 =>
    simp [Host.backwardDispatch, lr] at run
    obtain ⟨u, a, prepareRun, kernelRun⟩ := run
    cases u
    obtain ⟨schedule, actual⟩ := source_backward_success codec graph ops choices .real32 d _ a after kernelRun
    exact ⟨⟨a, schedule, (contracts.prepare Bool.true .real32 y.device).extension prepareRun, actual⟩⟩
  | complex64 =>
    simp [Host.backwardDispatch, lr] at run
    obtain ⟨u, a, prepareRun, sv, b, sRun, lv, c, lRun, yv, e, yRun,
      gsv, f, gsRun, gxv, g, gxRun, glv, h, glRun, gyv, j, gyRun, kernelRun⟩ := run
    cases u
    have hs := (contracts.realView s).success a sv b sRun
    have hl := (contracts.realView transition).success b lv c lRun
    have hy := (contracts.realView y).success c yv e yRun
    have hgs := (contracts.realView gs).success e gsv f gsRun
    have hgx := (contracts.realView gx).success f gxv g gxRun
    have hgl := (contracts.realView gl).success g glv h glRun
    have hgy := (contracts.realView gy).success h gyv j gyRun
    subst sv
    subst lv
    subst yv
    subst gsv
    subst gxv
    subst glv
    subst gyv
    obtain ⟨schedule, actual⟩ := source_backward_success codec graph ops choices .complex64 d _ j after kernelRun
    have e0 := (contracts.prepare Bool.true .complex64 y.device).extension prepareRun
    have e1 := e0.trans ((contracts.realView s).extension sRun)
    have e2 := e1.trans ((contracts.realView transition).extension lRun)
    have e3 := e2.trans ((contracts.realView y).extension yRun)
    have e4 := e3.trans ((contracts.realView gs).extension gsRun)
    have e5 := e4.trans ((contracts.realView gx).extension gxRun)
    have e6 := e5.trans ((contracts.realView gl).extension glRun)
    have extension := e6.trans ((contracts.realView gy).extension gyRun)
    exact ⟨⟨j, schedule, extension, actual⟩⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DiagSsmTriton.Dispatch
