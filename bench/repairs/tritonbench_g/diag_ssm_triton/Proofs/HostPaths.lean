import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.LibraryContracts

/-! Success inversion of the actual host program. These witnesses are derived
from the returned result, not assumed execution or output laws. Every stored
field records one actual, ordered local API call and its intermediate state. -/
namespace VeriTile.Bench.Repairs.DiagSsmTriton.HostPaths
open Kernels ByteMemory Execution Protocol Tensors Host LibraryContracts
set_option autoImplicit false

structure Choices where
  forward : ∀ (route : Route) (d : Dimensions) (views : ForwardViews), State → SourceActions.ForwardDecision d
  backward : ∀ (route : Route) (d : Dimensions) (views : BackwardViews), State → SourceActions.BackwardDecision d

def implementation (codec : Codec Word) (graph : Graph) (ops : Operations Word) (choices : Choices) : KernelActions :=
  SourceActions.kernels codec graph ops choices.forward choices.backward

structure ForwardPath (api : Services) (kernels : KernelActions) (d : Dimensions)
    (s x transition returned : Tensor) (before after : State) where
  initial : Tensor
  input : Tensor
  parameter : Tensor
  afterInitial : State
  afterInput : State
  afterParameter : State
  allocated : State
  launched : State
  normalizeInitial : Succeeded (Host.normalize api .initial s) before initial afterInitial
  normalizeInput : Succeeded (Host.normalize api .input x) afterInitial input afterInput
  normalizeParameter : Succeeded (Host.normalize api .transition transition) afterInput parameter afterParameter
  allocate : Succeeded (api.zerosLike input) afterParameter returned allocated
  dispatch : Succeeded (Host.forwardDispatch api kernels d initial input parameter returned) allocated () launched
  save : Succeeded (api.save initial returned parameter) launched () after

theorem forward_path (api : Services) (kernels : KernelActions) (d : Dimensions)
    (s x transition returned : Tensor) (before after : State)
    (sd : Dense s) (xd : Dense x) (ld : Dense transition)
    (shape : x.shape = [d.length, d.batch, d.dim])
    (run : Succeeded (Host.forward api kernels s x transition) before returned after) :
    Nonempty (ForwardPath api kernels d s x transition returned before after) := by
  simp only [Host.forward, Host.forwardFinish, Host.program_bind, succeeded_bind, succeeded_call] at run
  simp [Host.requireContiguous, sd.1, xd.1, ld.1, Host.shape3, shape] at run
  obtain ⟨ns, a, hs, nx, b, hx, nl, c, hl, p, hy, u, q, hkernel, v, hsave⟩ := run
  cases u
  cases v
  exact ⟨⟨ns, nx, nl, a, b, c, p, q, hs, hx, hl, hy, hkernel, hsave⟩⟩

structure BackwardPath (api : Services) (kernels : KernelActions)
    (gy : Tensor) (returned : Tensor × Tensor × Tensor) (before after : State) where
  restored : Tensor × Tensor × Tensor
  d : Dimensions
  normalized : Tensor
  contiguousGradient : Tensor
  gs : Tensor
  gx : Tensor
  gl : Tensor
  afterRestore : State
  afterNormalize : State
  afterContiguous : State
  afterS : State
  afterX : State
  afterL : State
  launched : State
  restore : Succeeded api.restore before restored afterRestore
  shape : Succeeded (Host.shape3 .output restored.2.1) afterRestore d afterRestore
  normalize : Succeeded (Host.normalize api .gradOutput gy) afterRestore normalized afterNormalize
  contiguous : Succeeded (api.contiguous normalized) afterNormalize contiguousGradient afterContiguous
  allocateS : Succeeded (api.emptyLike restored.1) afterContiguous gs afterS
  allocateX : Succeeded (api.emptyLike contiguousGradient) afterS gx afterX
  allocateL : Succeeded (api.emptyLike restored.1) afterX gl afterL
  dispatch : Succeeded (Host.backwardDispatch api kernels d restored.1 restored.2.1 restored.2.2
    contiguousGradient gs gx gl) afterL () launched
  sum : Succeeded (api.sumDim0 gl) launched returned.2.2 after
  initialIdentity : returned.1 = gs
  inputIdentity : returned.2.1 = gx

theorem shape_success (role : Role) (tensor : Tensor) (before after : State) (d : Dimensions)
    (run : Succeeded (Host.shape3 role tensor) before d after) :
    tensor.shape = [d.length, d.batch, d.dim] ∧ after = before := by
  unfold Host.shape3 at run
  simp only [succeeded_call] at run
  split at run
  · rename_i l b dim shape
    rcases succeeded_pure _ _ _ _ |>.mp run with ⟨hd, hs⟩
    subst d
    exact ⟨shape, hs⟩
  · simp [Succeeded, raise] at run

theorem backward_path (api : Services) (kernels : KernelActions) (gy : Tensor)
    (returned : Tensor × Tensor × Tensor) (before after : State)
    (run : Succeeded (Host.backward api kernels gy) before returned after) :
    Nonempty (BackwardPath api kernels gy returned before after) := by
  simp only [Host.backward, Host.backwardFinish, Host.program_bind, succeeded_bind, succeeded_call] at run
  obtain ⟨restored, a, hr, d, shapeState, hshape, ng, b, hn, cg, c, hc,
    gs, e, hs, gx, f, hx, gl, g, hl, u, h, hd, reduced, r, hsum, hpure⟩ := run
  have stateEq := (shape_success .output restored.2.1 a shapeState d hshape).2
  subst shapeState
  have resultEq := (succeeded_pure _ _ _ _).mp hpure
  rcases resultEq with ⟨heq, hstate⟩
  subst r
  cases u
  subst returned
  exact ⟨⟨restored, d, ng, cg, gs, gx, gl, a, b, c, e, f, g, h,
    hr, hshape, hn, hc, hs, hx, hl, hd, hsum, rfl, rfl⟩⟩

theorem source_forward_success (codec : Codec Word) (graph : Graph) (ops : Operations Word)
    (choices : Choices) (route : Route) (d : Dimensions) (views : ForwardViews) (before after : State)
    (run : Succeeded ((implementation codec graph ops choices).forward route d views) before () after) :
    ∃ schedule : SourceActions.Schedule d,
      after = {before with heap := forwardGrid codec route graph ops d views schedule.order before.heap} := by
  cases hc : choices.forward route d views before with
  | complete schedule =>
    refine ⟨schedule, ?_⟩
    simp [Succeeded, implementation, SourceActions.kernels, SourceActions.forward, hc] at run
    exact run.symm
  | failedSetup error => simp [Succeeded, implementation, SourceActions.kernels, SourceActions.forward, hc] at run
  | stoppedDuringSource point error => simp [Succeeded, implementation, SourceActions.kernels, SourceActions.forward, hc] at run

theorem source_backward_success (codec : Codec Word) (graph : Graph) (ops : Operations Word)
    (choices : Choices) (route : Route) (d : Dimensions) (views : BackwardViews) (before after : State)
    (run : Succeeded ((implementation codec graph ops choices).backward route d views) before () after) :
    ∃ schedule : SourceActions.Schedule d,
      after = {before with heap := backwardGrid codec route graph ops d views schedule.order before.heap} := by
  cases hc : choices.backward route d views before with
  | complete schedule =>
    refine ⟨schedule, ?_⟩
    simp [Succeeded, implementation, SourceActions.kernels, SourceActions.backward, hc] at run
    exact run.symm
  | failedSetup error => simp [Succeeded, implementation, SourceActions.kernels, SourceActions.backward, hc] at run
  | stoppedDuringSource point error => simp [Succeeded, implementation, SourceActions.kernels, SourceActions.backward, hc] at run

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DiagSsmTriton.HostPaths
