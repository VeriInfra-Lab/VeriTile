import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.Tensors
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.Reduction

/-! Exact selected-precision wrapper dataflow. Local Torch services may fail;
the continuation is entered only after each preceding operation succeeds.
Kernel actions are supplied by the source-execution adapter, never by a
desired-result oracle. This module defines ordering and return identities; the
following connection modules discharge each action's value/frame contract. -/
set_option autoImplicit false
namespace VeriTile.Bench.Repairs.DiagSsmTriton.Host
open Kernels ByteMemory Execution Tensors Protocol

structure State where
  heap : Heap
  saved : Option Saved
  live : List Backing
  version : Nat → Nat

local instance : Monad (Program State) where
  pure := Protocol.pure
  bind := Protocol.bind

@[simp] theorem program_bind {α β : Type} (first : Program State α) (next : α → Program State β) :
    (first >>= next) = Protocol.bind first next := rfl

structure Services where
  contiguous : Tensor → Program State Tensor
  zerosLike : Tensor → Program State Tensor
  emptyLike : Tensor → Program State Tensor
  realView : Tensor → Program State View
  prepare : Bool → Route → VeriTile.Torch.Device → Program State Unit
  save : Tensor → Tensor → Tensor → Program State Unit
  restore : Program State (Tensor × Tensor × Tensor)
  sumDim0 : Tensor → Program State Tensor

/-- These actions will be instantiated with the byte-memory interpreters and
their ordinary dispatch/failure controls; they are kept distinct from Torch
services so no tensor-library contract contains task-specific output laws. -/
structure KernelActions where
  forward : Route → Dimensions → ForwardViews → Program State Unit
  backward : Route → Dimensions → BackwardViews → Program State Unit

def isComplex (route : Route) : Bool := decide (route = .complex64)

def requireContiguous (role : Role) (tensor : Tensor) : Program State Unit :=
  call (.contiguousCheck role)
    (if tensor.contiguous then Protocol.pure () else raise .assertion)

def requireComplex (role : Role) (tensor : Tensor) : Program State Unit :=
  call (.complexCheck role)
    (if tensor.route = .complex64 then Protocol.pure () else raise .dtype)

def shape3 (role : Role) (tensor : Tensor) : Program State Dimensions :=
  call (.shape role) (match tensor.shape with
    | [length, batch, dim] => Protocol.pure ⟨length, batch, dim⟩
    | _ => raise .shape)

/-- Admission checks the logical flags; it never materializes a copy. The
historical helper name is retained so the grid/host proofs can be reused. -/
def normalize (_api : Services) (role : Role) (tensor : Tensor) : Program State Tensor :=
  call (.resolvedCheck role)
    (if tensor.negative || tensor.conjugated then raise .dtype else Protocol.pure tensor)

def forwardDispatch (api : Services) (kernels : KernelActions) (d : Dimensions)
    (s x transition y : Tensor) : Program State Unit := do
  let _ ← call .dtype (Protocol.pure ())
  match transition.route with
  | .complex64 =>
    let _ ← requireComplex .initial s
    let _ ← requireComplex .input x
    let _ ← call (.prepareKernel Bool.false Bool.true) (api.prepare Bool.false .complex64 x.device)
    -- Preserve the source's complex-forward argument order: s,x,y,Lambda.
    let sv ← call (.realView .initial) (api.realView s)
    let xv ← call (.realView .input) (api.realView x)
    let yv ← call (.realView .output) (api.realView y)
    let lv ← call (.realView .transition) (api.realView transition)
    call (.launch Bool.false Bool.true) (kernels.forward .complex64 d ⟨sv, xv, lv, yv⟩)
  | .real32 =>
    let _ ← call (.prepareKernel Bool.false Bool.false) (api.prepare Bool.false .real32 x.device)
    call (.launch Bool.false Bool.false)
      (kernels.forward .real32 d ⟨s.view, x.view, transition.view, y.view⟩)

def forwardFinish (api : Services) (kernels : KernelActions) (d : Dimensions)
    (s x transition y : Tensor) : Program State Tensor := do
  let _ ← forwardDispatch api kernels d s x transition y
  let _ ← call .save (api.save s y transition)
  Protocol.pure y

def forward (api : Services) (kernels : KernelActions) (s x transition : Tensor) : Program State Tensor := do
  -- Python's chained and assertion short-circuits left-to-right.
  let _ ← requireContiguous .initial s
  let _ ← requireContiguous .input x
  let _ ← requireContiguous .transition transition
  let d ← shape3 .input x
  -- Python n=batch*dim is unbounded host integer arithmetic. It is captured by
  -- the grid closure and does not launch anything before the later call.
  let s ← normalize api .initial s
  let x ← normalize api .input x
  let transition ← normalize api .transition transition
  let y ← call (.zerosLike .input) (api.zerosLike x)
  forwardFinish api kernels d s x transition y

def backwardDispatch (api : Services) (kernels : KernelActions) (d : Dimensions)
    (s y transition gy gs gx gl : Tensor) : Program State Unit := do
  let _ ← call .dtype (Protocol.pure ())
  match transition.route with
  | .complex64 =>
    let _ ← call (.prepareKernel Bool.true Bool.true) (api.prepare Bool.true .complex64 y.device)
    -- Preserve source order s,Lambda,y,gs,gx,gl,gy, distinct from forward.
    let sv ← call (.realView .initial) (api.realView s)
    let lv ← call (.realView .transition) (api.realView transition)
    let yv ← call (.realView .output) (api.realView y)
    let gsv ← call (.realView .gradInitial) (api.realView gs)
    let gxv ← call (.realView .gradInput) (api.realView gx)
    let glv ← call (.realView .gradTransition) (api.realView gl)
    let gyv ← call (.realView .gradOutput) (api.realView gy)
    call (.launch Bool.true Bool.true) (kernels.backward .complex64 d ⟨sv, lv, yv, gsv, gxv, glv, gyv⟩)
  | .real32 =>
    let _ ← call (.prepareKernel Bool.true Bool.false) (api.prepare Bool.true .real32 y.device)
    call (.launch Bool.true Bool.false)
      (kernels.backward .real32 d ⟨s.view, transition.view, y.view, gs.view, gx.view, gl.view, gy.view⟩)

def backwardFinish (api : Services) (kernels : KernelActions) (d : Dimensions)
    (s y transition gy gs gx gl : Tensor) : Program State (Tensor × Tensor × Tensor) := do
  let _ ← backwardDispatch api kernels d s y transition gy gs gx gl
  let reduced ← call .sumDim0 (api.sumDim0 gl)
  Protocol.pure (gs, gx, reduced)

def backward (api : Services) (kernels : KernelActions) (gy : Tensor) : Program State (Tensor × Tensor × Tensor) := do
  let restored ← call .restore api.restore
  let s := restored.1
  let y := restored.2.1
  let transition := restored.2.2
  let d ← shape3 .output y
  let gy ← normalize api .gradOutput gy
  let gy ← call (.contiguous .gradOutput) (api.contiguous gy)
  let gs ← call (.emptyLike .initial) (api.emptyLike s)
  let gx ← call (.emptyLike .gradOutput) (api.emptyLike gy)
  let gl ← call (.emptyLike .initial) (api.emptyLike s)
  backwardFinish api kernels d s y transition gy gs gx gl

theorem forward_return_identity (api : Services) (kernels : KernelActions) (d : Dimensions)
    (s x transition y : Tensor) (before afterKernel afterSave : State) (kernelTrace saveTrace : List Call)
    (kernel : forwardDispatch api kernels d s x transition y before = .success () afterKernel kernelTrace)
    (save : api.save s y transition afterKernel = .success () afterSave saveTrace) :
    forwardFinish api kernels d s x transition y before =
      .success y afterSave (kernelTrace ++ .save :: saveTrace) := by
  simp [forwardFinish, Protocol.bind, call, Protocol.pure, kernel, save, Result.prepend]

theorem backward_return_identity (api : Services) (kernels : KernelActions) (d : Dimensions)
    (s y transition gy gs gx gl reduced : Tensor) (before afterKernel afterSum : State)
    (kernelTrace sumTrace : List Call)
    (kernel : backwardDispatch api kernels d s y transition gy gs gx gl before = .success () afterKernel kernelTrace)
    (sum : api.sumDim0 gl afterKernel = .success reduced afterSum sumTrace) :
    backwardFinish api kernels d s y transition gy gs gx gl before =
      .success (gs, gx, reduced) afterSum (kernelTrace ++ .sumDim0 :: sumTrace) := by
  simp [backwardFinish, Protocol.bind, call, Protocol.pure, kernel, sum, Result.prepend]

theorem first_contiguity_failure (api : Services) (kernels : KernelActions) (s x transition : Tensor)
    (state : State) (bad : s.contiguous = Bool.false) :
    forward api kernels s x transition state =
      .failure .assertion state [.contiguousCheck .initial] := by
  simp [forward, requireContiguous, bad, call, Protocol.bind, Result.prepend, raise]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DiagSsmTriton.Host
