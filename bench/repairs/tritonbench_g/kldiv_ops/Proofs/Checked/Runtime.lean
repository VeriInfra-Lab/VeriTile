import bench.repairs.tritonbench_g.kldiv_ops.Proofs.Checked.Source
import bench.repairs.tritonbench_g.kldiv_triton.Proofs.Checked.Host

namespace VeriTile.Bench.Repairs.KldivOps.Contiguous
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open KldivTriton.Host (World Result ForwardCall BackwardCall Arithmetic Fresh Stable Summed Frame Complement)
noncomputable section
open Classical
set_option autoImplicit false

def reduction : KldivTriton.Math.Mode → Host.Reduction
  | .none => .none | .sum => .sum | .mean => .mean | .batchmean => .batchmean

def outputShape (call : ForwardCall) : List Nat :=
  (reduction call.mode).outputShape call.prediction.rows call.prediction.cols

/-- Ordinary API effects; setup contains no kernel/output callback. The sum
mode retains the source's actual sum(dim=0) versus sum() call. -/
structure Runtime (M : Type) where
  base : KldivTriton.Host.Runtime M
  zeros : List Nat → Nat → World M → Result M View
  setupForward : ForwardCall → View → World M → Result M Unit
  setupBackward : BackwardCall → World M → Result M Unit
  sum : (dim0 : Bool) → (v : View) → World M → Result M (Summed v.numel)

structure Library {M α β : Type} (ops : Ops M α β) (neg : Negation α)
    (arith : Arithmetic α) (rt : Runtime M) : Prop where
  base : KldivTriton.Host.Library ops neg arith rt.base
  zeros : ∀ shape owner before out after, rt.zeros shape owner before = .ok out after →
    out.shape = shape ∧ Dense out ∧ Fresh out owner before after ∧ after.device = before.device ∧
    out.storage.baseByte % 4 = 0
  zeros_stable : ∀ shape owner before, Stable ops before (rt.zeros shape owner before).state
  setupForward : ∀ call out before after, rt.setupForward call out before = .ok () after → after = before
  forward_stable : ∀ call out before, Stable ops before (rt.setupForward call out before).state
  setupBackward : ∀ call before after, rt.setupBackward call before = .ok () after → after = before
  backward_stable : ∀ call before, Stable ops before (rt.setupBackward call before).state
  sum : ∀ dim0 input before out after, rt.sum dim0 input before = .ok out after →
    out.view.shape = [] ∧ Fresh out.view input.storage.owner before after ∧
    logical ops neg after.memory out.view 0 =
      KldivTriton.HostReduction.eval arith.add arith.zero
        (fun i => logical ops neg before.memory input i.val) out.plan.tree
  sum_stable : ∀ dim0 input before, Stable ops before (rt.sum dim0 input before).state

def sumAndDivide {M : Type} (rt : Runtime M) (dim0 : Bool) (input : View)
    (denom : Option Nat) (s : World M) : Result M View :=
  (rt.sum dim0 input s).bind fun summed after =>
    match denom with
    | none => .ok summed.view after
    | some n => rt.base.perform (.divide summed.view n) after

def finishForward {M : Type} (rt : Runtime M) (call : ForwardCall) (output : View)
    (s : World M) : Result M View :=
  if call.mode = .none then .ok output s else
    sumAndDivide rt (decide (call.mode = .sum)) output
      (KldivTriton.Host.denominator call.mode call.prediction.rows call.prediction.cols) s

def forwardBody {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (a : Adapter α κ)
    (rt : Runtime M) (call : ForwardCall) (output : View) (eps : α) (s : World M) : Result M View :=
  (rt.setupForward call output s).bind fun _ before =>
    finishForward rt call output {before with memory := forwardKernel ops a before.memory call.prediction.view call.target.view output call.prediction.rows call.prediction.cols call.flag (reduction call.mode) eps}

def runForward {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (a : Adapter α κ)
    (rt : Runtime M) (p t : Tensor) (call : ForwardCall) (eps : α) (s : World M) : Result M View :=
  match checkForward (.tensor p) (.tensor t) (some call.flag)
      (KldivTriton.Contiguous.modeName call.mode) Bool.true with
  | .error reason => .error reason s
  | .ok () =>
    (rt.zeros (outputShape call) call.prediction.view.storage.owner s).bind fun output allocated =>
      (rt.base.perform (.enter call.prediction.view.storage.owner) allocated).bind fun token inside =>
        KldivTriton.Host.finallyDevice rt.base token (forwardBody ops a rt call output eps inside)

def backwardBody {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (a : Adapter α κ)
    (rt : Runtime M) (call : BackwardCall) (s : World M) : Result M View :=
  (rt.setupBackward call s).bind fun _ before =>
    KldivTriton.Host.finishBackward rt.base call call.gradient
      {before with memory := backwardKernel ops a before.memory call.target.view call.input.view call.input.rows call.input.cols call.flag}

def runBackward {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (a : Adapter α κ)
    (rt : Runtime M) (d t g : Tensor) (call : BackwardCall) (s : World M) : Result M View :=
  match checkBackward (.tensor d) (.tensor t) (.tensor g) (some call.flag) with
  | .error reason => .error reason s
  | .ok () =>
    (rt.base.perform (.enter call.input.view.storage.owner) s).bind fun token inside =>
      KldivTriton.Host.finallyDevice rt.base token (backwardBody ops a rt call inside)

end
end VeriTile.Bench.Repairs.KldivOps.Contiguous
