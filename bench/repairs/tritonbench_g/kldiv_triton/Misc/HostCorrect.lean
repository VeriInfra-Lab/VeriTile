import bench.repairs.tritonbench_g.kldiv_triton.Misc.ConditionalCorrect
import bench.repairs.tritonbench_g.kldiv_triton.Proofs.Checked.Frames
import bench.repairs.tritonbench_g.kldiv_triton.Proofs.Checked.Execution
import bench.repairs.tritonbench_g.kldiv_triton.Proofs.RealBridge
import bench.repairs.tritonbench_g.kldiv_triton.Proofs.Launch
import bench.repairs.tritonbench_g.kldiv_triton.Proofs.Context
import bench.repairs.tritonbench_g.kldiv_triton.Proofs.Grid

namespace VeriTile.Bench.Repairs.KldivTriton.Contiguous.Correct
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open Host
noncomputable section
set_option autoImplicit false

/-- Whole forward API in the independent Real interpretation. Ordinary host
failure is an explicit outcome with the complete caller frame. Empty means use
Real's total division convention; this is not a claim about native NaNs. -/
theorem forward {M β : Type} (ops : Ops M ℝ β) (rt : Runtime M)
    (lib : Library ops RealHost.negation RealHost.arithmetic rt)
    (a b : Tensor) (call : ForwardCall) (initial : World M)
    (admit : ForwardContract a b) (ya : MatrixBinding a call.prediction)
    (ta : MatrixBinding b call.target) (domain : ForwardDomain call initial) :
    Ensures (fun out after => RealHost.ForwardTarget ops call initial out after ∧
      Frame ops call.caller initial after ∧ after.device=initial.device)
      (fun _ after => Frame ops call.caller initial after)
      (runForward ops Storage.real rt a b call initial) := by
  have frame := runForward_frame ops RealHost.negation Storage.real RealHost.arithmetic rt lib a b call initial domain
  cases e : runForward ops Storage.real rt a b call initial with
  | error why after => simpa only [e,Result.state] using frame
  | ok out after =>
    obtain ⟨output,rows,cols,result,kept,device⟩ :=
      forward_success ops RealHost.negation Storage.real RealHost.arithmetic rt lib a b call initial after out admit ya ta domain e
    exact ⟨RealHost.forward_target ops call initial after output out rows cols result,kept,device⟩

/-- Independent target for the actual backward API: unscaled gradient written
into the supplied input, then either that exact object for scalar-one or the
actual broadcast product. Caller supplies any forward-reduction normalization. -/
def BackwardTarget {M β : Type} (ops : Ops M ℝ β) (call : BackwardCall) (initial : World M)
    (out : View) (after : World M) : Prop :=
  BackwardReturned ops RealHost.negation RealHost.arithmetic call.input call.gradient.shape
    (fun i => Math.gradient call.flag (logical ops RealHost.negation initial.memory call.target.view i))
    (fun i => logical ops RealHost.negation initial.memory call.gradient i) out after

theorem backward {M β : Type} (ops : Ops M ℝ β) (rt : Runtime M)
    (lib : Library ops RealHost.negation RealHost.arithmetic rt)
    (a b g : Tensor) (call : BackwardCall) (initial : World M)
    (admit : BackwardContract a b g) (ia : MatrixBinding a call.input)
    (ta : MatrixBinding b call.target) (ga : ViewBinding g call.gradient)
    (domain : BackwardDomain call initial) :
    Ensures (fun out after => BackwardTarget ops call initial out after ∧
      Complement ops call.input.view call.caller initial after ∧ after.device=initial.device)
      (fun _ after => Complement ops call.input.view call.caller initial after)
      (runBackward ops Storage.real rt a b g call initial) := by
  have frame := runBackward_complement ops RealHost.negation Storage.real RealHost.arithmetic rt lib a b g call initial domain
  cases e : runBackward ops Storage.real rt a b g call initial with
  | error why after => simpa only [e,Result.state] using frame
  | ok out after =>
    obtain ⟨result,_,device⟩ := backward_success ops RealHost.negation Storage.real RealHost.arithmetic rt lib a b g call initial after out admit ia ta ga domain e
    have values : backwardExpected ops RealHost.negation Storage.real call initial=
        fun i => Math.gradient call.flag (logical ops RealHost.negation initial.memory call.target.view i) := by
      funext i
      exact RealHost.derivative_real _ _
    rw [values] at result
    refine ⟨result,?_,device⟩
    simpa only [e,Result.state] using frame

#specNonCircular BackwardTarget avoiding [backwardExpected,Storage.backwardValue,Backward.sourceRun]
abbrev python_forward_checks_iff_preconditions := python_forward_checks_iff

abbrev python_backward_checks_iff_preconditions := python_backward_checks_iff

#axiomsClean forward
#axiomsClean backward
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.Contiguous.Correct
